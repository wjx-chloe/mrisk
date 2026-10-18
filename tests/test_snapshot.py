import copy
import io
import json
import sys
import urllib.error
import zipfile

import pandas as pd
import pytest

from mrisk.data import snapshot as snap
from mrisk.data.sources import SnapshotSource
from mrisk.db import schema as s
from mrisk.db.compare import compare_databases, identical
from mrisk.db.repo import get_engine, init_db, read_frame
from mrisk.jobs import daily as jobs
from mrisk.jobs import etl
from mrisk.jobs.demo import STEPS, run_demo

from fake_world import LAST_BACKFILL, make_env


def _raw(dates, close):
    return pd.DataFrame({"date": dates, "close": close, "volume": [100.0] * len(dates)})


# ---- merging date ranges ----------------------------------------------------------------------------------------
def test_merge_daily_unions_identical_overlap():
    a = _raw(["2025-01-02", "2025-01-03", "2025-01-06"], [10.0, 10.5, 11.0])
    b = _raw(["2025-01-03", "2025-01-06", "2025-01-07"], [10.5, 11.0, 11.2])
    merged, conflicts = snap.merge_daily({"r1": a, "r2": b})
    assert conflicts.empty
    assert merged["date"].tolist() == ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    assert merged["close"].tolist() == [10.0, 10.5, 11.0, 11.2]


def test_merge_daily_reports_every_disagreeing_value():
    a = _raw(["2025-01-02", "2025-01-03"], [10.0, 10.5])
    b = _raw(["2025-01-03", "2025-01-06"], [10.6, 11.0])
    b.loc[0, "volume"] = 200.0
    _, conflicts = snap.merge_daily({"r1": a, "r2": b})
    assert conflicts.to_dict("records") == [
        {"date": "2025-01-03", "column": "close", "range": "r1", "value": 10.5},
        {"date": "2025-01-03", "column": "close", "range": "r2", "value": 10.6},
        {"date": "2025-01-03", "column": "volume", "range": "r1", "value": 100.0},
        {"date": "2025-01-03", "column": "volume", "range": "r2", "value": 200.0},
    ]


def test_merge_daily_nan_equals_nan():
    a = _raw(["2025-01-02"], [float("nan")])
    _, conflicts = snap.merge_daily({"r1": a, "r2": a.copy()})
    assert conflicts.empty


def test_coverage_gaps_missing_date_and_other_close():
    db = pd.DataFrame({
        "trade_date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]),
        "close": [10.0, 10.5, 10.5, 11.0],
        "is_suspended": [False, False, True, False],  # 01-06 inferred suspended: no raw row expected
    })
    raw = _raw(["2025-01-02", "2025-01-03"], [10.0, 10.4])
    gaps = snap.coverage_gaps(db, raw)
    assert gaps["trade_date"].tolist() == [pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-07")]
    assert gaps["raw_close"].isna().tolist() == [False, True]


# ---- export / pack / extract ------------------------------------------------------------------------------------
def _write_cache(raw_dir, close_b=10.5):
    raw_dir.mkdir()
    for key in ["calendar", "list_sh_main", "list_sh_star", "list_sz", "list_bj", "delist_sh", "delist_sz"]:
        pd.DataFrame({"x": ["1"]}).to_parquet(raw_dir / f"{key}_20250110.parquet", index=False)
    pd.DataFrame({"行业代码": ["801010.SI"], "行业名称": ["农林牧渔"]}).to_parquet(raw_dir / "sw_l1_20250110.parquet", index=False)
    pd.DataFrame({"证券代码": ["600000"]}).to_parquet(raw_dir / "sw_comp_801010_20250110.parquet", index=False)
    pd.DataFrame({"date": ["1900-01-01"], "hfq_factor": ["1.0"]}).to_parquet(raw_dir / "hfq_SH600000_20250110.parquet", index=False)
    _raw(["2025-01-02", "2025-01-03"], [10.0, 10.5]).to_parquet(raw_dir / "daily_SH600000_19901219_20250103.parquet", index=False)
    _raw(["2025-01-03", "2025-01-06"], [close_b, 11.0]).to_parquet(raw_dir / "daily_SH600000_20250103_20250106.parquet", index=False)


INST = pd.DataFrame({"instrument_id": ["SH600000"], "exchange": ["SH"], "symbol": ["600000"]})
PX = pd.DataFrame({
    "trade_date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]), "instrument_id": "SH600000",
    "close": [10.0, 10.5, 11.0], "is_suspended": False,
})


def test_export_packs_merged_ranges_with_manifest(tmp_path):
    _write_cache(tmp_path / "raw")
    e = snap.export_snapshot(tmp_path / "raw", INST, PX, tmp_path / "out")
    assert e.zip_path.name == "mrisk-snapshot-20250106.zip"
    assert e.sha256 == snap.sha256_file(e.zip_path)
    assert e.manifest["price_end"] == "2025-01-06" and e.manifest["fetched"] == ["2025-01-10"]
    assert e.manifest["files"]["daily_SH600000.parquet"]["rows"] == 3
    assert "sw_comp_801010.parquet" in e.manifest["files"] and "calendar.parquet" in e.manifest["files"]
    again = snap.export_snapshot(tmp_path / "raw", INST, PX, tmp_path / "out2")
    assert again.sha256 == e.sha256  # deterministic zip


def test_export_refuses_conflicting_overlap_and_lists_it(tmp_path):
    _write_cache(tmp_path / "raw", close_b=10.6)
    with pytest.raises(snap.SnapshotError, match="overlapping dates disagree"):
        snap.export_snapshot(tmp_path / "raw", INST, PX, tmp_path / "out")
    rep = pd.read_csv(tmp_path / "out" / "snapshot_conflicts.csv")
    assert set(rep["value"]) == {10.5, 10.6} and set(rep["file"]) == {"daily_SH600000"}
    assert not list((tmp_path / "out").glob("*.zip"))


def test_export_refuses_coverage_gap(tmp_path):
    _write_cache(tmp_path / "raw")
    px = pd.concat([PX, PX.iloc[[0]].assign(trade_date=pd.Timestamp("2025-01-07"))])
    with pytest.raises(snap.SnapshotError, match="not reproduced"):
        snap.export_snapshot(tmp_path / "raw", INST, px, tmp_path / "out")
    assert pd.read_csv(tmp_path / "out" / "snapshot_coverage_gaps.csv")["trade_date"].tolist() == ["2025-01-07"]


def test_export_refuses_missing_files(tmp_path):
    _write_cache(tmp_path / "raw")
    (tmp_path / "raw" / "hfq_SH600000_20250110.parquet").unlink()
    with pytest.raises(snap.SnapshotError, match="hfq_SH600000"):
        snap.export_snapshot(tmp_path / "raw", INST, PX, tmp_path / "out")


def test_extract_verifies_and_refuses_unsafe_or_tampered(tmp_path):
    _write_cache(tmp_path / "raw")
    e = snap.export_snapshot(tmp_path / "raw", INST, PX, tmp_path / "out")
    m = snap.extract(e.zip_path, tmp_path / "snap")
    assert set(p.name for p in (tmp_path / "snap").iterdir()) == set(m["files"]) | {"manifest.json"}
    assert SnapshotSource(tmp_path / "snap").daily_raw("SH", "600000", "2025-01-03", "2025-01-06")["close"].tolist() == [10.5, 11.0]

    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("manifest.json", "{}")
        z.writestr("../escape.parquet", b"x")
    with pytest.raises(snap.SnapshotError, match="unsafe"):
        snap.extract(evil, tmp_path / "snap2")

    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(e.zip_path) as src, zipfile.ZipFile(tampered, "w") as dst:
        for n in src.namelist():
            dst.writestr(n, b"changed" if n == "calendar.parquet" else src.read(n))
    with pytest.raises(snap.SnapshotError, match="sha256 mismatch"):
        snap.extract(tampered, tmp_path / "snap3")


def test_snapshot_source_without_manifest_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch-snapshot"):
        SnapshotSource(tmp_path)


# ---- fetch ------------------------------------------------------------------------------------------------------
class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


@pytest.fixture
def release(tmp_path):
    _write_cache(tmp_path / "raw")
    e = snap.export_snapshot(tmp_path / "raw", INST, PX, tmp_path / "out")
    return {"repo": "o/r", "tag": "snapshot-20250106", "asset": e.zip_path.name, "sha256": e.sha256}, e.zip_path.read_bytes()


def test_fetch_anonymous_downloads_verifies_extracts(tmp_path, release, monkeypatch):
    rel, payload = release
    seen = []
    monkeypatch.setattr(snap, "_open", lambda req, timeout: seen.append(req) or _Resp(payload))
    m = snap.fetch_snapshot(rel, tmp_path / "rel", tmp_path / "snap")
    assert m["price_end"] == "2025-01-06" and (tmp_path / "snap" / "manifest.json").exists()
    assert [r.full_url for r in seen] == ["https://github.com/o/r/releases/download/snapshot-20250106/mrisk-snapshot-20250106.zip"]
    assert not seen[0].has_header("Authorization")
    snap.fetch_snapshot(rel, tmp_path / "rel", tmp_path / "snap")  # verified copy on disk: no second download
    assert len(seen) == 1


def test_fetch_with_token_uses_api_and_never_redirects_the_token(tmp_path, release, monkeypatch):
    rel, payload = release
    seen = []
    listing = json.dumps({"assets": [{"name": rel["asset"], "url": "https://api.github.com/repos/o/r/releases/assets/7"}]}).encode()
    monkeypatch.setattr(snap, "_open", lambda req, timeout: seen.append(req) or _Resp(listing if len(seen) == 1 else payload))
    snap.fetch_snapshot(rel, tmp_path / "rel", tmp_path / "snap", token="t0k")
    assert [r.full_url for r in seen] == [
        "https://api.github.com/repos/o/r/releases/tags/snapshot-20250106", "https://api.github.com/repos/o/r/releases/assets/7",
    ]
    for r in seen:  # unredirected: urllib does not copy it to the storage host the API redirects to
        assert r.unredirected_hdrs == {"Authorization": "Bearer t0k"} and "Authorization" not in r.headers
    assert seen[1].get_header("Accept") == "application/octet-stream"


def test_fetch_refuses_wrong_sha256(tmp_path, release, monkeypatch):
    rel, payload = release
    monkeypatch.setattr(snap, "_open", lambda req, timeout: _Resp(payload + b"x"))
    with pytest.raises(snap.SnapshotError, match="sha256 of the download"):
        snap.fetch_snapshot(rel, tmp_path / "rel", tmp_path / "snap")
    assert not (tmp_path / "snap").exists() and (tmp_path / "rel" / (rel["asset"] + ".bad")).exists()


def test_fetch_http_error_is_a_clear_error_with_private_hint(tmp_path, release, monkeypatch):
    rel, _ = release
    def not_found(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)
    monkeypatch.setattr(snap, "_open", not_found)
    with pytest.raises(snap.SnapshotError, match="HTTP 404 Not Found; the repository may be private: set GITHUB_TOKEN"):
        snap.fetch_snapshot(rel, tmp_path / "rel", tmp_path / "snap")
    with pytest.raises(snap.SnapshotError, match=r"HTTP 404 Not Found$"):
        snap.fetch_snapshot(rel, tmp_path / "rel", tmp_path / "snap", token="t0k")
    assert not list((tmp_path / "rel").iterdir())


# ---- export -> replay, end to end on the fake world ------------------------------------------------------------
def test_replay_from_exported_snapshot_is_bit_identical_and_offline(tmp_path, monkeypatch):
    """The fake world through the real AkshareSource (fake akshare module, raw cache in tmp/live/raw, a second date
    range for two stocks), then export -> extract -> mrisk demo with akshare unimportable: every table matches."""
    live = tmp_path / "live"
    live.mkdir()
    engine, cfg = make_env(live, monkeypatch, source="akshare")
    src = etl.make_source(cfg)
    for ex, sym in [("SH", "600000"), ("SZ", "000001")]:
        src.daily_raw(ex, sym, pd.Timestamp("2025-04-01"), LAST_BACKFILL)
    jobs.backfill(engine, cfg, end=LAST_BACKFILL)
    for _, run in STEPS:
        run(engine, cfg, cfg["portfolio"]["start"], LAST_BACKFILL)
    assert len(list((live / "raw").glob("daily_SH600000_*"))) == 2

    inst = read_frame(engine, s.instrument, date_col="list_date")
    px = read_frame(engine, s.daily_price)
    e = snap.export_snapshot(live / "raw", inst, px, tmp_path / "release")
    snap.extract(e.zip_path, tmp_path / "snapshot")

    monkeypatch.setitem(sys.modules, "akshare", None)  # any import of akshare now raises ImportError
    demo_cfg = copy.deepcopy(cfg)
    demo_cfg["data"].update(source="snapshot", snapshot_dir=str(tmp_path / "snapshot"), position_feed_dir=str(tmp_path / "demo_feed"))
    demo = get_engine(f"sqlite:///{(tmp_path / 'demo.db').as_posix()}")
    init_db(demo)
    steps = run_demo(demo, demo_cfg)
    assert steps["step"].tolist() == ["backfill", "risk", "pnl", "backtest", "attribution", "stress", "limits"]

    report = compare_databases(engine, demo)
    assert identical(report), report[report[["only_a", "only_b", "diff_rows"]].sum(axis=1) > 0]
    assert (report.set_index("table").loc[["risk_result", "stress_result", "limit_check"], "rows_a"] > 0).all()


def test_demo_refuses_live_source(tmp_path, monkeypatch):
    engine, cfg = make_env(tmp_path, monkeypatch)
    cfg["data"]["source"] = "akshare"
    with pytest.raises(ValueError, match="data.source must be snapshot"):
        run_demo(engine, cfg)
