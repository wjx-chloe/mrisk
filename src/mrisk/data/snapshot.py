"""Offline snapshot of the akshare raw cache: export from data/raw (checked against the database it fed), pack into
a zip for a GitHub Release, download, verify and extract. Read back by sources.SnapshotSource.

Layout: one parquet per raw cache key with the fetch date dropped (calendar, list_sh_main, ..., sw_comp_801010,
daily_SH600000, hfq_SH600000), the raw akshare frames unchanged, plus manifest.json (sha256 and rows per file).
The raw cache holds several date ranges of the same stock (daily_<id>_<start>_<end>); they are merged into one file,
and overlapping rows that disagree are reported, never resolved by picking one."""

import datetime as dt
import hashlib
import json
import logging
import os
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

FORMAT = 1
MANIFEST = "manifest.json"
DAILY = re.compile(r"^daily_(?P<sid>(?:SH|SZ|BJ)\d{6})_(?P<start>\d{8})_(?P<end>\d{8})$")
FETCHED = re.compile(r"^(?P<key>[A-Za-z0-9_]+)_(?P<fetched>\d{8})$")  # calendar_20260919, sw_comp_801010_20260919
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
ZIP_TIME = (1980, 1, 1, 0, 0, 0)
NOTICE = "行情经 akshare 取自公开网站，仅供学习研究；权利人要求即删除。"


class SnapshotError(RuntimeError):
    pass


@dataclass(frozen=True)
class Export:
    zip_path: Path
    sha256: str
    manifest: dict


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- export ----------------------------------------------------------------------------------------------------
def scan_raw(raw_dir: Path) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """Raw cache files grouped by snapshot key: (daily: daily_<id> -> its date ranges, other: key -> its fetch dates).
    A file name that fits neither pattern is an error."""
    daily, other, unknown = {}, {}, []
    for p in sorted(raw_dir.glob("*.parquet")):
        if m := DAILY.match(p.stem):
            daily.setdefault(f"daily_{m['sid']}", []).append(p)
        elif m := FETCHED.match(p.stem):
            other.setdefault(m["key"], []).append(p)
        else:
            unknown.append(p.name)
    if unknown:
        raise SnapshotError(f"unrecognised raw cache files: {unknown}")
    return daily, other


def merge_daily(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Union of one stock's date ranges -> (merged rows sorted by date, conflicts). conflicts has one row per
    (date, column, range) where a date present in several ranges does not carry the same value in all of them
    [date, column, range, value]; merged is only meaningful when conflicts is empty."""
    cols = [list(f.columns) for f in frames.values()]
    if any(c != cols[0] for c in cols):
        raise SnapshotError(f"date ranges have different columns: {dict(zip(frames, cols))}")
    allr = pd.concat([f.assign(_range=k) for k, f in frames.items()], ignore_index=True)
    value_cols = [c for c in cols[0] if c != "date"]
    distinct = allr.groupby("date")[value_cols].nunique(dropna=False)
    bad = distinct.index[(distinct > 1).any(axis=1)]
    conflicts = (
        allr[allr["date"].isin(bad)]
        .melt(id_vars=["date", "_range"], value_vars=value_cols, var_name="column", value_name="value")
        .rename(columns={"_range": "range"})
    )
    if len(conflicts):
        n = conflicts.groupby(["date", "column"])["value"].transform(lambda v: v.nunique(dropna=False))
        conflicts = conflicts[n > 1].sort_values(["date", "column", "range"], ignore_index=True)
    merged = allr.drop(columns="_range").drop_duplicates("date").sort_values("date", ignore_index=True)
    return merged, conflicts[["date", "column", "range", "value"]]


def coverage_gaps(db_prices: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Rows of daily_price with a real quote (not is_suspended) that the snapshot does not reproduce: the date is
    missing from the raw rows or carries another close. db_prices [trade_date, close, is_suspended] of one
    instrument; raw is its merged raw frame. Returns [trade_date, db_close, raw_close]."""
    real = db_prices.loc[~db_prices["is_suspended"].astype(bool), ["trade_date", "close"]]
    r = pd.DataFrame({"trade_date": pd.to_datetime(raw["date"]).astype("datetime64[ns]"), "raw_close": pd.to_numeric(raw["close"]).astype(float)})
    m = real.merge(r, on="trade_date", how="left").rename(columns={"close": "db_close"})
    return m[m["raw_close"].isna() | (m["raw_close"] != m["db_close"])].reset_index(drop=True)


def _same_content(paths: list[Path]) -> bool:
    first = pd.read_parquet(paths[0])
    return all(first.equals(pd.read_parquet(p)) for p in paths[1:])


def export_snapshot(raw_dir: Path, instruments: pd.DataFrame, prices: pd.DataFrame, out_dir: Path) -> Export:
    """Builds out_dir/mrisk-snapshot-<price_end>.zip from the raw cache. instruments [instrument_id, exchange, symbol]
    and prices [trade_date, instrument_id, close, is_suspended] are the database's instrument and daily_price
    tables: every real quote in daily_price must be reproduced by the snapshot. Conflicting overlaps, coverage
    gaps or missing files raise SnapshotError after writing the details to out_dir/*.csv; nothing is packed then."""
    daily, other = scan_raw(raw_dir)
    price_end = pd.Timestamp(prices["trade_date"].max())
    name = f"mrisk-snapshot-{price_end:%Y%m%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = out_dir / name
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    problems: list[str] = []

    fetched = set()
    for key, paths in other.items():
        fetched |= {FETCHED.match(p.stem)["fetched"] for p in paths}
        if len(paths) > 1 and not _same_content(paths):
            problems.append(f"{key}: {len(paths)} fetches with different content: {[p.name for p in paths]}")
            continue
        shutil.copyfile(paths[-1], stage / f"{key}.parquet")
    required = ["calendar", "list_sh_main", "list_sh_star", "list_sz", "list_bj", "delist_sh", "delist_sz", "sw_l1"]
    if "sw_l1" in other:
        first = pd.read_parquet(other["sw_l1"][-1])
        required += [f"sw_comp_{c.split('.')[0]}" for c in first["行业代码"]]
    sids = (instruments["exchange"] + instruments["symbol"]).tolist()
    required += [f"hfq_{sid}" for sid in sids]
    missing = [k for k in required if k not in other] + [f"daily_{sid}" for sid in sids if f"daily_{sid}" not in daily]
    if missing:
        problems.append(f"missing raw cache files: {missing}")

    conflicts, gaps = [], []
    id_of = dict(zip(sids, instruments["instrument_id"]))
    for key, paths in daily.items():
        frames = {DAILY.match(p.stem)["start"] + "_" + DAILY.match(p.stem)["end"]: pd.read_parquet(p) for p in paths}
        merged, bad = merge_daily(frames)
        if len(bad):
            conflicts.append(bad.assign(file=key))
            continue
        same = [p for p, f in zip(paths, frames.values()) if f.equals(merged)]
        if same:  # one range already holds the union: ship its bytes unchanged
            shutil.copyfile(same[0], stage / f"{key}.parquet")
        else:
            merged.to_parquet(stage / f"{key}.parquet", index=False)
        sid = key.removeprefix("daily_")
        if sid in id_of:
            g = coverage_gaps(prices[prices["instrument_id"] == id_of[sid]], merged)
            if len(g):
                gaps.append(g.assign(instrument_id=id_of[sid]))
    if conflicts:
        c = pd.concat(conflicts, ignore_index=True)
        c.to_csv(out_dir / "snapshot_conflicts.csv", index=False)
        problems.append(
            f"{c.groupby('file')['date'].nunique().sum()} overlapping dates disagree across date ranges in "
            f"{c['file'].nunique()} stocks; details in {out_dir / 'snapshot_conflicts.csv'}"
        )
    if gaps:
        g = pd.concat(gaps, ignore_index=True)
        g.to_csv(out_dir / "snapshot_coverage_gaps.csv", index=False)
        problems.append(f"{len(g)} real quotes in daily_price not reproduced by the raw cache; details in {out_dir / 'snapshot_coverage_gaps.csv'}")
    if problems:
        shutil.rmtree(stage)
        raise SnapshotError("snapshot not packed:\n  " + "\n  ".join(problems))

    files = {}
    for p in sorted(stage.glob("*.parquet")):
        files[p.name] = {"sha256": sha256_file(p), "rows": len(pd.read_parquet(p))}
    manifest = {
        "format": FORMAT,
        "price_end": f"{price_end:%Y-%m-%d}",
        "fetched": sorted(f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in fetched),
        "notice": NOTICE,
        "files": files,
    }
    (stage / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    zip_path = out_dir / f"{name}.zip"
    pack(stage, zip_path)
    return Export(zip_path, sha256_file(zip_path), manifest)


def pack(stage: Path, zip_path: Path) -> None:
    """Deterministic zip of the files directly in stage: sorted names, fixed timestamps, no directory entries."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(stage.iterdir()):
            info = zipfile.ZipInfo(p.name, date_time=ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, p.read_bytes(), compresslevel=9)


# ---- fetch -----------------------------------------------------------------------------------------------------
def release_url(repo: str, tag: str, asset: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{asset}"


def _open(req: urllib.request.Request, timeout: float):
    return urllib.request.urlopen(req, timeout=timeout)


def download(repo: str, tag: str, asset: str, dest: Path, token: str | None = None, timeout: float = 60) -> None:
    """Anonymous download of a release asset; with a token (private repository) through the GitHub API instead. The
    token goes only to api.github.com: it is an unredirected header, so the redirect to the storage host drops it."""
    if token:
        req = urllib.request.Request(f"https://api.github.com/repos/{repo}/releases/tags/{tag}", headers={"Accept": "application/vnd.github+json"})
        req.add_unredirected_header("Authorization", f"Bearer {token}")
        with _open(req, timeout) as r:
            assets = {a["name"]: a["url"] for a in json.load(r)["assets"]}
        if asset not in assets:
            raise SnapshotError(f"release {tag} of {repo} has no asset {asset}; it has {sorted(assets)}")
        req = urllib.request.Request(assets[asset], headers={"Accept": "application/octet-stream"})
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    else:
        req = urllib.request.Request(release_url(repo, tag, asset))
    part = dest.with_name(dest.name + ".part")
    with _open(req, timeout) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(part, dest)


def extract(zip_path: Path, target: Path) -> dict:
    """Unpacks a snapshot zip into target (replacing it) after checking every member: a plain file name, listed in
    the manifest with a matching sha256, and nothing listed missing. Returns the manifest."""
    tmp = target.with_name(target.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        bad = [n for n in names if not SAFE_NAME.match(n) or n in (".", "..")]
        if bad:
            raise SnapshotError(f"{zip_path.name}: unsafe member names {bad}")
        if MANIFEST not in names:
            raise SnapshotError(f"{zip_path.name}: no {MANIFEST}")
        for n in names:
            (tmp / n).write_bytes(z.read(n))
    manifest = json.loads((tmp / MANIFEST).read_text(encoding="utf-8"))
    listed, present = set(manifest["files"]), set(names) - {MANIFEST}
    if listed != present:
        raise SnapshotError(f"{zip_path.name}: files not in manifest {sorted(present - listed)}, missing {sorted(listed - present)}")
    wrong = [n for n, meta in manifest["files"].items() if sha256_file(tmp / n) != meta["sha256"]]
    if wrong:
        raise SnapshotError(f"{zip_path.name}: sha256 mismatch for {wrong}")
    if target.exists():
        shutil.rmtree(target)
    os.replace(tmp, target)
    return manifest


def fetch_snapshot(release: dict, release_dir: Path, snapshot_dir: Path, token: str | None = None) -> dict:
    """Downloads the release asset named in config (unless a copy with the pinned sha256 is already in release_dir),
    verifies it against the pinned sha256 and extracts it into snapshot_dir. Returns the manifest."""
    release_dir.mkdir(parents=True, exist_ok=True)
    dest = release_dir / release["asset"]
    if dest.exists() and sha256_file(dest) == release["sha256"]:
        log.info("%s already downloaded and verified", dest)
    else:
        log.info("downloading %s of %s (%s)", release["asset"], release["repo"], "with GITHUB_TOKEN" if token else "anonymous")
        try:
            download(release["repo"], release["tag"], release["asset"], dest, token)
        except urllib.error.HTTPError as e:
            hint = "; the repository may be private: set GITHUB_TOKEN" if e.code == 404 and not token else ""
            raise SnapshotError(f"download of {release['asset']} ({release['tag']}) failed: HTTP {e.code} {e.reason}{hint}") from None
        except urllib.error.URLError as e:
            raise SnapshotError(f"download of {release['asset']} failed: {e.reason}") from None
        got = sha256_file(dest)
        if got != release["sha256"]:
            bad = dest.with_name(dest.name + ".bad")
            os.replace(dest, bad)
            raise SnapshotError(f"sha256 of the download is {got}, config pins {release['sha256']}; kept as {bad}")
    manifest = extract(dest, snapshot_dir)
    log.info("snapshot %s (prices to %s, %d files) in %s", release["asset"], manifest["price_end"], len(manifest["files"]), snapshot_dir)
    return manifest


def read_manifest(snapshot_dir: Path) -> dict:
    path = snapshot_dir / MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"no snapshot in {snapshot_dir} ({MANIFEST} missing); run mrisk fetch-snapshot")
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot_price_end(snapshot_dir: Path) -> dt.date:
    return dt.date.fromisoformat(read_manifest(snapshot_dir)["price_end"])
