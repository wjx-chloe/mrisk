import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from mrisk.config import database_url, load_config, resolve_path
from mrisk.db.repo import get_engine, init_db, missing_columns, readonly_engine
from mrisk.jobs import daily as jobs
from mrisk.jobs import etl, gate
from mrisk.jobs import attribution as attribution_job
from mrisk.jobs import backtest as backtest_job
from mrisk.jobs import limits as limits_job
from mrisk.jobs import pnl as pnl_job
from mrisk.jobs import risk as risk_job
from mrisk.jobs import stress as stress_job

DEMO_CONFIG = "config.demo.yaml"

# date-range jobs: name -> (help, run(engine, cfg, start, end) -> per-date status, no dates is an error,
# statuses that are not errors)
RANGE_JOBS = {
    "risk": ("compute VaR/ES for position dates in a range and write risk_result", risk_job.run_risk, True, {"ok"}),
    "pnl": ("hypothetical P&L of each position date held to the next trading day -> pnl_hypo", pnl_job.run_pnl, False, {"ok"}),
    "backtest": (
        "rolling Kupiec / Christoffersen / traffic light for realisation dates in a range -> backtest_result",
        backtest_job.run_backtest, False, {"ok", "warming_up"},
    ),
    "attribution": (
        "split the day-on-day change of VaR / ES into quantity, revaluation and window effects -> var_attribution",
        attribution_job.run_attribution, False, {"ok"},
    ),
    "stress": ("historical and hypothetical scenarios on each position date -> stress_result", stress_job.run_stress, True, {"ok"}),
    # warn / breach / no_data are findings, not job failures
    "limits": ("evaluate active limit rules on each position date -> limit_check", limits_job.run_limits, True, {"ok", "warn", "breach", "no_data"}),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mrisk")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-universe", help="select the candidate pool and write seed/universe.csv")
    b = sub.add_parser("backfill", help="full load: calendar, instruments, code map, prices, positions, DQ")
    b.add_argument("--end")
    d = sub.add_parser("daily", help="incremental run for one trading day")
    d.add_argument("--date")
    q = sub.add_parser("dq", help="re-run DQ rules over a date range")
    q.add_argument("--start", required=True)
    q.add_argument("--end", required=True)
    for name, (help_, *_) in RANGE_JOBS.items():
        r = sub.add_parser(name, help=help_)
        rg = r.add_mutually_exclusive_group(required=True)
        rg.add_argument("--date")
        rg.add_argument("--start")
        r.add_argument("--end", help="with --start; defaults to --start")
    sub.add_parser("schedule", help="run the daily job every weekday at scheduler.daily_time (Asia/Shanghai)")
    x = sub.add_parser("export-snapshot", help="pack the akshare raw cache into an offline snapshot zip and print its sha256")
    x.add_argument("--out", default="data/release", help="output directory, relative to the config's directory")
    sub.add_parser(
        "fetch-snapshot",
        help=f"download the snapshot release pinned in the config, check its sha256, unpack it (default config {DEMO_CONFIG})",
    )
    m = sub.add_parser("demo", help=f"the whole pipeline on the offline snapshot into a new database, no network (default config {DEMO_CONFIG})")
    m.add_argument("--overwrite", action="store_true", help="delete the demo database first if it exists")
    v = sub.add_parser("dashboard", help="start the Streamlit dashboard on the config's database, reachable from this machine only")
    v.add_argument("--address", default="localhost", help="server address (default localhost; 0.0.0.0 opens it to the LAN)")
    v.add_argument("--port", type=int)
    sub.add_parser(
        "upgrade-db",
        help="bring a database made by an older version up to the current schema in place: add missing columns and "
             "compute only them (every other command that writes does this too); exit 2 if a date could not be filled",
    )
    c = sub.add_parser("diff-db", help="compare two databases row by row, run metadata left out (exit 2 when they differ)")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "diff-db":
        return _diff_db(args.a, args.b)
    config_path = args.config
    if config_path is None and args.cmd in ("demo", "fetch-snapshot"):
        config_path = os.environ.get("MRISK_CONFIG") or DEMO_CONFIG
    cfg = load_config(config_path)
    if args.cmd == "export-snapshot":
        return _export_snapshot(cfg, resolve_path(cfg, args.out))
    if args.cmd == "fetch-snapshot":
        return _fetch_snapshot(cfg)
    if args.cmd == "demo":
        return _demo(cfg, args.overwrite)
    if args.cmd == "dashboard":
        return _dashboard(cfg, args.address, args.port)
    engine = get_engine(database_url(cfg))
    added = init_db(engine)
    filled = _fill_new_columns(engine, cfg, added)
    if args.cmd == "upgrade-db":
        return 2 if (filled == "mismatch").any() else 0

    if args.cmd == "build-universe":
        uni = etl.build_universe_seed(cfg, etl.make_source(cfg))
        logging.info("universe: %d instruments written to seed", len(uni))
        return 0
    if args.cmd == "backfill":
        return 2 if len(jobs.backfill(engine, cfg, args.end)) else 0
    if args.cmd == "daily":
        blocked = jobs.daily(engine, cfg, args.date)
        return 2 if blocked is not None and len(blocked) else 0
    if args.cmd == "dq":
        res, blocked = etl.run_dq(engine, cfg, pd.Timestamp(args.start), pd.Timestamp(args.end))
        jobs._log_dq(res, blocked)
        gate.invalidate(engine, cfg, args.start, args.end)
        return 2 if len(blocked) else 0
    if args.cmd in RANGE_JOBS:
        if args.end and not args.start:
            ap.error("--end requires --start")
        _, run, empty_is_error, fine = RANGE_JOBS[args.cmd]
        status = run(engine, cfg, args.date or args.start, args.end)
        return 2 if (status.empty and empty_is_error) or (~status.isin(fine)).any() else 0
    if args.cmd == "schedule":
        from mrisk.jobs.scheduler import run_forever

        run_forever(engine, cfg)
        return 0
    return 1


def _fill_new_columns(engine, cfg, added: list[tuple[str, str]]) -> pd.Series:
    """Computes the values of columns an older database lacked (init_db has added them as NULL): only those columns,
    nothing else is recomputed. -> per-date status of the fill."""
    for name, col in added:
        logging.info("schema upgrade: added column %s.%s", name, col)
    status = risk_job.fill_es_contrib(engine, cfg)
    if len(status):
        counts = status.value_counts().to_dict()
        print(f"risk_result.es_hs_contrib filled on {counts.get('filled', 0)} date(s), "
              f"{counts.get('mismatch', 0)} date(s) not filled (stored risk differs from a recomputation, see the log)")
    return status


def _export_snapshot(cfg, out: Path) -> int:
    from mrisk.data.snapshot import SnapshotError, export_snapshot
    from mrisk.db import schema as s
    from mrisk.db.repo import read_frame

    engine = readonly_engine(database_url(cfg))
    inst = read_frame(engine, s.instrument, date_col="list_date", columns=["instrument_id", "exchange", "symbol"])
    px = read_frame(engine, s.daily_price, columns=["trade_date", "instrument_id", "close", "is_suspended"])
    try:
        e = export_snapshot(resolve_path(cfg, cfg["data"]["raw_cache_dir"]), inst, px, out)
    except SnapshotError as err:
        logging.error("%s", err)
        return 2
    tag = e.zip_path.stem.replace("mrisk-snapshot-", "snapshot-")
    for k, v in [("tag", tag), ("file", e.zip_path.name), ("size", f"{e.zip_path.stat().st_size:,} bytes"),
                 ("sha256", e.sha256), ("path", e.zip_path)]:
        print(f"{k:<8}{v}")
    print(f"prices to {e.manifest['price_end']}, fetched {', '.join(e.manifest['fetched'])}, {len(e.manifest['files'])} files")
    return 0


def _fetch_snapshot(cfg) -> int:
    from mrisk.data.snapshot import SnapshotError, fetch_snapshot

    rel = cfg.get("snapshot_release")
    if rel is None:
        logging.error("%s has no snapshot_release section (it is in %s)", cfg["_path"], DEMO_CONFIG)
        return 2
    try:
        fetch_snapshot(rel, resolve_path(cfg, "data/release"), resolve_path(cfg, cfg["data"]["snapshot_dir"]), os.environ.get("GITHUB_TOKEN") or None)
    except SnapshotError as err:
        logging.error("%s", err)
        return 2
    return 0


def _demo(cfg, overwrite: bool) -> int:
    from mrisk.jobs.demo import run_demo

    url = database_url(cfg)
    db = Path(url.removeprefix("sqlite:///")) if url.startswith("sqlite:///") else None
    if db is not None and db.exists():
        if not overwrite:
            logging.error("%s already exists; delete it or pass --overwrite", db)
            return 2
        db.unlink()
    engine = get_engine(url)
    init_db(engine)
    steps = run_demo(engine, cfg)
    print(steps.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"total {steps['seconds'].sum():.0f}s -> {db or url}")
    print(f"dashboard: uv run mrisk --config {_config_arg(cfg)} dashboard")
    return 0


def _config_arg(cfg) -> str:
    try:
        return os.path.relpath(cfg["_path"])
    except ValueError:  # another drive on Windows
        return str(cfg["_path"])


def _dashboard(cfg, address: str, port: int | None) -> int:
    """Streamlit is started from the config's directory (the repository root, where .streamlit/config.toml lives),
    and the address and usage-stats settings are passed explicitly as well, so the defaults hold from any directory."""
    app = Path(__file__).resolve().parent / "dashboard" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.address", address, "--browser.gatherUsageStats", "false"]
    if port:
        cmd += ["--server.port", str(port)]
    env = {**os.environ, "MRISK_CONFIG": str(cfg["_path"])}
    return subprocess.call(cmd, cwd=cfg["_root"], env=env)


def _diff_db(a: str, b: str) -> int:
    from mrisk.db.compare import RUN_METADATA, compare_databases, identical

    for p in (a, b):
        if not Path(p).exists():
            logging.error("%s does not exist", p)
            return 2
    ea, eb = (readonly_engine(f"sqlite:///{Path(p).resolve().as_posix()}") for p in (a, b))
    gaps = {p: missing_columns(e) for p, e in ((a, ea), (b, eb))}
    for p, cols in gaps.items():
        if cols:
            logging.error("%s lacks %s: made by an older version; run mrisk upgrade-db on it first (adds and fills only these)",
                          p, ", ".join(f"{t}.{c}" for t, c in cols))
    if any(gaps.values()):
        return 2
    report = compare_databases(ea, eb)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(report.to_string(index=False))
    same = identical(report)
    print(f"left out: {', '.join(RUN_METADATA)} -> {'identical' if same else 'DIFFERENT'}")
    return 0 if same else 2


if __name__ == "__main__":
    sys.exit(main())
