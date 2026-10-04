import argparse
import logging
import sys

import pandas as pd

from mrisk.config import database_url, load_config
from mrisk.db.repo import get_engine, init_db
from mrisk.jobs import daily as jobs
from mrisk.jobs import etl, gate
from mrisk.jobs import backtest as backtest_job
from mrisk.jobs import pnl as pnl_job
from mrisk.jobs import risk as risk_job

# date-range jobs: name -> (help, run(engine, cfg, start, end) -> per-date status, no dates is an error,
# statuses that are not errors)
RANGE_JOBS = {
    "risk": ("compute VaR/ES for position dates in a range and write risk_result", risk_job.run_risk, True, {"ok"}),
    "pnl": ("hypothetical P&L of each position date held to the next trading day -> pnl_hypo", pnl_job.run_pnl, False, {"ok"}),
    "backtest": (
        "rolling Kupiec / Christoffersen / traffic light for realisation dates in a range -> backtest_result",
        backtest_job.run_backtest, False, {"ok", "warming_up"},
    ),
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
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    engine = get_engine(database_url(cfg))
    init_db(engine)

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


if __name__ == "__main__":
    sys.exit(main())
