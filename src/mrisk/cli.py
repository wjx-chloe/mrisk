import argparse
import logging
import sys

import pandas as pd

from mrisk.config import database_url, load_config
from mrisk.db.repo import get_engine, init_db
from mrisk.jobs import daily as jobs
from mrisk.jobs import etl
from mrisk.jobs import risk as risk_job


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
    r = sub.add_parser("risk", help="compute VaR/ES for position dates in a range and write risk_result")
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
        return 2 if len(blocked) else 0
    if args.cmd == "risk":
        if args.end and not args.start:
            ap.error("--end requires --start")
        status = risk_job.run_risk(engine, cfg, args.date or args.start, args.end)
        return 2 if status.empty or (status != "ok").any() else 0
    if args.cmd == "schedule":
        from mrisk.jobs.scheduler import run_forever

        run_forever(engine, cfg)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
