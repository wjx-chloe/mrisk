"""mrisk demo: the whole pipeline on the offline snapshot into a fresh database, no network.
backfill (calendar, instruments, prices, mock positions, DQ) up to the snapshot's last price date, then
risk -> pnl -> backtest -> attribution -> stress -> limits over every position date."""

import logging
import time

import pandas as pd
from sqlalchemy import Engine

from mrisk.config import resolve_path
from mrisk.data.snapshot import snapshot_price_end
from mrisk.jobs import daily as jobs
from mrisk.jobs import attribution as attribution_job
from mrisk.jobs import backtest as backtest_job
from mrisk.jobs import limits as limits_job
from mrisk.jobs import pnl as pnl_job
from mrisk.jobs import risk as risk_job
from mrisk.jobs import stress as stress_job

log = logging.getLogger(__name__)

STEPS = [
    ("risk", risk_job.run_risk),
    ("pnl", pnl_job.run_pnl),
    ("backtest", backtest_job.run_backtest),
    ("attribution", attribution_job.run_attribution),
    ("stress", stress_job.run_stress),
    ("limits", limits_job.run_limits),
]


def run_demo(engine: Engine, cfg) -> pd.DataFrame:
    """Runs every step on an empty database; returns one row per step [step, seconds, days, statuses], where
    statuses counts the per-date status the step returned (backfill: the dates DQ blocked)."""
    if cfg["data"]["source"] != "snapshot":
        raise ValueError(f"mrisk demo replays the offline snapshot: data.source must be snapshot, got {cfg['data']['source']!r}")
    end = pd.Timestamp(snapshot_price_end(resolve_path(cfg, cfg["data"]["snapshot_dir"])))
    start = pd.Timestamp(cfg["portfolio"]["start"])
    rows = []
    t = time.perf_counter()
    blocked = jobs.backfill(engine, cfg, end)
    rows.append({"step": "backfill", "seconds": time.perf_counter() - t, "days": None,
                 "statuses": f"blocked {len(blocked)}" if len(blocked) else "ok"})
    for name, run in STEPS:
        t = time.perf_counter()
        status = run(engine, cfg, start, end)
        counts = status.value_counts()
        rows.append({"step": name, "seconds": time.perf_counter() - t, "days": len(status),
                     "statuses": ", ".join(f"{k} {v}" for k, v in counts.items())})
        log.info("demo %s: %.1fs, %s", name, rows[-1]["seconds"], rows[-1]["statuses"])
    return pd.DataFrame(rows)
