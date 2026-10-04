import logging

import pandas as pd
from sqlalchemy import Engine

from mrisk.jobs import etl
from mrisk.jobs import gate
from mrisk.jobs import risk as risk_job

log = logging.getLogger(__name__)


def backfill(engine: Engine, cfg, end=None) -> pd.DatetimeIndex:
    src = etl.make_source(cfg)
    log.info("calendar: %d rows", etl.fetch_trade_calendar(engine, src))
    end = pd.Timestamp(end) if end else etl.trading_days(engine, end=pd.Timestamp.today().normalize())[-1]
    log.info("instruments: %d rows", etl.fetch_instruments(engine, src))
    log.info("code map: %d rows", etl.apply_code_map(engine))
    start = pd.Timestamp(cfg["data"]["price_start"])
    log.info("prices %s..%s: %d rows", start.date(), end.date(), etl.fetch_prices(engine, src, start, end, "backfill", cfg["data"]["suspension_infer_min_coverage"]))
    path = etl.generate_positions(engine, cfg, end)
    log.info("positions: %d rows loaded from %s", etl.load_positions(engine, cfg, path), path)
    res, blocked = etl.run_dq(engine, cfg, start, end, run_id=f"backfill-{etl.now_cst():%Y%m%dT%H%M%S}")
    _log_dq(res, blocked)
    gate.invalidate(engine, cfg, start, end)
    return blocked


def daily(engine: Engine, cfg, date=None) -> pd.DatetimeIndex | None:
    """Incremental run for one trading day: prices -> positions -> DQ -> risk, the last only when DQ has no
    blocking failure. Returns the blocked dates (DQ or the risk gate), or None if not a trading day."""
    src = etl.make_source(cfg)
    etl.fetch_trade_calendar(engine, src)
    d = pd.Timestamp(date).normalize() if date else pd.Timestamp(etl.now_cst().date())
    if d not in etl.trading_days(engine, d, d):
        log.info("%s is not a trading day, nothing to do", d.date())
        return None
    log.info("prices %s: %d rows", d.date(), etl.fetch_prices(engine, src, d, d, "daily", cfg["data"]["suspension_infer_min_coverage"]))
    path = etl.generate_positions(engine, cfg, d)
    log.info("positions: %d rows loaded", etl.load_positions(engine, cfg, path))
    res, blocked = etl.run_dq(engine, cfg, d, d, run_id=f"daily-{etl.now_cst():%Y%m%dT%H%M%S}")
    _log_dq(res, blocked)
    gate.invalidate(engine, cfg, d, d)
    if len(blocked):
        log.error("%s: blocking DQ failure, risk not computed", d.date())
        return blocked
    status = risk_job.run_risk(engine, cfg, d)
    return pd.DatetimeIndex(status.index[status != "ok"])


def _log_dq(res: pd.DataFrame, blocked: pd.DatetimeIndex) -> None:
    summary = res.pivot_table(index="rule_id", columns="status", values="check_date", aggfunc="count", fill_value=0)
    log.info("DQ summary (days per status):\n%s", summary.to_string())
    if len(blocked):
        log.error("blocking DQ failures on %d day(s): %s", len(blocked), [d.date().isoformat() for d in blocked[:20]])
