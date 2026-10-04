"""Risk job: read from the db -> gate on DQ -> engine.compute_risk -> write risk_result.
Never fetches market data; everything comes from what the ETL already loaded and validated."""

import logging

import pandas as pd
from sqlalchemy import Engine

from mrisk.db import schema as s
from mrisk.db.repo import read_frame, replace_partition
from mrisk.engine.returns import RETURN_TYPES
from mrisk.engine.risk import compute_risk
from mrisk.jobs import etl
from mrisk.jobs.gate import dq_gate

log = logging.getLogger(__name__)

PRICE_COLUMNS = ["trade_date", "instrument_id", "close", "adj_close", "is_suspended"]


def _check_params(cfg) -> None:
    if cfg["holding_period_days"] != 1:
        raise ValueError(f"only a 1-day holding period is supported, got {cfg['holding_period_days']}")
    if cfg["parametric_dist"] != "normal":
        raise ValueError(f"only parametric_dist=normal is supported, got {cfg['parametric_dist']!r}")
    if cfg["return_type"] not in RETURN_TYPES:
        raise ValueError(f"return_type must be one of {RETURN_TYPES}, got {cfg['return_type']!r}")


def run_risk(engine: Engine, cfg, start, end=None, run_id: str | None = None) -> pd.Series:
    """Computes and stores risk for every position date in [start, end] that passes dq_gate. Each date's rows
    replace whatever risk_result held for (trade_date, portfolio_id); a blocked date's rows are deleted.
    Returns the per-date status: ok / blocked / no_dq."""
    _check_params(cfg)
    pid = cfg["portfolio"]["portfolio_id"]
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else start
    run_id = run_id or f"risk-{etl.now_cst():%Y%m%dT%H%M%S}"
    lookback = cfg["lookback_days"]

    pos = read_frame(engine, s.position, start, end, portfolio_id=pid, columns=["trade_date", "instrument_id", "quantity"])
    dates = pd.DatetimeIndex(sorted(pos["trade_date"].unique()))
    if dates.empty:
        log.warning("no positions for %s in %s..%s", pid, start.date(), end.date())
        return pd.Series(dtype=object, index=pd.DatetimeIndex([]))
    cal = etl.trading_days(engine, end=dates[0])
    window_start = cal[max(0, len(cal) - 1 - lookback)]
    prices = read_frame(
        engine, s.daily_price, window_start, end, instrument_id=sorted(pos["instrument_id"].unique()), columns=PRICE_COLUMNS
    )
    instruments = read_frame(engine, s.instrument, date_col="list_date", columns=["instrument_id", "industry_l1"])

    gate = dq_gate(engine, cfg, dates, "risk_result")
    status = gate["status"]
    for d in dates[status != "ok"]:
        log.error("%s: risk not computed (%s): %s", d.date(), status[d], gate.at[d, "reason"])
        if status[d] == "blocked":  # whatever was stored for d rests on rejected data
            replace_partition(engine, s.risk_result, pd.DataFrame(), trade_date=d.date(), portfolio_id=pid)
    for d in dates[status == "ok"]:
        res = compute_risk(
            pos.loc[pos["trade_date"] == d, ["instrument_id", "quantity"]],
            prices, instruments, d, pid, cfg["confidence"], lookback, cfg["return_type"],
        )
        _log_diagnostics(d, res)
        out = res.table.drop(columns="es_tail_n").assign(trade_date=d, portfolio_id=pid, run_id=run_id)
        replace_partition(engine, s.risk_result, out, trade_date=d.date(), portfolio_id=pid)
    ok = int((status == "ok").sum())
    log.info("risk %s: %d day(s) written, %d skipped, run_id %s", pid, ok, len(status) - ok, run_id)
    return status


def _log_diagnostics(d: pd.Timestamp, res) -> None:
    susp = res.suspended_days[res.suspended_days > 0]
    if len(susp):
        log.info("%s: suspended days in the return window (returns set to 0): %s", d.date(), susp.to_dict())
    empty = res.table[res.table["es_tail_n"] == 0]
    if len(empty):
        cells = [f"{r.level}:{r.level_key}@{r.confidence:g}" for r in empty.itertuples()]
        log.warning("%s: no P&L strictly beyond VaR, ES set to VaR for %s", d.date(), cells)
