"""Risk job: read from the db -> gate on DQ -> engine.compute_risk -> write risk_result.
Never fetches market data; everything comes from what the ETL already loaded and validated."""

import logging

import pandas as pd
from sqlalchemy import Engine, bindparam, select, update

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

    pos, prices, instruments = _inputs(engine, cfg, start, end)
    dates = pd.DatetimeIndex(sorted(pos["trade_date"].unique()))
    if dates.empty:
        log.warning("no positions for %s in %s..%s", pid, start.date(), end.date())
        return pd.Series(dtype=object, index=pd.DatetimeIndex([]))

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


def _inputs(engine: Engine, cfg, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(positions of [start, end], prices from the first date's window start to end, instruments) as compute_risk takes them."""
    pid = cfg["portfolio"]["portfolio_id"]
    pos = read_frame(engine, s.position, start, end, portfolio_id=pid, columns=["trade_date", "instrument_id", "quantity"])
    if pos.empty:
        return pos, pd.DataFrame(columns=PRICE_COLUMNS), pd.DataFrame(columns=["instrument_id", "industry_l1"])
    cal = etl.trading_days(engine, end=pos["trade_date"].min())
    window_start = cal[max(0, len(cal) - 1 - cfg["lookback_days"])]
    prices = read_frame(
        engine, s.daily_price, window_start, end, instrument_id=sorted(pos["instrument_id"].unique()), columns=PRICE_COLUMNS
    )
    instruments = read_frame(engine, s.instrument, date_col="list_date", columns=["instrument_id", "industry_l1"])
    return pos, prices, instruments


STORED = ["market_value", "var_hs", "var_param", "es_hs"]  # what fill_es_contrib checks before it writes
KEY = ["level", "level_key", "confidence"]


def fill_es_contrib(engine: Engine, cfg) -> pd.Series:
    """Fills risk_result.es_hs_contrib where it is NULL (rows stored before the column existed), one position date at
    a time, without touching any other column or run_id. The date is recomputed with compute_risk on the same inputs,
    and the column is written only if the recomputed market value, VaR (both methods) and ES equal the stored ones
    bit for bit on every row: then the contributions belong to exactly the numbers already in the table. Otherwise
    nothing is written for that date and the difference is logged (the inputs or the code changed since; rerun
    mrisk risk for it). -> per-date status: filled / mismatch (empty when nothing is NULL)."""
    _check_params(cfg)
    pid = cfg["portfolio"]["portfolio_id"]
    rr = s.risk_result.c
    with engine.connect() as conn:
        todo = pd.DatetimeIndex(sorted(pd.to_datetime([d for (d,) in conn.execute(
            select(rr.trade_date).distinct().where(rr.portfolio_id == pid, rr.es_hs_contrib.is_(None)))])))
    status = pd.Series(dtype=object, index=pd.DatetimeIndex([]))
    if todo.empty:
        return status
    log.info("risk_result.es_hs_contrib missing on %d date(s) %s..%s: filling", len(todo), todo[0].date(), todo[-1].date())
    pos, prices, instruments = _inputs(engine, cfg, todo[0], todo[-1])
    stmt = (
        update(s.risk_result)
        .where(rr.trade_date == bindparam("d"), rr.portfolio_id == pid, rr.level == bindparam("lv"),
               rr.level_key == bindparam("k"), rr.confidence == bindparam("c"))
        .values(es_hs_contrib=bindparam("v"))
    )
    for d in todo:
        stored = read_frame(engine, s.risk_result, d, d, portfolio_id=pid, columns=KEY + STORED)
        try:
            res = compute_risk(pos.loc[pos["trade_date"] == d, ["instrument_id", "quantity"]], prices, instruments, d, pid,
                               cfg["confidence"], cfg["lookback_days"], cfg["return_type"])
        except ValueError as e:  # e.g. the window no longer complete
            log.error("%s: cannot recompute risk (%s); es_hs_contrib not filled", d.date(), e)
            status[d] = "mismatch"
            continue
        m = stored.merge(res.table, on=KEY, how="outer", suffixes=("", "_new"), indicator=True)
        same = (m["_merge"] == "both").all() and all(
            ((m[c] == m[f"{c}_new"]) | (m[c].isna() & m[f"{c}_new"].isna())).all() for c in STORED)
        if not same:
            diff = {c: float((m[c] - m[f"{c}_new"]).abs().max()) for c in STORED}
            log.error("%s: stored risk differs from a recomputation (%d stored rows, %d recomputed, max abs diff %s); "
                      "es_hs_contrib not filled, rerun mrisk risk for this date", d.date(), len(stored), len(res.table), diff)
            status[d] = "mismatch"
            continue
        rows = [{"d": d.date(), "lv": r.level, "k": r.level_key, "c": r.confidence, "v": float(r.es_hs_contrib)}
                for r in m.itertuples()]
        with engine.begin() as conn:
            conn.execute(stmt, rows)
        status[d] = "filled"
    log.info("es_hs_contrib: %s", status.value_counts().to_dict())
    return status


def _log_diagnostics(d: pd.Timestamp, res) -> None:
    susp = res.suspended_days[res.suspended_days > 0]
    if len(susp):
        log.info("%s: suspended days in the return window (returns set to 0): %s", d.date(), susp.to_dict())
    empty = res.table[res.table["es_tail_n"] == 0]
    if len(empty):
        cells = [f"{r.level}:{r.level_key}@{r.confidence:g}" for r in empty.itertuples()]
        log.warning("%s: no P&L strictly beyond VaR, ES set to VaR for %s", d.date(), cells)
