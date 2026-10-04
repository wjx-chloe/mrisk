"""Hypothetical P&L job: positions of t held to the next trading day -> pnl_hypo (SPEC §3.12)."""

import logging

import pandas as pd
from sqlalchemy import Engine, func, select

from mrisk.db import schema as s
from mrisk.db.repo import read_frame, replace_dates
from mrisk.engine.pnl import hypothetical_pnl
from mrisk.jobs import etl
from mrisk.jobs.gate import dq_gate

log = logging.getLogger(__name__)

PRICE_COLUMNS = ["trade_date", "instrument_id", "close", "adj_close", "is_suspended"]


def run_pnl(engine: Engine, cfg, start, end=None, run_id: str | None = None) -> pd.Series:
    """pnl_hypo for every position date t in [start, end] whose next trading day already has prices.
    Each date's rows are replaced; a date the DQ gate refuses has its rows deleted.
    Returns the per-date status (ok / blocked / no_dq); dates not yet realised are only logged."""
    pid = cfg["portfolio"]["portfolio_id"]
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else start
    run_id = run_id or f"pnl-{etl.now_cst():%Y%m%dT%H%M%S}"

    pos = read_frame(engine, s.position, start, end, portfolio_id=pid, columns=["trade_date", "instrument_id", "quantity"])
    pos_dates = pd.DatetimeIndex(sorted(pos["trade_date"].unique()))
    cal = etl.trading_days(engine)
    with engine.connect() as conn:
        last_px = pd.Timestamp(conn.execute(select(func.max(s.daily_price.c.trade_date))).scalar_one())
    i = cal.get_indexer(pos_dates)
    nxt = pd.DatetimeIndex([cal[k + 1] if 0 <= k < len(cal) - 1 else pd.NaT for k in i])
    realisable = nxt.notna() & (nxt <= last_px)
    if (~realisable).any():
        log.info("pnl %s: %d position date(s) not yet realised, from %s", pid, int((~realisable).sum()), pos_dates[~realisable][0].date())
    dates = pos_dates[realisable]
    if dates.empty:
        return pd.Series(dtype=object, index=pd.DatetimeIndex([]))

    gate = dq_gate(engine, cfg, dates, "pnl_hypo")
    status = gate["status"]
    for d in dates[status != "ok"]:
        log.error("%s: hypothetical P&L not computed (%s): %s", d.date(), status[d], gate.at[d, "reason"])
    ok = dates[status == "ok"]
    next_day = pd.Series(nxt[realisable], index=dates)
    out = pd.DataFrame(columns=[c.name for c in s.pnl_hypo.columns])
    if len(ok):
        held = pos[pos["trade_date"].isin(ok)]
        prices = read_frame(
            engine, s.daily_price, ok.min(), next_day[ok].max(),
            instrument_id=sorted(held["instrument_id"].unique()), columns=PRICE_COLUMNS,
        )
        instruments = read_frame(engine, s.instrument, date_col="list_date", columns=["instrument_id", "industry_l1"])
        out = hypothetical_pnl(held, prices, instruments, pid).assign(portfolio_id=pid, run_id=run_id)
        got = out.groupby("position_date")["pnl_date"].first()
        if not (got.reindex(ok).to_numpy() == next_day[ok].to_numpy()).all():
            raise ValueError("pnl_date does not match the next trading day of the calendar")
    replace_dates(engine, s.pnl_hypo, out, "position_date", dates, portfolio_id=pid)
    log.info("pnl %s: %d day(s) written, %d skipped, run_id %s", pid, len(ok), len(dates) - len(ok), run_id)
    return status
