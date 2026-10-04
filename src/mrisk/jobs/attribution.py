"""Attribution job: explain the t-1 -> t change of VaR / ES -> var_attribution (SPEC §5.3)."""

import logging

import pandas as pd
from sqlalchemy import Engine

from mrisk.db import schema as s
from mrisk.db.repo import read_frame, replace_dates
from mrisk.engine.attribution import attribute_change
from mrisk.jobs import etl
from mrisk.jobs.gate import dq_gate

log = logging.getLogger(__name__)

PRICE_COLUMNS = ["trade_date", "instrument_id", "close", "adj_close", "is_suspended"]


def run_attribution(engine: Engine, cfg, start, end=None, run_id: str | None = None) -> pd.Series:
    """var_attribution for every position date t in [start, end] whose previous trading day also has positions.
    Each date's rows are replaced; a date the DQ gate refuses has its rows deleted.
    Returns the per-date status: ok / blocked / no_dq."""
    pid = cfg["portfolio"]["portfolio_id"]
    lookback = cfg["lookback_days"]
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else start
    run_id = run_id or f"attribution-{etl.now_cst():%Y%m%dT%H%M%S}"

    cal = etl.trading_days(engine)
    lo = cal[max(cal.searchsorted(start) - 1, 0)]
    pos = read_frame(engine, s.position, lo, end, portfolio_id=pid, columns=["trade_date", "instrument_id", "quantity"])
    pos_dates = pd.DatetimeIndex(sorted(pos["trade_date"].unique()))
    i = cal.get_indexer(pos_dates)
    prev = pd.Series([cal[k - 1] if k > 0 else pd.NaT for k in i], index=pos_dates)
    has_prev = prev.isin(pos_dates) & (pos_dates >= start)
    dates = pos_dates[has_prev.to_numpy()]
    skipped = pos_dates[(pos_dates >= start) & ~has_prev.to_numpy()]
    if len(skipped):
        log.info("attribution %s: no previous position date for %s", pid, [d.date().isoformat() for d in skipped])
    if dates.empty:
        return pd.Series(dtype=object, index=pd.DatetimeIndex([]))

    gate = dq_gate(engine, cfg, dates, "var_attribution")
    status = gate["status"]
    for d in dates[status != "ok"]:
        log.error("%s: attribution not computed (%s): %s", d.date(), status[d], gate.at[d, "reason"])
    ok = dates[status == "ok"]
    out = pd.DataFrame(columns=[c.name for c in s.var_attribution.columns])
    if len(ok):
        prices = read_frame(
            engine, s.daily_price, cal[cal.get_loc(ok.min()) - lookback - 1], ok.max(),
            instrument_id=sorted(pos["instrument_id"].unique()), columns=PRICE_COLUMNS,
        )
        by_date = {d: g[["instrument_id", "quantity"]] for d, g in pos.groupby("trade_date")}
        frames = [
            attribute_change(by_date[prev[d]], by_date[d], prices, prev[d], d, pid, cfg["confidence"], lookback, cfg["return_type"]).assign(trade_date=d)
            for d in ok
        ]
        out = pd.concat(frames, ignore_index=True).assign(portfolio_id=pid, run_id=run_id)
    replace_dates(engine, s.var_attribution, out, "trade_date", dates, portfolio_id=pid)
    log.info("attribution %s: %d day(s) written, %d skipped, run_id %s", pid, len(ok), len(dates) - len(ok), run_id)
    return status
