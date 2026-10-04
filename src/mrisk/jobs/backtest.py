"""Backtest job: risk_result VaR at t vs pnl_hypo t -> t+1, rolling window -> backtest_result (SPEC §5.4).
Reads only stored, already gated results; a missing pair (a day the DQ gate refused) leaves a gap, and any
window holding a gap gets no result."""

import logging

import numpy as np
import pandas as pd
from sqlalchemy import Engine, func, select

from mrisk.db import schema as s
from mrisk.db.repo import read_frame, replace_dates
from mrisk.engine.backtest import TRAFFIC_LIGHT_OBS, exceptions, rolling_backtest, traffic_light
from mrisk.jobs import etl

log = logging.getLogger(__name__)

METHODS = {"hs": "var_hs", "param": "var_param"}


def observations(engine: Engine, cfg) -> pd.DataFrame:
    """One row per position date t of the portfolio: pnl_date (next trading day), VaR per method x confidence
    (columns '<method>@<confidence>') and the hypothetical P&L; NaN where not stored."""
    pid = cfg["portfolio"]["portfolio_id"]
    pos_dates = pd.DatetimeIndex(sorted(read_frame(engine, s.position, portfolio_id=pid, columns=["trade_date"])["trade_date"].unique()))
    cal = etl.trading_days(engine)
    i = cal.get_indexer(pos_dates)
    obs = pd.DataFrame(index=pos_dates)
    obs["pnl_date"] = [cal[k + 1] if k + 1 < len(cal) else pd.NaT for k in i]
    rr = read_frame(engine, s.risk_result, portfolio_id=pid, level="portfolio")
    for method, col in METHODS.items():
        for conf in cfg["confidence"]:
            obs[f"{method}@{conf:g}"] = rr[rr["confidence"] == conf].set_index("trade_date")[col].reindex(pos_dates)
    ph = read_frame(engine, s.pnl_hypo, date_col="position_date", portfolio_id=pid, level="portfolio")
    obs["pnl"] = ph.set_index("position_date")["pnl"].reindex(pos_dates)
    return obs


def run_backtest(engine: Engine, cfg, start, end=None, run_id: str | None = None) -> pd.Series:
    """backtest_result for every as-of date a in [start, end], a being the realisation date of the window's last
    observation. Returns the per-date status: ok, warming_up (fewer than window observations since the portfolio
    started) or gap (a missing observation inside the window; its rows are deleted). Dates not yet realised (no prices yet) are left out."""
    pid = cfg["portfolio"]["portfolio_id"]
    W = cfg["backtest_window_days"]
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else start
    run_id = run_id or f"backtest-{etl.now_cst():%Y%m%dT%H%M%S}"

    obs = observations(engine, cfg)
    with engine.connect() as conn:
        last_px = pd.Timestamp(conn.execute(select(func.max(s.daily_price.c.trade_date))).scalar_one())
    obs = obs[obs["pnl_date"] <= last_px]  # the last position date may not be realised yet
    in_range = (obs["pnl_date"] >= start) & (obs["pnl_date"] <= end)
    if not in_range.any():
        log.warning("backtest %s: no realisation dates in %s..%s", pid, start.date(), end.date())
        return pd.Series(dtype=object, index=pd.DatetimeIndex([]))
    k = np.arange(len(obs))
    status = pd.Series(np.where(k + 1 < W, "warming_up", "ok"), index=obs["pnl_date"])

    rows = []
    for method in METHODS:
        for conf in cfg["confidence"]:
            var = obs[f"{method}@{conf:g}"]
            valid = (var.notna() & obs["pnl"].notna()).to_numpy()
            hit = np.zeros(len(obs), dtype=bool)
            hit[valid] = exceptions(var[valid], obs["pnl"][valid])
            bt = rolling_backtest(hit, valid, W, 1 - conf)
            status[(~bt["complete"].to_numpy()) & (k + 1 >= W)] = "gap"
            bt.index = obs["pnl_date"]
            bt = bt[bt["complete"].to_numpy() & in_range.to_numpy()]
            zone = traffic_light(bt["n_exceptions"]) if (method == "hs" and conf == 0.99 and W == TRAFFIC_LIGHT_OBS) else None
            rows.append(
                bt.drop(columns=["complete", "n00", "n01", "n10", "n11"]).assign(
                    method=method, confidence=conf, window_days=W, zone=zone
                )
            )
    status = status[in_range.to_numpy()]
    out = pd.concat(rows).rename_axis("asof_date").reset_index()
    out = out[out["asof_date"].isin(status.index[status == "ok"])].assign(portfolio_id=pid, run_id=run_id)
    replace_dates(engine, s.backtest_result, out, "asof_date", status.index, portfolio_id=pid)
    for d in status.index[status == "gap"]:
        log.error("%s: backtest window has a missing observation, no result", d.date())
    n_ok, n_warm = int((status == "ok").sum()), int((status == "warming_up").sum())
    log.info("backtest %s: %d day(s) written, %d warming up, %d with gaps, run_id %s", pid, n_ok, n_warm, len(status) - n_ok - n_warm, run_id)
    return status
