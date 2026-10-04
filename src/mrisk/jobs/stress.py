"""Stress job: scenarios (seed) x position dates -> stress_result (SPEC §6.1).
Historical scenarios are computed once per run for the whole pool; the price dates each one actually read are
recorded in scenario_price_date, which the DQ gate and the invalidation use."""

import json
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine

from mrisk.db import schema as s
from mrisk.db.repo import read_frame, replace_dates, replace_partition, upsert
from mrisk.engine.returns import returns_window
from mrisk.engine.stress import ScenarioPath, historical_stress, hypothetical_shocks, hypothetical_stress, scenario_path
from mrisk.engine.valuation import market_value
from mrisk.jobs import etl
from mrisk.jobs.gate import dq_gate

log = logging.getLogger(__name__)

SEED = Path(__file__).resolve().parents[1] / "data" / "seed" / "scenario.json"
PRICE_COLUMNS = ["trade_date", "instrument_id", "close", "adj_close", "is_suspended"]


def load_scenarios(engine: Engine) -> pd.DataFrame:
    """Upsert the seed scenarios and return them [scenario_id, type, definition]."""
    sc = pd.DataFrame(json.loads(SEED.read_text(encoding="utf-8")))
    upsert(engine, s.scenario, sc)
    return sc


def historical_paths(engine: Engine, cfg, scenarios: pd.DataFrame, pool: pd.DataFrame) -> dict[str, ScenarioPath]:
    """Path of every pool name per historical scenario; records the price dates read."""
    cal = etl.trading_days(engine)
    ind = pool.set_index("instrument_id")["industry_l1"]
    out = {}
    for r in scenarios[scenarios["type"] == "historical"].itertuples():
        base, end = pd.Timestamp(r.definition["base"]), pd.Timestamp(r.definition["end"])
        lo = cal[max(cal.get_loc(base) - cfg["lookback_days"], 0)]  # room for suspensions running into the base
        px = read_frame(engine, s.daily_price, lo, end, instrument_id=pool["instrument_id"].tolist(), columns=PRICE_COLUMNS)
        # rows from the first price actually loaded (price history starts at data.price_start); a suspension
        # reaching back past it still raises in scenario_path
        first = px["trade_date"].min()
        wide = px.pivot(index="trade_date", columns="instrument_id").reindex(cal[(cal >= max(lo, first)) & (cal <= end)])
        wide = wide.reindex(columns=pd.MultiIndex.from_product([["adj_close", "is_suspended"], pool["instrument_id"]]))
        path = scenario_path(wide["adj_close"], wide["is_suspended"], ind, base, end)
        replace_partition(
            engine, s.scenario_price_date, pd.DataFrame({"scenario_id": r.scenario_id, "price_date": path.price_dates}),
            scenario_id=r.scenario_id,
        )
        log.info(
            "scenario %s %s..%s: %d day(s), prices read %s..%s, proxy names: %d unlisted, %d suspended",
            r.scenario_id, base.date(), end.date(), len(path.returns), path.price_dates[0].date(), path.price_dates[-1].date(),
            int(path.unlisted.sum()), int(path.suspended.sum()),
        )
        out[r.scenario_id] = path
    return out


def run_stress(engine: Engine, cfg, start, end=None, run_id: str | None = None) -> pd.Series:
    """stress_result for every position date in [start, end] and every seed scenario. A (date, scenario) pair the
    DQ gate refuses gets no rows. Returns the per-date status: ok, or the worst gate status among the scenarios."""
    pid = cfg["portfolio"]["portfolio_id"]
    L = cfg["lookback_days"]
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else start
    run_id = run_id or f"stress-{etl.now_cst():%Y%m%dT%H%M%S}"

    scenarios = load_scenarios(engine)
    pool = read_frame(engine, s.instrument, date_col="list_date", columns=["instrument_id", "industry_l1", "status"])
    ind = pool.set_index("instrument_id")["industry_l1"]
    pos = read_frame(engine, s.position, start, end, portfolio_id=pid, columns=["trade_date", "instrument_id", "quantity"])
    dates = pd.DatetimeIndex(sorted(pos["trade_date"].unique()))
    if dates.empty:
        log.warning("no positions for %s in %s..%s", pid, start.date(), end.date())
        return pd.Series(dtype=object, index=pd.DatetimeIndex([]))
    paths = historical_paths(engine, cfg, scenarios, pool)

    gates = {}
    for r in scenarios.itertuples():
        g = dq_gate(engine, cfg, dates, f"stress_result:{r.type}", key=r.scenario_id)
        for d in dates[g["status"] != "ok"]:
            log.error("%s %s: stress not computed (%s): %s", d.date(), r.scenario_id, g.at[d, "status"], g.at[d, "reason"])
        gates[r.scenario_id] = g["status"]
    st = pd.DataFrame(gates)
    rank = {"ok": 0, "no_dq": 1, "blocked": 2}
    status = st.apply(lambda row: max(row, key=rank.get), axis=1)

    cal = etl.trading_days(engine)
    px = read_frame(engine, s.daily_price, cal[max(cal.get_loc(dates[0]) - L, 0)], end, columns=PRICE_COLUMNS)
    wide = px.pivot(index="trade_date", columns="instrument_id")
    active = pool.loc[pool["status"] == "active", "instrument_id"]
    hyp = scenarios[scenarios["type"] == "hypothetical"]
    rows = []
    for d in dates:
        todo = st.columns[st.loc[d] == "ok"]
        if todo.empty:
            continue
        p = pos.loc[pos["trade_date"] == d, ["instrument_id", "quantity"]]
        mv = market_value(p, wide["close"].loc[d], ind).set_index("instrument_id")["market_value"]
        if todo.isin(hyp["scenario_id"]).any():
            # the pool's VaR window of d, names with a full window (the factor universe)
            win = wide["adj_close"].loc[:d].tail(L + 1)
            full = active[active.isin(win.columns[win.notna().all()])]
            if len(full) < len(active) and d == dates[0]:
                log.info("factor pool on %s leaves out names without a full window: %s", d.date(), sorted(set(active) - set(full)))
            R_pool, _ = returns_window(wide["adj_close"][full], wide["is_suspended"][full], d, L, cfg["return_type"])
        for r in scenarios[scenarios["scenario_id"].isin(todo)].itertuples():
            if r.type == "historical":
                res = historical_stress(mv, ind, paths[r.scenario_id], pid)
            else:
                res = hypothetical_stress(mv, ind, hypothetical_shocks(r.definition, R_pool, ind, mv.index), pid)
            rows.append(res.assign(trade_date=d, scenario_id=r.scenario_id))
    out = pd.concat(rows, ignore_index=True).assign(portfolio_id=pid, run_id=run_id) if rows else pd.DataFrame(columns=[c.name for c in s.stress_result.columns])
    replace_dates(engine, s.stress_result, out, "trade_date", dates, portfolio_id=pid)
    n_ok = int((status == "ok").sum())
    log.info("stress %s: %d day(s) x %d scenario(s), %d day(s) with a refused scenario, run_id %s", pid, n_ok, len(scenarios), len(dates) - n_ok, run_id)
    return status
