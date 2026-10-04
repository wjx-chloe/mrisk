"""Limits job: active rules (seed) x position dates -> limit_check (SPEC §6.2, §6.3).
Every rule gets a row every day: ok / warn / breach when its input is there, no_data (with the reason) when not."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import Engine, delete

from mrisk.db import schema as s
from mrisk.db.repo import read_frame, upsert
from mrisk.engine.valuation import market_value
from mrisk.jobs.gate import dq_gate
from mrisk.limits.engine import evaluate, metric_rows

log = logging.getLogger(__name__)

SEED = Path(__file__).resolve().parents[1] / "limits" / "seed" / "limit_rule.csv"
RISK_METRICS = {"var_hs": "var", "es_hs": "es"}  # risk_result column -> metric prefix (var_hs_99, es_hs_95, ...)


def load_rules(engine: Engine, pid: str) -> pd.DataFrame:
    seed = pd.read_csv(SEED, dtype={"rule_id": str}).drop(columns="note", errors="ignore")
    seed["is_active"] = seed["is_active"].astype(str).str.lower().eq("true")
    upsert(engine, s.limit_rule, seed)
    return read_frame(engine, s.limit_rule, date_col="rule_id", portfolio_id=pid)


def _reason(engine, cfg, d, table, key=None) -> dict:
    g = dq_gate(engine, cfg, pd.DatetimeIndex([d]), table, key=key)
    st = g.at[d, "status"]
    name = table.split(":")[0]
    if st == "ok":
        return {"missing": name, "reason": "not computed for this date"}
    return {"missing": name, "reason": f"{st}: {g.at[d, 'reason']}"}


def run_limits(engine: Engine, cfg, start, end=None, run_id: str | None = None) -> pd.Series:
    """limit_check for every position date in [start, end]. Returns the per-date worst check status."""
    pid = cfg["portfolio"]["portfolio_id"]
    start = pd.Timestamp(start)
    end = pd.Timestamp(end) if end is not None else start
    rules = load_rules(engine, pid)
    active = rules[rules["is_active"]]
    pos = read_frame(engine, s.position, start, end, portfolio_id=pid, columns=["trade_date", "instrument_id", "quantity"])
    dates = pd.DatetimeIndex(sorted(pos["trade_date"].unique()))
    if dates.empty or active.empty:
        log.warning("limits %s: nothing to check in %s..%s", pid, start.date(), end.date())
        return pd.Series(dtype=object, index=pd.DatetimeIndex([]))

    px = read_frame(engine, s.daily_price, start, end, instrument_id=sorted(pos["instrument_id"].unique()), columns=["trade_date", "instrument_id", "close"])
    close = px.pivot(index="trade_date", columns="instrument_id", values="close")
    ind = read_frame(engine, s.instrument, date_col="list_date", columns=["instrument_id", "industry_l1"]).set_index("instrument_id")["industry_l1"]
    rr = read_frame(engine, s.risk_result, start, end, portfolio_id=pid, level="portfolio")
    sr = read_frame(engine, s.stress_result, start, end, portfolio_id=pid, level="portfolio")
    conc_gate = dq_gate(engine, cfg, dates, "limit_check:concentration")["status"]
    scen_type = read_frame(engine, s.scenario, date_col="scenario_id", columns=["scenario_id", "type"]).set_index("scenario_id")["type"]
    needed = set(active["metric"])

    rows, worst = [], {}
    order = {"ok": 0, "no_data": 1, "warn": 2, "breach": 3}
    for d in dates:
        metrics, missing = [], {}
        # concentration: positions and prices of d only
        conc = {"concentration_instrument", "concentration_industry"} & needed
        if conc:
            if conc_gate[d] == "ok":
                mv = market_value(pos.loc[pos["trade_date"] == d, ["instrument_id", "quantity"]], close.loc[d], ind)
                total = mv["market_value"].sum()
                metrics.append(metric_rows("concentration_instrument", "instrument", mv["instrument_id"], mv["market_value"], total))
                by_ind = mv.groupby("industry_l1")["market_value"].sum()
                metrics.append(metric_rows("concentration_industry", "industry", by_ind.index, by_ind.to_numpy(), total))
            else:
                why = _reason(engine, cfg, d, "limit_check:concentration")
                missing.update({m: {**why, "missing": "position / daily_price"} for m in conc})
        # VaR / ES from risk_result
        r = rr[rr["trade_date"] == d]
        for m in sorted(x for x in needed if x.startswith(("var_", "es_"))):
            col, conf = m.rsplit("_", 1)
            row = r[np.isclose(r["confidence"], int(conf) / 100)]
            if len(row) and col in row:
                metrics.append(metric_rows(m, "portfolio", [pid], [row[col].iloc[0]], row["market_value"].iloc[0]))
            else:
                missing[m] = _reason(engine, cfg, d, "risk_result")
        # stress losses (positive) from stress_result
        for m in sorted(x for x in needed if x.startswith("stress_loss:")):
            sid = m.split(":", 1)[1]
            row = sr[(sr["trade_date"] == d) & (sr["scenario_id"] == sid)]
            if len(row):
                metrics.append(metric_rows(m, "portfolio", [pid], [-row["pnl"].iloc[0]], row["market_value"].iloc[0]))
            else:
                if sid not in scen_type:
                    raise ValueError(f"limit on unknown scenario {sid}")
                missing[m] = _reason(engine, cfg, d, f"stress_result:{scen_type[sid]}", key=sid)
        mtab = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame(columns=["metric", "scope", "key", "value", "base"])
        chk = evaluate(active, mtab, missing).assign(trade_date=d)
        for c in chk.itertuples():
            if c.status == "breach":
                log.error("%s limit %s BREACH: %.4g > %.4g %s", d.date(), c.rule_id, c.actual_value, c.threshold, c.detail or "")
            elif c.status == "warn":
                log.warning("%s limit %s warn: %.4g past %.4g", d.date(), c.rule_id, c.actual_value, c.warn_threshold)
            elif c.status == "no_data":
                log.warning("%s limit %s no_data: %s", d.date(), c.rule_id, c.detail)
        worst[d] = max(chk["status"], key=order.get)
        rows.append(chk)
    out = pd.concat(rows, ignore_index=True)
    with engine.begin() as conn:
        conn.execute(delete(s.limit_check).where(s.limit_check.c.trade_date.in_([d.date() for d in dates]), s.limit_check.c.rule_id.in_(rules["rule_id"].tolist())))
    upsert(engine, s.limit_check, out)
    counts = out["status"].value_counts().to_dict()
    log.info("limits %s: %d day(s) x %d rule(s): %s", pid, len(dates), len(active), counts)
    return pd.Series(worst)
