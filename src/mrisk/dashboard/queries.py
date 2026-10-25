"""Read-only queries behind the dashboard pages: stored results in, frames out. Nothing here recomputes a risk number;
the pages only show what the jobs wrote."""

import logging

import numpy as np
import pandas as pd
from sqlalchemy import Engine, func, select

from mrisk.db import schema as s
from mrisk.db.repo import read_frame, readonly_engine  # noqa: F401  (readonly_engine: used by app.py)
from mrisk.jobs import etl
from mrisk.jobs.gate import dq_gate, stress_entry
from mrisk.dashboard import transforms as tf

log = logging.getLogger(__name__)

RISK_COLS = ["market_value", "var_hs", "var_param", "es_hs"]


def active_rules(engine: Engine, pid: str) -> pd.DataFrame:
    rules = read_frame(engine, s.limit_rule, date_col="rule_id", portfolio_id=pid)
    return rules[rules["is_active"].astype(bool)].reset_index(drop=True)


def complete_dates(engine: Engine, cfg) -> pd.DatetimeIndex:
    """Position dates with a complete result: portfolio risk_result on every configured confidence and a limit_check
    row for every active rule (the limits step can fail on its own, see docs/week4-plan.md §5.2). Oldest first."""
    pid = cfg["portfolio"]["portfolio_id"]
    rr = s.risk_result.c
    with engine.connect() as conn:
        risk = pd.read_sql(
            select(rr.trade_date, func.count(rr.confidence.distinct()).label("n"))
            .where(rr.portfolio_id == pid, rr.level == "portfolio", rr.confidence.in_(cfg["confidence"]))
            .group_by(rr.trade_date),
            conn,
        )
    rules = active_rules(engine, pid)["rule_id"].tolist()
    lc = s.limit_check.c
    with engine.connect() as conn:
        lim = pd.read_sql(
            select(lc.trade_date, func.count().label("n")).where(lc.rule_id.in_(rules)).group_by(lc.trade_date), conn
        )
    ok_risk = pd.to_datetime(risk.loc[risk["n"] == len(cfg["confidence"]), "trade_date"])
    ok_lim = pd.to_datetime(lim.loc[lim["n"] == len(rules), "trade_date"])
    return pd.DatetimeIndex(sorted(set(ok_risk) & set(ok_lim)))


def position_dates(engine: Engine, pid: str) -> pd.DatetimeIndex:
    pos = read_frame(engine, s.position, portfolio_id=pid, columns=["trade_date"])
    return pd.DatetimeIndex(sorted(pos["trade_date"].unique()))


def banner(engine: Engine, cfg, complete: pd.DatetimeIndex) -> dict | None:
    """What to say at the top when the latest position date is not the default date. -> {level, text} or None.
    level 'error' when the DQ gate refuses that date's risk, 'warning' when its results are only incomplete."""
    pid = cfg["portfolio"]["portfolio_id"]
    pos = position_dates(engine, pid)
    if pos.empty:
        return None
    latest = pos[-1]
    shown = complete[-1].date() if len(complete) else "无"
    if len(complete) and complete[-1] >= latest:
        return None
    try:
        g = dq_gate(engine, cfg, pd.DatetimeIndex([latest]), "risk_result")
        status, reason = g.at[latest, "status"], g.at[latest, "reason"]
    except ValueError as e:  # calendar too short for the window
        status, reason = "unknown", str(e)
    if status == "blocked":
        return {"level": "error", "text": f"最新交易日 {latest.date()} 被 DQ 门禁阻断，风险未计算：{reason}。默认显示 {shown}。"}
    if status == "no_dq":
        return {"level": "warning", "text": f"最新交易日 {latest.date()} 的窗口里有未做 DQ 的日期：{reason}。默认显示 {shown}。"}
    return {"level": "warning", "text": f"最新交易日 {latest.date()} 的风险结果或限额检查不完整（任务可能尚未跑完或单独失败）。默认显示 {shown}。"}


SOURCE_LABEL = {"snapshot": "离线快照", "akshare": "akshare 实时拉数"}


def data_status(engine: Engine, cfg, d) -> dict:
    """The data behind the page for d: latest price and position dates in the database, the source the config names
    (the database does not record where its prices came from), the DQ gate of d's VaR / ES and d's DQ results.
    -> {price_end, position_end, source, gate, gate_reason, dq_fail, dq_fail_blocking, dq_run}."""
    d = pd.Timestamp(d)
    pid = cfg["portfolio"]["portfolio_id"]
    dp, ps = s.daily_price.c, s.position.c
    with engine.connect() as conn:
        price_end = conn.execute(select(func.max(dp.trade_date))).scalar()
        position_end = conn.execute(select(func.max(ps.trade_date)).where(ps.portfolio_id == pid)).scalar()
    src = cfg["data"]["source"]
    source = SOURCE_LABEL.get(src, src)
    if src == "snapshot" and cfg.get("snapshot_release"):
        source += f" {cfg['snapshot_release']['tag']}"
    gate, reason = _gate(engine, cfg, d, "risk_result")
    dq = dq_today(engine, d)
    fail = dq["status"] == "fail"
    return {
        "price_end": pd.Timestamp(price_end) if price_end else None,
        "position_end": pd.Timestamp(position_end) if position_end else None,
        "source": source,
        "gate": gate,
        "gate_reason": reason,
        "dq_fail": int(fail.sum()),
        "dq_fail_blocking": int((fail & (dq["severity"] == "blocking")).sum()),
        "dq_run": int(dq["status"].notna().sum()),
    }


def names(engine: Engine) -> pd.DataFrame:
    return read_frame(engine, s.instrument, date_col="list_date", columns=["instrument_id", "name", "industry_l1"]).set_index("instrument_id")


def portfolio_risk(engine: Engine, pid: str, conf: float, start=None, end=None) -> pd.DataFrame:
    """Portfolio market value, VaR (HS, parametric) and ES per position date. Index trade_date."""
    rr = read_frame(engine, s.risk_result, start, end, portfolio_id=pid, level="portfolio")
    rr = rr[np.isclose(rr["confidence"], conf)]
    return rr.set_index("trade_date").sort_index()[RISK_COLS]


def overview(engine: Engine, pid: str, d, conf: float, complete: pd.DatetimeIndex) -> pd.DataFrame:
    """Rows 'today' and 'prev' (the previous complete date, absent on the first) of portfolio_risk."""
    d = pd.Timestamp(d)
    prev = complete[complete < d]
    days = [d] + ([prev[-1]] if len(prev) else [])
    r = portfolio_risk(engine, pid, conf, min(days), d).loc[lambda x: x.index.isin(days)]
    return r.rename(index={d: "today", **({prev[-1]: "prev"} if len(prev) else {})})


def limit_checks(engine: Engine, pid: str, end=None) -> pd.DataFrame:
    """Every stored check of the portfolio's rules up to end, with the rule's metric and scope."""
    rules = read_frame(engine, s.limit_rule, date_col="rule_id", portfolio_id=pid)
    lc = read_frame(engine, s.limit_check, end=end, rule_id=rules["rule_id"].tolist())
    return lc.merge(rules[["rule_id", "metric", "scope", "operator", "threshold_type"]], on="rule_id")


def limits_today(engine: Engine, pid: str, d) -> pd.DataFrame:
    """Checks on d with their run: how many consecutive position dates up to d the rule has been in today's status
    (no_data or a missing row breaks the count). new = warn / breach that started today."""
    d = pd.Timestamp(d)
    lc = limit_checks(engine, pid, end=d)
    pos = position_dates(engine, pid)
    wide = lc.pivot(index="trade_date", columns="rule_id", values="status").reindex(pos[pos <= d])
    runs = tf.trailing_runs(wide)
    today = lc[lc["trade_date"] == d].set_index("rule_id")
    today["run_days"] = runs.reindex(today.index).fillna(0).astype(int)
    today["new"] = today["status"].isin(["warn", "breach"]) & (today["run_days"] == 1)
    today["rank"] = today["status"].map(tf.STATUS_RANK)
    return today.sort_values(["rank", "rule_id"], ascending=[False, True]).reset_index()


def limit_episodes(engine: Engine, pid: str, d, status: str = "breach") -> pd.DataFrame:
    """Runs of `status` per rule up to d on the position-date calendar: [rule_id, start, end, days, peak]."""
    d = pd.Timestamp(d)
    lc = limit_checks(engine, pid, end=d)
    pos = position_dates(engine, pid)
    pos = pos[pos <= d]
    out = []
    for rid, g in lc.groupby("rule_id"):
        g = g.set_index("trade_date").reindex(pos)
        seg = tf.segments(g["status"].eq(status), g["actual_value"])
        out.append(seg.assign(rule_id=rid))
    cols = ["rule_id", "start", "end", "days", "peak"]
    return pd.concat(out, ignore_index=True)[cols].sort_values("end", ascending=False).reset_index(drop=True) if out else pd.DataFrame(columns=cols)


def exposure(engine: Engine, pid: str, d, conf: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(industries [industry, market_value, weight], names [instrument_id, name, industry, market_value, weight]),
    largest first, from the risk_result rows of d."""
    rr = read_frame(engine, s.risk_result, d, d, portfolio_id=pid)
    rr = rr[np.isclose(rr["confidence"], conf)]
    total = rr.loc[rr["level"] == "portfolio", "market_value"].iloc[0]
    ind = rr[rr["level"] == "industry"][["level_key", "market_value"]].rename(columns={"level_key": "industry"})
    inst = rr[rr["level"] == "instrument"][["level_key", "market_value"]].rename(columns={"level_key": "instrument_id"})
    inst = inst.join(names(engine), on="instrument_id").rename(columns={"industry_l1": "industry"})
    for f in (ind, inst):
        f["weight"] = f["market_value"] / total
    return (
        ind.sort_values("market_value", ascending=False).reset_index(drop=True),
        inst.sort_values("market_value", ascending=False).reset_index(drop=True)[["instrument_id", "name", "industry", "market_value", "weight"]],
    )


def backtest(engine: Engine, cfg, d, conf: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(results as of d per method, observations). d is the realisation date of the window's last observation, so the
    view as of d covers P&L realised up to d. observations: the window's [position_date, pnl_date, pnl, var_hs,
    var_param] (empty while warming up)."""
    pid, W = cfg["portfolio"]["portfolio_id"], cfg["backtest_window_days"]
    d = pd.Timestamp(d)
    br = read_frame(engine, s.backtest_result, d, d, date_col="asof_date", portfolio_id=pid)
    br = br[np.isclose(br["confidence"], conf)].set_index("method")
    ph = read_frame(engine, s.pnl_hypo, end=d, date_col="pnl_date", portfolio_id=pid, level="portfolio")
    ph = ph.sort_values("pnl_date").tail(W)[["position_date", "pnl_date", "pnl"]]
    risk = portfolio_risk(engine, pid, conf, ph["position_date"].min(), ph["position_date"].max()) if len(ph) else pd.DataFrame(columns=RISK_COLS)
    obs = ph.join(risk[["var_hs", "var_param"]], on="position_date").reset_index(drop=True)
    return br, obs


def attribution(engine: Engine, pid: str, d, conf: float, metric: str) -> tuple[pd.Series, pd.DataFrame]:
    """(portfolio effects: effect -> value, names: [instrument_id, name, industry, effect columns...] by |total|)."""
    va = read_frame(engine, s.var_attribution, d, d, portfolio_id=pid, metric=metric)
    va = va[np.isclose(va["confidence"], conf)]
    port = va[va["level"] == "portfolio"].set_index("effect")["value"]
    inst = va[va["level"] == "instrument"].pivot(index="level_key", columns="effect", values="value")
    inst = inst.join(names(engine)).rename(columns={"industry_l1": "industry"})
    inst = inst.reindex(inst["total"].abs().sort_values(ascending=False).index) if "total" in inst else inst
    return port, inst.rename_axis("instrument_id").reset_index()


def scenarios(engine: Engine) -> pd.DataFrame:
    sc = read_frame(engine, s.scenario, date_col="scenario_id")
    return tf.label_scenarios(sc).assign(entry=[stress_entry(r.type, r.definition) for r in sc.itertuples()])


def stress(engine: Engine, pid: str, d) -> pd.DataFrame:
    """Portfolio rows of every scenario on d, labelled [scenario_id, shock, mode, pnl, loss_pct, ...], worst first.
    loss_pct and proxy_pct are shares of the market value, losses positive."""
    sr = read_frame(engine, s.stress_result, d, d, portfolio_id=pid, level="portfolio")
    sr = sr.merge(scenarios(engine), on="scenario_id")
    sr["loss_pct"] = -sr["pnl"] / sr["market_value"]
    sr["proxy_pct"] = -sr["pnl_proxy"] / sr["market_value"]
    sr["max_daily_pct"] = sr["max_daily_loss"] / sr["market_value"]
    return sr.sort_values("loss_pct", ascending=False).reset_index(drop=True)


def stress_detail(engine: Engine, pid: str, d, scenario_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(industries, names) of one scenario on d. names carry `proxied`: the name took a proxy path in the scenario
    (unlisted or suspended); its own number is not reliable (docs/week3b-stress-limits.md §5.5 a)."""
    sr = read_frame(engine, s.stress_result, d, d, portfolio_id=pid, scenario_id=scenario_id)
    total = sr.loc[sr["level"] == "portfolio", "market_value"].iloc[0]
    sr["loss_pct"] = -sr["pnl"] / total  # share of the portfolio's market value
    ind = sr[sr["level"] == "industry"].rename(columns={"level_key": "industry"})
    inst = sr[sr["level"] == "instrument"].rename(columns={"level_key": "instrument_id"}).join(names(engine), on="instrument_id")
    inst["proxied"] = (inst["proxy_mv_unlisted"] > 0) | (inst["proxy_mv_suspended"] > 0)
    inst["own_return"] = inst["pnl"] / inst["market_value"]
    return ind.sort_values("loss_pct", ascending=False).reset_index(drop=True), inst.sort_values("loss_pct", ascending=False).reset_index(drop=True)


def dq_today(engine: Engine, d) -> pd.DataFrame:
    """The latest DQ run's result per rule for check date d, with the rule's name, category and severity. Static rules
    (judged by their latest run overall) show that run."""
    rules = read_frame(engine, s.dq_rule, date_col="rule_id")
    res = read_frame(engine, s.dq_result, d, d, date_col="check_date")
    res = res[res["created_at"] == res.groupby("rule_id")["created_at"].transform("max")]
    out = rules.merge(res[["rule_id", "status", "n_checked", "n_failed", "detail_json", "run_id", "created_at"]], on="rule_id", how="left")
    return out.sort_values("rule_id").reset_index(drop=True)


GATE_TABLES = {
    "risk_result": "VaR / ES",
    "var_attribution": "VaR 归因",
    "limit_check:concentration": "集中度限额",
}
STRESS_ENTRIES = {"stress_result:historical": "历史情景", "stress_result:hypothetical": "β 传导情景", "stress_result:raw": "其余不变情景"}
GATE_RANK = ["ok", "no_dq", "unknown", "blocked"]


def gate_today(engine: Engine, cfg, d) -> pd.DataFrame:
    """DQ gate status of d for each result the dashboard shows. The hypothetical P&L is the one realised on d (position
    date d - 1). Historical scenarios rest on their own dates, so each is asked and the worst shown; the scenarios of
    the other stress entries share one dependency, so one of them answers for all."""
    d = pd.Timestamp(d)
    rows = [(label, *_gate(engine, cfg, d, table)) for table, label in GATE_TABLES.items()]
    cal = etl.trading_days(engine, end=d)
    if len(cal) >= 2:
        rows.insert(1, ("假设损益（当日实现）", *_gate(engine, cfg, cal[-2], "pnl_hypo")))
    sc = scenarios(engine)
    for entry, label in STRESS_ENTRIES.items():
        keys = sc.loc[sc["entry"] == entry, "scenario_id"].tolist()
        if keys:
            res = [_gate(engine, cfg, d, entry, k) for k in (keys if entry == "stress_result:historical" else keys[:1])]
            rows.append((label, *max(res, key=lambda r: GATE_RANK.index(r[0]))))
    return pd.DataFrame(rows, columns=["结果", "status", "reason"])


def _gate(engine, cfg, d, table, key=None) -> tuple[str, str]:
    try:
        g = dq_gate(engine, cfg, pd.DatetimeIndex([d]), table, key=key)
        return g.at[d, "status"], g.at[d, "reason"]
    except ValueError as e:  # the calendar does not cover the window
        return "unknown", str(e)
