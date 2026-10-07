"""DQ gate and invalidation for every table derived from prices and positions.

Blocking rules are classified by how long the data they check feeds a calculation (dq.blocking_scope):
position 1 day, price as long as the price sits in some window, static reference data every day. Each
derived table declares, for a row dated t, which price dates and position dates it rests on, relative to t
and, for historical stress scenarios, as absolute dates the calculation recorded (scenario_price_date). The gate
and the invalidation both read that declaration, so they always agree on what a failure touches. limit_check
rests on its rules' inputs: when an input row is invalidated, the checks on it turn into no_data."""

import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import Engine, Table, delete, update

from mrisk.db import schema as s
from mrisk.db.repo import read_frame
from mrisk.jobs import etl

log = logging.getLogger(__name__)

SCOPES = ("price", "position", "static")


@dataclass(frozen=True)
class Dependency:
    """A row dated t rests on price dates t - price[0] .. t + price[1] and position dates
    t - position[0] .. t + position[1] (trading days). With key_col / keys the entry covers only rows whose
    key_col is in keys; fixed maps a key to absolute price dates all of that key's rows rest on."""

    table: Table
    date_col: str
    price: tuple[int, int]
    position: tuple[int, int]
    key_col: str | None = None
    keys: tuple[str, ...] = ()
    fixed: dict[str, pd.DatetimeIndex] = field(default_factory=dict)


def stress_entry(scenario_type: str, definition: dict) -> str:
    """Dependency entry of a scenario's stress_result rows: historical, hypothetical (factor shocks, betas on the VaR
    window) or raw (shocks given per name: market values on t only)."""
    if scenario_type == "historical":
        return "stress_result:historical"
    return "stress_result:raw" if "raw" in definition else "stress_result:hypothetical"


def dependencies(cfg, engine: Engine) -> dict[str, Dependency]:
    L, W = cfg["lookback_days"], cfg["backtest_window_days"]
    deps = {
        # VaR at t: the lookback + 1 prices ending on t, positions of t
        "risk_result": Dependency(s.risk_result, "trade_date", (L, 0), (0, 0)),
        # positions of t held to t + 1
        "pnl_hypo": Dependency(s.pnl_hypo, "position_date", (0, 1), (0, 0)),
        # VaR on the windows of t - 1 and t, positions of t - 1 and t
        "var_attribution": Dependency(s.var_attribution, "trade_date", (L + 1, 0), (1, 0)),
        # as of a: W pairs (VaR at t, P&L t -> t + 1) for position dates t = a - W .. a - 1
        "backtest_result": Dependency(s.backtest_result, "asof_date", (W + L, 0), (W, -1)),
    }
    scen = read_frame(engine, s.scenario, date_col="scenario_id", columns=["scenario_id", "type", "definition"]) if engine is not None else pd.DataFrame()
    if len(scen):
        entry = pd.Series([stress_entry(r.type, r.definition) for r in scen.itertuples()], index=scen["scenario_id"])
        hist = tuple(entry.index[entry == "stress_result:historical"])
        hyp = tuple(entry.index[entry == "stress_result:hypothetical"])
        raw = tuple(entry.index[entry == "stress_result:raw"])
        spd = read_frame(engine, s.scenario_price_date, date_col="price_date")
        fixed = {k: pd.DatetimeIndex(sorted(g["price_date"])) for k, g in spd.groupby("scenario_id")}
        # historical: market values on t, positions of t, and the scenario's recorded price dates
        deps["stress_result:historical"] = Dependency(s.stress_result, "trade_date", (0, 0), (0, 0), "scenario_id", hist, fixed)
        # hypothetical: betas on the VaR window of t, positions of t
        deps["stress_result:hypothetical"] = Dependency(s.stress_result, "trade_date", (L, 0), (0, 0), "scenario_id", hyp)
        # raw: shocks given per name, market values on t, positions of t
        deps["stress_result:raw"] = Dependency(s.stress_result, "trade_date", (0, 0), (0, 0), "scenario_id", raw)
    # concentration limits: prices and positions of t only; other limits follow their inputs (see invalidate)
    deps["limit_check:concentration"] = Dependency(s.limit_check, "trade_date", (0, 0), (0, 0))
    return deps


def blocking_scope(cfg) -> dict[str, list[str]]:
    """dq.blocking_scope -> {class: rules}. Each blocking rule in exactly one class."""
    unknown = set(cfg["dq"]["blocking_scope"]) - set(SCOPES)
    if unknown:
        raise ValueError(f"dq.blocking_scope: unknown classes {sorted(unknown)}, expected {SCOPES}")
    scope = {k: list(cfg["dq"]["blocking_scope"].get(k) or []) for k in SCOPES}
    seen: dict[str, str] = {}
    for k, rules in scope.items():
        for r in rules:
            if r in seen:
                raise ValueError(f"dq.blocking_scope: {r} classified as both {seen[r]} and {k}")
            seen[r] = k
    if set(seen) != set(cfg["dq"]["blocking"]):
        raise ValueError(f"dq.blocking_scope {sorted(seen)} must cover exactly dq.blocking {sorted(cfg['dq']['blocking'])}")
    return scope


def _latest_failures(engine: Engine, start, end) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Latest DQ run per check date in [start, end] -> (dates that have a result, failed rows of those runs).
    A failure counts until a later run of the same date passes: that is what "unresolved" means."""
    res = read_frame(engine, s.dq_result, start, end, date_col="check_date")
    latest = res[res["created_at"] == res.groupby("check_date")["created_at"].transform("max")]
    return pd.DatetimeIndex(latest["check_date"].unique()), latest[latest["status"] == "fail"]


def _static_failures(engine: Engine, rules: list[str]) -> pd.DataFrame:
    """Static rules re-check today's reference tables whatever check date they are filed under, so each is judged
    by its most recent run overall, not per date: an old date backfilled later still carries its old pass."""
    if not rules:
        return pd.DataFrame(columns=["rule_id", "run_id", "created_at"])
    res = read_frame(
        engine, s.dq_result, date_col="check_date", rule_id=rules, columns=["rule_id", "run_id", "status", "created_at"]
    )
    latest = res[res["created_at"] == res.groupby("rule_id")["created_at"].transform("max")]
    return latest.loc[latest["status"] == "fail", ["rule_id", "run_id", "created_at"]].drop_duplicates("rule_id")


def dq_gate(engine: Engine, cfg, dates: pd.DatetimeIndex, table: str = "risk_result", key: str | None = None) -> pd.DataFrame:
    """Per row date t of the dependency `table` (and `key` for keyed entries) -> [status, reason]. 'blocked' if a
    price rule failed on any price date t rests on (relative or fixed), a position rule on any position date, or a
    static rule in its latest run; else 'no_dq' if one of those dates has no DQ result; else 'ok'."""
    dep = dependencies(cfg, engine)[table]
    if dep.key_col is not None and key not in dep.keys:
        raise ValueError(f"{table}: key {key!r} not among {dep.keys}")
    fixed = dep.fixed.get(key, pd.DatetimeIndex([])) if key is not None else pd.DatetimeIndex([])
    scope = blocking_scope(cfg)
    cal = etl.trading_days(engine)
    i = cal.get_indexer(dates)
    if (i < 0).any():
        raise ValueError(f"not trading days: {[d.date().isoformat() for d in dates[i < 0]]}")
    plo, phi = i - dep.price[0], i + dep.price[1]
    qlo, qhi = i - dep.position[0], i + dep.position[1]
    lo, hi = np.minimum(plo, qlo), np.maximum(phi, qhi)
    if (lo < 0).any():
        raise ValueError(f"calendar starts after the data {dates[lo < 0][0].date()} rests on ({table})")
    if (hi >= len(cal)).any():
        raise ValueError(f"calendar ends before the data {dates[hi >= len(cal)][0].date()} rests on ({table})")
    span = pd.DatetimeIndex([cal[lo.min()], cal[hi.max()]]).union(fixed)
    checked, fail = _latest_failures(engine, span.min(), span.max())
    px_fail = fail[fail["rule_id"].isin(scope["price"])]
    pos_fail = fail[fail["rule_id"].isin(scope["position"])]
    st_fail = _static_failures(engine, scope["static"])
    fixed_fail = px_fail[px_fail["check_date"].isin(fixed)]
    fixed_unchecked = fixed.difference(checked)

    # prefix sums over the calendar: count flagged dates in [a, b] in O(1) per row
    def count(days, a, b) -> np.ndarray:
        c = np.concatenate([[0], np.cumsum(cal.isin(pd.DatetimeIndex(days)))])
        return c[b + 1] - c[a]

    blocked = (count(px_fail["check_date"], plo, phi) > 0) | (count(pos_fail["check_date"], qlo, qhi) > 0)
    blocked |= (len(st_fail) > 0) | (len(fixed_fail) > 0)
    unchecked = ((phi - plo + 1) - count(checked, plo, phi) > 0) | ((qhi - qlo + 1) - count(checked, qlo, qhi) > 0)
    unchecked |= len(fixed_unchecked) > 0

    out = pd.DataFrame({"status": "ok", "reason": ""}, index=dates)
    out.loc[unchecked & ~blocked, "status"] = "no_dq"
    out.loc[blocked, "status"] = "blocked"
    head = [f"static {r.rule_id} failed in its latest run {r.run_id}" for r in st_fail.itertuples()]
    if len(fixed_fail):
        hits = sorted(f"{r.rule_id}@{r.check_date.date()}" for r in fixed_fail.itertuples())
        head.append(f"price on dates {key} rests on: {hits[:10]}")
    for k in np.flatnonzero(out["status"].to_numpy() != "ok"):
        d = dates[k]
        p0, p1, q0, q1 = cal[plo[k]], cal[phi[k]], cal[qlo[k]], cal[qhi[k]]
        pf = pos_fail[(pos_fail["check_date"] >= q0) & (pos_fail["check_date"] <= q1)]
        xf = px_fail[(px_fail["check_date"] >= p0) & (px_fail["check_date"] <= p1)]
        parts = list(head)
        if len(pf):
            hits = sorted(f"{r.rule_id}@{r.check_date.date()}" for r in pf.itertuples())
            parts.append(f"position on {q0.date()}..{q1.date()}: {hits}" if q0 != q1 else f"position {sorted(pf['rule_id'])} on {d.date()}")
        if len(xf):
            hits = sorted(f"{r.rule_id}@{r.check_date.date()}" for r in xf.itertuples())
            parts.append(f"price in window {p0.date()}..{p1.date()}: {hits[:10]}{' ...' if len(hits) > 10 else ''}")
        if not parts:
            need = cal[min(plo[k], qlo[k]) : max(phi[k], qhi[k]) + 1].union(fixed)
            missing = need[~need.isin(checked)]
            parts.append(f"no DQ result for {len(missing)} day(s) the data rests on, first {missing[0].date()}")
        out.at[d, "reason"] = "; ".join(parts)
    return out


def _span_hits(cal: pd.DatetimeIndex, j: np.ndarray, before: int, after: int) -> np.ndarray:
    """Calendar mask of rows t with t - before <= f <= t + after for some failure index f in j."""
    hit = np.zeros(len(cal) + 1, dtype=int)
    a = np.clip(j - after, 0, len(cal))
    b = np.clip(j + before + 1, 0, len(cal))
    np.add.at(hit, a, 1)
    np.add.at(hit, b, -1)
    return np.cumsum(hit)[: len(cal)] > 0


def invalidate(engine: Engine, cfg, start, end) -> dict[str, pd.DatetimeIndex]:
    """Deletes the rows of every derived table that rest on data a blocking rule rejected and returns the affected
    dates per entry ('<entry>' or '<entry>/<key>'). A static rule failing in its latest run removes every row.
    Otherwise, per the latest DQ run of each check date f in [start, end]: a row dated t goes if f is one of the
    price (price rule) or position (position rule) dates it rests on, even when t lies outside [start, end]; a
    failure on a fixed date of a key removes all of that key's rows. limit_check rows whose inputs went are turned
    into no_data rather than deleted, so the page shows the day as unchecked. Recompute once the data is fixed."""
    scope = blocking_scope(cfg)
    pid = cfg["portfolio"]["portfolio_id"]
    deps = dependencies(cfg, engine)
    out: dict[str, pd.DatetimeIndex] = {}
    st_fail = _static_failures(engine, scope["static"])
    cal = etl.trading_days(engine)

    if len(st_fail):
        why = f"static {sorted(st_fail['rule_id'])} failed in their latest run"
        for name, dep in deps.items():
            if dep.table is s.limit_check:
                continue
            stored = read_frame(engine, dep.table, date_col=dep.date_col, portfolio_id=pid, columns=[dep.date_col])[dep.date_col]
            with engine.begin() as conn:
                n = conn.execute(delete(dep.table).where(dep.table.c.portfolio_id == pid)).rowcount
            out[name] = pd.DatetimeIndex(sorted(stored.unique()))
            if n:
                log.warning("%s: all %d %s row(s) of %s deleted", why, n, name, pid)
        out["limit_check"] = _limits_to_no_data(engine, pid, None, None, {"invalidated": "all inputs", "reason": why})
        return out

    _, fail = _latest_failures(engine, start, end)
    px_days = pd.DatetimeIndex(fail.loc[fail["rule_id"].isin(scope["price"]), "check_date"].unique())
    pos_days = pd.DatetimeIndex(fail.loc[fail["rule_id"].isin(scope["position"]), "check_date"].unique())
    px_j, pos_j = cal.get_indexer(px_days), cal.get_indexer(pos_days)
    reason = {"price": [d.date().isoformat() for d in px_days], "position": [d.date().isoformat() for d in pos_days]}
    for name, dep in deps.items():
        affected = cal[_span_hits(cal, px_j, *dep.price) | _span_hits(cal, pos_j, *dep.position)]
        col = dep.table.c
        if dep.table is s.limit_check:  # concentration checks: prices and positions of t
            rules = _rules(engine, pid)
            ids = rules.loc[rules["metric"].str.startswith("concentration_"), "rule_id"].tolist()
            out[name] = _limits_to_no_data(engine, pid, affected, ids, {"invalidated": "prices / positions of the day", "dq_failures": reason})
            continue
        # a keyed entry with no keys covers no rows (it must not fall back to the unkeyed, whole-table delete)
        for key in [None] if dep.key_col is None else list(dep.keys):
            cond = [col.portfolio_id == pid] + ([col[dep.key_col] == key] if key is not None else [])
            hit_all = key is not None and px_days.isin(dep.fixed.get(key, pd.DatetimeIndex([]))).any()
            label = name if key is None else f"{name}/{key}"
            if hit_all:  # a failure on a date the scenario itself was computed from
                stored = read_frame(engine, dep.table, date_col=dep.date_col, portfolio_id=pid, **{dep.key_col: key}, columns=[dep.date_col])
                days = pd.DatetimeIndex(sorted(stored[dep.date_col].unique()))
            else:
                days = affected
            out[label] = days
            if days.empty:
                continue
            with engine.begin() as conn:
                n = conn.execute(delete(dep.table).where(*cond, col[dep.date_col].in_([d.date() for d in days]))).rowcount
            if n:
                log.warning(
                    "blocking DQ failures in %s..%s invalidate %s on %d day(s) %s..%s; %d row(s) deleted",
                    pd.Timestamp(start).date(), pd.Timestamp(end).date(), label, len(days), days[0].date(), days[-1].date(), n,
                )
            # limits resting on what was just deleted
            ids = _rules_on(engine, pid, name, key)
            if ids:
                out[f"limit_check<-{label}"] = _limits_to_no_data(engine, pid, days, ids, {"invalidated": label, "dq_failures": reason})
    return out


def _rules(engine: Engine, pid: str) -> pd.DataFrame:
    return read_frame(engine, s.limit_rule, date_col="rule_id", portfolio_id=pid, columns=["rule_id", "metric"])


def _rules_on(engine: Engine, pid: str, name: str, key: str | None) -> list[str]:
    """limit rules whose metric is computed from the rows of dependency entry `name` (and scenario `key`)."""
    rules = _rules(engine, pid)
    if name == "risk_result":
        return rules.loc[rules["metric"].str.match(r"^(var|es)_"), "rule_id"].tolist()
    if name.startswith("stress_result") and key is not None:
        return rules.loc[rules["metric"] == f"stress_loss:{key}", "rule_id"].tolist()
    return []


def _limits_to_no_data(engine: Engine, pid: str, days, rule_ids, detail: dict) -> pd.DatetimeIndex:
    """Turn stored checks into no_data (actual cleared, detail says why); days / rule_ids None mean all."""
    lc = s.limit_check.c
    if rule_ids is not None and not rule_ids:
        return pd.DatetimeIndex([])
    if days is not None and len(days) == 0:
        return pd.DatetimeIndex([])
    pid_rules = [r for r in _rules(engine, pid)["rule_id"] if rule_ids is None or r in rule_ids]
    cond = [lc.rule_id.in_(pid_rules)]
    if days is not None:
        cond.append(lc.trade_date.in_([d.date() for d in days]))
    with engine.begin() as conn:
        n = conn.execute(
            update(s.limit_check).where(*cond).values(status="no_data", actual_value=None, detail=json.loads(json.dumps(detail)))
        ).rowcount
    if n:
        log.warning("%d limit_check row(s) turned into no_data: %s", n, detail.get("invalidated"))
    return pd.DatetimeIndex([]) if days is None else pd.DatetimeIndex(days)
