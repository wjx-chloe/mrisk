"""DQ gate and invalidation for every table derived from prices and positions.

Blocking rules are classified by how long the data they check feeds a calculation (dq.blocking_scope):
position 1 day, price as long as the price sits in some window, static reference data every day. Each
derived table declares, for a row dated t, which price dates and position dates it rests on; the gate and
the invalidation both read that declaration, so they always agree on what a failure touches."""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import Engine, Table, delete

from mrisk.db import schema as s
from mrisk.db.repo import read_frame
from mrisk.jobs import etl

log = logging.getLogger(__name__)

SCOPES = ("price", "position", "static")


@dataclass(frozen=True)
class Dependency:
    """A row dated t rests on price dates t - price[0] .. t + price[1] and position dates
    t - position[0] .. t + position[1], counted in trading days."""

    table: Table
    date_col: str
    price: tuple[int, int]
    position: tuple[int, int]


def dependencies(cfg) -> dict[str, Dependency]:
    L, W = cfg["lookback_days"], cfg["backtest_window_days"]
    return {
        # VaR at t: the lookback + 1 prices ending on t, positions of t
        "risk_result": Dependency(s.risk_result, "trade_date", (L, 0), (0, 0)),
        # positions of t held to t + 1
        "pnl_hypo": Dependency(s.pnl_hypo, "position_date", (0, 1), (0, 0)),
        # VaR on the windows of t - 1 and t, positions of t - 1 and t
        "var_attribution": Dependency(s.var_attribution, "trade_date", (L + 1, 0), (1, 0)),
        # as of a: W pairs (VaR at t, P&L t -> t + 1) for position dates t = a - W .. a - 1
        "backtest_result": Dependency(s.backtest_result, "asof_date", (W + L, 0), (W, -1)),
    }


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


def dq_gate(engine: Engine, cfg, dates: pd.DatetimeIndex, table: str = "risk_result") -> pd.DataFrame:
    """Per row date t of `table` -> [status, reason]. 'blocked' if a price rule failed on any price date t rests
    on, a position rule on any position date, or a static rule in its latest run; else 'no_dq' if one of those
    dates has no DQ result; else 'ok'. Failures are judged by the latest DQ run of each date."""
    dep = dependencies(cfg)[table]
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
    checked, fail = _latest_failures(engine, cal[lo.min()], cal[hi.max()])
    px_fail = fail[fail["rule_id"].isin(scope["price"])]
    pos_fail = fail[fail["rule_id"].isin(scope["position"])]
    st_fail = _static_failures(engine, scope["static"])

    # prefix sums over the calendar: count flagged dates in [a, b] in O(1) per row
    def count(days, a, b) -> np.ndarray:
        c = np.concatenate([[0], np.cumsum(cal.isin(pd.DatetimeIndex(days)))])
        return c[b + 1] - c[a]

    blocked = (count(px_fail["check_date"], plo, phi) > 0) | (count(pos_fail["check_date"], qlo, qhi) > 0) | (len(st_fail) > 0)
    unchecked = ((phi - plo + 1) - count(checked, plo, phi) > 0) | ((qhi - qlo + 1) - count(checked, qlo, qhi) > 0)

    out = pd.DataFrame({"status": "ok", "reason": ""}, index=dates)
    out.loc[unchecked & ~blocked, "status"] = "no_dq"
    out.loc[blocked, "status"] = "blocked"
    static_reason = [f"static {r.rule_id} failed in its latest run {r.run_id}" for r in st_fail.itertuples()]
    for k in np.flatnonzero(out["status"].to_numpy() != "ok"):
        d = dates[k]
        p0, p1, q0, q1 = cal[plo[k]], cal[phi[k]], cal[qlo[k]], cal[qhi[k]]
        pf = pos_fail[(pos_fail["check_date"] >= q0) & (pos_fail["check_date"] <= q1)]
        xf = px_fail[(px_fail["check_date"] >= p0) & (px_fail["check_date"] <= p1)]
        parts = list(static_reason)
        if len(pf):
            hits = sorted(f"{r.rule_id}@{r.check_date.date()}" for r in pf.itertuples())
            parts.append(f"position on {q0.date()}..{q1.date()}: {hits}" if q0 != q1 else f"position {sorted(pf['rule_id'])} on {d.date()}")
        if len(xf):
            hits = sorted(f"{r.rule_id}@{r.check_date.date()}" for r in xf.itertuples())
            parts.append(f"price in window {p0.date()}..{p1.date()}: {hits[:10]}{' ...' if len(hits) > 10 else ''}")
        if not parts:
            need = cal[min(plo[k], qlo[k]) : max(phi[k], qhi[k]) + 1]
            missing = need[~need.isin(checked)]
            parts.append(f"no DQ result for {len(missing)} day(s) in window, first {missing[0].date()}")
        out.at[d, "reason"] = "; ".join(parts)
    return out


def invalidate(engine: Engine, cfg, start, end) -> dict[str, pd.DatetimeIndex]:
    """Deletes the rows of every derived table that rest on data a blocking rule rejected; returns the affected
    dates per table. A static rule failing in its latest run removes every row. Otherwise, per the latest DQ run
    of each check date f in [start, end]: a row dated t goes if f is one of the price (price rule) or position
    (position rule) dates it rests on, even when t lies outside [start, end]. Recompute once the data is fixed."""
    scope = blocking_scope(cfg)
    pid = cfg["portfolio"]["portfolio_id"]
    deps = dependencies(cfg)
    out: dict[str, pd.DatetimeIndex] = {}
    st_fail = _static_failures(engine, scope["static"])
    if len(st_fail):
        for name, dep in deps.items():
            col = dep.table.c
            stored = read_frame(engine, dep.table, date_col=dep.date_col, portfolio_id=pid, columns=[dep.date_col])[dep.date_col]
            with engine.begin() as conn:
                n = conn.execute(delete(dep.table).where(col.portfolio_id == pid)).rowcount
            out[name] = pd.DatetimeIndex(sorted(stored.unique()))
            if n:
                log.warning("static rules %s failed in their latest run: all %d %s row(s) of %s deleted", sorted(st_fail["rule_id"]), n, name, pid)
        return out

    _, fail = _latest_failures(engine, start, end)
    cal = etl.trading_days(engine)
    px_j = cal.get_indexer(pd.DatetimeIndex(fail.loc[fail["rule_id"].isin(scope["price"]), "check_date"].unique()))
    pos_j = cal.get_indexer(pd.DatetimeIndex(fail.loc[fail["rule_id"].isin(scope["position"]), "check_date"].unique()))
    for name, dep in deps.items():
        # failure on f touches rows t with t - before <= f <= t + after, i.e. t in [f - after, f + before]
        hit = np.zeros(len(cal) + 1, dtype=int)
        for j, (before, after) in ((px_j, dep.price), (pos_j, dep.position)):
            a = np.clip(j - after, 0, len(cal))
            b = np.clip(j + before + 1, 0, len(cal))
            np.add.at(hit, a, 1)
            np.add.at(hit, b, -1)
        affected = cal[np.cumsum(hit)[: len(cal)] > 0]
        out[name] = affected
        if affected.empty:
            continue
        col = dep.table.c
        with engine.begin() as conn:
            n = conn.execute(
                delete(dep.table).where(col.portfolio_id == pid, col[dep.date_col].in_([d.date() for d in affected]))
            ).rowcount
        log.warning(
            "blocking DQ failures in %s..%s invalidate %s on %d day(s) %s..%s; %d row(s) deleted",
            pd.Timestamp(start).date(), pd.Timestamp(end).date(), name, len(affected), affected[0].date(), affected[-1].date(), n,
        )
    return out
