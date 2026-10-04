"""Risk job: read from the db -> gate on DQ -> engine.compute_risk -> write risk_result.
Never fetches market data; everything comes from what the ETL already loaded and validated."""

import logging

import numpy as np
import pandas as pd
from sqlalchemy import Engine, delete

from mrisk.db import schema as s
from mrisk.db.repo import read_frame, replace_partition
from mrisk.engine.returns import RETURN_TYPES
from mrisk.engine.risk import compute_risk
from mrisk.jobs import etl

log = logging.getLogger(__name__)

PRICE_COLUMNS = ["trade_date", "instrument_id", "close", "adj_close", "is_suspended"]


def _check_params(cfg) -> None:
    if cfg["holding_period_days"] != 1:
        raise ValueError(f"only a 1-day holding period is supported, got {cfg['holding_period_days']}")
    if cfg["parametric_dist"] != "normal":
        raise ValueError(f"only parametric_dist=normal is supported, got {cfg['parametric_dist']!r}")
    if cfg["return_type"] not in RETURN_TYPES:
        raise ValueError(f"return_type must be one of {RETURN_TYPES}, got {cfg['return_type']!r}")


SCOPES = ("price", "position", "static")


def blocking_scope(cfg) -> dict[str, list[str]]:
    """dq.blocking_scope -> {class: rules}. Classes follow how long the checked data feeds the calculation:
    position 1 day, price lookback + 1 days, static reference data every day. Each blocking rule in exactly one."""
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


def dq_gate(engine: Engine, cfg, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Per risk date t -> [status, reason]. Position rules are checked on t only; price rules on every price
    date of t's window (the lookback + 1 trading days ending on t), since one bad price feeds every window
    holding it; static rules by their most recent run, for every t. status: 'blocked' if any such failure,
    else 'no_dq' if a date that must be checked has no DQ result, else 'ok'."""
    scope = blocking_scope(cfg)
    lookback = cfg["lookback_days"]
    cal = etl.trading_days(engine, end=dates.max())
    i = cal.get_indexer(dates)
    if (i < 0).any():
        raise ValueError(f"not trading days: {[d.date().isoformat() for d in dates[i < 0]]}")
    lo = i - lookback
    if (lo < 0).any():
        raise ValueError(f"calendar holds fewer than {lookback + 1} days up to {dates[lo < 0][0].date()}")
    checked, fail = _latest_failures(engine, cal[lo.min()], dates.max())
    px_fail = fail[fail["rule_id"].isin(scope["price"])]
    pos_fail = fail[fail["rule_id"].isin(scope["position"])]
    st_fail = _static_failures(engine, scope["static"])

    # prefix sums over the calendar: failures / DQ results inside each window in O(1) per date
    def window_count(days: pd.DatetimeIndex) -> np.ndarray:
        c = np.concatenate([[0], np.cumsum(cal.isin(days))])
        return c[i + 1] - c[lo]

    n_px = window_count(pd.DatetimeIndex(px_fail["check_date"]))
    n_unchecked = (lookback + 1) - window_count(checked)
    blocked = (n_px > 0) | dates.isin(pos_fail["check_date"]) | (len(st_fail) > 0)

    out = pd.DataFrame({"status": "ok", "reason": ""}, index=dates)
    out.loc[(n_unchecked > 0) & ~blocked, "status"] = "no_dq"
    out.loc[blocked, "status"] = "blocked"
    static_reason = [f"static {r.rule_id} failed in its latest run {r.run_id}" for r in st_fail.itertuples()]
    for d, lo_d in zip(dates, cal[lo]):
        if out.at[d, "status"] == "ok":
            continue
        pf = pos_fail[pos_fail["check_date"] == d]
        xf = px_fail[(px_fail["check_date"] >= lo_d) & (px_fail["check_date"] <= d)]
        parts = list(static_reason)
        if len(pf):
            parts.append(f"position {sorted(pf['rule_id'])} on {d.date()}")
        if len(xf):
            hits = sorted(f"{r.rule_id}@{r.check_date.date()}" for r in xf.itertuples())
            parts.append(f"price in window {lo_d.date()}..{d.date()}: {hits[:10]}{' ...' if len(hits) > 10 else ''}")
        if not parts:
            window = cal[(cal >= lo_d) & (cal <= d)]
            missing = window[~window.isin(checked)]
            parts.append(f"no DQ result for {len(missing)} day(s) in window, first {missing[0].date()}")
        out.at[d, "reason"] = "; ".join(parts)
    return out


def invalidate_risk(engine: Engine, cfg, start, end) -> pd.DatetimeIndex:
    """Deletes the risk_result rows that rest on data a blocking rule rejected. A static rule failing in its most
    recent run removes every date. Otherwise, per the latest DQ run of each check date in [start, end]: a position
    failure on f removes f; a price failure on f removes f and the next lookback trading days (every window
    holding f), even beyond end. Re-run risk after the data is fixed. Returns the affected dates."""
    scope = blocking_scope(cfg)
    lookback = cfg["lookback_days"]
    pid = cfg["portfolio"]["portfolio_id"]
    rr = s.risk_result.c
    st_fail = _static_failures(engine, scope["static"])
    if len(st_fail):
        stored = read_frame(engine, s.risk_result, portfolio_id=pid, columns=["trade_date"])["trade_date"]
        with engine.begin() as conn:
            n = conn.execute(delete(s.risk_result).where(rr.portfolio_id == pid)).rowcount
        log.warning(
            "static rules %s failed in their latest run: all %d risk_result row(s) of %s deleted",
            sorted(st_fail["rule_id"]), n, pid,
        )
        return pd.DatetimeIndex(sorted(stored.unique()))
    _, fail = _latest_failures(engine, start, end)
    cal = etl.trading_days(engine)
    hit = np.zeros(len(cal) + lookback + 2, dtype=int)
    j = cal.get_indexer(pd.DatetimeIndex(fail.loc[fail["rule_id"].isin(scope["price"]), "check_date"].unique()))
    np.add.at(hit, j, 1)
    np.add.at(hit, j + lookback + 1, -1)
    affected = cal[np.cumsum(hit)[: len(cal)] > 0].union(
        pd.DatetimeIndex(fail.loc[fail["rule_id"].isin(scope["position"]), "check_date"].unique())
    )
    if affected.empty:  # nothing failed, or only static rules whose latest run has since passed
        return affected
    with engine.begin() as conn:
        n = conn.execute(
            delete(s.risk_result).where(rr.portfolio_id == pid, rr.trade_date.in_([d.date() for d in affected]))
        ).rowcount
    log.warning(
        "blocking DQ failures in %s..%s invalidate risk on %d day(s) %s..%s; %d risk_result row(s) deleted",
        pd.Timestamp(start).date(), pd.Timestamp(end).date(), len(affected), affected[0].date(), affected[-1].date(), n,
    )
    return affected


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

    gate = dq_gate(engine, cfg, dates)
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
