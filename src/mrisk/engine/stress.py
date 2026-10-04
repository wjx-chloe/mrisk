"""Stress testing (SPEC §6.1): historical scenarios (base close -> end close, buy and hold) with gap proxies,
and hypothetical factor shocks propagated to names by beta. Matrices are (dates x names); no loop over dates."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_PROXY_NAMES = 3
STRESS_COLUMNS = [
    "level", "level_key", "market_value", "pnl", "max_daily_loss", "proxy_mv_unlisted", "proxy_mv_suspended", "pnl_proxy",
]


def proxy_returns(adj: pd.DataFrame, suspended: pd.DataFrame, industry: pd.Series, min_names: int = MIN_PROXY_NAMES) -> pd.DataFrame:
    """Per date (from the second row) and name: the equal-weight return of the name's industry peers that really
    traded on that day and the day before; the whole pool's when fewer than `min_names` peers did.
    adj / suspended: (dates x pool names); industry: name -> industry for every column."""
    a = adj.to_numpy(dtype=float)
    s = suspended.reindex_like(adj).fillna(False).to_numpy(dtype=bool)
    listed = ~np.isnan(a)
    clean = listed[1:] & listed[:-1] & ~s[1:] & ~s[:-1]  # traded on d and d - 1: a plain one-day return
    r = np.where(clean, np.divide(a[1:], a[:-1], out=np.ones_like(a[1:]), where=clean) - 1.0, 0.0)
    ind = industry.reindex(adj.columns).to_numpy()
    if pd.isna(ind).any():
        raise ValueError(f"missing industry for {adj.columns[pd.isna(ind)].tolist()}")
    inds = np.unique(ind)
    G = (ind[:, None] == inds[None, :]).astype(float)  # (N, K)
    n_k = clean.astype(float) @ G
    mean_k = np.divide(r @ G, n_k, out=np.full(n_k.shape, np.nan), where=n_k > 0)
    n_pool = clean.sum(axis=1)
    if (n_pool == 0).any():
        bad = adj.index[1:][n_pool == 0]
        raise ValueError(f"no name traded on {[d.date().isoformat() for d in bad]}, no proxy possible")
    pool = r.sum(axis=1) / n_pool
    per_ind = np.where(n_k >= min_names, mean_k, pool[:, None])  # (D-1, K)
    return pd.DataFrame(per_ind @ G.T, index=adj.index[1:], columns=adj.columns)


@dataclass(frozen=True)
class ScenarioPath:
    returns: pd.DataFrame  # (window days, names): daily returns on (base, end]
    is_proxy: pd.DataFrame  # same shape: the return is a proxy (unlisted or suspended day)
    unlisted: pd.Series  # name had an unlisted (proxied) day in the window
    suspended: pd.Series  # name had a suspended (proxied) day in the window and none unlisted
    price_dates: pd.DatetimeIndex  # every price date the window's numbers were read from


def scenario_path(adj: pd.DataFrame, suspended: pd.DataFrame, industry: pd.Series, base, end) -> ScenarioPath:
    """Daily returns of every column over (base, end] with gap proxies, reconciled per suspension segment:
    on the full series, suspended days take the proxy and the resumption day r takes the residual that makes the
    segment compound to the real P_r / P_{s-1}; then only days in (base, end] are kept. Unlisted days (no price,
    or the first listed day) take the proxy too. `adj` must reach back to the day before every suspension segment
    that overlaps the window, and to base."""
    base, end = pd.Timestamp(base), pd.Timestamp(end)
    idx = adj.index
    if base not in idx or end not in idx or base >= end:
        raise ValueError(f"need prices on base {base.date()} and end {end.date()}, base before end")
    a = adj.to_numpy(dtype=float)
    s = suspended.reindex_like(adj).fillna(False).to_numpy(dtype=bool)
    listed = ~np.isnan(a)
    s &= listed
    proxy = proxy_returns(adj, suspended, industry).to_numpy()  # rows 1..D-1
    D, N = a.shape

    # last traded (listed, not suspended) row at or before each row; -1 if none
    traded = listed & ~s
    rows = np.arange(D)[:, None]
    last_traded = np.maximum.accumulate(np.where(traded, rows, -1), axis=0)
    # running product of (1 + proxy) over suspended days, for segment products by ratio
    grow = np.vstack([np.ones((1, N)), np.where(s[1:], 1.0 + proxy, 1.0)])
    C = np.cumprod(grow, axis=0)

    ret = np.full((D, N), np.nan)
    unl = np.zeros((D, N), dtype=bool)
    sus = np.zeros((D, N), dtype=bool)
    r1 = slice(1, D)
    first_listed = listed[1:] & ~listed[:-1]
    unl[r1] = ~listed[1:] | first_listed
    sus[r1] = s[1:] & ~unl[r1]
    resume = traded[1:] & s[:-1]  # traded today, suspended yesterday
    plain = traded[1:] & traded[:-1]
    ret[r1] = np.where(unl[r1] | sus[r1], proxy, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret[r1] = np.where(plain, a[1:] / a[:-1] - 1.0, ret[r1])
        prev = last_traded[:-1]  # s - 1 for a segment ending yesterday
        k = np.clip(prev, 0, D - 1)
        cols = np.arange(N)[None, :]
        seg = C[:-1] / C[k, cols]  # product of (1 + proxy) over s .. r - 1
        resid = (a[1:] / a[k, cols]) / seg - 1.0
    ret[r1] = np.where(resume, resid, ret[r1])

    i0, i1 = idx.get_loc(base), idx.get_loc(end)
    win = slice(i0 + 1, i1 + 1)
    # suspension segments that put a number into the window read prices back to their s - 1:
    # a suspended window row's last_traded is s - 1; a resumption row's s - 1 is last_traded of the row before
    sus_win = s[win]
    res_win = resume[i0:i1]  # resume[k] is about row k + 1, so this covers rows i0 + 1 .. i1
    starts = np.concatenate([last_traded[win][sus_win], last_traded[i0:i1][res_win]])
    if (starts < 0).any():
        bad = adj.columns[(sus_win & (last_traded[win] < 0)).any(axis=0) | (res_win & (last_traded[i0:i1] < 0)).any(axis=0)]
        raise ValueError(f"suspension starting before the first price row for {bad.tolist()}; read prices further back")
    lo = int(min(i0, starts.min())) if len(starts) else i0

    window = idx[win]
    R = pd.DataFrame(ret[win], index=window, columns=adj.columns)
    if R.isna().any().any():
        raise ValueError("internal: unresolved return in the scenario window")
    U = pd.DataFrame(unl[win], index=window, columns=adj.columns)
    S = pd.DataFrame(sus[win], index=window, columns=adj.columns)
    u_any = U.any(axis=0)
    return ScenarioPath(R, U | S, u_any, S.any(axis=0) & ~u_any, idx[lo : i1 + 1])


def historical_stress(mv: pd.Series, industry: pd.Series, path: ScenarioPath, portfolio_id: str) -> pd.DataFrame:
    """mv: held name -> market value on the position date. Buy and hold through the path:
    pnl = mv x (prod(1 + r) - 1); daily P&L along the path gives max_daily_loss (positive, 0 if no loss day);
    pnl_proxy is the part of pnl earned on proxy days. -> STRESS_COLUMNS, portfolio / industry / instrument."""
    ids = mv.index
    missing = ids.difference(path.returns.columns)
    if len(missing):
        raise ValueError(f"no scenario path for {missing.tolist()}")
    R = path.returns[ids].to_numpy()
    P = path.is_proxy[ids].to_numpy()
    v = mv.to_numpy(dtype=float)
    G = np.vstack([np.ones((1, len(ids))), np.cumprod(1.0 + R, axis=0)])
    daily = (G[1:] - G[:-1]) * v  # (D, n)
    pnl = daily.sum(axis=0)
    pnl_proxy = (daily * P).sum(axis=0)
    return _levels(
        ids, industry, portfolio_id, v, pnl, daily, pnl_proxy,
        v * path.unlisted.reindex(ids).to_numpy(dtype=float), v * path.suspended.reindex(ids).to_numpy(dtype=float),
    )


def betas(R: pd.DataFrame, factor: pd.Series) -> pd.Series:
    """beta_i = cov(r_i, f) / var(f) over the window (sample moments, mean removed)."""
    f = factor.to_numpy(dtype=float)
    fc = f - f.mean()
    var = fc @ fc
    if var <= 0:
        raise ValueError("factor has zero variance over the window")
    X = R.to_numpy(dtype=float)
    return pd.Series((X - X.mean(axis=0)).T @ fc / var, index=R.columns)


def hypothetical_shocks(definition: dict, R_pool: pd.DataFrame, industry: pd.Series, names) -> pd.Series:
    """Per-name return shock. Factor mode ({market: s} or {industry: {k: s}}, one factor): E[r_i | f = s] = beta_i s,
    the market factor being the pool's equal-weight return and an industry factor its members' over the window
    R_pool (the VaR window of the position date, every pool name). Raw mode ({raw: {...}}): shocks as given,
    instrument > industry > market."""
    names = pd.Index(names)
    if "raw" in definition:
        raw = definition["raw"]
        ind = industry.reindex(names)
        out = pd.Series(float(raw.get("market", 0.0)), index=names)
        out = out.where(~ind.isin(list(raw.get("industry", {}))), ind.map(raw.get("industry", {})).astype(float))
        inst = raw.get("instrument", {})
        out = out.where(~names.isin(list(inst)), names.map(inst).astype(float))
        return out
    keys = [k for k in ("market", "industry") if k in definition]
    if len(keys) != 1 or (keys == ["industry"] and len(definition["industry"]) != 1):
        raise ValueError(f"factor scenario needs exactly one factor shock, got {definition}")
    missing = names.difference(R_pool.columns)
    if len(missing):
        raise ValueError(f"names outside the pool window: {missing.tolist()}")
    if keys == ["market"]:
        factor, s = R_pool.mean(axis=1), float(definition["market"])
    else:
        (k, s), = definition["industry"].items()
        members = industry.index[industry == k].intersection(R_pool.columns)
        if members.empty:
            raise ValueError(f"no pool names in industry {k}")
        factor, s = R_pool[members].mean(axis=1), float(s)
    return betas(R_pool[names], factor) * s


def hypothetical_stress(mv: pd.Series, industry: pd.Series, shocks: pd.Series, portfolio_id: str) -> pd.DataFrame:
    """pnl = mv x shock per name, aggregated. -> STRESS_COLUMNS (no path, so max_daily_loss is empty)."""
    ids = mv.index
    v = mv.to_numpy(dtype=float)
    pnl = v * shocks.reindex(ids).to_numpy(dtype=float)
    out = _levels(ids, industry, portfolio_id, v, pnl, None, np.zeros(len(ids)), np.zeros(len(ids)), np.zeros(len(ids)))
    out["max_daily_loss"] = np.nan
    return out


def rank_scenarios(portfolio_rows: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Portfolio-level stress rows [scenario_id, pnl, ...] -> the top_n largest losses, worst first."""
    return portfolio_rows.sort_values("pnl", kind="stable").head(top_n).reset_index(drop=True)


def _levels(ids, industry, portfolio_id, v, pnl, daily, pnl_proxy, mv_unl, mv_sus) -> pd.DataFrame:
    ind = industry.reindex(ids)
    if ind.isna().any():
        raise ValueError(f"missing industry for {ind.index[ind.isna()].tolist()}")
    inds = np.unique(ind.to_numpy())
    G = (ind.to_numpy()[:, None] == inds[None, :]).astype(float)
    ones = np.ones((len(ids), 1))
    blocks = []
    for level, keys, M in (("portfolio", [portfolio_id], ones), ("industry", list(inds), G), ("instrument", list(ids), np.eye(len(ids)))):
        mdl = np.maximum(-(daily @ M).min(axis=0), 0.0) if daily is not None and len(daily) else np.full(M.shape[1], np.nan)
        blocks.append(
            pd.DataFrame(
                {
                    "level": level, "level_key": keys, "market_value": v @ M, "pnl": pnl @ M, "max_daily_loss": mdl,
                    "proxy_mv_unlisted": mv_unl @ M, "proxy_mv_suspended": mv_sus @ M, "pnl_proxy": pnl_proxy @ M,
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)[STRESS_COLUMNS]
