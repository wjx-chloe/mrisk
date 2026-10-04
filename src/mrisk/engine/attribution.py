"""VaR change attribution (SPEC §5.3): quantity -> revaluation -> window, with Euler per-name contributions.

V(W, P) is a risk measure on return window W with market-value vector P; q quantities, c closes:
    V0 = V(W_{t-1}, q_{t-1} c_{t-1})   yesterday's figure
    V1 = V(W_{t-1}, q_t     c_{t-1})   quantity effect   = V1 - V0
    V2 = V(W_{t-1}, q_t     c_t)       revaluation effect = V2 - V1
    V3 = V(W_t,     q_t     c_t)       window effect     = V3 - V2   (today's figure)
The effects telescope to V3 - V0 exactly. Each V is evaluated by its own call on the same code path, so an
unchanged input gives a bit-identical V and the effect is exactly 0."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from mrisk.engine.returns import returns_window
from mrisk.engine.var import hs_es, hs_var, param_var

METRICS = ("var_hs", "var_param", "es_hs")
EFFECTS = ("quantity", "revaluation", "window", "total")


def risk_components(R: np.ndarray, e: np.ndarray, confidence) -> tuple[dict, dict]:
    """Window R (T, N), market values e (N,) -> (totals {metric: (C,)}, components {metric: (C, N)}).
    Totals are the engine's own hs_var / hs_es / param_var; components are Euler allocations summing to them:
    var_hs: -[(1 - lam) R[d_lo] + lam R[d_hi]] * e on the two scenario days the linear quantile interpolates;
    es_hs: -mean over tail days of R * e (VaR components when the tail is empty, as ES = VaR then);
    var_param: z * e_i (S e)_i / sigma (0 when sigma = 0)."""
    R = np.asarray(R, dtype=float)
    e = np.asarray(e, dtype=float)
    conf = np.atleast_1d(np.asarray(confidence, dtype=float))
    T = R.shape[0]
    port = R @ e  # (T,)
    E = e[:, None]
    var = hs_var(port[:, None], conf)[:, 0]
    es, n_tail = hs_es(port[:, None], var[:, None])
    totals = {"var_hs": var, "es_hs": es[:, 0], "var_param": param_var(R, E, conf)[:, 0]}

    contrib = R * e[None, :]  # (T, N) per-name P&L scenarios
    order = np.argsort(port, kind="stable")
    h = (1.0 - conf) * (T - 1)
    lo = np.floor(h).astype(int)
    lam = h - lo
    hi = np.minimum(lo + 1, T - 1)
    c_var = -((1 - lam)[:, None] * contrib[order[lo]] + lam[:, None] * contrib[order[hi]])

    tail = port[None, :] < -var[:, None]  # (C, T)
    n = tail.sum(axis=1)
    c_es = np.where(n[:, None] > 0, -(tail.astype(float) @ contrib) / np.maximum(n, 1)[:, None], c_var)

    cov = np.atleast_2d(np.cov(R, rowvar=False, ddof=1))
    se = cov @ e
    sigma = np.sqrt(max(float(e @ se), 0.0))
    z = norm.ppf(conf)
    c_par = z[:, None] * (e * se)[None, :] / sigma if sigma > 0 else np.zeros((len(conf), len(e)))
    return totals, {"var_hs": c_var, "es_hs": c_es, "var_param": c_par}


def attribute_change(
    positions_prev: pd.DataFrame,
    positions_cur: pd.DataFrame,
    prices: pd.DataFrame,
    prev,
    asof,
    portfolio_id: str,
    confidence,
    lookback: int,
    return_type: str = "simple",
) -> pd.DataFrame:
    """positions_*[instrument_id, quantity] at the closes of `prev` (t - 1) and `asof` (t); prices[trade_date,
    instrument_id, close, adj_close, is_suspended] covering both windows. -> [metric, confidence, effect, level,
    level_key, value, window_dropped]: portfolio rows carry the effects, instrument rows the Euler contributions
    (summing to the portfolio row); window_dropped is the return date leaving the window, on 'window' rows."""
    prev, asof = pd.Timestamp(prev), pd.Timestamp(asof)
    conf = np.sort(np.atleast_1d(np.asarray(confidence, dtype=float)))
    ids = sorted(set(positions_prev["instrument_id"]) | set(positions_cur["instrument_id"]))
    px = prices[prices["instrument_id"].isin(ids) & (prices["trade_date"] <= asof)]
    wide = px.pivot(index="trade_date", columns="instrument_id")
    wide = wide.reindex(columns=pd.MultiIndex.from_product([wide.columns.levels[0], ids]))
    R0, _ = returns_window(wide["adj_close"], wide["is_suspended"], prev, lookback, return_type)
    R1, _ = returns_window(wide["adj_close"], wide["is_suspended"], asof, lookback, return_type)
    if R0.index[1:].tolist() != R1.index[:-1].tolist():
        raise ValueError(f"{prev.date()} is not the trading day before {asof.date()}")

    def closes(d):
        c = wide["close"].loc[d, ids].to_numpy(dtype=float)
        if np.isnan(c).any() or (c <= 0).any():
            raise ValueError(f"missing or non-positive close on {d.date()} for {[i for i, v in zip(ids, c) if not v > 0]}")
        return c

    def qty(pos):
        q = pos.set_index("instrument_id")["quantity"].reindex(ids).fillna(0.0).to_numpy(dtype=float)
        if (q < 0).any():
            raise ValueError("negative quantity (long only)")
        return q

    c0, c1, q0, q1 = closes(prev), closes(asof), qty(positions_prev), qty(positions_cur)
    evals = [
        risk_components(R0.to_numpy(), q0 * c0, conf),
        risk_components(R0.to_numpy(), q1 * c0, conf),
        risk_components(R0.to_numpy(), q1 * c1, conf),
        risk_components(R1.to_numpy(), q1 * c1, conf),
    ]
    steps = {"quantity": (1, 0), "revaluation": (2, 1), "window": (3, 2), "total": (3, 0)}
    dropped = R0.index[0]
    rows = []
    for metric in METRICS:
        for effect, (b, a) in steps.items():
            tot = evals[b][0][metric] - evals[a][0][metric]
            comp = evals[b][1][metric] - evals[a][1][metric]
            for ci, c in enumerate(conf):
                wd = dropped if effect == "window" else pd.NaT
                rows.append((metric, c, effect, "portfolio", portfolio_id, tot[ci], wd))
                rows.extend((metric, c, effect, "instrument", iid, v, wd) for iid, v in zip(ids, comp[ci]))
    return pd.DataFrame(rows, columns=["metric", "confidence", "effect", "level", "level_key", "value", "window_dropped"])
