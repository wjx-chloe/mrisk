"""1-day VaR and ES (SPEC §5.2 steps 4-6). Every function works column-wise on a (T, K) P&L matrix or a
(N, K) exposure matrix, so one call covers all risk units and all confidence levels. Losses are positive."""

import numpy as np
from scipy.stats import norm


def _confidence(confidence) -> np.ndarray:
    c = np.atleast_1d(np.asarray(confidence, dtype=float))
    if c.ndim != 1 or not ((c > 0) & (c < 1)).all():
        raise ValueError(f"confidence must be a list of levels in (0, 1), got {confidence!r}")
    return c


def _matrix(a, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {a.shape}")
    return a


def hs_var(pnl, confidence) -> np.ndarray:
    """Historical-simulation VaR: -quantile(pnl, 1 - c) per column, numpy linear interpolation. (T, K) -> (C, K)."""
    pnl = _matrix(pnl, "pnl")
    return -np.quantile(pnl, 1.0 - _confidence(confidence), axis=0, method="linear")


def hs_es(pnl, var) -> tuple[np.ndarray, np.ndarray]:
    """Historical-simulation ES: -mean(pnl[pnl < -VaR]) per column. (T, K), (C, K) -> (es, n_tail), both (C, K).
    When no observation lies strictly beyond VaR (ties, or a flat series), ES = VaR; n_tail = 0 flags it."""
    pnl = _matrix(pnl, "pnl")
    var = _matrix(var, "var")
    tail = pnl[None, :, :] < -var[:, None, :]
    n_tail = tail.sum(axis=1)
    loss = -np.where(tail, pnl[None, :, :], 0.0).sum(axis=1)
    es = np.divide(loss, n_tail, out=var.copy(), where=n_tail > 0)
    return es, n_tail


def param_sigma(R, E) -> np.ndarray:
    """P&L standard deviation per risk unit: sqrt(diag(E' S E)) with S = cov(R), sample mean removed, ddof=1.
    (T, N), (N, K) -> (K,)."""
    R = _matrix(R, "R")
    E = _matrix(E, "E")
    if R.shape[1] != E.shape[0]:
        raise ValueError(f"R has {R.shape[1]} instruments, E has {E.shape[0]}")
    cov = np.atleast_2d(np.cov(R, rowvar=False, ddof=1))
    q = ((cov @ E) * E).sum(axis=0)
    # E'SE >= 0 for a covariance matrix; clip round-off below zero, e.g. a column that never moved
    return np.sqrt(np.maximum(q, 0.0))


def param_var(R, E, confidence) -> np.ndarray:
    """Variance-covariance VaR under normality with zero mean: z_c x sigma, z_c = norm.ppf(c) exactly.
    (T, N), (N, K) -> (C, K)."""
    z = norm.ppf(_confidence(confidence))
    return z[:, None] * param_sigma(R, E)[None, :]


def hs_var_components(parts, port, confidence) -> np.ndarray:
    """Euler allocation of the historical-simulation VaR of `port` to the columns of `parts` (T, K), the P&L of the
    pieces port is made of (port = parts summed over the pieces): -[(1 - lam) parts[d_lo] + lam parts[d_hi]] on the
    two scenario days the linear quantile interpolates. (T, K), (T,) -> (C, K)."""
    parts = _matrix(parts, "parts")
    port = np.asarray(port, dtype=float)
    conf = _confidence(confidence)
    T = port.shape[0]
    order = np.argsort(port, kind="stable")
    h = (1.0 - conf) * (T - 1)
    lo = np.floor(h).astype(int)
    lam = h - lo
    hi = np.minimum(lo + 1, T - 1)
    return -((1 - lam)[:, None] * parts[order[lo]] + lam[:, None] * parts[order[hi]])


def hs_es_components(parts, port, var, confidence) -> np.ndarray:
    """Euler allocation of the historical-simulation ES of `port` to the columns of `parts`: minus the mean of each
    column over the tail days of port (port < -VaR, the days hs_es averages), so the pieces of port sum to its ES.
    A piece can contribute a negative amount (it gains on the tail days on average). With no day strictly beyond VaR,
    ES = VaR and the VaR components are returned. (T, K), (T,), (C,) -> (C, K)."""
    parts = _matrix(parts, "parts")
    port = np.asarray(port, dtype=float)
    var = np.asarray(var, dtype=float)
    tail = port[None, :] < -var[:, None]  # (C, T)
    n = tail.sum(axis=1)
    return np.where(n[:, None] > 0, -(tail.astype(float) @ parts) / np.maximum(n, 1)[:, None],
                    hs_var_components(parts, port, confidence))
