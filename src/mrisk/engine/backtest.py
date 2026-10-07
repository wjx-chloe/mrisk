"""VaR backtesting (SPEC §5.4): exceptions, Kupiec POF, Christoffersen independence, Basel traffic light.
0 x ln 0 is taken as 0 throughout (scipy.special.xlogy), which covers x = 0, x = T and empty transition counts."""

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binom, chi2

CRITICAL_5PCT = float(chi2.ppf(0.95, 1))  # 3.8415, exact rather than the rounded 3.84
GREEN_CUM, YELLOW_CUM = 0.95, 0.9999  # Basel cut-offs on the cumulative exception probability


def exceptions(var, pnl) -> np.ndarray:
    """An exception is a loss strictly beyond VaR: pnl < -var."""
    var, pnl = np.asarray(var, dtype=float), np.asarray(pnl, dtype=float)
    if np.isnan(var).any() or np.isnan(pnl).any():
        raise ValueError("VaR or P&L has missing values")
    return pnl < -var


def kupiec_pof(n, x, p) -> tuple[np.ndarray, np.ndarray]:
    """LR = -2 ln[(1-p)^(n-x) p^x] + 2 ln[(1-x/n)^(n-x) (x/n)^x] ~ chi2(1) -> (LR, p-value). Broadcasts."""
    n, x, p = (np.asarray(a, dtype=float) for a in (n, x, p))
    if (n <= 0).any() or (x < 0).any() or (x > n).any():
        raise ValueError("need n > 0 and 0 <= x <= n")
    phat = x / n
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(n - x, 1 - phat) + xlogy(x, phat)
    lr = np.maximum(-2.0 * (ll0 - ll1), 0.0)  # >= 0 at the MLE; clip round-off
    return lr, chi2.sf(lr, 1)


def christoffersen_ind(n00, n01, n10, n11) -> tuple[np.ndarray, np.ndarray]:
    """Independence LR from transition counts n_ij (state i yesterday -> j today) ~ chi2(1) -> (LR, p-value).
    pi01 = n01/(n00+n01), pi11 = n11/(n10+n11), pi = (n01+n11)/total; a rate with a zero denominator only
    multiplies zero counts, so it is set to 0."""
    n00, n01, n10, n11 = (np.asarray(a, dtype=float) for a in (n00, n01, n10, n11))

    def rate(num, den):
        return np.divide(num, den, out=np.zeros(np.broadcast(num, den).shape), where=den > 0)

    pi01, pi11, pi = rate(n01, n00 + n01), rate(n11, n10 + n11), rate(n01 + n11, n00 + n01 + n10 + n11)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = np.maximum(-2.0 * (ll0 - ll1), 0.0)
    return lr, chi2.sf(lr, 1)


def zone_bounds(p: float, n: int) -> tuple[int, int]:
    """Basel traffic-light bounds for n observations at exception rate p, X ~ B(n, p): k exceptions are green while
    P(X <= k) < 95%, yellow while P(X <= k) < 99.99%, red beyond. -> (largest green k, largest yellow k), -1 where
    no count qualifies. n = 250: 4 / 9 at p = 0.01, 17 / 26 at p = 0.05."""
    if not 0 < p < 1 or n < 1:
        raise ValueError(f"need 0 < p < 1 and n >= 1, got p={p}, n={n}")
    c = binom.cdf(np.arange(n + 1), n, p)  # non-decreasing in k
    return int((c < GREEN_CUM).sum()) - 1, int((c < YELLOW_CUM).sum()) - 1


def traffic_light(x, p: float, n: int) -> np.ndarray:
    """Zone of each exception count x over n observations at exception rate p (bounds from zone_bounds)."""
    green, yellow = zone_bounds(p, n)
    x = np.asarray(x)
    return np.select([x <= green, x <= yellow], ["green", "yellow"], "red")


def rolling_backtest(hit, valid, window: int, p: float) -> pd.DataFrame:
    """Per observation k (in order), the backtest over observations k - window + 1 .. k.
    hit: exception flags; valid: whether observation k exists (a gap from a DQ block is invalid).
    Returns [complete, n_obs, n_exceptions, n00, n01, n10, n11, lr_pof, p_value_pof, lr_ind, p_value_ind];
    statistics are NaN where the window is not complete. Window sums via prefix sums, no loop over dates."""
    hit = np.asarray(hit, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if hit.shape != valid.shape or hit.ndim != 1:
        raise ValueError("hit and valid must be 1-D and the same length")
    h = hit & valid
    M = len(h)

    def wsum(a, w):  # sum of a[k - w + 1 .. k]; incomplete leading windows are masked by `complete`
        c = np.concatenate([[0], np.cumsum(a)])
        k = np.arange(len(a))
        return c[k + 1] - c[np.maximum(k + 1 - w, 0)]

    n_obs = wsum(valid, window)
    complete = n_obs == window
    x = wsum(h, window)
    # transitions (k' - 1 -> k') for k' inside the window: the window - 1 pairs ending at k
    a, b = h[:-1], h[1:]
    pairs = {"n00": ~a & ~b, "n01": ~a & b, "n10": a & ~b, "n11": a & b}
    counts = {k: np.concatenate([[0], wsum(v, window - 1)[: M - 1]]) for k, v in pairs.items()}

    out = pd.DataFrame({"complete": complete, "n_obs": n_obs, "n_exceptions": x, **counts})
    full = complete.copy()
    lr, pv = kupiec_pof(np.where(full, window, 1), np.where(full, x, 0), p)
    li, pi = christoffersen_ind(*(np.where(full, out[k], 0) for k in ("n00", "n01", "n10", "n11")))
    out["lr_pof"] = np.where(full, lr, np.nan)
    out["p_value_pof"] = np.where(full, pv, np.nan)
    out["lr_ind"] = np.where(full, li, np.nan)
    out["p_value_ind"] = np.where(full, pi, np.nan)
    return out
