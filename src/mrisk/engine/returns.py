"""Return matrix over the lookback window (SPEC §5.2 step 2)."""

import numpy as np
import pandas as pd

RETURN_TYPES = ("simple", "log")


def returns_window(
    adj_close: pd.DataFrame,
    suspended: pd.DataFrame,
    asof,
    lookback: int,
    kind: str = "simple",
) -> tuple[pd.DataFrame, pd.Series]:
    """Wide adj_close / is_suspended frames (trade_date x instrument_id) -> (R, suspended_days).
    R is the (lookback, N) matrix of the `lookback` daily returns ending on `asof` inclusive, built from the
    last lookback + 1 prices. Returns on suspended days are set to 0 (SPEC §5.2); suspended_days counts
    them per instrument. Never shortens the window: too little history or any gap raises."""
    if kind not in RETURN_TYPES:
        raise ValueError(f"return type must be one of {RETURN_TYPES}, got {kind!r}")
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2, got {lookback}")
    idx = adj_close.index
    if not (idx.is_unique and idx.is_monotonic_increasing):
        raise ValueError("adj_close index must be unique and sorted")
    asof = pd.Timestamp(asof)
    if asof not in idx:
        raise ValueError(f"no prices on {asof.date()}")
    end = idx.get_loc(asof) + 1
    if end < lookback + 1:
        raise ValueError(f"need {lookback + 1} prices up to {asof.date()}, have {end}")
    px = adj_close.iloc[end - lookback - 1 : end]
    gaps = px.columns[px.isna().any()]
    if len(gaps):
        raise ValueError(f"missing adj_close in the window ending {asof.date()} for {gaps.tolist()}")
    p = px.to_numpy(dtype=float)
    if (p <= 0).any():
        raise ValueError(f"non-positive adj_close for {px.columns[(p <= 0).any(axis=0)].tolist()}")
    susp = suspended.reindex(index=px.index[1:], columns=px.columns)
    if susp.isna().any().any():
        raise ValueError("is_suspended does not cover the return window")
    susp = susp.to_numpy(dtype=bool)
    ratio = p[1:] / p[:-1]
    r = ratio - 1.0 if kind == "simple" else np.log(ratio)
    r = np.where(susp, 0.0, r)
    return (
        pd.DataFrame(r, index=px.index[1:], columns=px.columns),
        pd.Series(susp.sum(axis=0), index=px.columns, name="suspended_days"),
    )
