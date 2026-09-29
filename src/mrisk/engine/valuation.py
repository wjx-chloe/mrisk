"""Market value per holding and the exposure matrix that maps holdings to risk units (SPEC §5.2 steps 1, 3)."""

import numpy as np
import pandas as pd

LEVELS = ("portfolio", "industry", "instrument")


def market_value(positions: pd.DataFrame, close: pd.Series, industry: pd.Series) -> pd.DataFrame:
    """positions[instrument_id, quantity] for one date, close and industry indexed by instrument_id
    -> [instrument_id, industry_l1, quantity, close, market_value, weight], sorted by instrument_id.
    mv = quantity x unadjusted close; weight = mv / total mv. Long only."""
    pos = positions[["instrument_id", "quantity"]]
    if pos.empty:
        raise ValueError("no positions")
    dup = pos["instrument_id"][pos["instrument_id"].duplicated()]
    if len(dup):
        raise ValueError(f"duplicate positions for {sorted(dup.unique())}")
    if (pos["quantity"] <= 0).any():
        raise ValueError(f"non-positive quantity (long only) for {sorted(pos.loc[pos['quantity'] <= 0, 'instrument_id'])}")
    out = pos.sort_values("instrument_id", ignore_index=True)
    out["close"] = out["instrument_id"].map(close)
    bad = out.loc[out["close"].isna() | (out["close"] <= 0), "instrument_id"]
    if len(bad):
        raise ValueError(f"missing or non-positive close for {bad.tolist()}")
    ind = out["instrument_id"].map(industry)
    bad = out.loc[ind.isna() | (ind.astype(str).str.strip() == ""), "instrument_id"]
    if len(bad):
        raise ValueError(f"missing industry_l1 for {bad.tolist()}")
    out.insert(1, "industry_l1", ind)
    out["market_value"] = out["quantity"] * out["close"]
    out["weight"] = out["market_value"] / out["market_value"].sum()
    return out


def exposure_matrix(mv: pd.DataFrame, portfolio_key: str) -> tuple[np.ndarray, pd.DataFrame]:
    """market_value() output -> (E, units). E is (N, K): column k holds the market values of the holdings
    in risk unit k and 0 elsewhere, so R @ E is the P&L of every unit at once. Units are ordered
    portfolio, industries (sorted), instruments (as in mv); units has [level, level_key, market_value]."""
    v = mv["market_value"].to_numpy(dtype=float)
    ind = mv["industry_l1"].to_numpy()
    industries = np.unique(ind)
    E = np.hstack([v[:, None], np.where(ind[:, None] == industries[None, :], v[:, None], 0.0), np.diag(v)])
    units = pd.DataFrame(
        {
            "level": ["portfolio"] + ["industry"] * len(industries) + ["instrument"] * len(v),
            "level_key": [portfolio_key, *industries.tolist(), *mv["instrument_id"].tolist()],
        }
    )
    units["market_value"] = E.sum(axis=0)
    return E, units
