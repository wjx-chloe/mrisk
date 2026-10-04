"""Hypothetical P&L for backtesting (SPEC §3.12): hold the close positions of t to the next price date t + 1."""

import numpy as np
import pandas as pd

PNL_COLUMNS = ["position_date", "pnl_date", "level", "level_key", "market_value", "pnl"]


def hypothetical_pnl(
    positions: pd.DataFrame, prices: pd.DataFrame, instruments: pd.DataFrame, portfolio_id: str
) -> pd.DataFrame:
    """positions[trade_date, instrument_id, quantity] for any number of position dates t;
    prices[trade_date, instrument_id, close, adj_close, is_suspended] covering each t and its next price date;
    instruments[instrument_id, industry_l1].
    -> PNL_COLUMNS for every t that has a next price date, at portfolio / industry / instrument level:
    market_value = quantity x close_t, pnl = market_value x (adj_close_{t+1} / adj_close_t - 1), the same basis
    as the VaR P&L; 0 where the name is suspended on t + 1. Vectorised across dates and names."""
    q = positions.pivot(index="trade_date", columns="instrument_id", values="quantity").fillna(0.0)
    days = pd.DatetimeIndex(np.sort(prices["trade_date"].unique()))
    pos_idx = days.get_indexer(q.index)
    if (pos_idx < 0).any():
        raise ValueError(f"no prices on position dates {[d.date().isoformat() for d in q.index[pos_idx < 0]]}")
    realised = pos_idx + 1 < len(days)
    q, pos_idx = q[realised], pos_idx[realised]
    if q.empty:
        return pd.DataFrame(columns=PNL_COLUMNS)
    t0, t1 = days[pos_idx], days[pos_idx + 1]
    ids = q.columns

    wide = prices[prices["instrument_id"].isin(ids)].pivot(index="trade_date", columns="instrument_id")
    wide = wide.reindex(columns=pd.MultiIndex.from_product([["close", "adj_close", "is_suspended"], ids]))
    close0 = wide["close"].reindex(t0).to_numpy(dtype=float)
    adj0 = wide["adj_close"].reindex(t0).to_numpy(dtype=float)
    adj1 = wide["adj_close"].reindex(t1).to_numpy(dtype=float)
    susp1 = wide["is_suspended"].reindex(t1)

    Q = q.to_numpy()
    held = Q > 0
    bad = held & (np.isnan(close0) | np.isnan(adj0) | np.isnan(adj1) | susp1.isna().to_numpy())
    if bad.any():
        r, c = np.nonzero(bad)
        pairs = sorted({(t0[i].date().isoformat(), ids[j]) for i, j in zip(r, c)})
        raise ValueError(f"missing prices on t or t + 1 for held (position date, name): {pairs[:10]}")
    ret = np.where(held & ~susp1.fillna(False).to_numpy(dtype=bool), adj1 / np.where(held, adj0, 1.0) - 1.0, 0.0)
    mv = np.where(held, Q * np.nan_to_num(close0), 0.0)
    pnl = mv * ret

    ind = instruments.set_index("instrument_id")["industry_l1"].reindex(ids)
    if ind[held.any(axis=0)].isna().any():
        raise ValueError(f"missing industry_l1 for {ind[held.any(axis=0) & ind.isna().to_numpy()].index.tolist()}")
    industries = np.unique(ind.dropna().to_numpy())
    G = (ind.to_numpy()[:, None] == industries[None, :]).astype(float)  # (N, K) membership

    frames = [
        _long(t0, t1, "portfolio", [portfolio_id], mv.sum(axis=1, keepdims=True), pnl.sum(axis=1, keepdims=True)),
        _long(t0, t1, "industry", industries, mv @ G, pnl @ G),
        _long(t0, t1, "instrument", ids, mv, pnl, keep=held),
    ]
    out = pd.concat(frames, ignore_index=True)
    out = out[(out["level"] != "industry") | (out["market_value"] > 0)]  # industries not held that day
    return out.sort_values(["position_date", "level", "level_key"], ignore_index=True)[PNL_COLUMNS]


def _long(t0, t1, level, keys, mv, pnl, keep=None) -> pd.DataFrame:
    D, K = mv.shape
    m = np.ones((D, K), dtype=bool) if keep is None else keep
    r, c = np.nonzero(m)
    return pd.DataFrame(
        {
            "position_date": t0[r],
            "pnl_date": t1[r],
            "level": level,
            "level_key": np.asarray(keys)[c],
            "market_value": mv[r, c],
            "pnl": pnl[r, c],
        }
    )
