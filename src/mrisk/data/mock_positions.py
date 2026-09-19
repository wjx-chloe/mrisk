"""Mock long-only portfolio generator.

Positions change only on the first trading day of each week (small resizes, occasional
same-industry swap). Between rebalances quantities follow corporate actions: quantity is
scaled by the hfq-factor ratio since the last rebalance and rounded to a 100-share lot,
so bonus/split shares are received and sub-lot cash-dividend reinvestment is ignored.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

LOT = 100


@dataclass(frozen=True)
class MockParams:
    portfolio_id: str
    start: pd.Timestamp
    initial_capital: float
    n_holdings: int
    min_industries: int
    rebalance_weight_change: float
    swap_probability: float
    seed: int
    max_initial_weight: float = 0.08
    resize_per_week: int = 4


def _round_lot(q: np.ndarray) -> np.ndarray:
    return np.maximum(np.round(q / LOT), 1) * LOT


def _pick_initial(pool: pd.DataFrame, eligible: set[str], p: MockParams, must: list[str], rng) -> list[str]:
    cand = pool[pool["instrument_id"].isin(eligible)]
    if cand["industry_l1"].nunique() < p.min_industries:
        raise ValueError("pool has fewer eligible industries than min_industries")
    chosen = list(must)
    by_ind = {
        ind: list(rng.permutation(g["instrument_id"].to_numpy()))
        for ind, g in cand[~cand["instrument_id"].isin(must)].groupby("industry_l1")
    }
    order = list(rng.permutation(sorted(by_ind)))
    while len(chosen) < p.n_holdings and any(by_ind.values()):
        for ind in order:
            if by_ind[ind] and len(chosen) < p.n_holdings:
                chosen.append(by_ind[ind].pop())
    if len(chosen) < p.n_holdings:
        raise ValueError(f"only {len(chosen)} eligible instruments, need {p.n_holdings}")
    return chosen


def generate_positions(
    pool: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: pd.Series,
    params: MockParams,
    must_include: list[str] = (),
) -> pd.DataFrame:
    """pool[instrument_id, industry_l1]; prices[trade_date, instrument_id, close, adj_close, is_suspended]
    -> position[trade_date, portfolio_id, instrument_id, quantity]."""
    rng = np.random.default_rng(params.seed)
    days = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values()
    days = days[days >= params.start]
    px = prices[prices["trade_date"].isin(days)].copy()
    px["factor"] = px["adj_close"] / px["close"]
    # upstream positions exist on every trading day even when quotes are missing: carry prices, no trades
    close = px.pivot(index="trade_date", columns="instrument_id", values="close").reindex(days).ffill()
    factor = px.pivot(index="trade_date", columns="instrument_id", values="factor").reindex(days).ffill()
    tradable = px[~px["is_suspended"]].pivot(index="trade_date", columns="instrument_id", values="close").reindex(days).notna()
    industry = pool.set_index("instrument_id")["industry_l1"]

    week = days.to_period("W")
    rebal_days = days[np.r_[True, week[1:] != week[:-1]]]

    d0 = rebal_days[0]
    eligible = set(tradable.columns[tradable.loc[d0].to_numpy()])
    missing = set(must_include) - eligible
    if missing:
        raise ValueError(f"must_include not tradable on start date: {sorted(missing)}")
    held = _pick_initial(pool, eligible, params, list(must_include), rng)
    w = rng.uniform(0.5, 1.5, len(held))
    w = np.minimum(w / w.sum(), params.max_initial_weight)
    base_q = pd.Series(_round_lot(w * params.initial_capital / close.loc[d0, held].to_numpy()), index=held)

    segments = []
    for i, d in enumerate(rebal_days):
        if i > 0:
            base_q = _rebalance(base_q, d, prev_d, factor, close, tradable, industry, must_include, params, rng)
        seg_end = rebal_days[i + 1] if i + 1 < len(rebal_days) else days[-1] + pd.Timedelta(days=1)
        seg_days = days[(days >= d) & (days < seg_end)]
        ratio = factor.loc[seg_days, base_q.index] / factor.loc[d, base_q.index]
        q = _round_lot(ratio.to_numpy() * base_q.to_numpy())
        segments.append(
            pd.DataFrame(q, index=seg_days, columns=base_q.index).rename_axis("trade_date").reset_index()
            .melt(id_vars="trade_date", var_name="instrument_id", value_name="quantity")
        )
        prev_d = seg_days[-1]
        base_q = pd.Series(q[-1], index=base_q.index)
    out = pd.concat(segments, ignore_index=True)
    out.insert(1, "portfolio_id", params.portfolio_id)
    return out.sort_values(["trade_date", "instrument_id"], ignore_index=True)


def _rebalance(base_q, d, prev_d, factor, close, tradable, industry, must_include, p: MockParams, rng) -> pd.Series:
    q = base_q.copy()
    # carry corporate actions from the last day of the previous segment to d
    q[:] = _round_lot(q.to_numpy() * (factor.loc[d, q.index] / factor.loc[prev_d, q.index]).to_numpy())
    can_trade = tradable.loc[d]
    held_tradable = [s for s in q.index if can_trade.get(s, False)]
    k = min(p.resize_per_week, len(held_tradable))
    for s in rng.choice(held_tradable, size=k, replace=False) if k else []:
        q[s] = _round_lot(np.array([q[s] * (1 + rng.uniform(-p.rebalance_weight_change, p.rebalance_weight_change))]))[0]
    if rng.random() < p.swap_probability:
        out_cands = [s for s in held_tradable if s not in must_include]
        if out_cands:
            s_out = rng.choice(out_cands)
            ind = industry[s_out]
            in_cands = [
                s for s in industry.index[industry == ind] if s not in q.index and can_trade.get(s, False)
            ]
            if in_cands:
                s_in = rng.choice(in_cands)
                mv = q[s_out] * close.loc[d, s_out]
                q = q.drop(s_out)
                q[s_in] = _round_lot(np.array([mv / close.loc[d, s_in]]))[0]
    return q


def render_upstream_feed(
    positions: pd.DataFrame, instruments: pd.DataFrame, code_map: pd.DataFrame
) -> pd.DataFrame:
    """Express positions as an upstream system would: exchange + the symbol valid on that date."""
    out = positions.merge(instruments[["instrument_id", "exchange", "symbol"]], on="instrument_id")
    cm = code_map[["instrument_id", "old_symbol", "effective_date"]]
    out = out.merge(cm, on="instrument_id", how="left")
    use_old = out["old_symbol"].notna() & (out["trade_date"] < out["effective_date"])
    out.loc[use_old, "symbol"] = out.loc[use_old, "old_symbol"]
    return out[["trade_date", "portfolio_id", "exchange", "symbol", "quantity"]]
