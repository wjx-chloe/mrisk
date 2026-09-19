import numpy as np
import pandas as pd
import pytest

from mrisk.data.code_map import resolve_instrument_id
from mrisk.data.mock_positions import MockParams, generate_positions, render_upstream_feed

DAYS = pd.bdate_range("2025-01-06", periods=40)
IDS = [f"S{i:02d}" for i in range(12)]
INDS = ["银行", "电子", "医药生物", "汽车"]
SPLIT_ID, SPLIT_DAY = "S01", DAYS[12]  # 10送10 on a Wednesday


def _pool():
    return pd.DataFrame({"instrument_id": IDS, "industry_l1": [INDS[i % 4] for i in range(12)]})


def _prices():
    rng = np.random.default_rng(0)
    rows = []
    for k, s in enumerate(IDS):
        close = 10 * np.exp(np.cumsum(rng.normal(0, 0.01, len(DAYS)))) + k
        factor = np.ones(len(DAYS))
        if s == SPLIT_ID:
            post = DAYS >= SPLIT_DAY
            close[post] /= 2
            factor[post] = 2.0
        rows.append(pd.DataFrame({"trade_date": DAYS, "instrument_id": s, "close": close, "adj_close": close * factor, "is_suspended": False}))
    return pd.concat(rows, ignore_index=True)


def _params(**kw):
    base = dict(
        portfolio_id="P1", start=DAYS[0], initial_capital=1e7, n_holdings=6, min_industries=3,
        rebalance_weight_change=0.1, swap_probability=0.5, seed=7,
    )
    base.update(kw)
    return MockParams(**base)


@pytest.fixture(scope="module")
def pos():
    return generate_positions(_pool(), _prices(), pd.Series(DAYS), _params(), must_include=[SPLIT_ID])


def test_shape_and_invariants(pos):
    assert set(pos["trade_date"]) == set(DAYS)
    assert (pos.groupby("trade_date").size() == 6).all()
    assert (pos["quantity"] > 0).all()
    assert (pos["quantity"] % 100 == 0).all()
    inds = pos.merge(_pool(), on="instrument_id").groupby("trade_date")["industry_l1"].nunique()
    assert (inds >= 3).all()
    assert pos["instrument_id"].nunique() > 6  # at least one swap happened


def test_changes_only_on_rebalance_days_or_corporate_action(pos):
    wide = pos.pivot(index="trade_date", columns="instrument_id", values="quantity").fillna(0)
    changed = wide.ne(wide.shift()).iloc[1:]
    week = DAYS.to_period("W")
    first_of_week = set(DAYS[np.r_[True, week[1:] != week[:-1]]])
    for day, row in changed.iterrows():
        if day not in first_of_week:
            assert set(row.index[row]) <= {SPLIT_ID}, day


def test_split_doubles_quantity_and_keeps_mv(pos):
    q = pos[pos["instrument_id"] == SPLIT_ID].set_index("trade_date")["quantity"]
    i = DAYS.get_loc(SPLIT_DAY)
    assert q[DAYS[i]] == 2 * q[DAYS[i - 1]]


def test_initial_mv_close_to_capital(pos):
    px = _prices()
    d0 = pos[pos["trade_date"] == DAYS[0]].merge(px, on=["trade_date", "instrument_id"])
    mv = (d0["quantity"] * d0["close"]).sum()
    assert 0.4e7 < mv <= 1e7


def test_deterministic_given_seed():
    a = generate_positions(_pool(), _prices(), pd.Series(DAYS), _params(), must_include=[SPLIT_ID])
    b = generate_positions(_pool(), _prices(), pd.Series(DAYS), _params(), must_include=[SPLIT_ID])
    pd.testing.assert_frame_equal(a, b)


def test_must_include_held_throughout(pos):
    assert (pos.groupby("trade_date")["instrument_id"].apply(lambda s: SPLIT_ID in set(s))).all()


def test_positions_extend_over_days_without_quotes():
    px = _prices()
    px = px[px["trade_date"] < DAYS[-2]]
    pos = generate_positions(_pool(), px, pd.Series(DAYS), _params(), must_include=[SPLIT_ID])
    wide = pos.pivot(index="trade_date", columns="instrument_id", values="quantity")
    assert wide.index[-1] == DAYS[-1]
    pd.testing.assert_series_equal(wide.loc[DAYS[-1]], wide.loc[DAYS[-3]], check_names=False)


def test_too_few_industries_raises():
    with pytest.raises(ValueError):
        generate_positions(_pool(), _prices(), pd.Series(DAYS), _params(min_industries=5))


def test_upstream_feed_uses_old_code_before_switch_and_resolves_back(pos):
    instruments = pd.DataFrame({"instrument_id": IDS, "exchange": "BJ", "symbol": [f"92{i:04d}" for i in range(12)]})
    switch = DAYS[20]
    cm = pd.DataFrame({"instrument_id": [SPLIT_ID], "old_symbol": ["834001"], "new_symbol": ["920001"], "effective_date": [switch]})
    feed = render_upstream_feed(pos, instruments, cm)
    s = feed[feed["symbol"].isin(["834001", "920001"])]
    assert set(s.loc[s["trade_date"] < switch, "symbol"]) == {"834001"}
    assert set(s.loc[s["trade_date"] >= switch, "symbol"]) == {"920001"}
    back = resolve_instrument_id(feed, instruments, cm)
    pd.testing.assert_frame_equal(
        back[["trade_date", "portfolio_id", "instrument_id", "quantity"]].sort_values(["trade_date", "instrument_id"], ignore_index=True),
        pos,
    )
