import numpy as np
import pandas as pd
import pytest

from mrisk.engine.pnl import PNL_COLUMNS, hypothetical_pnl

D = pd.bdate_range("2025-01-06", periods=4)
# A (银行): plain moves; suspended on D4 with a moved adj_close that must be ignored
# B (电子): 10-for-10 bonus on D3, close halves but adj_close (factor 1 -> 2) is flat
PRICES = pd.DataFrame(
    {
        "trade_date": list(D) * 2,
        "instrument_id": ["A"] * 4 + ["B"] * 4,
        "close": [10.0, 11.0, 10.5, 10.5, 20.0, 20.0, 10.0, 10.2],
        "adj_close": [10.0, 11.0, 10.5, 11.0, 20.0, 20.0, 20.0, 20.4],
        "is_suspended": [False, False, False, True, False, False, False, False],
    }
)
POS = pd.DataFrame(
    {
        "trade_date": list(D) * 2,
        "instrument_id": ["A"] * 4 + ["B"] * 4,
        "quantity": [100.0, 100.0, 200.0, 200.0, 50.0, 50.0, 100.0, 100.0],
    }
)
INST = pd.DataFrame({"instrument_id": ["A", "B"], "industry_l1": ["银行", "电子"]})


def test_hypothetical_pnl_hand_computed():
    # t = D1: A 100 x 10 = 1000, r 11/10 - 1 = 0.1 -> 100 ; B 50 x 20 = 1000, r 0 -> 0       ; P 2000, 100
    # t = D2: A 100 x 11 = 1100, r 10.5/11 - 1 -> -50  ; B 50 x 20 = 1000, ex-date: adj 20/20 -> 0
    #         (an unadjusted price difference would book 50 x (10 - 20) = -500 on the bonus issue)      ; P 2100, -50
    # t = D3: A 200 x 10.5 = 2100, D4 suspended -> 0 (adj 11/10.5 ignored) ; B 100 x 10 = 1000, r 0.02 -> 20 ; P 3100, 20
    # t = D4: no next price date, not realised
    raw = hypothetical_pnl(POS, PRICES, INST, "P")
    assert list(raw.columns) == PNL_COLUMNS
    out = raw.set_index(["position_date", "level_key"])
    exp = {
        (D[0], "P"): (2000.0, 100.0), (D[0], "A"): (1000.0, 100.0), (D[0], "B"): (1000.0, 0.0),
        (D[1], "P"): (2100.0, -50.0), (D[1], "A"): (1100.0, -50.0), (D[1], "B"): (1000.0, 0.0),
        (D[2], "P"): (3100.0, 20.0), (D[2], "A"): (2100.0, 0.0), (D[2], "B"): (1000.0, 20.0),
        (D[0], "银行"): (1000.0, 100.0), (D[0], "电子"): (1000.0, 0.0),
        (D[2], "银行"): (2100.0, 0.0), (D[2], "电子"): (1000.0, 20.0),
    }
    for key, (mv, pnl) in exp.items():
        assert out.loc[key, "market_value"] == pytest.approx(mv, abs=1e-9), key
        assert out.loc[key, "pnl"] == pytest.approx(pnl, abs=1e-9), key
    assert D[3] not in out.index.get_level_values(0)
    assert (out.reset_index().groupby("position_date")["pnl_date"].first() == pd.Series(D[1:4], index=D[:3])).all()


def test_levels_add_up():
    out = hypothetical_pnl(POS, PRICES, INST, "P")
    by = out.pivot_table(index="position_date", columns="level", values=["pnl", "market_value"], aggfunc="sum")
    for v in ("pnl", "market_value"):
        np.testing.assert_allclose(by[v]["instrument"], by[v]["portfolio"])
        np.testing.assert_allclose(by[v]["industry"], by[v]["portfolio"])


def test_uses_positions_of_t_not_t_plus_1():
    pos = POS[~((POS["instrument_id"] == "B") & (POS["trade_date"] == D[1]))]  # B sold on D2
    out = hypothetical_pnl(pos, PRICES, INST, "P").set_index(["position_date", "level_key"])
    assert out.loc[(D[0], "B"), "market_value"] == 1000.0  # still held at the close of D1
    assert (D[1], "B") not in out.index


@pytest.mark.parametrize(
    "prices, msg",
    [
        (PRICES[~((PRICES["instrument_id"] == "B") & (PRICES["trade_date"] == D[2]))], "missing prices on t or t \\+ 1"),
        (PRICES[PRICES["trade_date"] != D[1]], "no prices on position dates"),
    ],
)
def test_rejects_missing_prices(prices, msg):
    with pytest.raises(ValueError, match=msg):
        hypothetical_pnl(POS, prices, INST, "P")


def test_missing_industry_raises():
    with pytest.raises(ValueError, match="missing industry_l1"):
        hypothetical_pnl(POS, PRICES, INST[INST["instrument_id"] == "A"], "P")
