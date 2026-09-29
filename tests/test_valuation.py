import numpy as np
import pandas as pd
import pytest

from mrisk.engine.valuation import exposure_matrix, market_value

POS = pd.DataFrame({"instrument_id": ["C", "A", "B"], "quantity": [300.0, 1000.0, 2000.0]})
CLOSE = pd.Series({"A": 10.0, "B": 5.5, "C": 30.0, "Z": 1.0})
IND = pd.Series({"A": "银行", "B": "银行", "C": "电子", "Z": "计算机"})


def test_market_value_hand_computed():
    # A 1000 x 10 = 10000, B 2000 x 5.5 = 11000, C 300 x 30 = 9000, total 30000
    mv = market_value(POS, CLOSE, IND)
    assert mv["instrument_id"].tolist() == ["A", "B", "C"]
    assert mv["industry_l1"].tolist() == ["银行", "银行", "电子"]
    assert mv["market_value"].tolist() == [10000.0, 11000.0, 9000.0]
    assert mv["weight"].to_numpy() == pytest.approx([1 / 3, 11 / 30, 0.3], abs=1e-12)


@pytest.mark.parametrize(
    "pos, close, ind, msg",
    [
        (POS.assign(quantity=[300.0, 0.0, 2000.0]), CLOSE, IND, "non-positive quantity"),
        (pd.concat([POS, POS.iloc[[0]]]), CLOSE, IND, "duplicate"),
        (POS, CLOSE.drop("B"), IND, "missing or non-positive close"),
        (POS, CLOSE.replace(30.0, 0.0), IND, "missing or non-positive close"),
        (POS, CLOSE, IND.drop("C"), "missing industry_l1"),
        (POS, CLOSE, IND.replace("电子", " "), "missing industry_l1"),
        (POS.iloc[:0], CLOSE, IND, "no positions"),
    ],
)
def test_market_value_rejects_bad_inputs(pos, close, ind, msg):
    with pytest.raises(ValueError, match=msg):
        market_value(pos, close, ind)


def test_exposure_matrix_hand_computed():
    E, units = exposure_matrix(market_value(POS, CLOSE, IND), "P001")
    assert units["level"].tolist() == ["portfolio", "industry", "industry", "instrument", "instrument", "instrument"]
    assert units["level_key"].tolist() == ["P001", "电子", "银行", "A", "B", "C"]
    expected = np.array(
        [
            # P       电子    银行     A        B        C
            [10000.0, 0.0, 10000.0, 10000.0, 0.0, 0.0],
            [11000.0, 0.0, 11000.0, 0.0, 11000.0, 0.0],
            [9000.0, 9000.0, 0.0, 0.0, 0.0, 9000.0],
        ]
    )
    np.testing.assert_array_equal(E, expected)
    assert units["market_value"].tolist() == [30000.0, 9000.0, 21000.0, 10000.0, 11000.0, 9000.0]
