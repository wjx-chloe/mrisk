import numpy as np
import pandas as pd
import pytest

from mrisk.engine.returns import returns_window

DATES = pd.bdate_range("2025-01-06", periods=5)
ADJ = pd.DataFrame({"A": [10.0, 11.0, 9.9, 9.9, 10.89], "B": [20.0, 20.0, 21.0, 21.0, 21.0]}, index=DATES)
SUSP = pd.DataFrame(False, index=DATES, columns=ADJ.columns)


def test_simple_returns_hand_computed():
    # A: 11/10-1 = 0.1, 9.9/11-1 = -0.1, 0, 10.89/9.9-1 = 0.1 ; B: 0, 0.05, 0, 0
    R, sd = returns_window(ADJ, SUSP, DATES[-1], 4, "simple")
    assert R.index.equals(DATES[1:])
    np.testing.assert_allclose(R["A"], [0.1, -0.1, 0.0, 0.1], atol=1e-12)
    np.testing.assert_allclose(R["B"], [0.0, 0.05, 0.0, 0.0], atol=1e-12)
    assert sd.tolist() == [0, 0]


def test_log_returns_hand_computed():
    R, _ = returns_window(ADJ, SUSP, DATES[-1], 4, "log")
    np.testing.assert_allclose(R["A"], [np.log(1.1), np.log(0.9), 0.0, np.log(1.1)], atol=1e-12)


def test_window_ends_on_asof_and_ignores_later_rows():
    R, _ = returns_window(ADJ, SUSP, DATES[3], 2, "simple")
    assert R.index.equals(DATES[2:4])
    np.testing.assert_allclose(R["A"], [-0.1, 0.0], atol=1e-12)


def test_suspended_days_are_zero_and_counted_inside_the_window_only():
    adj = ADJ.copy()
    adj.loc[DATES[3], "B"] = 22.0  # a suspended row whose price moved anyway: SPEC says the return is 0
    susp = SUSP.copy()
    susp.loc[DATES[3], "B"] = True
    susp.loc[DATES[0], "A"] = True  # first price row of the window, not a return date
    R, sd = returns_window(adj, susp, DATES[-1], 4, "simple")
    assert R.loc[DATES[3], "B"] == 0.0
    assert R.loc[DATES[4], "B"] == pytest.approx(21 / 22 - 1)  # resumption day keeps its real move
    assert sd.to_dict() == {"A": 0, "B": 1}


@pytest.mark.parametrize(
    "adj, susp, asof, lookback, kind, msg",
    [
        (ADJ, SUSP, "2025-01-11", 2, "simple", "no prices on"),
        (ADJ, SUSP, DATES[-1], 5, "simple", "need 6 prices"),
        (ADJ.assign(B=[np.nan, 20.0, 21.0, 21.0, 21.0]), SUSP, DATES[-1], 4, "simple", r"missing adj_close .*\['B'\]"),
        (ADJ.assign(B=[20.0, 0.0, 21.0, 21.0, 21.0]), SUSP, DATES[-1], 4, "simple", "non-positive"),
        (ADJ, SUSP.iloc[:3], DATES[-1], 4, "simple", "does not cover"),
        (ADJ, SUSP, DATES[-1], 4, "pct", "return type"),
        (ADJ, SUSP, DATES[-1], 1, "simple", "lookback must be"),
        (ADJ.iloc[::-1], SUSP, DATES[-1], 2, "simple", "sorted"),
    ],
)
def test_returns_window_rejects_bad_inputs(adj, susp, asof, lookback, kind, msg):
    with pytest.raises(ValueError, match=msg):
        returns_window(adj, susp, asof, lookback, kind)


def test_gap_before_window_is_fine():
    adj = ADJ.assign(B=[np.nan, 20.0, 21.0, 21.0, 21.0])
    R, _ = returns_window(adj, SUSP, DATES[-1], 3, "simple")
    assert not R.isna().any().any()
