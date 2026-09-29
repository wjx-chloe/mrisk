import time

import numpy as np
import pandas as pd
import pytest

from mrisk.engine.risk import RESULT_COLUMNS, compute_risk

CONF = [0.95, 0.99]


def make_prices(returns: dict[str, list[float]], close_asof: dict[str, float], start="2025-01-02"):
    """Long price frame whose adj_close reproduces `returns` exactly; close is flat at the as-of close
    (only the as-of close enters the market value, and it differs from adj_close on purpose)."""
    T = len(next(iter(returns.values())))
    dates = pd.bdate_range(start, periods=T + 1)
    rows = []
    for iid, r in returns.items():
        adj = 50.0 * np.concatenate([[1.0], np.cumprod(1.0 + np.asarray(r))])
        rows.append(
            pd.DataFrame(
                {"trade_date": dates, "instrument_id": iid, "close": close_asof[iid], "adj_close": adj, "is_suspended": False}
            )
        )
    return pd.concat(rows, ignore_index=True), dates


# ---- 3 stocks, 2 industries, T = 10: every number below is worked out by hand ----------------------------------
# market value: A 1000 x 10 = 10000, B 2000 x 5.5 = 11000, C 300 x 30 = 9000, total 30000
# daily simple returns (%):
RET = {
    "A": [1, -2, 3, -1, 0, 2, -3, 1, -1, 2],
    "B": [-1, 1, 2, -2, 1, -1, -4, 2, 0, 1],
    "C": [2, -1, -3, 1, 2, 0, -2, -1, 3, -2],
}
RET = {k: [x / 100 for x in v] for k, v in RET.items()}
POS = pd.DataFrame({"instrument_id": ["A", "B", "C"], "quantity": [1000.0, 2000.0, 300.0]})
INST = pd.DataFrame({"instrument_id": ["A", "B", "C"], "industry_l1": ["银行", "银行", "电子"]})
CLOSE = {"A": 10.0, "B": 5.5, "C": 30.0}
# P&L = MV x r:
#   A   :  100 -200  300 -100    0  200 -300  100 -100  200
#   B   : -110  110  220 -220  110 -110 -440  220    0  110
#   C   :  180  -90 -270   90  180    0 -180  -90  270 -180   (= 电子)
#   银行:  -10  -90  520 -320  110   90 -740  320 -100  310   (A + B)
#   P   :  170 -180  250 -230  290   90 -920  230  170  130   (银行 + 电子)
# HS: linear quantile position (1-c)(T-1) = 0.45 (95%) / 0.09 (99%), both between the two worst days x1, x2:
#   VaR = -(x1 + pos (x2 - x1)); the tail < -VaR is {x1} for both, so ES95 = ES99 = -x1
#   P   : x1 -920, x2 -230 -> VaR95 920 - 0.45 x 690 = 609.5, VaR99 920 - 0.09 x 690 = 857.9, ES 920
#   电子: x1 -270, x2 -180 -> 270 - 40.5 = 229.5,  270 - 8.1 = 261.9,  ES 270
#   银行: x1 -740, x2 -320 -> 740 - 189 = 551.0,   740 - 37.8 = 702.2, ES 740
#   A   : x1 -300, x2 -200 -> 300 - 45 = 255.0,    300 - 9 = 291.0,    ES 300
#   B   : x1 -440, x2 -220 -> 440 - 99 = 341.0,    440 - 19.8 = 420.2, ES 440
# Parametric: sigma^2 = (sum x^2 - T mean^2) / (T - 1), VaR = z sigma, z95 = 1.6448536270, z99 = 2.3263478740
#   P   : sum 0,    sum x^2 1214000 -> 1214000 / 9            = 134888.8889, sigma 367.2722272
#   电子: sum -90,  sum x^2 299700  -> (299700 - 810) / 9     = 33210.0000,  sigma 182.2361106
#   银行: sum 90,   sum x^2 1157300 -> (1157300 - 810) / 9    = 128498.8889, sigma 358.4674168
#   A   : sum 200,  sum x^2 340000  -> (340000 - 4000) / 9    = 37333.3333,  sigma 193.2183566
#   B   : sum -110, sum x^2 399300  -> (399300 - 1210) / 9    = 44232.2222,  sigma 210.3145792
EXPECTED = pd.DataFrame(
    [
        # level, key, mv, var95, var99, es, pvar95, pvar99
        ("portfolio", "P001", 30000.0, 609.5, 857.9, 920.0, 604.1091, 854.4030),
        ("industry", "电子", 9000.0, 229.5, 261.9, 270.0, 299.7517, 423.9446),
        ("industry", "银行", 21000.0, 551.0, 702.2, 740.0, 589.6264, 833.9199),
        ("instrument", "A", 10000.0, 255.0, 291.0, 300.0, 317.8159, 449.4931),
        ("instrument", "B", 11000.0, 341.0, 420.2, 440.0, 345.9367, 489.2649),
        ("instrument", "C", 9000.0, 229.5, 261.9, 270.0, 299.7517, 423.9446),
    ],
    columns=["level", "level_key", "mv", "v95", "v99", "es", "p95", "p99"],
)


def hand_example(**kw):
    prices, dates = make_prices(RET, CLOSE)
    return compute_risk(POS, prices, INST, dates[-1], "P001", CONF, 10, **kw), dates


def test_compute_risk_matches_hand_calculation_to_4_decimals():
    res, _ = hand_example()
    t = res.table
    assert list(t.columns) == RESULT_COLUMNS
    assert len(t) == 6 * 2
    assert t[["level", "level_key"]].drop_duplicates().to_records(index=False).tolist() == list(
        EXPECTED[["level", "level_key"]].itertuples(index=False, name=None)
    )
    assert t["confidence"].tolist() == CONF * 6
    lo, hi = t[t["confidence"] == 0.95].reset_index(drop=True), t[t["confidence"] == 0.99].reset_index(drop=True)
    tol = dict(atol=5e-5, rtol=0)
    np.testing.assert_allclose(lo["market_value"], EXPECTED["mv"], **tol)
    np.testing.assert_allclose(lo["var_hs"], EXPECTED["v95"], **tol)
    np.testing.assert_allclose(hi["var_hs"], EXPECTED["v99"], **tol)
    np.testing.assert_allclose(lo["es_hs"], EXPECTED["es"], **tol)
    np.testing.assert_allclose(hi["es_hs"], EXPECTED["es"], **tol)
    np.testing.assert_allclose(lo["var_param"], EXPECTED["p95"], **tol)
    np.testing.assert_allclose(hi["var_param"], EXPECTED["p99"], **tol)
    assert (t["es_tail_n"] == 1).all()
    assert res.suspended_days.to_dict() == {"A": 0, "B": 0, "C": 0}


def test_var_is_not_additive_across_levels_diversification():
    """Layer sums differ from the portfolio figure, and that is allowed. Here diversification makes them larger."""
    t = hand_example()[0].table
    t = t[t["confidence"] == 0.99].set_index("level_key")
    port = t.loc["P001", "var_hs"]
    by_inst = t.loc[["A", "B", "C"], "var_hs"].sum()  # 291 + 420.2 + 261.9 = 973.1
    by_ind = t.loc[["电子", "银行"], "var_hs"].sum()  # 261.9 + 702.2 = 964.1
    assert by_inst == pytest.approx(973.1) and by_ind == pytest.approx(964.1) and port == pytest.approx(857.9)
    assert by_inst != pytest.approx(port) and by_ind != pytest.approx(port)


def test_hs_var_can_be_superadditive_while_parametric_is_not():
    """Two names that each crash on a different day: each alone has one bad day in 20, the portfolio has two.
    At 95% (position 0.05 x 19 = 0.95): each name VaR = 1000 - 0.95 x 1000 = 50, portfolio VaR = 1000 > 50 + 50."""
    ret = {"X": [-0.1] + [0.0] * 19, "Y": [0.0, -0.1] + [0.0] * 18}
    prices, dates = make_prices(ret, {"X": 10.0, "Y": 10.0})
    pos = pd.DataFrame({"instrument_id": ["X", "Y"], "quantity": [1000.0, 1000.0]})
    inst = pd.DataFrame({"instrument_id": ["X", "Y"], "industry_l1": ["银行", "电子"]})
    t = compute_risk(pos, prices, inst, dates[-1], "P", [0.95], 20).table.set_index("level_key")
    assert t.loc["X", "var_hs"] == pytest.approx(50.0) and t.loc["Y", "var_hs"] == pytest.approx(50.0)
    assert t.loc["P", "var_hs"] == pytest.approx(1000.0)
    assert t.loc["P", "var_hs"] > t.loc[["X", "Y"], "var_hs"].sum()
    # normal VaR is z x sigma and sigma obeys the triangle inequality, so it never exceeds the sum
    assert t.loc["P", "var_param"] <= t.loc[["X", "Y"], "var_param"].sum()


def test_market_value_uses_unadjusted_close_and_ignores_rows_after_asof():
    prices, dates = make_prices(RET, CLOSE)
    later = prices[prices["trade_date"] == dates[-1]].assign(trade_date=dates[-1] + pd.Timedelta(days=3), close=999.0)
    res = compute_risk(POS, pd.concat([prices, later]), INST, dates[-1], "P001", CONF, 10)
    assert res.table.iloc[0]["market_value"] == 30000.0
    assert res.table.iloc[1]["var_hs"] == pytest.approx(857.9)


def test_suspended_days_are_returned_and_zeroed():
    ret = {k: list(v) for k, v in RET.items()}
    ret["C"][4] = 0.0  # suspended on return day 5: price carried, return 0
    prices, dates = make_prices(ret, CLOSE)
    prices.loc[(prices["instrument_id"] == "C") & (prices["trade_date"] == dates[5]), "is_suspended"] = True
    res = compute_risk(POS, prices, INST, dates[-1], "P001", CONF, 10)
    assert res.suspended_days.to_dict() == {"A": 0, "B": 0, "C": 1}


def test_empty_tail_is_reported_not_logged():
    ret = dict(RET, C=[0.0] * 10)  # flat all window long
    prices, dates = make_prices(ret, CLOSE)
    t = compute_risk(POS, prices, INST, dates[-1], "P001", CONF, 10).table.set_index("level_key")
    assert (t.loc["C", "es_tail_n"] == 0).all()
    np.testing.assert_allclose(t.loc["C", "es_hs"], t.loc["C", "var_hs"])
    assert (t.loc["P001", "es_tail_n"] > 0).all()


def test_log_returns_option():
    simple = hand_example()[0].table
    log = hand_example(return_type="log")[0].table
    # log(1 + r) < r, so every loss is a little larger under log returns
    assert (log["var_hs"] > simple["var_hs"]).all()


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda p, i: (p[p["instrument_id"] != "B"], i), r"missing adj_close .*\['B'\]"),
        (lambda p, i: (p, i[i["instrument_id"] != "C"]), "missing industry_l1"),
        (lambda p, i: (p[p["trade_date"] > p["trade_date"].min()], i), "need 11 prices"),
    ],
)
def test_compute_risk_raises_instead_of_shrinking(mutate, msg):
    prices, dates = make_prices(RET, CLOSE)
    p, i = mutate(prices, INST)
    with pytest.raises(ValueError, match=msg):
        compute_risk(POS, p, i, dates[-1], "P001", CONF, 10)


def test_performance_50_names_250_days_under_1s():
    rng = np.random.default_rng(0)
    ids = [f"S{i:02d}" for i in range(50)]
    ret = {iid: rng.normal(0, 0.02, 250).tolist() for iid in ids}
    prices, dates = make_prices(ret, {iid: 10.0 for iid in ids})
    pos = pd.DataFrame({"instrument_id": ids, "quantity": 1000.0})
    inst = pd.DataFrame({"instrument_id": ids, "industry_l1": [f"I{i % 8}" for i in range(50)]})
    compute_risk(pos, prices, inst, dates[-1], "P", CONF, 250)  # warm-up
    t0 = time.perf_counter()
    res = compute_risk(pos, prices, inst, dates[-1], "P", CONF, 250)
    elapsed = time.perf_counter() - t0
    assert len(res.table) == (1 + 8 + 50) * 2
    assert elapsed < 1.0, f"{elapsed:.3f}s"
