import numpy as np
import pandas as pd
import pytest

from mrisk.engine.attribution import attribute_change, risk_components
from mrisk.engine.risk import compute_risk
from mrisk.engine.var import hs_es, hs_var, param_var

# ---- hand-computed: 2 names, T = 4, 95% (linear quantile position 0.05 x 3 = 0.15) ----------------------------
D = pd.bdate_range("2025-01-06", periods=7)  # returns r1..r6 on D1..D6; t - 1 = D5, t = D6
RET = {"A": [0.0, -0.10, 0.02, -0.04, 0.01, -0.02], "B": [0.0, 0.01, -0.06, 0.03, -0.02, -0.03]}
CLOSE_T1 = {"A": 10.0, "B": 20.0}  # closes on t - 1


def _prices(ret=RET):
    """Closes move with adj_close (a constant factor per name, no corporate action) and equal CLOSE_T1 on D5."""
    rows = []
    for iid, r in ret.items():
        adj = 50.0 * np.concatenate([[1.0], np.cumprod(1 + np.array(r))])
        close = adj * (CLOSE_T1[iid] / adj[5])
        rows.append(pd.DataFrame({"trade_date": D, "instrument_id": iid, "close": close, "adj_close": adj, "is_suspended": False}))
    return pd.concat(rows, ignore_index=True)


Q0 = pd.DataFrame({"instrument_id": ["A", "B"], "quantity": [100.0, 100.0]})
Q1 = pd.DataFrame({"instrument_id": ["A", "B"], "quantity": [100.0, 200.0]})  # bought 100 B


def test_attribution_hand_computed():
    # c0 = (10, 20), c1 = c0 x (1 + r6) = (9.8, 19.4); no corporate action, so q~_t = q_t
    # P0 = (1000, 2000); P1 = q~1 c0 = (1000, 4000); P2 = q1 c1 = (980, 3880); W0 = r2..r5, W1 = r3..r6
    # scenario P&L (A + B):
    #   V0 on W0: r2 -100+20=-80  r3 20-120=-100  r4 -40+60=20  r5 10-40=-30            -> lo r3, hi r2
    #   V1 on W0: r2 -100+40=-60  r3 20-240=-220  r4 -40+120=80 r5 10-80=-70            -> lo r3, hi r5
    #   V2 on W0: r2 -98+38.8=-59.2  r3 19.6-232.8=-213.2  r4 -39.2+116.4=77.2  r5 9.8-77.6=-67.8 -> lo r3, hi r5
    #   V3 on W1: r3 -213.2  r4 77.2  r5 -67.8  r6 -19.6-116.4=-136                     -> lo r3, hi r6
    # VaR95 = -(x_lo + 0.15 (x_hi - x_lo)):
    #   V0 = 100 - 0.15 x 20 = 97 ; V1 = 220 - 0.15 x 150 = 197.5
    #   V2 = 213.2 - 0.15 x 145.4 = 191.39 ; V3 = 213.2 - 0.15 x 77.2 = 201.62
    # lambda = (980 + 3880) / (1000 + 4000) = 0.972
    # effects: quantity 100.5 ; scale -0.028 x 197.5 = -5.53 ; drift 191.39 - 0.972 x 197.5 = -0.58 ;
    #          window 201.62 - 191.39 = 10.23 ; total 104.62
    # Euler, -(0.85 x name P&L on lo + 0.15 x name P&L on hi):
    #   A: V0 -(17 - 15) = -2 ; V1 -(17 + 1.5) = -18.5 ; V2 -(16.66 + 1.47) = -18.13 ; V3 -(16.66 - 2.94) = -13.72
    #   B: V0 -(-102 + 3) = 99 ; V1 -(-204 - 12) = 216 ; V2 -(-197.88 - 11.64) = 209.52 ; V3 -(-197.88 - 17.46) = 215.34
    #   A: quantity -16.5, scale 0.518, drift -18.13 + 0.972 x 18.5 = -0.148, window 4.41, total -11.72
    #   B: quantity 117, scale -6.048, drift 209.52 - 0.972 x 216 = -0.432, window 5.82, total 116.34
    # ES95 (tail strictly below -VaR is the worst day only): V0 100, V1 220, V2 213.2, V3 213.2
    out = attribute_change(Q0, Q1, _prices(), D[5], D[6], "P", [0.95], 4)
    v = out[out["metric"] == "var_hs"].set_index(["effect", "level_key"])["value"]
    exp = {
        ("quantity", "P"): 100.5, ("revaluation_scale", "P"): -5.53, ("revaluation_drift", "P"): -0.58,
        ("window", "P"): 10.23, ("total", "P"): 104.62,
        ("quantity", "A"): -16.5, ("revaluation_scale", "A"): 0.518, ("revaluation_drift", "A"): -0.148,
        ("window", "A"): 4.41, ("total", "A"): -11.72,
        ("quantity", "B"): 117.0, ("revaluation_scale", "B"): -6.048, ("revaluation_drift", "B"): -0.432,
        ("window", "B"): 5.82, ("total", "B"): 116.34,
    }
    for key, val in exp.items():
        assert v[key] == pytest.approx(val, abs=5e-5), key
    es = out[(out["metric"] == "es_hs") & (out["level"] == "portfolio")].set_index("effect")["value"]
    # ES revaluation 213.2 - 220 = -6.8: scale -0.028 x 220 = -6.16, drift 213.2 - 0.972 x 220 = -0.64
    assert es.to_dict() == pytest.approx(
        {"quantity": 120.0, "revaluation_scale": -6.16, "revaluation_drift": -0.64, "window": 0.0, "total": 113.2}, abs=5e-5
    )
    w = out[out["effect"] == "window"]
    assert (w["window_dropped"] == D[2]).all() and out.loc[out["effect"] != "window", "window_dropped"].isna().all()


def test_effects_telescope_and_contributions_sum():
    rng = np.random.default_rng(9)
    ids = [f"S{i}" for i in range(12)]
    dates = pd.bdate_range("2025-01-02", periods=62)
    rows = []
    for iid in ids:
        adj = 20 * np.cumprod(1 + rng.normal(0, 0.02, len(dates)))
        rows.append(pd.DataFrame({"trade_date": dates, "instrument_id": iid, "close": adj * rng.uniform(0.5, 1),
                                  "adj_close": adj, "is_suspended": False}))
    prices = pd.concat(rows, ignore_index=True)
    q0 = pd.DataFrame({"instrument_id": ids[:10], "quantity": rng.integers(1, 50, 10) * 100.0})
    q1 = pd.DataFrame({"instrument_id": ids[2:], "quantity": rng.integers(1, 50, 10) * 100.0})  # swaps names
    conf = [0.95, 0.99]
    out = attribute_change(q0, q1, prices, dates[-2], dates[-1], "P", conf, 60)
    port = out[out["level"] == "portfolio"].set_index(["metric", "confidence", "effect"])["value"]
    inst = out[out["level"] == "instrument"].groupby(["metric", "confidence", "effect"])["value"].sum()
    np.testing.assert_allclose(inst.reindex(port.index), port, rtol=1e-9, atol=1e-6)
    for m in ("var_hs", "var_param", "es_hs"):
        for c in conf:
            parts = sum(port[(m, c, e)] for e in ("quantity", "revaluation_scale", "revaluation_drift", "window"))
            assert parts == pytest.approx(port[(m, c, "total")], rel=1e-12, abs=1e-9)
    # total = today's figure - yesterday's figure as compute_risk reports them
    inst_tbl = pd.DataFrame({"instrument_id": ids, "industry_l1": "X"})
    r0 = compute_risk(q0, prices, inst_tbl, dates[-2], "P", conf, 60).table
    r1 = compute_risk(q1, prices, inst_tbl, dates[-1], "P", conf, 60).table
    for m in ("var_hs", "var_param", "es_hs"):
        for c in conf:
            a = r0[(r0["level"] == "portfolio") & (r0["confidence"] == c)][m].iloc[0]
            b = r1[(r1["level"] == "portfolio") & (r1["confidence"] == c)][m].iloc[0]
            assert port[(m, c, "total")] == pytest.approx(b - a, rel=1e-9)


def test_quantity_effect_is_exactly_zero_without_rebalancing():
    out = attribute_change(Q0, Q0, _prices(), D[5], D[6], "P", [0.95, 0.99], 4)
    q = out[out["effect"] == "quantity"]["value"]
    assert (q == 0.0).all()  # exactly, not approximately


def test_weight_drift_is_zero_when_every_name_moves_alike():
    ret = {k: v[:-1] + [0.037] for k, v in RET.items()}  # every name +3.7% on t, close and adj_close alike
    out = attribute_change(Q0, Q1, _prices(ret), D[5], D[6], "P", [0.95, 0.99], 4)
    v = out.set_index(["metric", "confidence", "effect", "level_key"])["value"]
    for m in ("var_hs", "var_param", "es_hs"):
        for c in (0.95, 0.99):
            scale = v[(m, c, "revaluation_scale", "P")]
            assert scale != 0.0
            assert v[(m, c, "revaluation_drift", "P")] == pytest.approx(0.0, abs=1e-9 * abs(scale) / 0.037)
            for iid in ("A", "B"):
                assert v[(m, c, "revaluation_drift", iid)] == pytest.approx(0.0, abs=1e-9 * abs(scale) / 0.037)


def test_measures_are_homogeneous_of_degree_one():
    rng = np.random.default_rng(4)
    R = rng.standard_t(4, (250, 6)) * 0.02
    e = rng.uniform(1e5, 1e6, 6)
    tot, comp = risk_components(R, e, [0.95, 0.99])
    tot2, comp2 = risk_components(R, 1.37 * e, [0.95, 0.99])
    for m in tot:
        np.testing.assert_allclose(tot2[m], 1.37 * tot[m], rtol=1e-12)
        np.testing.assert_allclose(comp2[m], 1.37 * comp[m], rtol=1e-11)


def _one_name(close, factor):
    """Single name over D0..D5 (lookback 4: t - 1 = D4, t = D5); returns r1..r4 on the closes up to D4."""
    rows = pd.DataFrame({"trade_date": D[:6], "instrument_id": "A", "close": close, "is_suspended": False})
    rows["adj_close"] = rows["close"] * np.asarray(factor)
    return rows


def test_pure_bonus_issue_is_not_a_trade():
    # 10-for-10 on D5, no trade: shares 1000 -> 2000, close halves, factor 1 -> 2, so q~_t = 2000 x 1/2 = 1000
    close = [10.0, 9.0, 9.18, 8.8128, 8.901, 4.6]
    factor = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
    prices = _one_name(close, factor)
    q0 = pd.DataFrame({"instrument_id": ["A"], "quantity": [1000.0]})
    q1 = pd.DataFrame({"instrument_id": ["A"], "quantity": [2000.0]})
    out = attribute_change(q0, q1, prices, D[4], D[5], "P", [0.95, 0.99], 4)
    assert (out[out["effect"] == "quantity"]["value"] == 0.0).all()
    # the move is all revaluation: lambda = 2000 x 4.6 / (1000 x 8.901), a -(1 - 9200/8901) total return
    # VaR95 of t - 1 = 809.991 = 0.091 x 8901 (see the dividend test below) ; scale = (lambda - 1) x 809.991
    #   = (9200 - 8901) / 8901 x 0.091 x 8901 = 0.091 x 299 = 27.209
    v = out[(out["metric"] == "var_hs") & (out["level"] == "portfolio") & (out["confidence"] == 0.95)].set_index("effect")["value"]
    assert v["revaluation_scale"] == pytest.approx((9200 / 8901 - 1) * 809.991, rel=1e-9)
    assert v["revaluation_scale"] == pytest.approx(27.209, abs=5e-5)


def test_small_cash_dividend_not_reinvested_is_a_net_sale():
    # one name, q = 1000 both days (the dividend rounded to less than half a lot), factor 1.00 -> 1.02
    # window of t - 1 (r1..r4): closes 10, 9, 9.18, 8.8128, 8.901 -> r = -0.10, +0.02, -0.04, +0.01
    # P0 = 1000 x 8.901 = 8901 ; scenario P&L: -890.1, 178.02, -356.04, 89.01
    # VaR95 = -(x1 + 0.15 (x2 - x1)) = 890.1 - 0.15 x (890.1 - 356.04) = 890.1 - 80.109 = 809.991
    # q~_t = 1000 x 1.00 / 1.02 ; quantity effect = VaR0 x (1/1.02 - 1) = 809.991 x (-0.0196078) = -15.8822
    close = [10.0, 9.0, 9.18, 8.8128, 8.901, 8.80]
    factor = [1.0, 1.0, 1.0, 1.0, 1.0, 1.02]
    prices = _one_name(close, factor)
    q = pd.DataFrame({"instrument_id": ["A"], "quantity": [1000.0]})
    out = attribute_change(q, q, prices, D[4], D[5], "P", [0.95], 4)
    v = out[(out["metric"] == "var_hs") & (out["level"] == "portfolio")].set_index("effect")["value"]
    assert v["quantity"] == pytest.approx(-15.8822, abs=5e-5)
    assert v["quantity"] == pytest.approx(809.991 * (1 / 1.02 - 1), rel=1e-12)


def test_risk_components_sum_to_engine_totals():
    rng = np.random.default_rng(1)
    R = rng.standard_t(4, (250, 8)) * 0.02
    e = rng.uniform(1e5, 1e6, 8)
    conf = [0.95, 0.99]
    tot, comp = risk_components(R, e, conf)
    pnl = (R @ e)[:, None]
    var = hs_var(pnl, conf)
    np.testing.assert_allclose(tot["var_hs"], var[:, 0], rtol=1e-12)
    np.testing.assert_allclose(tot["es_hs"], hs_es(pnl, var)[0][:, 0], rtol=1e-12)
    np.testing.assert_allclose(tot["var_param"], param_var(R, e[:, None], conf)[:, 0], rtol=1e-12)
    for m in tot:
        np.testing.assert_allclose(comp[m].sum(axis=1), tot[m], rtol=1e-9)


def test_rejects_non_consecutive_dates():
    with pytest.raises(ValueError, match="not the trading day before"):
        attribute_change(Q0, Q1, _prices(), D[4], D[6], "P", [0.95], 3)
