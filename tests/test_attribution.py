import numpy as np
import pandas as pd
import pytest

from mrisk.engine.attribution import attribute_change, risk_components
from mrisk.engine.risk import compute_risk
from mrisk.engine.var import hs_es, hs_var, param_var

# ---- hand-computed: 2 names, T = 4, 95% (linear quantile position 0.05 x 3 = 0.15) ----------------------------
D = pd.bdate_range("2025-01-06", periods=7)  # returns r1..r6 on D1..D6; t - 1 = D5, t = D6
RET = {"A": [0.0, -0.10, 0.02, -0.04, 0.01, -0.02], "B": [0.0, 0.01, -0.06, 0.03, -0.02, 0.04]}
CLOSE = {"A": {D[5]: 10.0, D[6]: 10.5}, "B": {D[5]: 20.0, D[6]: 19.0}}


def _prices():
    rows = []
    for iid, r in RET.items():
        adj = 50.0 * np.concatenate([[1.0], np.cumprod(1 + np.array(r))])
        close = [CLOSE[iid].get(d, 10.0) for d in D]
        rows.append(pd.DataFrame({"trade_date": D, "instrument_id": iid, "close": close, "adj_close": adj, "is_suspended": False}))
    return pd.concat(rows, ignore_index=True)


Q0 = pd.DataFrame({"instrument_id": ["A", "B"], "quantity": [100.0, 100.0]})
Q1 = pd.DataFrame({"instrument_id": ["A", "B"], "quantity": [100.0, 200.0]})  # bought 100 B


def test_attribution_hand_computed():
    # P0 = q0 c0 = (1000, 2000); P1 = q1 c0 = (1000, 4000); P2 = q1 c1 = (1050, 3800); W0 = r2..r5, W1 = r3..r6
    # scenario P&L (A + B):
    #   V0 on W0: r2 -100+20=-80  r3 20-120=-100  r4 -40+60=20   r5 10-40=-30   -> lo r3, hi r2
    #   V1 on W0: r2 -100+40=-60  r3 20-240=-220  r4 -40+120=80  r5 10-80=-70   -> lo r3, hi r5
    #   V2 on W0: r2 -105+38=-67  r3 21-228=-207  r4 -42+114=72  r5 10.5-76=-65.5 -> lo r3, hi r2
    #   V3 on W1: r3 -207  r4 72  r5 -65.5  r6 -21+152=131                     -> lo r3, hi r5
    # VaR95 = -(x_lo + 0.15 (x_hi - x_lo)):
    #   V0 = 100 - 0.15 x 20 = 97 ; V1 = 220 - 0.15 x 150 = 197.5 ; V2 = 207 - 0.15 x 140 = 186 ; V3 = 207 - 0.15 x 141.5 = 185.775
    # effects: quantity 100.5, revaluation -11.5, window -0.225, total 88.775
    # Euler, -(0.85 x name P&L on lo + 0.15 x name P&L on hi):
    #   A: V0 -(17 - 15) = -2 ; V1 -(17 + 1.5) = -18.5 ; V2 -(17.85 - 15.75) = -2.1 ; V3 -(17.85 + 1.575) = -19.425
    #   B: V0 -(-102 + 3) = 99 ; V1 -(-204 - 12) = 216 ; V2 -(-193.8 + 5.7) = 188.1 ; V3 -(-193.8 - 11.4) = 205.2
    # ES95 (tail strictly below -VaR is the worst day only): V0 100, V1 220, V2 207, V3 207
    out = attribute_change(Q0, Q1, _prices(), D[5], D[6], "P", [0.95], 4)
    v = out[out["metric"] == "var_hs"].set_index(["effect", "level_key"])["value"]
    exp = {
        ("quantity", "P"): 100.5, ("revaluation", "P"): -11.5, ("window", "P"): -0.225, ("total", "P"): 88.775,
        ("quantity", "A"): -16.5, ("revaluation", "A"): 16.4, ("window", "A"): -17.325, ("total", "A"): -17.425,
        ("quantity", "B"): 117.0, ("revaluation", "B"): -27.9, ("window", "B"): 17.1, ("total", "B"): 106.2,
    }
    for key, val in exp.items():
        assert v[key] == pytest.approx(val, abs=5e-5), key
    es = out[(out["metric"] == "es_hs") & (out["level"] == "portfolio")].set_index("effect")["value"]
    assert es.to_dict() == pytest.approx({"quantity": 120.0, "revaluation": -13.0, "window": 0.0, "total": 107.0}, abs=5e-5)
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
            parts = sum(port[(m, c, e)] for e in ("quantity", "revaluation", "window"))
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
