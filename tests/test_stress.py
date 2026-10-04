import numpy as np
import pandas as pd
import pytest

from mrisk.engine.stress import (
    STRESS_COLUMNS,
    ScenarioPath,
    betas,
    historical_stress,
    hypothetical_shocks,
    hypothetical_stress,
    proxy_returns,
    rank_scenarios,
    scenario_path,
)

D = pd.bdate_range("2025-03-03", periods=9)  # D0..D8
# three 银行 peers that always trade: their equal-weight return is the 银行 proxy whenever X is not trading
PEER_RET = {
    "Y1": [0.01, -0.02, 0.03, -0.01, 0.02, 0.00, -0.01, 0.01],
    "Y2": [0.00, -0.01, 0.02, -0.03, 0.01, 0.01, 0.00, -0.02],
    "Y3": [0.02, -0.03, 0.01, -0.02, 0.00, 0.02, -0.02, 0.00],
}
# peer mean on D1..D8: +1%, -2%, +2%, -2%, +1%, +1%, -1%, -1/3%
PROXY = [0.01, -0.02, 0.02, -0.02, 0.01, 0.01, -0.01, -1 / 300]
IND = pd.Series({"X": "银行", "Y1": "银行", "Y2": "银行", "Y3": "银行", "Z": "电子"})


def frame(x_close, x_susp):
    """X's closes (NaN = not listed) and suspension flags on D0..D8 plus the peers and a lone 电子 name Z.
    adj_close = close (no corporate action); suspended rows carry the last real price, as the ETL fills them."""
    adj = {k: 10 * np.concatenate([[1.0], np.cumprod(1 + np.array(r))]) for k, r in PEER_RET.items()}
    adj["Z"] = 20 * np.concatenate([[1.0], np.cumprod(1 + np.array([0.03, -0.01, 0.0, 0.02, -0.04, 0.01, 0.0, 0.01]))])
    adj["X"] = np.array(x_close, dtype=float)
    A = pd.DataFrame(adj, index=D)[["X", "Y1", "Y2", "Y3", "Z"]]
    S = pd.DataFrame(False, index=D, columns=A.columns)
    S["X"] = x_susp
    return A, S


def test_proxy_uses_clean_peers_and_falls_back_to_the_pool():
    x = [10, 10.5, 10.5, 10.5, 9.8, 10.29, 10.0842, 10.0842, 10.0842]
    A, S = frame(x, [False, False, True, True, False, False, False, False, False])
    P = proxy_returns(A, S, IND)
    # X suspended on D2, D3 and resuming on D4: none of those is a clean X return, so 银行 = mean of Y1..Y3
    np.testing.assert_allclose(P.loc[D[2:5], "X"], PROXY[1:4], atol=1e-12)
    # on D1 X is clean (10 -> 10.5): 银行 = mean of X +5%, Y1 +1%, Y2 0%, Y3 +2% = 2%
    assert P.loc[D[1], "Y1"] == pytest.approx(0.02, abs=1e-12)
    # 电子 has one name (< 3): the pool's mean of clean returns, D1 = (5 + 1 + 0 + 2 + 3) / 5 = 2.2%
    assert P.loc[D[1], "Z"] == pytest.approx(0.022, abs=1e-12)
    # D3: X suspended, so the pool is Y1..Y3 and Z: (3 + 2 + 1 + 0) / 4 = 1.5%
    assert P.loc[D[3], "Z"] == pytest.approx(0.015, abs=1e-12)


def test_resumes_inside_the_window():
    # X: suspended D2..D3 (s = D2, s - 1 = D1), resumes D4; window (D1, D6]
    # D2 proxy -2%, D3 proxy +2%; D4 residual = (9.8 / 10.5) / (0.98 x 1.02) - 1 = 0.9333333 / 0.9996 - 1 = 0.9337068 - 1 = -0.0662932
    # D5 +5%, D6 -2% -> cumulative = P6 / P1 - 1 = 10.0842 / 10.5 - 1 = -0.0396
    x = [10, 10.5, 10.5, 10.5, 9.8, 10.29, 10.0842, 10.0842, 10.0842]
    A, S = frame(x, [False, False, True, True, False, False, False, False, False])
    p = scenario_path(A, S, IND, D[1], D[6])
    r = p.returns["X"]
    np.testing.assert_allclose(r, [-0.02, 0.02, -0.0662932, 0.05, -0.02], atol=5e-8)
    assert (1 + r).prod() - 1 == pytest.approx(10.0842 / 10.5 - 1, abs=1e-12)
    assert p.is_proxy["X"].tolist() == [True, True, False, False, False]
    assert p.suspended["X"] and not p.unlisted["X"]
    assert p.price_dates.equals(D[1:7])  # s - 1 is the base itself


def test_suspended_past_the_end():
    # X: trades D1 10.5, D2 10.29, D3 10.5, suspended from D4 to after the end D6
    # window (D1, D6]: D2 -2%, D3 +2.0408%, D4..D6 proxies -2%, +1%, +1%
    # cumulative = (10.5 / 10.5) x 0.98 x 1.01 x 1.01 - 1 = -0.000302 ; all of the D4..D6 P&L is proxy
    x = [10, 10.5, 10.29, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5]
    A, S = frame(x, [False] * 4 + [True] * 5)
    p = scenario_path(A, S, IND, D[1], D[6])
    r = p.returns["X"]
    assert (1 + r).prod() - 1 == pytest.approx(0.98 * 1.01 * 1.01 - 1, abs=1e-12)
    assert p.is_proxy["X"].tolist() == [False, False, True, True, True]
    out = historical_stress(pd.Series({"X": 1000.0}), IND, p, "P").set_index("level_key")
    assert out.at["X", "pnl"] == pytest.approx(1000 * (0.98 * 1.01 * 1.01 - 1), abs=1e-9)
    # P&L earned on proxy days = 1000 x (G6 - G3) with G3 = 1
    assert out.at["X", "pnl_proxy"] == pytest.approx(1000 * (0.98 * 1.01 * 1.01 - 1), abs=1e-9)
    assert out.at["X", "proxy_mv_suspended"] == 1000.0 and out.at["X", "proxy_mv_unlisted"] == 0.0


def test_base_inside_a_suspension():
    # X: suspended D1..D3 (s = D1, s - 1 = D0 at 10), resumes D4 at 9.8; base D2 is inside the suspension
    # full series: D1 proxy +1%, D2 proxy -2%, D3 proxy +2%, D4 residual = (9.8 / 10) / (1.01 x 0.98 x 1.02) - 1
    # window (D2, D6] keeps D3 (+2%), D4, D5 (+5%), D6 (-2%):
    #   cumulative = 1.02 x (0.98 / 1.009596) x 1.05 x 0.98 - 1 = (P6 / P0) / (1.01 x 0.98) - 1
    #              = 1.00842 / 0.9898 - 1 = 0.0188119 ; the pre-base part (D1, D2 proxies) stays outside
    x = [10, 10, 10, 10, 9.8, 10.29, 10.0842, 10.0842, 10.0842]
    A, S = frame(x, [False, True, True, True, False, False, False, False, False])
    p = scenario_path(A, S, IND, D[2], D[6])
    r = p.returns["X"]
    assert r.iloc[1] == pytest.approx(0.98 / (1.01 * 0.98 * 1.02) - 1, abs=1e-12)
    assert (1 + r).prod() - 1 == pytest.approx(1.00842 / 0.9898 - 1, abs=1e-12)
    assert (1 + r).prod() - 1 == pytest.approx(0.0188119, abs=5e-8)
    assert p.price_dates.equals(D[0:7])  # read back to s - 1 = D0, before the base


def test_unlisted_name_takes_the_proxy():
    # X lists on D4: D2..D4 are proxies (the listing day has no previous price), D5 onwards real
    x = [np.nan, np.nan, np.nan, np.nan, 9.8, 10.29, 10.0842, 10.0842, 10.0842]
    A, S = frame(x, [False] * 9)
    p = scenario_path(A, S, IND, D[1], D[6])
    assert p.is_proxy["X"].tolist() == [True, True, True, False, False]
    assert p.unlisted["X"] and not p.suspended["X"]
    np.testing.assert_allclose(p.returns["X"].iloc[:3], PROXY[1:4], atol=1e-12)


def test_rejects_a_suspension_starting_before_the_data():
    x = [10, 10, 10, 10, 9.8, 10.29, 10.0842, 10.0842, 10.0842]
    A, S = frame(x, [True, True, True, True, False, False, False, False, False])
    with pytest.raises(ValueError, match="read prices further back"):
        scenario_path(A, S, IND, D[2], D[6])


def test_historical_stress_hand_computed_and_additive():
    # MV 1000 each; A (银行) +10%, -5% ; B (银行) -2%, 0% ; C (电子) +1%, -10%
    # A: G 1.1, 1.045 -> daily +100, -55, pnl 45 ; B: daily -20, 0, pnl -20 ; C: G 1.01, 0.909 -> +10, -101, pnl -91
    # portfolio daily +90, -156 -> pnl -66, max daily loss 156 ; 银行 +80, -55 -> 25, 55 ; 电子 -91, 101
    idx = pd.bdate_range("2025-01-02", periods=2)
    R = pd.DataFrame({"A": [0.10, -0.05], "B": [-0.02, 0.0], "C": [0.01, -0.10]}, index=idx)
    path = ScenarioPath(R, R * 0 != 0, pd.Series(False, index=R.columns), pd.Series(False, index=R.columns), idx)
    ind = pd.Series({"A": "银行", "B": "银行", "C": "电子"})
    out = historical_stress(pd.Series({"A": 1000.0, "B": 1000.0, "C": 1000.0}), ind, path, "P")
    assert list(out.columns) == STRESS_COLUMNS
    o = out.set_index("level_key")
    exp = {"P": (-66, 156), "银行": (25, 55), "电子": (-91, 101), "A": (45, 55), "B": (-20, 20), "C": (-91, 101)}
    for k, (pnl, mdl) in exp.items():
        assert o.at[k, "pnl"] == pytest.approx(pnl, abs=1e-9) and o.at[k, "max_daily_loss"] == pytest.approx(mdl, abs=1e-9), k
    # unlike VaR, stress P&L adds up across levels
    assert out.groupby("level")["pnl"].sum().nunique() == 1


# ---- hypothetical -----------------------------------------------------------------------------------------------
# window of 4 days; industry 银行 = {A, B}, 电子 = {C}
RW = pd.DataFrame(
    {"A": [0.03, -0.03, 0.01, -0.01], "B": [0.01, -0.01, 0.05, -0.05], "C": [0.02, 0.01, -0.01, -0.02]},
    index=pd.bdate_range("2025-01-02", periods=4),
)
HIND = pd.Series({"A": "银行", "B": "银行", "C": "电子"})


def test_industry_shock_by_beta_hand_computed():
    # 银行 factor f = (A + B) / 2 = 2%, -2%, 3%, -3% (mean 0); var-sum 4 + 4 + 9 + 9 = 26e-4
    # cov-sum(A, f) = 6 + 6 + 3 + 3 = 18e-4 -> beta_A = 18/26 = 0.692308 ; cov-sum(B, f) = 2 + 2 + 15 + 15 = 34e-4 -> 1.307692
    # shock 银行 -10%: A -6.92308%, B -13.07692%; their equal-weight index falls exactly 10%
    sh = hypothetical_shocks({"industry": {"银行": -0.10}}, RW, HIND, ["A", "B", "C"])
    assert sh["A"] == pytest.approx(-0.0692308, abs=5e-8) and sh["B"] == pytest.approx(-0.1307692, abs=5e-8)
    assert sh[["A", "B"]].mean() == pytest.approx(-0.10, abs=1e-12)
    out = hypothetical_stress(pd.Series({"A": 1000.0, "B": 1000.0}), HIND, sh, "P").set_index("level_key")
    assert out.at["P", "pnl"] == pytest.approx(-200.0, abs=1e-9)  # equal weights = the industry index
    assert out["max_daily_loss"].isna().all() and (out["pnl_proxy"] == 0).all()


def test_member_betas_average_to_one():
    rng = np.random.default_rng(3)
    R = pd.DataFrame(rng.normal(0, 0.02, (250, 9)), columns=[f"S{i}" for i in range(9)])
    ind = pd.Series(["a"] * 4 + ["b"] * 5, index=R.columns)
    assert betas(R, R.mean(axis=1)).mean() == pytest.approx(1.0, abs=1e-12)  # market factor over the pool
    members = ind.index[ind == "b"]
    assert betas(R[members], R[members].mean(axis=1)).mean() == pytest.approx(1.0, abs=1e-12)
    sh = hypothetical_shocks({"industry": {"b": -0.10}}, R, ind, members)
    assert sh.mean() == pytest.approx(-0.10, abs=1e-12)


def test_raw_shocks_and_precedence():
    sh = hypothetical_shocks({"raw": {"market": -0.05, "industry": {"银行": -0.10}, "instrument": {"B": -0.30}}}, RW, HIND, ["A", "B", "C"])
    assert sh.to_dict() == {"A": -0.10, "B": -0.30, "C": -0.05}


@pytest.mark.parametrize("definition", [{"market": -0.05, "industry": {"银行": -0.1}}, {"industry": {"银行": -0.1, "电子": -0.1}}, {}])
def test_factor_mode_takes_exactly_one_factor(definition):
    with pytest.raises(ValueError, match="exactly one factor"):
        hypothetical_shocks(definition, RW, HIND, ["A"])


def test_rank_scenarios():
    rows = pd.DataFrame({"scenario_id": ["a", "b", "c"], "pnl": [-5.0, -20.0, 3.0]})
    assert rank_scenarios(rows, 2)["scenario_id"].tolist() == ["b", "a"]
