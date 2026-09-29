import numpy as np
import pytest

from mrisk.engine.var import hs_es, hs_var, param_sigma, param_var

# T = 10 P&L observations; sorted: -80, -50, -35, -20, -10, 5, 10, 20, 30, 40
PNL = np.array([-50.0, 30.0, -20.0, 10.0, -80.0, 40.0, -10.0, 5.0, 20.0, -35.0])[:, None]


def test_hs_var_es_hand_computed():
    # linear quantile position = (1 - c) x (T - 1)
    # c = 0.99: pos 0.09 -> -80 + 0.09 x (-50 + 80) = -77.3   tail < -77.3: {-80}             ES 80
    # c = 0.95: pos 0.45 -> -80 + 0.45 x 30        = -66.5   tail < -66.5: {-80}             ES 80
    # c = 0.70: pos 2.70 -> -35 + 0.70 x (-20 + 35) = -24.5  tail < -24.5: {-80, -50, -35}   ES 55
    var = hs_var(PNL, [0.99, 0.95, 0.7])
    np.testing.assert_allclose(var[:, 0], [77.3, 66.5, 24.5], atol=1e-10)
    es, n = hs_es(PNL, var)
    np.testing.assert_allclose(es[:, 0], [80.0, 80.0, 55.0], atol=1e-10)
    assert n[:, 0].tolist() == [1, 1, 3]
    # with T = 10 the 95% and 99% tails both contain only the worst day, so ES95 == ES99
    assert es[0, 0] == es[1, 0]


def test_hs_es_empty_tail_falls_back_to_var():
    ties = np.array([-5.0, -5.0, -5.0, -5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # 0.7: between x3 = x4 = -5
    flat = np.zeros(10)  # a name suspended for the whole window
    pnl = np.column_stack([ties, flat])
    var = hs_var(pnl, [0.7])
    np.testing.assert_allclose(var, [[5.0, 0.0]], atol=1e-12)
    es, n = hs_es(pnl, var)
    np.testing.assert_allclose(es, var, atol=1e-12)
    assert n.tolist() == [[0, 0]]


def test_param_var_hand_computed():
    # R = 0.02, -0.02, 0.01, -0.01: mean 0, sample variance (4 + 4 + 1 + 1)e-4 / 3 = 1/3000
    # sigma = sqrt(1/3000) = 0.0182574186 ; MV 1e6 -> 18257.4186
    # VaR99 = 2.3263478740 x 18257.4186 = 42473.1069 ; VaR95 = 1.6448536270 x 18257.4186 = 30030.7812
    R = np.array([0.02, -0.02, 0.01, -0.01])[:, None]
    var = param_var(R, [[1e6]], [0.95, 0.99])
    np.testing.assert_allclose(var[:, 0], [30030.7812, 42473.1069], atol=5e-5)


def test_param_var_does_not_add_the_mean():
    R = np.array([0.03, 0.01, 0.04, 0.00])  # mean 0.02, deviations 0.01, -0.01, 0.02, -0.02
    var = param_var(R[:, None], [[1e6]], [0.99])
    np.testing.assert_allclose(var, param_var((R - 0.02)[:, None], [[1e6]], [0.99]), rtol=1e-12)


def test_sigma_two_paths_agree():
    rng = np.random.default_rng(7)
    R = rng.normal(0.0005, 0.02, (250, 12)) @ rng.normal(0, 1, (12, 12)) * 0.3
    E = np.hstack([rng.uniform(1e5, 1e6, (12, 1)), np.diag(rng.uniform(1e5, 1e6, 12)), rng.uniform(0, 1e6, (12, 3))])
    np.testing.assert_allclose(param_sigma(R, E), np.std(R @ E, axis=0, ddof=1), rtol=1e-12)


def _reference(pnl, R, E, conf):
    """Loop-based reference, one column and one level at a time."""
    C, K = len(conf), pnl.shape[1]
    var, es, pv = np.empty((C, K)), np.empty((C, K)), np.empty((C, K))
    from scipy.stats import norm

    for i, c in enumerate(conf):
        for k in range(K):
            x = np.sort(pnl[:, k])
            h = (1 - c) * (len(x) - 1)
            lo = int(np.floor(h))
            q = x[lo] + (h - lo) * (x[min(lo + 1, len(x) - 1)] - x[lo])
            var[i, k] = -q
            tail = [v for v in x if v < q]
            es[i, k] = -sum(tail) / len(tail) if tail else -q
            e = E[:, k]
            mu = R.mean(axis=0)
            s2 = sum(float(e @ (R[t] - mu)) ** 2 for t in range(len(R))) / (len(R) - 1)
            pv[i, k] = norm.ppf(c) * np.sqrt(s2)
    return var, es, pv


def test_vectorised_matches_loop_reference():
    rng = np.random.default_rng(3)
    R = rng.standard_t(4, (250, 6)) * 0.015
    E = np.hstack([rng.uniform(1e5, 1e6, (6, 1)), np.diag(rng.uniform(1e5, 1e6, 6))])
    pnl = R @ E
    conf = [0.95, 0.99, 0.975]
    var = hs_var(pnl, conf)
    es, _ = hs_es(pnl, var)
    rv, re, rp = _reference(pnl, R, E, conf)
    np.testing.assert_allclose(var, rv, rtol=1e-12)
    np.testing.assert_allclose(es, re, rtol=1e-12)
    np.testing.assert_allclose(param_var(R, E, conf), rp, rtol=1e-10)


def test_properties():
    rng = np.random.default_rng(11)
    R = rng.normal(0, 0.02, (250, 5))
    E = np.hstack([np.full((5, 1), 2e5), np.diag(np.full(5, 2e5))])
    pnl = R @ E
    var = hs_var(pnl, [0.95, 0.99])
    es, _ = hs_es(pnl, var)
    assert (es >= var).all()
    assert (var[1] >= var[0]).all()
    pv = param_var(R, E, [0.95, 0.99])
    assert (pv[1] > pv[0]).all()
    # homogeneity: doubling every position doubles every measure
    np.testing.assert_allclose(hs_var(R @ (2 * E), [0.95, 0.99]), 2 * var, rtol=1e-12)
    np.testing.assert_allclose(param_var(R, 2 * E, [0.95, 0.99]), 2 * pv, rtol=1e-12)


@pytest.mark.parametrize("conf", [0.0, 1.0, [0.95, 1.2], [[0.95]]])
def test_rejects_bad_confidence(conf):
    with pytest.raises(ValueError, match="confidence"):
        hs_var(PNL, conf)


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="instruments"):
        param_sigma(np.zeros((5, 3)), np.zeros((2, 1)))
    with pytest.raises(ValueError, match="2-D"):
        hs_var(np.zeros(5), [0.99])
