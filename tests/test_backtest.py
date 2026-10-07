import numpy as np
import pytest

from mrisk.engine.backtest import (
    CRITICAL_5PCT,
    christoffersen_ind,
    exceptions,
    kupiec_pof,
    rolling_backtest,
    traffic_light,
    zone_bounds,
)


def test_kupiec_hand_computed():
    # T = 250, x = 7, p = 0.01
    # ln L0 = 243 ln 0.99 + 7 ln 0.01 = 243 x (-0.0100503) + 7 x (-4.6051702) = -2.4422316 - 32.2361913 = -34.6784229
    # ln L1 = 243 ln 0.972 + 7 ln 0.028 = 243 x (-0.0283995) + 7 x (-3.5755508) = -6.9010723 - 25.0288554 = -31.9299277
    # LR = -2 (-34.6784229 + 31.9299277) = 5.4970 > 3.8415 -> reject at 5%
    lr, pv = kupiec_pof(250, 7, 0.01)
    assert lr == pytest.approx(5.4970, abs=5e-5)
    assert pv == pytest.approx(0.0190, abs=5e-5)
    assert lr > CRITICAL_5PCT
    assert CRITICAL_5PCT == pytest.approx(3.8415, abs=5e-5)


def test_kupiec_edges_zero_log_zero():
    # x = 0: ln L1 = 250 ln 1 + 0 ln 0 = 0, LR = -2 x 250 ln 0.99 = 5.0252
    lr, _ = kupiec_pof(250, 0, 0.01)
    assert lr == pytest.approx(5.0252, abs=5e-5)
    # x = T: ln L1 = 0 ln 0 + 250 ln 1 = 0, LR = -2 x 250 ln 0.01 = 2302.5851
    lr, pv = kupiec_pof(250, 250, 0.01)
    assert lr == pytest.approx(2302.5851, abs=5e-5) and pv == pytest.approx(0.0)
    # x = pT exactly: LR = 0
    lr, pv = kupiec_pof(200, 2, 0.01)
    assert lr == pytest.approx(0.0, abs=1e-12) and pv == pytest.approx(1.0)
    assert np.isfinite(kupiec_pof([250, 250], [0, 250], 0.05)[0]).all()


def test_christoffersen_hand_computed():
    # 1 1 1 0 0 0 0 0 0 0: pairs 11 11 10 00 00 00 00 00 00 -> n00 6, n01 0, n10 1, n11 2
    # pi01 = 0, pi11 = 2/3, pi = 2/9
    # ln L0 = 7 ln(7/9) + 2 ln(2/9) = -4.7673558 ; ln L1 = 6 ln 1 + 0 ln 0 + 1 ln(1/3) + 2 ln(2/3) = -1.9095425
    # LR = -2 (-4.7673558 + 1.9095425) = 5.7156  (clustered exceptions)
    lr, pv = christoffersen_ind(6, 0, 1, 2)
    assert lr == pytest.approx(5.7156, abs=5e-5)
    assert pv == pytest.approx(0.0168, abs=5e-5)


def test_christoffersen_edges():
    # 0 0 1 1 0 0 0 1 0 0: n00 4, n01 2, n10 2, n11 1 -> pi01 = pi11 = 1/3 = pi -> LR 0
    assert christoffersen_ind(4, 2, 2, 1)[0] == pytest.approx(0.0, abs=1e-12)
    # no exceptions (n01 = n10 = n11 = 0) and all exceptions (n00 = n01 = n10 = 0): every term is 0 ln 0
    for counts in [(249, 0, 0, 0), (0, 0, 0, 249)]:
        lr, pv = christoffersen_ind(*counts)
        assert lr == 0.0 and pv == 1.0
    # one isolated exception at the end: n10 = n11 = 0, pi11 has no denominator
    lr, _ = christoffersen_ind(248, 1, 0, 0)
    assert lr == pytest.approx(0.0, abs=1e-12)


def test_traffic_light_boundaries_basel_250():
    # Basel table: 99% -> <= 4 green, 5-9 yellow, >= 10 red; same rule at 95% -> 17 / 26
    assert zone_bounds(0.01, 250) == (4, 9)
    assert zone_bounds(0.05, 250) == (17, 26)
    assert traffic_light([0, 4, 5, 9, 10, 30], 1 - 0.99, 250).tolist() == ["green", "green", "yellow", "yellow", "red", "red"]
    assert traffic_light([17, 18, 26, 27], 1 - 0.95, 250).tolist() == ["green", "yellow", "yellow", "red"]


def test_zone_bounds_hand_computed():
    # n = 2, p = 0.5: P(X <= 0) = 0.25, P(X <= 1) = 0.75, P(X <= 2) = 1
    # green while < 0.95 -> k <= 1; yellow while < 0.9999 -> also k <= 1; so 2 is red with no yellow count
    assert zone_bounds(0.5, 2) == (1, 1)
    assert traffic_light([0, 1, 2], 0.5, 2).tolist() == ["green", "green", "red"]
    # n = 1, p = 0.06: P(X <= 0) = 0.94 < 0.95 green, P(X <= 1) = 1 red
    assert zone_bounds(0.06, 1) == (0, 0)
    with pytest.raises(ValueError):
        zone_bounds(0.0, 250)


def test_exceptions_are_strict():
    assert exceptions([10.0, 10.0, 10.0], [-10.0, -10.0001, 5.0]).tolist() == [False, True, False]
    with pytest.raises(ValueError, match="missing"):
        exceptions([10.0, np.nan], [0.0, 0.0])


def _loop_reference(hit, valid, w, p):
    rows = []
    for k in range(len(hit)):
        lo = k - w + 1
        if lo < 0 or not valid[lo : k + 1].all():
            rows.append(None)
            continue
        h = hit[lo : k + 1]
        c = {"n00": 0, "n01": 0, "n10": 0, "n11": 0}
        for a, b in zip(h[:-1], h[1:]):
            c[f"n{int(a)}{int(b)}"] += 1
        lr, _ = kupiec_pof(w, h.sum(), p)
        li, _ = christoffersen_ind(c["n00"], c["n01"], c["n10"], c["n11"])
        rows.append((int(h.sum()), c, float(lr), float(li)))
    return rows


def test_rolling_matches_loop_reference_with_gaps():
    rng = np.random.default_rng(5)
    M, w = 400, 60
    hit = rng.random(M) < 0.05
    hit[100:104] = True  # a cluster
    valid = np.ones(M, dtype=bool)
    valid[[150, 151, 300]] = False
    out = rolling_backtest(hit, valid, w, 0.01)
    ref = _loop_reference(hit, valid, w, 0.01)
    for k, r in enumerate(ref):
        if r is None:
            assert not out.at[k, "complete"] and np.isnan(out.at[k, "lr_pof"])
            continue
        x, c, lr, li = r
        row = out.loc[k]
        assert row["complete"] and row["n_obs"] == w and row["n_exceptions"] == x
        assert {key: row[key] for key in c} == c
        assert row["lr_pof"] == pytest.approx(lr, rel=1e-12) and row["lr_ind"] == pytest.approx(li, rel=1e-12, abs=1e-12)
    assert out["complete"].sum() == sum(r is not None for r in ref)
    full = out[out["complete"]]
    assert (full[["n00", "n01", "n10", "n11"]].sum(axis=1) == w - 1).all()
