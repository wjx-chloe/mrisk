import pandas as pd
import pytest

from mrisk.data.transform import apply_hfq, build_daily_price, drop_before_listing, fill_suspended, infer_board

D = pd.to_datetime


def test_infer_board():
    ex = pd.Series(["SH", "SH", "SZ", "SZ", "SZ", "BJ"])
    sym = pd.Series(["600000", "688001", "000001", "300750", "301001", "920682"])
    assert infer_board(ex, sym).tolist() == ["main", "star", "main", "chinext", "chinext", "bj"]


def test_apply_hfq_uses_factor_in_force_hand_computed():
    daily = pd.DataFrame(
        {
            "instrument_id": ["A"] * 3,
            "trade_date": D(["2025-01-02", "2025-01-03", "2025-01-06"]),
            "close": [10.0, 9.5, 9.6],
            "volume": [1, 1, 1],
        }
    )
    # ex-date 2025-01-03: factor 1.0 -> 1.05
    factors = pd.DataFrame(
        {"instrument_id": ["A", "A"], "effective_date": D(["1900-01-01", "2025-01-03"]), "hfq_factor": [1.0, 1.05]}
    )
    out = apply_hfq(daily, factors)
    assert out["adj_close"].tolist() == pytest.approx([10.0, 9.975, 10.08])


def test_apply_hfq_tolerates_mixed_datetime_resolutions():
    daily = pd.DataFrame(
        {"instrument_id": ["A"], "trade_date": D(["2025-01-02"]).astype("datetime64[us]"), "close": [2.0], "volume": [1]}
    )
    factors = pd.DataFrame(
        {"instrument_id": ["A"], "effective_date": D(["1900-01-01"]).astype("datetime64[s]"), "hfq_factor": [1.5]}
    )
    assert apply_hfq(daily, factors)["adj_close"].tolist() == [3.0]


def test_apply_hfq_missing_factor_raises():
    daily = pd.DataFrame({"instrument_id": ["A"], "trade_date": D(["2025-01-02"]), "close": [1.0], "volume": [1]})
    factors = pd.DataFrame({"instrument_id": ["A"], "effective_date": D(["2025-06-01"]), "hfq_factor": [1.0]})
    with pytest.raises(ValueError):
        apply_hfq(daily, factors)


def test_drop_before_listing():
    daily = pd.DataFrame({"instrument_id": ["BJ1", "BJ1", "SH1"], "trade_date": D(["2019-01-02", "2020-07-27", "2019-01-02"])})
    lst = pd.DataFrame({"instrument_id": ["BJ1", "SH1"], "list_date": D(["2020-07-27", "1999-01-01"])})
    out = drop_before_listing(daily, lst)
    assert list(zip(out["instrument_id"], out["trade_date"])) == [("BJ1", D("2020-07-27")), ("SH1", D("2019-01-02"))]


def test_fill_suspended_interior_only():
    cal = pd.Series(D(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]))
    daily = pd.DataFrame(
        {
            "instrument_id": ["A", "A", "A", "B"],
            "trade_date": D(["2025-01-02", "2025-01-06", "2025-01-07", "2025-01-03"]),
            "close": [10.0, 11.0, 12.0, 5.0],
            "adj_close": [20.0, 22.0, 24.0, 5.0],
            "volume": [100, 0, 300, 50],
        }
    )
    out = fill_suspended(daily, cal)
    a = out[out["instrument_id"] == "A"]
    assert a["trade_date"].tolist() == D(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]).tolist()
    assert a["close"].tolist() == [10.0, 10.0, 11.0, 12.0]
    assert a["adj_close"].tolist() == [20.0, 20.0, 22.0, 24.0]
    assert a["volume"].tolist() == [100, 0, 0, 300]
    # 01-03 synthesised, 01-06 quoted with zero volume
    assert a["is_suspended"].tolist() == [False, True, True, False]
    # trailing day 01-08 is not filled for either instrument
    assert D("2025-01-08") not in set(out["trade_date"])
    assert len(out[out["instrument_id"] == "B"]) == 1


def test_build_daily_price_columns():
    cal = pd.Series(D(["2025-01-02", "2025-01-03"]))
    daily = pd.DataFrame({"instrument_id": ["A"], "trade_date": D(["2025-01-02"]), "close": [2.0], "volume": [10]})
    factors = pd.DataFrame({"instrument_id": ["A"], "effective_date": D(["1900-01-01"]), "hfq_factor": [3.0]})
    lst = pd.DataFrame({"instrument_id": ["A"], "list_date": D(["2000-01-01"])})
    out = build_daily_price(daily, factors, lst, cal)
    assert out.columns.tolist() == ["trade_date", "instrument_id", "close", "adj_close", "volume", "is_suspended"]
    assert out["adj_close"].tolist() == [6.0]
