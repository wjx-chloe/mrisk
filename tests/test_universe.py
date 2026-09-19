import pandas as pd
import pytest

from mrisk.data.universe import build_instruments, select_universe

D = pd.to_datetime

LISTINGS = pd.DataFrame(
    {
        "exchange": ["SH", "SH", "SZ", "SZ", "SH", "BJ"],
        "symbol": ["600000", "600036", "000001", "300750", "600001", "920682"],
        "name": ["浦发银行", "招商银行", "平安银行", "宁德时代", "*ST某某", "球冠电缆"],
        "list_date": D(["1999-11-10", "2002-04-09", "1991-04-03", "2018-06-11", "2000-01-01", "2020-07-27"]),
    }
)
SW = pd.DataFrame(
    {
        "symbol": ["600000", "600036", "000001", "300750", "600001", "920682"],
        "industry_l1": ["银行", "银行", "银行", "电力设备", "电力设备", "电力设备"],
        "sw_weight": ["1.0", "5.0", "3.0", "9.0", "8.0", "0.1"],
    }
)


def test_select_universe_filters_and_ranks():
    u = select_universe(LISTINGS, SW, ["银行"], 2, D("2015-01-01"), ["BJ920682"])
    assert u["instrument_id"].tolist() == ["BJ920682", "SH600036", "SZ000001"]


def test_select_universe_industry_without_eligible_members_raises():
    # 电力设备: 宁德时代 listed 2018, 600001 is ST, 920682 is BJ -> nothing eligible
    with pytest.raises(ValueError):
        select_universe(LISTINGS, SW, ["银行", "电力设备"], 2, D("2015-01-01"), [])


def test_build_instruments():
    u = pd.DataFrame({"instrument_id": ["SH600000", "BJ920682"], "exchange": ["SH", "BJ"], "symbol": ["600000", "920682"]})
    delisted = pd.DataFrame({"exchange": ["SH"], "symbol": ["600000"]})
    inst = build_instruments(u, LISTINGS, SW, delisted)
    assert inst["board"].tolist() == ["main", "bj"]
    assert inst["status"].tolist() == ["delisted", "active"]
    assert inst["industry_l1"].tolist() == ["银行", "电力设备"]


def test_industry_override_fills_only_uncovered_names():
    u = pd.DataFrame({"instrument_id": ["SH600000", "BJ920682"], "exchange": ["SH", "BJ"], "symbol": ["600000", "920682"]})
    sw = SW[SW["symbol"] != "920682"]
    ov = pd.DataFrame({"instrument_id": ["BJ920682"], "industry_l1": ["电力设备"]})
    inst = build_instruments(u, LISTINGS, sw, LISTINGS.iloc[:0], ov)
    assert inst["industry_l1"].tolist() == ["银行", "电力设备"]
    with pytest.raises(ValueError):
        build_instruments(u, LISTINGS, sw, LISTINGS.iloc[:0], pd.DataFrame({"instrument_id": ["SH600000"], "industry_l1": ["电子"]}))


def test_build_instruments_missing_industry_raises():
    u = pd.DataFrame({"instrument_id": ["SH600000"], "exchange": ["SH"], "symbol": ["600000"]})
    with pytest.raises(ValueError):
        build_instruments(u, LISTINGS, SW.iloc[1:], LISTINGS.iloc[:0])
