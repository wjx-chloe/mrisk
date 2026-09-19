import datetime as dt
import os

import pandas as pd
import pytest

from mrisk.data import sources
from mrisk.data.sources import AkshareSource


def test_clean_name_normalizes_fullwidth_and_spaces():
    assert sources.clean_name("万  科Ａ") == "万科A"


def test_norm_listings_maps_all_exchanges():
    sh = pd.DataFrame({"证券代码": ["600000"], "证券简称": ["浦发银行"], "上市日期": ["1999-11-10"]})
    star = pd.DataFrame({"证券代码": ["688001"], "证券简称": ["华兴源创"], "上市日期": ["2019-07-22"]})
    sz = pd.DataFrame({"A股代码": ["1"], "A股简称": ["平安银行"], "A股上市日期": ["1991-04-03"]})
    bj = pd.DataFrame({"证券代码": ["920682"], "证券简称": ["球冠电缆"], "上市日期": ["2020-07-27"]})
    out = sources.norm_listings(sh, star, sz, bj)
    assert out["exchange"].tolist() == ["SH", "SH", "SZ", "BJ"]
    assert out["symbol"].tolist() == ["600000", "688001", "000001", "920682"]
    assert out["list_date"].iloc[3] == pd.Timestamp("2020-07-27")


def test_norm_daily_types_and_sort():
    raw = pd.DataFrame({"date": ["2025-01-03", "2025-01-02"], "close": ["10.5", "10.0"], "volume": ["200.0", "100.0"]})
    out = sources.norm_daily(raw)
    assert out["close"].tolist() == [10.0, 10.5]
    assert out["volume"].tolist() == [100, 200]
    assert str(out["volume"].dtype) == "int64"


def test_norm_hfq_factor_sorted_ascending():
    raw = pd.DataFrame({"date": ["2025-07-16", "1900-01-01"], "hfq_factor": ["16.6", "1.0"]})
    out = sources.norm_hfq_factor(raw)
    assert out["hfq_factor"].tolist() == [1.0, 16.6]


def test_norm_sw_members():
    comps = {"银行": pd.DataFrame({"证券代码": ["600000"], "最新权重": ["1.5"]})}
    out = sources.norm_sw_members(comps)
    assert out.to_dict("records") == [{"symbol": "600000", "industry_l1": "银行", "sw_weight": "1.5"}]


class _FakeAk:
    def __init__(self, fail_times):
        self.calls = 0
        self.fail_times = fail_times
        self.no_proxy_seen = None

    def tool_trade_date_hist_sina(self):
        self.calls += 1
        self.no_proxy_seen = os.environ.get("NO_PROXY")
        if self.calls <= self.fail_times:
            raise ConnectionError("boom")
        return pd.DataFrame({"trade_date": ["2025-01-02", "2025-01-03"]})


@pytest.fixture
def fake_ak(monkeypatch):
    import akshare

    fake = _FakeAk(fail_times=1)
    monkeypatch.setattr(akshare, "tool_trade_date_hist_sina", fake.tool_trade_date_hist_sina)
    monkeypatch.setattr(sources.time, "sleep", lambda s: None)
    return fake


def test_retry_then_cache(tmp_path, fake_ak):
    src = AkshareSource(tmp_path, request_interval_sec=0, max_retries=2, today=dt.date(2025, 1, 3))
    cal = src.trade_calendar()
    assert fake_ak.calls == 2
    assert fake_ak.no_proxy_seen == "*"
    assert len(cal) == 2
    src.trade_calendar()
    assert fake_ak.calls == 2


def test_retry_exhausted_raises(tmp_path, fake_ak):
    fake_ak.fail_times = 10
    src = AkshareSource(tmp_path, request_interval_sec=0, max_retries=1)
    with pytest.raises(ConnectionError):
        src.trade_calendar()


@pytest.mark.network
def test_live_sina_daily_and_factor(tmp_path):
    src = AkshareSource(tmp_path)
    px = src.daily_raw("BJ", "920682", dt.date(2025, 4, 28), dt.date(2025, 5, 8))
    assert pd.Timestamp("2025-05-06") in set(px["trade_date"])
    assert not src.hfq_factor("SH", "600000").empty
