import datetime as dt

import numpy as np
import pandas as pd
import pytest

from mrisk.config import load_config
from mrisk.db import schema as s
from mrisk.db.repo import get_engine, init_db, read_frame
from mrisk.jobs import daily as jobs
from mrisk.jobs import etl
from mrisk.jobs.scheduler import build_trigger

CAL = pd.bdate_range("2025-03-03", "2025-06-30")
LAST_BACKFILL = pd.Timestamp("2025-06-27")
INDS = ["银行", "电子", "医药生物", "汽车"]
SYMS = [("SH", f"6000{i:02d}") for i in range(6)] + [("SZ", f"0000{i:02d}") for i in range(6)]
BJ = ("BJ", "920682")


class FakeSource:
    def __init__(self):
        rng = np.random.default_rng(1)
        self.px = {}
        for ex, sym in SYMS + [BJ]:
            r = rng.uniform(-0.03, 0.03, len(CAL))
            self.px[(ex, sym)] = np.round(10 * np.exp(np.cumsum(r)), 2)

    def trade_calendar(self):
        return pd.DataFrame({"trade_date": CAL})

    def listings(self):
        rows = [{"exchange": ex, "symbol": sym, "name": f"股票{sym}", "list_date": pd.Timestamp("2005-01-04")} for ex, sym in SYMS]
        rows.append({"exchange": "BJ", "symbol": "920682", "name": "球冠电缆", "list_date": pd.Timestamp("2020-07-27")})
        return pd.DataFrame(rows)

    def sw_l1_members(self):
        rows = [{"symbol": sym, "industry_l1": INDS[i % 4], "sw_weight": str(1 + i)} for i, (_, sym) in enumerate(SYMS)]
        rows.append({"symbol": "920682", "industry_l1": "电力设备", "sw_weight": "0.1"})
        return pd.DataFrame(rows)

    def delisted(self):
        return pd.DataFrame({"exchange": ["SH"], "symbol": ["600999"]})

    def daily_raw(self, exchange, symbol, start, end):
        m = (CAL >= pd.Timestamp(start)) & (CAL <= pd.Timestamp(end))
        out = pd.DataFrame({"trade_date": CAL[m], "close": self.px[(exchange, symbol)][m], "volume": 1000})
        if symbol == "600003":  # suspended one day
            out = out[out["trade_date"] != pd.Timestamp("2025-05-14")]
        if symbol == "600004":  # suspended across price_start (2025-03-10)
            out = out[(out["trade_date"] < pd.Timestamp("2025-03-06")) | (out["trade_date"] > pd.Timestamp("2025-03-11"))]
        return out.reset_index(drop=True)

    def hfq_factor(self, exchange, symbol):
        return pd.DataFrame({"effective_date": pd.to_datetime(["1900-01-01"]), "hfq_factor": [1.0]})


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = load_config("config.yaml")
    cfg["_root"] = tmp_path
    cfg["data"]["price_start"] = "2025-03-10"
    cfg["portfolio"].update(start="2025-04-01", n_holdings=6, min_industries=3, initial_capital=1e7)
    cfg["universe"].update(industries=INDS, per_industry=3, listed_before="2015-01-01")
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "code_map.csv").write_text(
        "instrument_id,old_symbol,new_symbol,effective_date\nBJ920682,834682,920682,2025-05-06\n", encoding="utf-8"
    )
    (seed / "industry_override.csv").write_text("instrument_id,industry_l1,note\n", encoding="utf-8")
    monkeypatch.setattr(etl, "SEED_DIR", seed)
    src = FakeSource()
    monkeypatch.setattr(etl, "make_source", lambda cfg: src)
    engine = get_engine(f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    init_db(engine)
    etl.build_universe_seed(cfg, src)
    return engine, cfg


def test_backfill_then_daily(env, monkeypatch):
    engine, cfg = env
    blocked = jobs.backfill(engine, cfg, end=LAST_BACKFILL)
    assert len(blocked) == 0

    res = read_frame(engine, s.dq_result, date_col="check_date")
    assert set(res["status"]) <= {"pass", "skip"}, res[res["status"] == "fail"]
    assert (res[res["rule_id"] == "DQ10"]["status"] == "skip").all()

    inst = read_frame(engine, s.instrument, date_col="list_date")
    assert len(inst) == 13 and inst.set_index("instrument_id").loc["BJ920682", "board"] == "bj"

    px = read_frame(engine, s.daily_price)
    susp = px[(px["instrument_id"] == "SH600003") & (px["trade_date"] == pd.Timestamp("2025-05-14"))]
    assert susp["is_suspended"].tolist() == [True]
    lead = px[(px["instrument_id"] == "SH600004") & (px["trade_date"] <= pd.Timestamp("2025-03-11"))]
    assert lead["trade_date"].min() == pd.Timestamp("2025-03-10")
    assert lead["is_suspended"].all() and len(lead) == 2
    src = etl.make_source(cfg)
    last_real = src.px[("SH", "600004")][CAL.get_loc(pd.Timestamp("2025-03-05"))]
    assert (lead["close"] == last_real).all()

    feed = pd.read_csv(etl.feed_path(cfg), dtype={"symbol": str}, parse_dates=["trade_date"])
    bj = feed[feed["exchange"] == "BJ"]
    assert set(bj.loc[bj["trade_date"] < "2025-05-06", "symbol"]) == {"834682"}
    assert set(bj.loc[bj["trade_date"] >= "2025-05-06", "symbol"]) == {"920682"}
    pos = read_frame(engine, s.position)
    bj_pos = pos[pos["instrument_id"] == "BJ920682"]
    assert bj_pos["trade_date"].nunique() == pos["trade_date"].nunique()

    monkeypatch.setattr(etl, "now_cst", lambda: dt.datetime(2025, 6, 30, 16, 31))
    assert len(jobs.daily(engine, cfg, "2025-06-30")) == 0
    px = read_frame(engine, s.daily_price, start="2025-06-30")
    assert set(px["load_type"]) == {"daily"} and len(px) == 13
    assert read_frame(engine, s.daily_price, end="2025-06-27")["load_type"].eq("backfill").all()
    last = read_frame(engine, s.dq_result, start="2025-06-30", date_col="check_date")
    assert last.set_index("rule_id").loc["DQ10", "status"] == "pass"


def test_daily_skips_non_trading_day(env):
    engine, cfg = env
    assert jobs.daily(engine, cfg, "2025-06-28") is None


def test_scheduler_trigger_weekday_1630():
    t = build_trigger("16:30")
    nxt = t.get_next_fire_time(None, pd.Timestamp("2026-09-19 12:00", tz="Asia/Shanghai").to_pydatetime())
    assert (nxt.weekday(), nxt.hour, nxt.minute) == (0, 16, 30)
