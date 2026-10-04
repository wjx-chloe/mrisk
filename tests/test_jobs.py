import datetime as dt

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import delete

from mrisk.config import load_config
from mrisk.db import schema as s
from mrisk.db.repo import get_engine, init_db, read_frame, upsert
from mrisk.jobs import daily as jobs
from mrisk.jobs import etl
from mrisk.jobs import risk as risk_job
from mrisk.jobs.scheduler import build_trigger

CAL = pd.bdate_range("2025-03-03", "2025-06-30")
LAST_BACKFILL = pd.Timestamp("2025-06-27")
INDS = ["银行", "电子", "医药生物", "汽车"]
SYMS = [("SH", f"6000{i:02d}") for i in range(6)] + [("SZ", f"0000{i:02d}") for i in range(6)]
BJ = ("BJ", "920682")


class FakeSource:
    def __init__(self):
        self.no_quote: set[tuple[str, pd.Timestamp]] = set()
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
        drop = [d for s, d in self.no_quote if s in (symbol, "*")]
        out = out[~out["trade_date"].isin(drop)]
        return out.reset_index(drop=True)

    def hfq_factor(self, exchange, symbol):
        return pd.DataFrame({"effective_date": pd.to_datetime(["1900-01-01"]), "hfq_factor": [1.0]})


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = load_config("config.yaml")
    cfg["_root"] = tmp_path
    cfg["data"]["price_start"] = "2025-03-10"
    cfg["portfolio"].update(start="2025-04-01", n_holdings=6, min_industries=3, initial_capital=1e7)
    cfg["lookback_days"] = 20  # the fake calendar holds ~80 days
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
    rr = read_frame(engine, s.risk_result)
    assert set(rr["trade_date"]) == {pd.Timestamp("2025-06-30")} and set(rr["confidence"]) == {0.95, 0.99}


def test_daily_infers_suspension_for_held_stock_without_quote(env, monkeypatch):
    engine, cfg = env
    jobs.backfill(engine, cfg, end=LAST_BACKFILL)
    held = read_frame(engine, s.position, start=LAST_BACKFILL)["instrument_id"].iloc[0]
    src = etl.make_source(cfg)
    src.no_quote.add((held[2:], pd.Timestamp("2025-06-30")))
    monkeypatch.setattr(etl, "now_cst", lambda: dt.datetime(2025, 6, 30, 16, 31))
    assert len(jobs.daily(engine, cfg, "2025-06-30")) == 0
    row = read_frame(engine, s.daily_price, start="2025-06-30", instrument_id=held)
    prev = read_frame(engine, s.daily_price, start=LAST_BACKFILL, end=LAST_BACKFILL, instrument_id=held)
    assert row["is_suspended"].tolist() == [True]
    assert row["close"].tolist() == prev["close"].tolist()
    res = read_frame(engine, s.dq_result, start="2025-06-30", date_col="check_date")
    assert set(res["status"]) <= {"pass", "skip"}


def test_daily_does_not_infer_when_whole_day_missing(env, monkeypatch):
    engine, cfg = env
    jobs.backfill(engine, cfg, end=LAST_BACKFILL)
    etl.make_source(cfg).no_quote.add(("*", pd.Timestamp("2025-06-30")))
    monkeypatch.setattr(etl, "now_cst", lambda: dt.datetime(2025, 6, 30, 16, 31))
    blocked = jobs.daily(engine, cfg, "2025-06-30")
    assert blocked.tolist() == [pd.Timestamp("2025-06-30")]
    assert read_frame(engine, s.daily_price, start="2025-06-30").empty
    assert len(read_frame(engine, s.position, start="2025-06-30")) == 6
    res = read_frame(engine, s.dq_result, start="2025-06-30", date_col="check_date").set_index("rule_id")["status"]
    assert res["DQ01"] == "fail" and res["DQ02"] == "fail"
    assert read_frame(engine, s.risk_result).empty


def test_daily_skips_non_trading_day(env):
    engine, cfg = env
    assert jobs.daily(engine, cfg, "2025-06-28") is None


def test_scheduler_trigger_weekday_1630():
    t = build_trigger("16:30")
    nxt = t.get_next_fire_time(None, pd.Timestamp("2026-09-19 12:00", tz="Asia/Shanghai").to_pydatetime())
    assert (nxt.weekday(), nxt.hour, nxt.minute) == (0, 16, 30)


# ---- risk job ------------------------------------------------------------------------------------------------
SWITCH = pd.Timestamp("2025-05-06")
RISK_DATE = pd.Timestamp("2025-05-20")  # a 20-return window ending here starts 2025-04-23 and spans SWITCH


@pytest.fixture
def risk_env(env):
    engine, cfg = env
    assert len(jobs.backfill(engine, cfg, end=LAST_BACKFILL)) == 0
    return engine, cfg


def test_risk_job_writes_all_levels_and_bj_series_is_continuous_across_920_switch(risk_env):
    engine, cfg = risk_env
    status = risk_job.run_risk(engine, cfg, RISK_DATE, run_id="r1")
    assert status.to_dict() == {RISK_DATE: "ok"}

    rr = read_frame(engine, s.risk_result)
    pos = read_frame(engine, s.position, RISK_DATE, RISK_DATE)
    n_ind = read_frame(engine, s.instrument, date_col="list_date").set_index("instrument_id").loc[pos["instrument_id"], "industry_l1"].nunique()
    assert len(rr) == (1 + n_ind + len(pos)) * 2
    assert set(rr["confidence"]) == {0.95, 0.99} and set(rr["run_id"]) == {"r1"}
    assert rr.loc[rr["level"] == "portfolio", "market_value"].iloc[0] == pytest.approx((pos["quantity"] * read_frame(
        engine, s.daily_price, RISK_DATE, RISK_DATE).set_index("instrument_id").loc[pos["instrument_id"], "close"].to_numpy()).sum())
    assert "BJ920682" in set(rr["level_key"]) and not rr["level_key"].str.contains("834682").any()

    # BJ920682 is one series across the code switch: 21 prices, no gap, and its instrument VaR is exactly
    # -quantile(MV x r) over returns that include the switch day
    px = read_frame(engine, s.daily_price, "2025-04-22", RISK_DATE, instrument_id="BJ920682").set_index("trade_date")
    assert len(px) == 21 and SWITCH in px.index and not px["is_suspended"].any()
    r = (px["adj_close"] / px["adj_close"].shift(1) - 1).dropna()
    assert r[SWITCH] == pytest.approx(px.loc[SWITCH, "adj_close"] / px["adj_close"].iloc[px.index.get_loc(SWITCH) - 1] - 1)
    mv = pos.set_index("instrument_id").loc["BJ920682", "quantity"] * px.loc[RISK_DATE, "close"]
    bj = rr[rr["level_key"] == "BJ920682"].set_index("confidence")
    assert bj.loc[0.99, "var_hs"] == pytest.approx(-np.quantile(mv * r.to_numpy(), 0.01), rel=1e-12)
    assert bj.loc[0.95, "var_param"] == pytest.approx(1.6448536269514722 * mv * r.std(ddof=1), rel=1e-12)


def test_risk_job_rerun_replaces_the_date(risk_env):
    engine, cfg = risk_env
    risk_job.run_risk(engine, cfg, RISK_DATE, run_id="r1")
    dropped = read_frame(engine, s.position, RISK_DATE, RISK_DATE)["instrument_id"].iloc[0]
    with engine.begin() as conn:
        conn.execute(delete(s.position).where(s.position.c.trade_date == RISK_DATE.date(), s.position.c.instrument_id == dropped))
    risk_job.run_risk(engine, cfg, RISK_DATE, run_id="r2")
    rr = read_frame(engine, s.risk_result)
    assert set(rr["run_id"]) == {"r2"}
    assert dropped not in set(rr["level_key"])


def _dq(engine, day, rule, status="fail", created=dt.datetime(2030, 1, 1)):
    """A later DQ run for one check date; it becomes the latest run for that date."""
    upsert(engine, s.dq_result, pd.DataFrame([{
        "run_id": f"later-{created:%Y%m%d}", "check_date": pd.Timestamp(day), "rule_id": rule, "status": status,
        "n_checked": 6, "n_failed": int(status == "fail"), "detail_json": None, "created_at": created}]))


def _cal(engine, start, end):
    return etl.trading_days(engine, pd.Timestamp(start), pd.Timestamp(end))


F = pd.Timestamp("2025-05-21")


def test_dq_gate_position_failure_blocks_only_that_day(risk_env):
    engine, cfg = risk_env
    _dq(engine, F, "DQ07")
    dates = _cal(engine, "2025-05-20", "2025-06-27")
    gate = risk_job.dq_gate(engine, cfg, dates)
    assert gate.index[gate["status"] != "ok"].tolist() == [F]
    assert gate.at[F, "reason"] == "position ['DQ07'] on 2025-05-21"


@pytest.mark.parametrize("rule", ["DQ02", "DQ03", "DQ11"])
def test_dq_gate_price_rules_including_dq11_block_the_whole_window(risk_env, rule):
    engine, cfg = risk_env
    _dq(engine, F, rule)
    cal = etl.trading_days(engine)
    i = cal.get_loc(F)
    gate = risk_job.dq_gate(engine, cfg, _cal(engine, "2025-05-20", "2025-06-27"))
    assert gate.index[gate["status"] == "blocked"].equals(cal[i : i + 21])


def test_dq_gate_price_failure_blocks_every_window_that_holds_it(risk_env):
    engine, cfg = risk_env
    _dq(engine, F, "DQ03")
    cal = etl.trading_days(engine)
    i = cal.get_loc(F)
    dates = _cal(engine, "2025-05-20", "2025-06-27")
    gate = risk_job.dq_gate(engine, cfg, dates)
    # F and the next lookback (20) trading days hold F in their window; the day before and the 21st after do not
    assert gate.index[gate["status"] == "blocked"].equals(cal[i : i + 21])
    assert gate.at[cal[i - 1], "status"] == "ok" and gate.at[cal[i + 21], "status"] == "ok"
    assert "DQ03@2025-05-21" in gate.at[cal[i + 20], "reason"]
    # fixed and re-checked: the latest run for F passes, the failure is resolved
    _dq(engine, F, "DQ03", "pass", dt.datetime(2031, 1, 1))
    assert (risk_job.dq_gate(engine, cfg, dates)["status"] == "ok").all()


def test_dq_gate_missing_dq_in_window_and_precedence(risk_env):
    engine, cfg = risk_env
    gap = pd.Timestamp("2025-05-12")
    with engine.begin() as conn:
        conn.execute(delete(s.dq_result).where(s.dq_result.c.check_date == gap.date()))
    _dq(engine, F, "DQ07")
    cal = etl.trading_days(engine)
    i = cal.get_loc(gap)
    gate = risk_job.dq_gate(engine, cfg, cal[i - 1 : i + 22])
    st = gate["status"]
    assert st[cal[i - 1]] == "ok" and st[cal[i + 21]] == "ok"
    assert (st[cal[i : i + 21].drop(F)] == "no_dq").all()
    assert st[F] == "blocked"  # a known failure wins over a missing check
    assert gate.at[gap, "reason"] == "no DQ result for 1 day(s) in window, first 2025-05-12"


def test_run_risk_deletes_a_blocked_day_it_had_stored(risk_env, caplog):
    engine, cfg = risk_env
    risk_job.run_risk(engine, cfg, F, run_id="r1")
    _dq(engine, F, "DQ07")
    status = risk_job.run_risk(engine, cfg, F, run_id="r2")
    assert status.to_dict() == {F: "blocked"}
    assert read_frame(engine, s.risk_result).empty
    assert "risk not computed (blocked): position ['DQ07'] on 2025-05-21" in caplog.text


def test_invalidate_risk_scopes_deletion_by_rule_class(risk_env):
    engine, cfg = risk_env
    risk_job.run_risk(engine, cfg, "2025-05-20", "2025-06-27")
    stored = set(read_frame(engine, s.risk_result)["trade_date"])
    cal = etl.trading_days(engine)
    i = cal.get_loc(F)
    _dq(engine, F, "DQ03")
    gone = risk_job.invalidate_risk(engine, cfg, F, F)
    assert gone.equals(cal[i : i + 21])
    left = set(read_frame(engine, s.risk_result)["trade_date"])
    assert left == stored - set(cal[i : i + 21])

    g = pd.Timestamp("2025-06-25")
    _dq(engine, g, "DQ07")
    assert risk_job.invalidate_risk(engine, cfg, g, g).tolist() == [g]
    assert set(read_frame(engine, s.risk_result)["trade_date"]) == left - {g}
    assert risk_job.invalidate_risk(engine, cfg, "2025-06-02", "2025-06-06").empty  # nothing failed there


def test_daily_skips_risk_when_window_has_an_unresolved_price_failure(risk_env, monkeypatch):
    engine, cfg = risk_env
    _dq(engine, "2025-06-20", "DQ02")
    monkeypatch.setattr(etl, "now_cst", lambda: dt.datetime(2025, 6, 30, 16, 31))
    assert jobs.daily(engine, cfg, "2025-06-30").tolist() == [pd.Timestamp("2025-06-30")]
    assert read_frame(engine, s.risk_result).empty


def test_static_rule_is_judged_by_its_latest_run_not_by_check_date(risk_env):
    engine, cfg = risk_env
    dates = _cal(engine, "2025-05-20", "2025-05-23")
    # the latest DQ08 run (filed under 2025-06-27) fails; 2025-05-20's own DQ08 row is still the backfill's pass
    _dq(engine, "2025-06-27", "DQ08")
    own = read_frame(engine, s.dq_result, "2025-05-20", "2025-05-20", date_col="check_date", rule_id="DQ08")
    assert set(own["status"]) == {"pass"}
    gate = risk_job.dq_gate(engine, cfg, dates)
    assert (gate["status"] == "blocked").all()
    assert gate.at[dates[0], "reason"] == "static DQ08 failed in its latest run later-20300101"
    # a newer DQ08 run passes, wherever it is filed: resolved for every date
    _dq(engine, "2025-04-01", "DQ08", "pass", dt.datetime(2031, 1, 1))
    assert (risk_job.dq_gate(engine, cfg, dates)["status"] == "ok").all()


def test_invalidate_risk_static_failure_deletes_every_date(risk_env):
    engine, cfg = risk_env
    risk_job.run_risk(engine, cfg, "2025-05-20", "2025-05-30")
    stored = set(read_frame(engine, s.risk_result)["trade_date"])
    _dq(engine, "2025-06-27", "DQ09")
    gone = risk_job.invalidate_risk(engine, cfg, "2025-06-27", "2025-06-27")
    assert set(gone) == stored
    assert read_frame(engine, s.risk_result).empty


@pytest.mark.parametrize(
    "scope, msg",
    [
        ({"price": ["DQ02", "DQ03", "DQ11"], "position": ["DQ07"], "static": ["DQ08"]}, "must cover exactly"),
        ({"price": ["DQ02", "DQ03", "DQ11"], "position": ["DQ07", "DQ11"], "static": ["DQ08", "DQ09"]}, "both price and position"),
        ({"price": ["DQ02", "DQ03", "DQ11", "DQ07", "DQ08", "DQ09"], "reference": []}, "unknown classes"),
    ],
)
def test_blocking_scope_must_classify_every_blocking_rule_once(risk_env, scope, msg):
    engine, cfg = risk_env
    cfg["dq"]["blocking_scope"] = scope
    with pytest.raises(ValueError, match=msg):
        risk_job.run_risk(engine, cfg, RISK_DATE)


def test_risk_job_short_calendar_raises_and_unchecked_window_is_no_dq(risk_env):
    engine, cfg = risk_env
    # DQ starts at price_start 2025-03-10; the 20-day window of 2025-04-01 begins 2025-03-04
    assert risk_job.run_risk(engine, cfg, "2025-04-01").to_dict() == {pd.Timestamp("2025-04-01"): "no_dq"}
    cfg["lookback_days"] = 60
    with pytest.raises(ValueError, match="fewer than 61 days"):
        risk_job.run_risk(engine, cfg, "2025-04-01")


def test_risk_job_logs_suspension_and_empty_es_tail(risk_env, caplog):
    engine, cfg = risk_env
    iid = read_frame(engine, s.position, RISK_DATE, RISK_DATE)["instrument_id"].iloc[0]
    # suspend one held name for the whole window: flat price, every return 0, so its HS tail is empty
    px = read_frame(engine, s.daily_price, "2025-04-22", RISK_DATE, instrument_id=iid)
    flat = px.assign(adj_close=px["adj_close"].iloc[0], close=px["close"].iloc[0], is_suspended=True)
    upsert(engine, s.daily_price, flat)
    caplog.set_level("INFO")
    risk_job.run_risk(engine, cfg, RISK_DATE)
    assert f"'{iid}': 20" in caplog.text
    assert "ES set to VaR for" in caplog.text
    assert f"'instrument:{iid}@0.95'" in caplog.text and f"'instrument:{iid}@0.99'" in caplog.text
    assert "portfolio:" not in caplog.text
    rr = read_frame(engine, s.risk_result)
    one = rr[rr["level_key"] == iid]
    assert (one["var_hs"] == 0).all() and (one["es_hs"] == 0).all() and (one["var_param"] == 0).all()


@pytest.mark.parametrize("key, value, msg", [("holding_period_days", 10, "1-day"), ("parametric_dist", "t", "normal"), ("return_type", "pct", "return_type")])
def test_risk_job_rejects_unsupported_params(risk_env, key, value, msg):
    engine, cfg = risk_env
    cfg[key] = value
    with pytest.raises(ValueError, match=msg):
        risk_job.run_risk(engine, cfg, RISK_DATE)
