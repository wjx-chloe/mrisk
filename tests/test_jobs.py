import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import delete

from mrisk.config import load_config
from mrisk.db import schema as s
from mrisk.db.repo import get_engine, init_db, read_frame, upsert
from mrisk.jobs import daily as jobs
from mrisk.jobs import etl
from mrisk.jobs import gate
from mrisk.jobs import limits as limits_job
from mrisk.jobs import stress as stress_job
from mrisk.jobs import risk as risk_job
from mrisk.jobs.scheduler import build_trigger

CAL = pd.bdate_range("2025-03-03", "2025-06-30")
LAST_BACKFILL = pd.Timestamp("2025-06-27")
INDS = ["银行", "电子", "医药生物", "汽车"]
SYMS = [("SH", f"6000{i:02d}") for i in range(6)] + [("SZ", f"0000{i:02d}") for i in range(6)]
BJ = ("BJ", "920682")


def stress_job_rules_csv():
    """The repo's limit rules plus a stress-loss rule on HIST_B, for the fake portfolio. The repo's own stress-loss
    rules are dropped: their scenarios (HIST_2015, ...) are not in the fake scenario seed."""
    from mrisk.jobs.limits import SEED as REPO_RULES

    lines = REPO_RULES.read_text(encoding="utf-8").splitlines()
    body = "\n".join(x for x in lines if ",stress_loss:" not in x)
    return body + "\nL_STRESS_B,P001,stress_loss:HIST_B,portfolio,<=,0.05,pct_of_mv,0.8,true,test\n"


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
    # stress scenarios inside the fake calendar: SH600003 is suspended on 2025-05-14 only
    scen = tmp_path / "scenario.json"
    scen.write_text(json.dumps([
        {"scenario_id": "HIST_A", "type": "historical", "definition": {"base": "2025-05-12", "end": "2025-05-16"}},
        {"scenario_id": "HIST_B", "type": "historical", "definition": {"base": "2025-05-14", "end": "2025-05-20"}},
        {"scenario_id": "MKT_-5", "type": "hypothetical", "definition": {"market": -0.05}},
        {"scenario_id": "IND_银行_-10", "type": "hypothetical", "definition": {"industry": {"银行": -0.10}}},
        {"scenario_id": "IND_银行_-10_RAW", "type": "hypothetical", "definition": {"raw": {"industry": {"银行": -0.10}}}},
    ], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(stress_job, "SEED", scen)
    rules = tmp_path / "limit_rule.csv"
    rules.write_text(
        stress_job_rules_csv(), encoding="utf-8"
    )
    monkeypatch.setattr(limits_job, "SEED", rules)
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
    ph = read_frame(engine, s.pnl_hypo, date_col="position_date")
    assert set(ph["position_date"]) == {LAST_BACKFILL} and set(ph["pnl_date"]) == {pd.Timestamp("2025-06-30")}
    assert set(read_frame(engine, s.var_attribution)["trade_date"]) == {pd.Timestamp("2025-06-30")}
    assert read_frame(engine, s.backtest_result, date_col="asof_date").empty  # warming up: 1 observation


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
    gt = gate.dq_gate(engine, cfg, dates)
    assert gt.index[gt["status"] != "ok"].tolist() == [F]
    assert gt.at[F, "reason"] == "position ['DQ07'] on 2025-05-21"


@pytest.mark.parametrize("rule", ["DQ02", "DQ03", "DQ11"])
def test_dq_gate_price_rules_including_dq11_block_the_whole_window(risk_env, rule):
    engine, cfg = risk_env
    _dq(engine, F, rule)
    cal = etl.trading_days(engine)
    i = cal.get_loc(F)
    gt = gate.dq_gate(engine, cfg, _cal(engine, "2025-05-20", "2025-06-27"))
    assert gt.index[gt["status"] == "blocked"].equals(cal[i : i + 21])


def test_dq_gate_price_failure_blocks_every_window_that_holds_it(risk_env):
    engine, cfg = risk_env
    _dq(engine, F, "DQ03")
    cal = etl.trading_days(engine)
    i = cal.get_loc(F)
    dates = _cal(engine, "2025-05-20", "2025-06-27")
    gt = gate.dq_gate(engine, cfg, dates)
    # F and the next lookback (20) trading days hold F in their window; the day before and the 21st after do not
    assert gt.index[gt["status"] == "blocked"].equals(cal[i : i + 21])
    assert gt.at[cal[i - 1], "status"] == "ok" and gt.at[cal[i + 21], "status"] == "ok"
    assert "DQ03@2025-05-21" in gt.at[cal[i + 20], "reason"]
    # fixed and re-checked: the latest run for F passes, the failure is resolved
    _dq(engine, F, "DQ03", "pass", dt.datetime(2031, 1, 1))
    assert (gate.dq_gate(engine, cfg, dates)["status"] == "ok").all()


def test_dq_gate_missing_dq_in_window_and_precedence(risk_env):
    engine, cfg = risk_env
    gap = pd.Timestamp("2025-05-12")
    with engine.begin() as conn:
        conn.execute(delete(s.dq_result).where(s.dq_result.c.check_date == gap.date()))
    _dq(engine, F, "DQ07")
    cal = etl.trading_days(engine)
    i = cal.get_loc(gap)
    gt = gate.dq_gate(engine, cfg, cal[i - 1 : i + 22])
    st = gt["status"]
    assert st[cal[i - 1]] == "ok" and st[cal[i + 21]] == "ok"
    assert (st[cal[i : i + 21].drop(F)] == "no_dq").all()
    assert st[F] == "blocked"  # a known failure wins over a missing check
    assert gt.at[gap, "reason"] == "no DQ result for 1 day(s) the data rests on, first 2025-05-12"


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
    gone = gate.invalidate(engine, cfg, F, F)["risk_result"]
    assert gone.equals(cal[i : i + 21])
    left = set(read_frame(engine, s.risk_result)["trade_date"])
    assert left == stored - set(cal[i : i + 21])

    g = pd.Timestamp("2025-06-25")
    _dq(engine, g, "DQ07")
    assert gate.invalidate(engine, cfg, g, g)["risk_result"].tolist() == [g]
    assert set(read_frame(engine, s.risk_result)["trade_date"]) == left - {g}
    assert gate.invalidate(engine, cfg, "2025-06-02", "2025-06-06")["risk_result"].empty  # nothing failed there


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
    gt = gate.dq_gate(engine, cfg, dates)
    assert (gt["status"] == "blocked").all()
    assert gt.at[dates[0], "reason"] == "static DQ08 failed in its latest run later-20300101"
    # a newer DQ08 run passes, wherever it is filed: resolved for every date
    _dq(engine, "2025-04-01", "DQ08", "pass", dt.datetime(2031, 1, 1))
    assert (gate.dq_gate(engine, cfg, dates)["status"] == "ok").all()


def test_invalidate_risk_static_failure_deletes_every_date(risk_env):
    engine, cfg = risk_env
    risk_job.run_risk(engine, cfg, "2025-05-20", "2025-05-30")
    stored = set(read_frame(engine, s.risk_result)["trade_date"])
    _dq(engine, "2025-06-27", "DQ09")
    gone = gate.invalidate(engine, cfg, "2025-06-27", "2025-06-27")["risk_result"]
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
    with pytest.raises(ValueError, match="calendar starts after the data 2025-04-01 rests on"):
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


# ---- hypothetical P&L job -------------------------------------------------------------------------------------
from mrisk.jobs import pnl as pnl_job  # noqa: E402


def test_pnl_job_matches_db_recompute_and_skips_unrealised(risk_env):
    engine, cfg = risk_env
    status = pnl_job.run_pnl(engine, cfg, "2025-05-02", LAST_BACKFILL, run_id="p1")
    cal = etl.trading_days(engine)
    expect = _cal(engine, "2025-05-02", LAST_BACKFILL)[:-1]  # 06-27 has no next-day prices yet
    assert status.index.equals(expect) and (status == "ok").all()
    ph = read_frame(engine, s.pnl_hypo, date_col="position_date")
    port = ph[ph["level"] == "portfolio"].set_index("position_date")
    px = read_frame(engine, s.daily_price).set_index(["trade_date", "instrument_id"])
    pos = read_frame(engine, s.position)
    for t in expect:
        n = cal[cal.get_loc(t) + 1]
        p = pos[pos["trade_date"] == t]
        a0 = px.loc[[(t, i) for i in p["instrument_id"]]]
        a1 = px.loc[[(n, i) for i in p["instrument_id"]]]
        r = np.where(a1["is_suspended"].to_numpy(), 0.0, a1["adj_close"].to_numpy() / a0["adj_close"].to_numpy() - 1)
        mv = p["quantity"].to_numpy() * a0["close"].to_numpy()
        assert port.at[t, "pnl_date"] == n
        assert port.at[t, "market_value"] == pytest.approx(mv.sum(), rel=1e-12)
        assert port.at[t, "pnl"] == pytest.approx((mv * r).sum(), rel=1e-12, abs=1e-6)
    # BJ920682 across the 920 switch: held on 05-05, realised on 05-06 under the new code, one series
    bj = ph[(ph["level_key"] == "BJ920682") & (ph["position_date"] == pd.Timestamp("2025-05-05"))]
    assert len(bj) == 1 and bj["pnl_date"].iloc[0] == SWITCH


def test_pnl_gate_and_invalidation_follow_each_tables_dependencies(risk_env):
    engine, cfg = risk_env
    risk_job.run_risk(engine, cfg, "2025-05-20", "2025-06-26")
    pnl_job.run_pnl(engine, cfg, "2025-05-15", "2025-06-26")
    cal = etl.trading_days(engine)
    i = cal.get_loc(F)
    _dq(engine, F, "DQ03")
    # price on F feeds the P&L of F - 1 (as t + 1) and of F (as t)
    st = pnl_job.run_pnl(engine, cfg, "2025-05-20", "2025-06-26")
    assert st.index[st != "ok"].tolist() == [cal[i - 1], F]
    gone = gate.invalidate(engine, cfg, F, F)
    assert gone["pnl_hypo"].tolist() == [cal[i - 1], F]
    assert gone["risk_result"].equals(cal[i : i + 21])
    assert gone["var_attribution"].equals(cal[i : i + 22])  # windows of t - 1 and t
    assert gone["backtest_result"].equals(cal[i : i + 20 + 250 + 1].intersection(cal))
    left = set(read_frame(engine, s.pnl_hypo, date_col="position_date")["position_date"])
    assert not left & {cal[i - 1], F} and cal[i - 2] in left and cal[i + 1] in left

    _dq(engine, F, "DQ03", "pass", dt.datetime(2031, 1, 1))
    g = cal[i + 3]
    _dq(engine, g, "DQ07")
    gone = gate.invalidate(engine, cfg, g, g)
    assert gone["pnl_hypo"].tolist() == [g] and gone["risk_result"].tolist() == [g]
    assert gone["var_attribution"].tolist() == [g, cal[i + 4]]  # positions of t - 1 and t


# ---- backtest job ---------------------------------------------------------------------------------------------
from mrisk.engine.backtest import christoffersen_ind, kupiec_pof, traffic_light  # noqa: E402
from mrisk.jobs import backtest as backtest_job  # noqa: E402


@pytest.fixture
def bt_env(env):
    engine, cfg = env
    cfg["lookback_days"] = 5
    cfg["backtest_window_days"] = 15
    assert len(jobs.backfill(engine, cfg, end=LAST_BACKFILL)) == 0
    risk_job.run_risk(engine, cfg, "2025-04-01", "2025-06-26")
    pnl_job.run_pnl(engine, cfg, "2025-04-01", "2025-06-26")
    return engine, cfg


def test_backtest_job_matches_loop_recompute(bt_env):
    engine, cfg = bt_env
    status = backtest_job.run_backtest(engine, cfg, "2025-04-01", "2025-06-30", run_id="b1")
    cal = etl.trading_days(engine)
    first_ok = cal[cal.get_loc(pd.Timestamp("2025-04-01")) + 15]  # realisation date of the 15th observation
    assert (status[status.index < first_ok] == "warming_up").all() and (status[status.index >= first_ok] == "ok").all()
    br = read_frame(engine, s.backtest_result, date_col="asof_date")
    assert len(br) == (status == "ok").sum() * 4
    # every method x confidence gets a zone, bounds from (1 - confidence, window of 15)
    assert br["zone"].notna().all()
    for conf, g in br.groupby("confidence"):
        assert g["zone"].tolist() == traffic_light(g["n_exceptions"], 1 - conf, 15).tolist()

    rr = read_frame(engine, s.risk_result, level="portfolio").set_index(["trade_date", "confidence"])
    ph = read_frame(engine, s.pnl_hypo, date_col="position_date", level="portfolio").set_index("pnl_date")
    for a in [first_ok, pd.Timestamp("2025-06-02"), pd.Timestamp("2025-06-27")]:
        win = ph.loc[:a].tail(15)
        for method, col in (("hs", "var_hs"), ("param", "var_param")):
            for conf in (0.95, 0.99):
                var = np.array([rr.at[(t, conf), col] for t in win["position_date"]])
                hit = win["pnl"].to_numpy() < -var
                c = {"00": 0, "01": 0, "10": 0, "11": 0}
                for x, y in zip(hit[:-1], hit[1:]):
                    c[f"{int(x)}{int(y)}"] += 1
                row = br[(br["asof_date"] == a) & (br["method"] == method) & (br["confidence"] == conf)].iloc[0]
                assert row["n_obs"] == 15 and row["n_exceptions"] == hit.sum()
                assert row["lr_pof"] == pytest.approx(float(kupiec_pof(15, hit.sum(), 1 - conf)[0]), rel=1e-12)
                assert row["lr_ind"] == pytest.approx(float(christoffersen_ind(c["00"], c["01"], c["10"], c["11"])[0]), rel=1e-12, abs=1e-12)


def test_backtest_job_leaves_no_result_on_windows_with_a_gap(bt_env):
    engine, cfg = bt_env
    backtest_job.run_backtest(engine, cfg, "2025-04-01", "2025-06-30")
    hole = pd.Timestamp("2025-05-21")  # e.g. a day the DQ gate refused
    with engine.begin() as conn:
        conn.execute(delete(s.pnl_hypo).where(s.pnl_hypo.c.position_date == hole.date()))
    status = backtest_job.run_backtest(engine, cfg, "2025-04-01", "2025-06-30")
    cal = etl.trading_days(engine)
    i = cal.get_loc(hole)
    gap = cal[i + 1 : i + 16]  # realisation dates whose 15-observation window holds the hole
    assert status.index[status == "gap"].equals(gap)
    stored = set(read_frame(engine, s.backtest_result, date_col="asof_date")["asof_date"])
    assert not stored & set(gap) and cal[i] in stored and cal[i + 16] in stored


# ---- attribution job ------------------------------------------------------------------------------------------
from mrisk.jobs import attribution as attribution_job  # noqa: E402


def test_attribution_job_telescopes_to_stored_risk_changes(risk_env):
    engine, cfg = risk_env
    risk_job.run_risk(engine, cfg, "2025-05-19", "2025-06-27")
    status = attribution_job.run_attribution(engine, cfg, "2025-05-20", "2025-06-27", run_id="a1")
    assert (status == "ok").all() and len(status) == len(_cal(engine, "2025-05-20", "2025-06-27"))
    va = read_frame(engine, s.var_attribution)
    rr = read_frame(engine, s.risk_result, level="portfolio").set_index(["trade_date", "confidence"])
    cal = etl.trading_days(engine)
    port = va[(va["level"] == "portfolio") & (va["effect"] == "total")]
    for r in port.itertuples():
        prev = cal[cal.get_loc(r.trade_date) - 1]
        diff = rr.at[(r.trade_date, r.confidence), r.metric] - rr.at[(prev, r.confidence), r.metric]
        assert r.value == pytest.approx(diff, rel=1e-9, abs=1e-6)
    # window rows name the return date that left the window: the first return of t - 1's window
    w = va[(va["effect"] == "window") & (va["level"] == "portfolio")].drop_duplicates("trade_date").set_index("trade_date")
    for t, d in w["window_dropped"].items():
        assert d == cal[cal.get_loc(t) - 1 - cfg["lookback_days"] + 1]
    # days without a trade and without a share-count change: quantity effect exactly 0
    pos = read_frame(engine, s.position, "2025-05-19", "2025-06-27")
    q = pos.pivot(index="trade_date", columns="instrument_id", values="quantity").fillna(0)
    unchanged = q.index[1:][(q.diff().iloc[1:] == 0).all(axis=1).to_numpy()]
    assert len(unchanged) > 10
    qe = va[(va["effect"] == "quantity") & va["trade_date"].isin(unchanged)]["value"]
    assert (qe == 0.0).all()


def test_attribution_job_gate_uses_both_days(risk_env):
    engine, cfg = risk_env
    _dq(engine, F, "DQ07")  # positions of F feed the attribution of F (as t) and of the next day (as t - 1)
    status = attribution_job.run_attribution(engine, cfg, "2025-05-20", "2025-05-23")
    cal = etl.trading_days(engine)
    assert status.index[status != "ok"].tolist() == [F, cal[cal.get_loc(F) + 1]]


# ---- stress and limits jobs -----------------------------------------------------------------------------------
from mrisk.engine.returns import returns_window  # noqa: E402
from mrisk.engine.stress import betas  # noqa: E402


def test_stress_job_recomputes_from_db(risk_env):
    engine, cfg = risk_env
    t = pd.Timestamp("2025-05-20")
    status = stress_job.run_stress(engine, cfg, t, run_id="s1")
    assert status.to_dict() == {t: "ok"}
    sr = read_frame(engine, s.stress_result, t, t)
    assert set(sr["scenario_id"]) == {"HIST_A", "HIST_B", "MKT_-5", "IND_银行_-10", "IND_银行_-10_RAW"}
    # stress P&L adds up across levels
    for sid, g in sr.groupby("scenario_id"):
        sums = g.groupby("level")["pnl"].sum()
        assert sums["industry"] == pytest.approx(sums["portfolio"]) and sums["instrument"] == pytest.approx(sums["portfolio"])
    port = sr[sr["level"] == "portfolio"].set_index("scenario_id")
    pos = read_frame(engine, s.position, t, t).set_index("instrument_id")["quantity"]
    px = read_frame(engine, s.daily_price).pivot(index="trade_date", columns="instrument_id")
    mv = pos * px["close"].loc[t, pos.index]
    # HIST_A 05-12 -> 05-16: every held name trades on both ends (SH600003's 05-14 suspension resumes inside),
    # so each name's cumulative return is the real P_end / P_base - 1
    cum = px["adj_close"].loc[pd.Timestamp("2025-05-16"), pos.index] / px["adj_close"].loc[pd.Timestamp("2025-05-12"), pos.index] - 1
    assert port.at["HIST_A", "pnl"] == pytest.approx((mv * cum).sum(), rel=1e-12)
    # the price dates each historical scenario read are recorded; HIST_B's base 05-14 sits in SH600003's suspension
    spd = read_frame(engine, s.scenario_price_date, date_col="price_date").groupby("scenario_id")["price_date"]
    assert spd.min()["HIST_A"] == pd.Timestamp("2025-05-12") and spd.max()["HIST_A"] == pd.Timestamp("2025-05-16")
    assert spd.min()["HIST_B"] == pd.Timestamp("2025-05-13")  # s - 1, before the base
    # IND_银行_-10: beta to the 银行 equal-weight factor on the VaR window of t, times -10%
    inst = read_frame(engine, s.instrument, date_col="list_date").set_index("instrument_id")
    R, _ = returns_window(px["adj_close"], px["is_suspended"].astype(bool), t, cfg["lookback_days"], cfg["return_type"])
    bank = inst.index[inst["industry_l1"] == "银行"]
    b = betas(R[pos.index], R[bank].mean(axis=1))
    assert port.at["IND_银行_-10", "pnl"] == pytest.approx((mv * b * -0.10).sum(), rel=1e-9)
    # the raw twin: -10% on the bank holdings only, everything else unchanged
    held_bank = pos.index.intersection(bank)
    assert len(held_bank) and port.at["IND_银行_-10_RAW", "pnl"] == pytest.approx(-0.10 * mv[held_bank].sum(), rel=1e-12)
    assert port.at["MKT_-5", "max_daily_loss"] != port.at["MKT_-5", "max_daily_loss"]  # NaN for hypothetical


def test_stress_fixed_dates_gate_and_cascade_to_limits(risk_env):
    engine, cfg = risk_env
    days = ("2025-06-17", "2025-06-20")  # far enough from 05-19 that no relative window holds it
    risk_job.run_risk(engine, cfg, *days)
    stress_job.run_stress(engine, cfg, *days)
    limits_job.run_limits(engine, cfg, *days)
    lc = read_frame(engine, s.limit_check)
    assert (lc[lc["rule_id"] == "L_STRESS_B"]["status"] != "no_data").all()
    f = pd.Timestamp("2025-05-19")  # inside HIST_B (05-13..05-20), outside HIST_A (05-12..05-16)
    _dq(engine, f, "DQ03")
    g = gate.dq_gate(engine, cfg, _cal(engine, *days), "stress_result:historical", key="HIST_B")
    assert (g["status"] == "blocked").all() and "price on dates HIST_B rests on: ['DQ03@2025-05-19']" in g["reason"].iloc[0]
    assert (gate.dq_gate(engine, cfg, _cal(engine, *days), "stress_result:historical", key="HIST_A")["status"] == "ok").all()
    gone = gate.invalidate(engine, cfg, f, f)
    assert gone["stress_result:historical/HIST_B"].equals(_cal(engine, *days))
    sr = read_frame(engine, s.stress_result, *days)
    assert "HIST_B" not in set(sr["scenario_id"]) and {"HIST_A", "MKT_-5", "IND_银行_-10"} <= set(sr["scenario_id"])
    lc = read_frame(engine, s.limit_check).set_index(["trade_date", "rule_id"])
    stress_checks = lc.xs("L_STRESS_B", level="rule_id")
    assert (stress_checks["status"] == "no_data").all() and stress_checks["actual_value"].isna().all()
    assert stress_checks["detail"].iloc[0]["invalidated"] == "stress_result:historical/HIST_B"
    assert (lc.xs("L_VAR99", level="rule_id")["status"] != "no_data").all()  # other limits untouched
    # recomputing after the gate refuses HIST_B: a no_data row with the gate's reason, never a missing row
    stress_job.run_stress(engine, cfg, *days)
    limits_job.run_limits(engine, cfg, *days)
    lc = read_frame(engine, s.limit_check)
    b = lc[lc["rule_id"] == "L_STRESS_B"]
    assert len(b) == 4 and (b["status"] == "no_data").all()
    assert b["detail"].iloc[0]["missing"] == "stress_result" and b["detail"].iloc[0]["reason"].startswith("blocked: price on dates HIST_B")


def test_raw_scenario_rests_on_prices_of_t_only(risk_env):
    """A bad price inside t's VaR window refuses the beta-propagated scenario but not its raw twin."""
    engine, cfg = risk_env
    cal = etl.trading_days(engine)
    days = _cal(engine, "2025-06-02", "2025-06-06")
    t = days[2]
    _dq(engine, cal[cal.get_loc(t) - 3], "DQ03")
    stress_job.run_stress(engine, cfg, days[0], days[-1])
    assert gate.dq_gate(engine, cfg, pd.DatetimeIndex([t]), "stress_result:hypothetical", key="IND_银行_-10").at[t, "status"] == "blocked"
    assert gate.dq_gate(engine, cfg, pd.DatetimeIndex([t]), "stress_result:raw", key="IND_银行_-10_RAW").at[t, "status"] == "ok"
    on_t = set(read_frame(engine, s.stress_result, t, t)["scenario_id"])
    assert "IND_银行_-10_RAW" in on_t and "IND_银行_-10" not in on_t and "MKT_-5" not in on_t
    # invalidation follows the same declaration: the raw rows of t survive, the factor rows of t are gone
    stress_job.run_stress(engine, cfg, days[0], days[-1])
    f = cal[cal.get_loc(t) - 3]
    gone = gate.invalidate(engine, cfg, f, f)
    assert gone["stress_result:raw/IND_银行_-10_RAW"].equals(pd.DatetimeIndex([f]))  # only the failure day itself
    assert t in gone["stress_result:hypothetical/IND_银行_-10"]
    assert "IND_银行_-10_RAW" in set(read_frame(engine, s.stress_result, t, t)["scenario_id"])


def test_invalidate_keyed_entry_without_keys_deletes_nothing(risk_env, tmp_path, monkeypatch):
    """With no hypothetical scenarios at all, a window failure must not wipe the historical rows of the days whose
    window holds it (a keyed entry with no keys used to fall back to an unkeyed delete of every scenario)."""
    engine, cfg = risk_env
    scen = tmp_path / "hist_only.json"
    scen.write_text(json.dumps([{"scenario_id": "HIST_A", "type": "historical", "definition": {"base": "2025-05-12", "end": "2025-05-16"}}]), encoding="utf-8")
    monkeypatch.setattr(stress_job, "SEED", scen)
    cal = etl.trading_days(engine)
    days = _cal(engine, "2025-06-02", "2025-06-06")
    stress_job.run_stress(engine, cfg, days[0], days[-1])
    f = cal[cal.get_loc(days[2]) - 3]  # in the VaR window of days[2], not a date HIST_A rests on
    _dq(engine, f, "DQ03")
    gate.invalidate(engine, cfg, f, f)
    assert set(read_frame(engine, s.stress_result, days[0], days[-1])["trade_date"]) == set(days)


def test_limits_job_writes_every_rule_every_day_with_no_data_where_inputs_are_refused(risk_env):
    engine, cfg = risk_env
    cal = etl.trading_days(engine)
    days = _cal(engine, "2025-06-02", "2025-06-06")
    t = days[2]
    _dq(engine, cal[cal.get_loc(t) - 3], "DQ03")  # a bad price inside t's VaR window, not on t itself
    risk_job.run_risk(engine, cfg, days[0], days[-1])
    stress_job.run_stress(engine, cfg, days[0], days[-1])
    limits_job.run_limits(engine, cfg, days[0], days[-1])
    lc = read_frame(engine, s.limit_check)
    rules = set(read_frame(engine, s.limit_rule, date_col="rule_id")["rule_id"])
    assert len(lc) == len(days) * len(rules)  # one row per rule per day, no gaps
    at = lc[lc["trade_date"] == t].set_index("rule_id")
    # VaR / ES rest on the window: no_data with the gate's reason; concentration rests on t only: evaluated
    assert at.at["L_VAR99", "status"] == "no_data" and at.at["L_VAR99", "detail"]["missing"] == "risk_result"
    assert "DQ03@" in at.at["L_VAR99", "detail"]["reason"]
    assert at.at["L_CONC_INST", "status"] in ("ok", "warn", "breach") and at.at["L_CONC_INST", "actual_value"] > 0
    # 6 names of ~1e7: single-name weights well above 10% -> breach, the offenders listed in detail
    conc = lc[lc["rule_id"] == "L_CONC_INST"]
    assert (conc["status"] == "breach").all()
    assert all(m["value"] > 0.08 for m in conc["detail"].iloc[0]["members"])
