"""Dashboard: pure transforms (hand-checked), chart figures (Figure.data), read-only queries on a fully computed fake
world, and every page through Streamlit's AppTest (numbers asserted on st.metric / st.dataframe, which AppTest can
read; Plotly data it cannot, so figures are checked in the charts tests)."""

import datetime as dt
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sqlalchemy import delete, update
from streamlit.testing.v1 import AppTest

from fake_world import LAST_BACKFILL, make_env
from mrisk.config import load_config
from mrisk.dashboard import charts
from mrisk.dashboard import queries as q
from mrisk.dashboard import transforms as tf
from mrisk.dashboard.common import LIMIT_LABEL, wan
from mrisk.db import schema as s
from mrisk.db.repo import get_engine, read_frame, upsert
from mrisk.engine.backtest import traffic_light, zone_bounds
from mrisk.jobs import attribution as attribution_job
from mrisk.jobs import backtest as backtest_job
from mrisk.jobs import daily as jobs
from mrisk.jobs import gate
from mrisk.jobs import limits as limits_job
from mrisk.jobs import pnl as pnl_job
from mrisk.jobs import risk as risk_job
from mrisk.jobs import stress as stress_job

APP = Path(__file__).resolve().parents[1] / "src" / "mrisk" / "dashboard" / "app.py"
PAGES = ["overview", "trend", "exposure", "backtest", "attribution", "stress", "limits", "dq"]


# ---- transforms ----------------------------------------------------------------------------------------------

def test_trailing_runs_hand_computed():
    d = pd.bdate_range("2025-06-02", periods=5)
    st = pd.DataFrame({
        "A": ["breach", "breach", "no_data", "breach", "breach"],  # no_data breaks: 2
        "B": ["warn", "ok", "warn", "warn", "warn"],  # 3
        "C": ["ok"] * 5,  # 5
        "D": ["breach", "breach", "breach", None, "breach"],  # a missing row breaks: 1
        "E": ["breach", "breach", "breach", "breach", None],  # no row on the last day: 0
    }, index=d)
    assert tf.trailing_runs(st).to_dict() == {"A": 2, "B": 3, "C": 5, "D": 1, "E": 0}


def test_segments_hand_computed():
    d = pd.bdate_range("2025-06-02", periods=7)
    flag = pd.Series([True, True, False, True, False, True, True], index=d)
    seg = tf.segments(flag, pd.Series([0.11, 0.12, 0.0, 0.105, 0.0, 0.2, np.nan], index=d))
    assert seg["days"].tolist() == [2, 1, 2]
    assert seg["start"].tolist() == [d[0], d[3], d[5]] and seg["end"].tolist() == [d[1], d[3], d[6]]
    assert seg["peak"].tolist() == [0.12, 0.105, 0.2]
    assert tf.segments(pd.Series([False, False], index=d[:2])).empty


def test_describe_scenario_pairs_beta_and_raw_on_the_same_shock():
    assert tf.describe_scenario("historical", {"base": "2015-06-12", "end": "2015-07-08"}) == ("2015-06-12 → 2015-07-08", "历史")
    assert tf.describe_scenario("hypothetical", {"market": -0.1}) == ("全市场 -10%", "β 传导")
    beta = tf.describe_scenario("hypothetical", {"industry": {"银行": -0.1}})
    raw = tf.describe_scenario("hypothetical", {"raw": {"industry": {"银行": -0.1}}})
    assert beta == ("银行 -10%", "β 传导") and raw == ("银行 -10%", "其余不变")
    assert tf.describe_scenario("hypothetical", {"raw": {"market": -0.05, "instrument": {"SH600000": -0.2}}}) == ("全市场 -5%、1 只个股", "其余不变")


def test_side_by_side_only_shocks_with_both_versions():
    port = pd.DataFrame({
        "shock": ["银行 -10%", "银行 -10%", "电子 -10%", "电子 -10%", "全市场 -10%", "2015-06-12 → 2015-07-08"],
        "mode": ["β 传导", "其余不变", "β 传导", "其余不变", "β 传导", "历史"],
        "loss_pct": [-0.0322, 0.0075, 0.039, 0.023, 0.1225, 0.3768],
    })
    out = tf.side_by_side(port)
    assert out["shock"].tolist() == ["电子 -10%", "银行 -10%"]  # worst beta loss first; market and history left out
    assert out["gap"].tolist() == pytest.approx([0.039 - 0.023, -0.0322 - 0.0075])


def test_calendar_grid():
    d = pd.DatetimeIndex(["2025-06-05", "2025-06-06", "2025-06-09"])  # Thu, Fri, next Mon
    g = tf.calendar_grid(d, [False, True, False])
    assert g["weekday"].tolist() == [3, 4, 0]
    assert g["week"].tolist() == [pd.Timestamp("2025-06-02")] * 2 + [pd.Timestamp("2025-06-09")]
    assert g["hit"].tolist() == [False, True, False]


# ---- charts ----------------------------------------------------------------------------------------------------

def test_backtest_chart_marks_exactly_the_exceptions():
    obs = pd.DataFrame({
        "pnl_date": pd.bdate_range("2025-06-02", periods=4),
        "pnl": [-5.0, -12.0, 3.0, -10.0],
        "var_hs": [10.0, 10.0, 10.0, 10.0],
    })
    fig = charts.backtest(obs, "var_hs", "t")
    bars, line, ex = fig.data
    assert list(bars.y) == [-5.0, -12.0, 3.0, -10.0]
    assert list(line.y) == [-10.0] * 4
    assert list(ex.y) == [-12.0] and ex.name == "例外（1 次）"  # -10 equals -VaR: not an exception (strict)
    assert ex.marker.color == charts.STATUS["critical"]


def test_lines_one_trace_per_series_in_slot_order():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=pd.bdate_range("2025-06-02", periods=2))
    fig = charts.lines(df, {"b": "B", "a": "A"}, "t")
    assert [t.name for t in fig.data] == ["B", "A"]
    assert [t.line.color for t in fig.data] == charts.SERIES[:2]
    assert list(fig.data[0].y) == [3.0, 4.0]
    assert len(fig.layout.to_plotly_json().get("yaxis2", {})) == 0  # one axis


def test_waterfall_steps_add_up():
    w = charts.waterfall({"x": 5.0, "y": -2.0}, "t").data[0]
    assert list(w.x) == ["x", "y", "总变化"] and list(w.y) == [5.0, -2.0, 0.0]  # the total bar is computed by plotly
    assert list(w.measure) == ["relative", "relative", "total"]
    assert list(w.text) == ["+5", "-2", "+3"]  # the total is labelled with the sum of the steps


def test_hbar_and_exception_calendar_and_limit_history():
    fig = charts.hbar(["电子", "银行"], [0.23, -0.01], "t")
    assert list(fig.data[0].y) == ["电子", "银行"] and list(fig.data[0].x) == [0.23, -0.01]
    lo, hi = fig.layout.xaxis.range
    assert lo < -0.01 and hi > 0.23  # padded for the outside labels
    g = tf.calendar_grid(pd.DatetimeIndex(["2025-06-05", "2025-06-06"]), [False, True])
    z = np.array(charts.exception_calendar(g, "t").data[0].z)
    assert z[3, 0] == 0.0 and z[4, 0] == 1.0 and np.isnan(z[0, 0])
    hist = pd.DataFrame({"trade_date": pd.bdate_range("2025-06-02", periods=3), "actual_value": [0.09, 0.11, 0.08],
                         "status": ["warn", "breach", "ok"]})
    fig = charts.limit_history(hist, 0.10, 0.08, "t")
    assert list(fig.data[1].y) == [0.11] and fig.data[1].name == "突破（1 天）"
    assert sorted(sh.y0 for sh in fig.layout.shapes) == [0.08, 0.10]


# ---- a fully computed fake world -----------------------------------------------------------------------------

@pytest.fixture(scope="session")
def built_world(tmp_path_factory):
    """Backfill and every job over the fake portfolio's dates; a short window (5) and backtest window (15) so that
    backtest results exist. -> (database file, config dict without _root)."""
    root = tmp_path_factory.mktemp("dash")
    with pytest.MonkeyPatch.context() as mp:
        engine, cfg = make_env(root, mp)
        cfg["lookback_days"] = 5
        cfg["backtest_window_days"] = 15
        assert len(jobs.backfill(engine, cfg, end=LAST_BACKFILL)) == 0
        a, b = "2025-04-01", LAST_BACKFILL
        for run in (risk_job.run_risk, pnl_job.run_pnl, backtest_job.run_backtest, attribution_job.run_attribution,
                    stress_job.run_stress, limits_job.run_limits):
            run(engine, cfg, a, b)
    engine.dispose()
    return root / "m.db", {k: v for k, v in cfg.items() if k != "_root"}


@pytest.fixture
def world(built_world, tmp_path, monkeypatch):
    """A private copy of the built world and a config file pointing at it (MRISK_CONFIG set for the app).
    -> (writable engine for setting up a case, config as the app loads it)."""
    db, cfg = built_world
    copy = tmp_path / "m.db"
    shutil.copy(db, copy)
    cfg = {**cfg, "database": {"url": f"sqlite:///{copy.as_posix()}"}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("MRISK_CONFIG", str(path))
    engine = get_engine(f"sqlite:///{copy.as_posix()}")
    yield engine, load_config(path)
    engine.dispose()


def _pid(cfg):
    return cfg["portfolio"]["portfolio_id"]


def _dq_fail(engine, day, rule="DQ03"):
    upsert(engine, s.dq_result, pd.DataFrame([{
        "run_id": "later", "check_date": pd.Timestamp(day), "rule_id": rule, "status": "fail", "n_checked": 6,
        "n_failed": 1, "detail_json": None, "created_at": dt.datetime(2030, 1, 1)}]))


def _mark_proxied(engine, d, scenario_id) -> str:
    """Store one held name of d as having taken the suspension proxy in the scenario (the fake world's only suspended
    name is never held), so the dashboard's handling of proxied names is exercised. -> that name."""
    sr = read_frame(engine, s.stress_result, d, d, scenario_id=scenario_id, level="instrument")
    r = sr.iloc[0]
    c = s.stress_result.c
    with engine.begin() as conn:
        conn.execute(update(s.stress_result).where(c.trade_date == d.date(), c.scenario_id == scenario_id, c.level == "instrument",
                                                   c.level_key == r["level_key"]).values(proxy_mv_suspended=r["market_value"]))
    return r["level_key"]


def test_readonly_engine_refuses_writes(world):
    engine, cfg = world
    ro = q.readonly_engine(cfg["database"]["url"])
    with pytest.raises(Exception, match="readonly"):
        with ro.begin() as conn:
            conn.execute(delete(s.limit_check))
    ro.dispose()


def test_complete_dates_need_every_confidence_and_every_active_rule(world):
    engine, cfg = world
    pos = q.position_dates(engine, _pid(cfg))
    full = q.complete_dates(engine, cfg)
    assert full.equals(pos)  # every job ran on every position date
    last, mid = pos[-1], pos[10]
    with engine.begin() as conn:
        conn.execute(delete(s.limit_check).where(s.limit_check.c.trade_date == last.date(), s.limit_check.c.rule_id == "L_VAR99"))
        conn.execute(delete(s.risk_result).where(s.risk_result.c.trade_date == mid.date(), s.risk_result.c.confidence == 0.95))
    assert q.complete_dates(engine, cfg).equals(pos.drop([mid, last]))
    note = q.banner(engine, cfg, q.complete_dates(engine, cfg))
    assert note["level"] == "warning" and str(last.date()) in note["text"] and str(pos[-2].date()) in note["text"]


def test_banner_when_the_latest_day_is_blocked_by_dq(world):
    engine, cfg = world
    last = q.position_dates(engine, _pid(cfg))[-1]
    assert q.banner(engine, cfg, q.complete_dates(engine, cfg)) is None
    _dq_fail(engine, last)
    gate.invalidate(engine, cfg, last, last)  # what the daily job does: risk of that day goes
    complete = q.complete_dates(engine, cfg)
    assert last not in complete
    note = q.banner(engine, cfg, complete)
    assert note["level"] == "error" and "DQ 门禁阻断" in note["text"] and "DQ03" in note["text"]


def test_limit_runs_count_consecutive_days_and_no_data_breaks(world):
    engine, cfg = world
    pid = _pid(cfg)
    pos = q.position_dates(engine, pid)
    d = pos[-1]
    today = q.limits_today(engine, pid, d).set_index("rule_id")
    # six names of ~1e7: every single-name check is a breach on every day
    assert today.at["L_CONC_INST", "status"] == "breach" and today.at["L_CONC_INST", "run_days"] == len(pos)
    assert not today.at["L_CONC_INST", "new"]
    lc = s.limit_check.c
    with engine.begin() as conn:
        conn.execute(update(s.limit_check).where(lc.rule_id == "L_CONC_INST", lc.trade_date == pos[-6].date()).values(status="no_data"))
    assert q.limits_today(engine, pid, d).set_index("rule_id").at["L_CONC_INST", "run_days"] == 5
    with engine.begin() as conn:
        conn.execute(update(s.limit_check).where(lc.rule_id == "L_CONC_INST", lc.trade_date == pos[-2].date()).values(status="ok"))
    t = q.limits_today(engine, pid, d).set_index("rule_id")
    assert t.at["L_CONC_INST", "run_days"] == 1 and t.at["L_CONC_INST", "new"]
    eps = q.limit_episodes(engine, pid, d, "breach")
    conc = eps[eps["rule_id"] == "L_CONC_INST"].sort_values("start")
    assert conc["days"].tolist() == [len(pos) - 6, 3, 1]  # split by the no_data day and the ok day


def test_exposure_and_overview_agree_with_stored_risk(world):
    engine, cfg = world
    pid, complete = _pid(cfg), q.complete_dates(engine, cfg)
    d = complete[-1]
    ind, inst = q.exposure(engine, pid, d, 0.99)
    assert ind["weight"].sum() == pytest.approx(1.0) and inst["weight"].sum() == pytest.approx(1.0)
    assert ind.set_index("industry")["market_value"].to_dict() == pytest.approx(inst.groupby("industry")["market_value"].sum().to_dict())
    ov = q.overview(engine, pid, d, 0.95, complete)
    rr = read_frame(engine, s.risk_result, d, d, level="portfolio")
    assert ov.at["today", "var_hs"] == rr.loc[rr["confidence"] == 0.95, "var_hs"].iloc[0]
    assert set(ov.index) == {"today", "prev"}


def test_backtest_query_zone_follows_the_binomial_rule_at_95(world):
    engine, cfg = world
    d = q.complete_dates(engine, cfg)[-1]
    br, obs = q.backtest(engine, cfg, d, 0.95)
    assert set(br.index) == {"hs", "param"} and br["zone"].notna().all()
    assert br["zone"].tolist() == traffic_light(br["n_exceptions"], 0.05, 15).tolist()
    assert len(obs) == 15 and obs["pnl_date"].max() == d
    assert (obs["pnl"] < -obs["var_hs"]).sum() == br.at["hs", "n_exceptions"]


def test_stress_queries_label_modes_and_flag_proxied_names(world):
    engine, cfg = world
    pid = _pid(cfg)
    sr = q.stress(engine, pid, q.complete_dates(engine, cfg)[-1])
    lab = sr.set_index("scenario_id")
    assert lab.at["IND_银行_-10", "shock"] == lab.at["IND_银行_-10_RAW", "shock"] == "银行 -10%"
    assert (lab.at["IND_银行_-10", "mode"], lab.at["IND_银行_-10_RAW", "mode"]) == ("β 传导", "其余不变")
    assert tf.side_by_side(sr)["shock"].tolist() == ["银行 -10%"]
    d = q.complete_dates(engine, cfg)[-1]
    name = _mark_proxied(engine, d, "HIST_A")
    ind, inst = q.stress_detail(engine, pid, d, "HIST_A")
    assert inst.set_index("instrument_id")["proxied"].to_dict() == {i: i == name for i in inst["instrument_id"]}
    assert ind["loss_pct"].sum() == pytest.approx(inst["loss_pct"].sum())  # proxied names still count in the totals


# ---- pages through AppTest -----------------------------------------------------------------------------------

def _app(page: str | None = None, conf: float | None = None) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    if conf is not None:
        at.sidebar.radio(key="conf").set_value(conf)
    if page is not None:
        at.switch_page(f"views/{page}.py")
    if conf is not None or page is not None:
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("conf", [0.99, 0.95])
def test_every_page_renders(world, conf):
    at = _app(conf=conf)
    for page in PAGES:
        at.switch_page(f"views/{page}.py")
        at.run()
        assert not at.exception, (page, [e.value for e in at.exception])
        assert at.header[0].value  # every page titles itself


def test_overview_metrics_and_limit_lights_match_the_database(world):
    engine, cfg = world
    complete = q.complete_dates(engine, cfg)
    d, prev = complete[-1], complete[-2]
    at = _app()
    assert at.sidebar.selectbox(key="date").value == d
    rr = read_frame(engine, s.risk_result, prev, d, level="portfolio")
    rr = rr[rr["confidence"] == 0.99].set_index("trade_date")
    m = {x.label: x for x in at.metric}
    assert m["组合市值"].value == wan(rr.at[d, "market_value"])
    assert m["VaR 历史模拟 99%"].value == wan(rr.at[d, "var_hs"])
    assert m["ES 历史模拟 99%"].delta == f"{(rr.at[d, 'es_hs'] - rr.at[prev, 'es_hs']) / 1e4:+,.1f} 万"
    lights = at.dataframe[0].value.set_index("规则")
    lc = read_frame(engine, s.limit_check, d, d).set_index("rule_id")
    assert set(lights.index) == set(lc.index)
    for rid in lc.index:
        assert lights.at[rid, "状态"] == LIMIT_LABEL[lc.at[rid, "status"]]


def test_confidence_switch_changes_the_numbers(world):
    engine, cfg = world
    d = q.complete_dates(engine, cfg)[-1]
    at = _app(conf=0.95)
    var95 = read_frame(engine, s.risk_result, d, d, level="portfolio").set_index("confidence").at[0.95, "var_hs"]
    assert {x.label: x.value for x in at.metric}["VaR 历史模拟 95%"] == wan(var95)


def test_backtest_page_at_95_shows_binomial_bounds(world):
    engine, cfg = world
    at = _app("backtest", 0.95)
    g, y = zone_bounds(0.05, 15)
    assert any(f"≤ {g} 绿，{g + 1}–{y} 黄，≥ {y + 1} 红" in c.value for c in at.caption)
    tab = at.dataframe[0].value.set_index("方法")
    br, _ = q.backtest(engine, cfg, q.complete_dates(engine, cfg)[-1], 0.95)
    assert tab.at["历史模拟", "例外"] == br.at["hs", "n_exceptions"]


def test_limits_page_separates_new_from_continuing(world):
    engine, cfg = world
    pos = q.position_dates(engine, _pid(cfg))
    lc = s.limit_check.c
    with engine.begin() as conn:
        conn.execute(update(s.limit_check).where(lc.rule_id == "L_CONC_INST", lc.trade_date == pos[-2].date()).values(status="ok"))
    at = _app("limits")
    m = {x.label: x.value for x in at.metric}
    tab = at.dataframe[0].value.set_index("规则")
    assert tab.at["L_CONC_INST", "触发"] == "今日新触发" and int(m["今日新触发"]) >= 1
    others = tab[(tab.index != "L_CONC_INST") & tab["status"].isin(["warn", "breach"])]
    assert all(v.startswith("已持续 ") for v in others["触发"])


def test_stress_page_ranks_with_mode_and_hides_proxied_names(world):
    engine, cfg = world
    name = _mark_proxied(engine, q.complete_dates(engine, cfg)[-1], "HIST_A")
    at = _app("stress")
    ranking, pair = at.dataframe[0].value, at.dataframe[1].value
    assert {"历史", "β 传导", "其余不变"} <= set(ranking["模式"])
    assert pair.columns[1] == "β 传导（基于近 250 日相关性）" and pair["冲击"].tolist() == ["银行 -10%"]
    at.selectbox(key="stress_scenario").set_value("HIST_A").run()
    names = at.dataframe[2].value
    proxied = names["说明"] != ""
    assert names.loc[proxied, "代码"].tolist() == [name]
    assert (names.loc[proxied, "损失占组合市值"] == "—").all() and (names.loc[proxied, "个股情景收益"] == "—").all()
    assert (names.loc[~proxied, "损失占组合市值"] != "—").all()


def test_blocked_latest_day_puts_a_banner_on_every_page(world):
    engine, cfg = world
    last = q.position_dates(engine, _pid(cfg))[-1]
    _dq_fail(engine, last)
    gate.invalidate(engine, cfg, last, last)
    at = _app("dq")
    assert any("DQ 门禁阻断" in e.value for e in at.error)
    assert at.sidebar.selectbox(key="date").value < last
