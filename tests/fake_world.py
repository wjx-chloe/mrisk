"""The fake world the job and dashboard tests run on: 12 stocks in 4 industries plus BJ920682 across its code
switch, ~80 trading days, SH600003 suspended on 2025-05-14, SH600004 across price_start."""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

from mrisk.config import load_config
from mrisk.db.repo import get_engine, init_db
from mrisk.jobs import etl
from mrisk.jobs import limits as limits_job
from mrisk.jobs import stress as stress_job

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


def fake_akshare(src: FakeSource) -> types.SimpleNamespace:
    """The akshare functions AkshareSource calls, serving src's world in akshare's raw frame layout (Chinese column
    names, string dates), so AkshareSource's own caching and normalisation run on it."""
    lst = src.listings()
    def listing(ex, code, name, date):
        x = lst[lst["exchange"] == ex]
        return pd.DataFrame({code: x["symbol"], name: x["name"], date: x["list_date"].dt.strftime("%Y-%m-%d")})
    sw = src.sw_l1_members()
    codes = {ind: f"8010{i:02d}" for i, ind in enumerate(sorted(sw["industry_l1"].unique()))}
    def components(symbol):
        x = sw[sw["industry_l1"] == {c: i for i, c in codes.items()}[symbol]]
        return pd.DataFrame({"证券代码": x["symbol"], "最新权重": x["sw_weight"]})
    def daily(symbol, start_date=None, end_date=None, adjust=""):
        ex, sym = symbol[:2].upper(), symbol[2:]
        if adjust == "hfq-factor":
            f = src.hfq_factor(ex, sym)
            return pd.DataFrame({"date": f["effective_date"].dt.strftime("%Y-%m-%d"), "hfq_factor": f["hfq_factor"].astype(str)})
        d = src.daily_raw(ex, sym, pd.Timestamp(start_date), pd.Timestamp(end_date))
        return pd.DataFrame({"date": d["trade_date"].dt.strftime("%Y-%m-%d"), "open": d["close"], "close": d["close"],
                             "volume": d["volume"].astype(float)})
    return types.SimpleNamespace(
        tool_trade_date_hist_sina=lambda: pd.DataFrame({"trade_date": CAL.strftime("%Y-%m-%d")}),
        stock_info_sh_name_code=lambda symbol: listing("SH", "证券代码", "证券简称", "上市日期").iloc[: 0 if symbol == "科创板" else None],
        stock_info_sz_name_code=lambda symbol: listing("SZ", "A股代码", "A股简称", "A股上市日期"),
        stock_info_bj_name_code=lambda: listing("BJ", "证券代码", "证券简称", "上市日期"),
        stock_info_sh_delist=lambda symbol: pd.DataFrame({"公司代码": src.delisted().query("exchange == 'SH'")["symbol"]}),
        stock_info_sz_delist=lambda symbol: pd.DataFrame({"证券代码": src.delisted().query("exchange == 'SZ'")["symbol"]}),
        sw_index_first_info=lambda: pd.DataFrame({"行业代码": [f"{c}.SI" for c in codes.values()], "行业名称": list(codes)}),
        index_component_sw=components,
        stock_zh_a_daily=daily,
    )


def make_env(tmp_path: Path, monkeypatch, source: str = "fake") -> tuple:
    """Config + empty database for the fake world under tmp_path; seeds, scenarios, rules and the source patched in
    with monkeypatch (a pytest MonkeyPatch). The calendar holds ~80 trading days. source="fake" patches FakeSource in
    as the source; source="akshare" keeps the real AkshareSource (cache in tmp_path/raw) over fake_akshare."""
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
    if source == "fake":
        monkeypatch.setattr(etl, "make_source", lambda cfg: src)
    else:
        monkeypatch.setitem(sys.modules, "akshare", fake_akshare(src))
        cfg["data"].update(source="akshare", raw_cache_dir="raw", request_interval_sec=0)
        src = etl.make_source(cfg)
    engine = get_engine(f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    init_db(engine)
    etl.build_universe_seed(cfg, src)
    return engine, cfg
