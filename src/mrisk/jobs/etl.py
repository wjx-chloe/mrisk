"""ETL jobs: pull -> validate -> write. The only layer that combines the network, files and the db."""

import datetime as dt
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine

from mrisk.config import resolve_path
from mrisk.data import mock_positions as mock
from mrisk.data.code_map import resolve_instrument_id
from mrisk.data.dq import RULES, DQContext, blocked_dates, run_rules
from mrisk.data.sources import AkshareSource
from mrisk.data.transform import build_daily_price
from mrisk.data.universe import build_instruments, select_universe
from mrisk.db import schema as s
from mrisk.db.repo import read_frame, replace_partition, upsert

log = logging.getLogger(__name__)

SEED_DIR = Path(__file__).resolve().parents[1] / "data" / "seed"
CST = dt.timezone(dt.timedelta(hours=8))
PRICE_CONTEXT_DAYS = 15


def now_cst() -> dt.datetime:
    return dt.datetime.now(CST).replace(tzinfo=None, microsecond=0)


def make_source(cfg) -> AkshareSource:
    d = cfg["data"]
    return AkshareSource(
        resolve_path(cfg, d["raw_cache_dir"]),
        request_interval_sec=d["request_interval_sec"],
        max_retries=d["max_retries"],
        bypass_proxy=d["bypass_proxy"],
    )


def load_seed(name: str) -> pd.DataFrame:
    df = pd.read_csv(SEED_DIR / name, dtype=str)
    if "effective_date" in df:
        df["effective_date"] = pd.to_datetime(df["effective_date"])
    return df


def trading_days(engine: Engine, start=None, end=None) -> pd.DatetimeIndex:
    cal = read_frame(engine, s.trade_calendar, start, end)
    return pd.DatetimeIndex(cal["trade_date"]).sort_values()


def build_universe_seed(cfg, src: AkshareSource) -> pd.DataFrame:
    u = cfg["universe"]
    uni = select_universe(
        src.listings(), src.sw_l1_members(), u["industries"], u["per_industry"],
        pd.Timestamp(u["listed_before"]), u["must_include"],
    )
    uni.to_csv(SEED_DIR / "universe.csv", index=False)
    return uni


def fetch_trade_calendar(engine: Engine, src: AkshareSource) -> int:
    return upsert(engine, s.trade_calendar, src.trade_calendar())


def fetch_instruments(engine: Engine, src: AkshareSource) -> int:
    inst = build_instruments(
        load_seed("universe.csv"), src.listings(), src.sw_l1_members(), src.delisted(),
        load_seed("industry_override.csv")[["instrument_id", "industry_l1"]],
    )
    return upsert(engine, s.instrument, inst)


def apply_code_map(engine: Engine) -> int:
    return upsert(engine, s.instrument_code_map, load_seed("code_map.csv"))


def fetch_prices(engine: Engine, src: AkshareSource, start, end, load_type: str) -> int:
    """Writes daily_price rows in [start, end]. A few earlier trading days are pulled as context so
    suspensions at the start of the window can be carried forward, but are not rewritten."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cal = trading_days(engine)
    before = cal[cal < start]
    ctx_start = before[-PRICE_CONTEXT_DAYS] if len(before) >= PRICE_CONTEXT_DAYS else start
    inst = read_frame(engine, s.instrument, date_col="list_date")
    daily, factors = [], []
    for i, row in enumerate(inst.itertuples(), 1):
        log.info("[%d/%d] %s", i, len(inst), row.instrument_id)
        d = src.daily_raw(row.exchange, row.symbol, ctx_start, end)
        daily.append(d.assign(instrument_id=row.instrument_id))
        factors.append(src.hfq_factor(row.exchange, row.symbol).assign(instrument_id=row.instrument_id))
    px = build_daily_price(
        pd.concat(daily, ignore_index=True), pd.concat(factors, ignore_index=True),
        inst[["instrument_id", "list_date"]], pd.Series(cal[(cal >= ctx_start) & (cal <= end)]),
    )
    px = px[(px["trade_date"] >= start) & (px["trade_date"] <= end)].assign(ingested_at=now_cst(), load_type=load_type)
    return upsert(engine, s.daily_price, px)


def _mock_params(cfg) -> mock.MockParams:
    p = cfg["portfolio"]
    return mock.MockParams(
        portfolio_id=p["portfolio_id"], start=pd.Timestamp(p["start"]), initial_capital=p["initial_capital"],
        n_holdings=p["n_holdings"], min_industries=p["min_industries"],
        rebalance_weight_change=p["rebalance_weight_change"], swap_probability=p["swap_probability"], seed=p["seed"],
    )


def feed_path(cfg) -> Path:
    return resolve_path(cfg, cfg["data"]["position_feed_dir"]) / f"positions_{cfg['portfolio']['portfolio_id']}.csv"


def generate_positions(engine: Engine, cfg, end=None) -> Path:
    """Mock upstream system: writes a position file keyed by exchange + the symbol valid on each date."""
    params = _mock_params(cfg)
    inst = read_frame(engine, s.instrument, date_col="list_date")
    cm = read_frame(engine, s.instrument_code_map, date_col="effective_date")
    prices = read_frame(engine, s.daily_price, params.start, end, columns=["trade_date", "instrument_id", "close", "adj_close", "is_suspended"])
    pool = inst.loc[inst["status"] == "active", ["instrument_id", "industry_l1"]]
    pos = mock.generate_positions(
        pool, prices, pd.Series(trading_days(engine, params.start, end)), params, must_include=cfg["universe"]["must_include"]
    )
    feed = mock.render_upstream_feed(pos, inst, cm)
    path = feed_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    feed.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


def load_positions(engine: Engine, cfg, path: Path | None = None) -> int:
    """Ingest an upstream position file, resolving old symbols through instrument_code_map."""
    feed = pd.read_csv(path or feed_path(cfg), dtype={"symbol": str}, parse_dates=["trade_date"])
    inst = read_frame(engine, s.instrument, date_col="list_date")
    cm = read_frame(engine, s.instrument_code_map, date_col="effective_date")
    pos = resolve_instrument_id(feed, inst, cm)[["trade_date", "portfolio_id", "instrument_id", "quantity"]]
    p = cfg["portfolio"]
    upsert(engine, s.portfolio, pd.DataFrame({"portfolio_id": [p["portfolio_id"]], "name": [p["name"]], "base_currency": ["CNY"]}))
    return replace_partition(engine, s.position, pos, portfolio_id=p["portfolio_id"])


def run_dq(engine: Engine, cfg, start, end, run_id: str | None = None) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Evaluates all rules for each trading day in [start, end]; writes dq_rule and dq_result."""
    cal = trading_days(engine)
    check_dates = cal[(cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end))]
    before = cal[cal < pd.Timestamp(start)]
    ctx_start = before[-PRICE_CONTEXT_DAYS] if len(before) >= PRICE_CONTEXT_DAYS else pd.Timestamp(start)
    ctx = DQContext(
        check_dates=check_dates,
        calendar=pd.Series(cal),
        instruments=read_frame(engine, s.instrument, date_col="list_date"),
        code_map=read_frame(engine, s.instrument_code_map, date_col="effective_date"),
        prices=read_frame(engine, s.daily_price, ctx_start, end),
        positions=read_frame(engine, s.position, ctx_start, end),
        params=cfg["dq"],
    )
    blocking = set(cfg["dq"]["blocking"])
    res = run_rules(ctx, RULES, blocking)
    upsert(
        engine, s.dq_rule,
        pd.DataFrame(
            [{"rule_id": r.rule_id, "name": r.name, "category": r.category, "description": r.description,
              "severity": "blocking" if r.rule_id in blocking else "warn", "is_active": True} for r in RULES]
        ),
    )
    run_id = run_id or f"dq-{now_cst():%Y%m%dT%H%M%S}"
    out = res.drop(columns="severity").assign(run_id=run_id, created_at=now_cst())
    upsert(engine, s.dq_result, out)
    return res, blocked_dates(res)
