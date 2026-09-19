"""The 12 first-batch data-quality rules (SPEC §4.3). Each is a pure, vectorised function of DQContext."""

import numpy as np
import pandas as pd

from mrisk.data.dq.base import DQContext, RuleOutput, RuleSpec, broadcast_static, per_date_count

CHINEXT_20PCT_FROM = pd.Timestamp("2020-08-24")


def _on_check_dates(df: pd.DataFrame, ctx: DQContext, col: str = "trade_date") -> pd.DataFrame:
    return df[df[col].isin(ctx.check_dates)].rename(columns={col: "check_date"})


def _with_prev(prices: pd.DataFrame) -> pd.DataFrame:
    px = prices.sort_values(["instrument_id", "trade_date"]).copy()
    g = px.groupby("instrument_id")
    px["prev_close"] = g["close"].shift()
    px["prev_adj_close"] = g["adj_close"].shift()
    return px


def dq01_price_complete(ctx: DQContext) -> RuleOutput:
    inst = ctx.instruments.loc[ctx.instruments["status"] == "active", ["instrument_id", "list_date"]]
    grid = pd.DataFrame({"check_date": ctx.check_dates}).merge(inst, how="cross")
    grid = grid[grid["list_date"].isna() | (grid["check_date"] >= grid["list_date"])]
    have = ctx.prices[["trade_date", "instrument_id"]].rename(columns={"trade_date": "check_date"})
    m = grid.merge(have, on=["check_date", "instrument_id"], how="left", indicator=True)
    fails = m.loc[m["_merge"] == "left_only", ["check_date", "instrument_id"]]
    return RuleOutput(per_date_count(grid["check_date"], ctx.check_dates), fails)


def dq02_position_has_price(ctx: DQContext) -> RuleOutput:
    pos = _on_check_dates(ctx.positions, ctx)[["check_date", "portfolio_id", "instrument_id"]]
    have = ctx.prices[["trade_date", "instrument_id"]].rename(columns={"trade_date": "check_date"})
    m = pos.merge(have, on=["check_date", "instrument_id"], how="left", indicator=True)
    fails = m.loc[m["_merge"] == "left_only", ["check_date", "portfolio_id", "instrument_id"]]
    return RuleOutput(per_date_count(pos["check_date"], ctx.check_dates), fails)


def dq03_close_positive(ctx: DQContext) -> RuleOutput:
    px = _on_check_dates(ctx.prices, ctx)
    ok = (px["close"] > 0) & (px["adj_close"] > 0)
    fails = px.loc[~ok, ["check_date", "instrument_id", "close", "adj_close"]]
    return RuleOutput(per_date_count(px["check_date"], ctx.check_dates), fails)


def price_limit(board: pd.Series, is_st: pd.Series, trade_date: pd.Series) -> np.ndarray:
    return np.select(
        [board == "bj", board == "star", (board == "chinext") & (trade_date >= CHINEXT_20PCT_FROM), board == "chinext", is_st],
        [0.30, 0.20, 0.20, 0.10, 0.05],
        default=0.10,
    )


def dq04_within_price_limit(ctx: DQContext) -> RuleOutput:
    p = ctx.params
    px = _with_prev(ctx.prices)
    px = px.merge(ctx.instruments[["instrument_id", "board", "name", "list_date"]], on="instrument_id", how="left")
    cal = np.sort(pd.to_datetime(ctx.calendar).to_numpy())
    list_idx = np.searchsorted(cal, px["list_date"].to_numpy())
    list_idx = np.where(px["list_date"].isna() | (px["list_date"] < cal[0]), -(10**9), list_idx)
    days_listed = np.searchsorted(cal, px["trade_date"].to_numpy()) - list_idx
    px = px[
        px["trade_date"].isin(ctx.check_dates)
        & px["prev_close"].notna()
        & ~px["is_suspended"]
        & (days_listed >= p["new_listing_exempt_days"])
    ].copy()
    limit = price_limit(px["board"], px["name"].str.contains("ST", na=False), px["trade_date"])
    # reference price on ex-dates: previous close restated in today's adjustment basis
    factor = px["adj_close"] / px["close"]
    ref = px["prev_adj_close"] / factor
    # on ex-dates the factor-implied reference only approximates the exchange's ex-rights price (rights issues)
    is_ex = (factor / (px["prev_adj_close"] / px["prev_close"]) - 1).abs() > 1e-9
    pct = np.where(is_ex, p["price_limit_tolerance_pct_exdate"], p["price_limit_tolerance_pct"])
    tol = np.maximum(pct * ref, p["price_limit_tolerance_abs"])
    ok = (px["close"] <= ref * (1 + limit) + tol) & (px["close"] >= ref * (1 - limit) - tol)
    px["ret"] = px["adj_close"] / px["prev_adj_close"] - 1
    px["limit"] = limit
    fails = px.loc[~ok, ["trade_date", "instrument_id", "board", "prev_close", "close", "ret", "limit"]]
    return RuleOutput(per_date_count(px["trade_date"], ctx.check_dates), fails.rename(columns={"trade_date": "check_date"}))


def dq05_adj_factor_monotone(ctx: DQContext) -> RuleOutput:
    px = _with_prev(ctx.prices)
    px = px[px["trade_date"].isin(ctx.check_dates) & px["prev_close"].notna()].copy()
    px["factor"] = px["adj_close"] / px["close"]
    px["prev_factor"] = px["prev_adj_close"] / px["prev_close"]
    ok = px["factor"] >= px["prev_factor"] * (1 - 1e-9)
    fails = px.loc[~ok, ["trade_date", "instrument_id", "prev_factor", "factor"]]
    return RuleOutput(per_date_count(px["trade_date"], ctx.check_dates), fails.rename(columns={"trade_date": "check_date"}))


def dq06_lot_size(ctx: DQContext) -> RuleOutput:
    pos = _on_check_dates(ctx.positions, ctx).merge(ctx.instruments[["instrument_id", "exchange"]], on="instrument_id", how="left")
    q = pos["quantity"]
    is_bj = pos["exchange"] == "BJ"
    ok = np.where(is_bj, (q >= 100) & (q == np.round(q)), (q > 0) & (q % 100 == 0))
    fails = pos.loc[~ok, ["check_date", "portfolio_id", "instrument_id", "exchange", "quantity"]]
    return RuleOutput(per_date_count(pos["check_date"], ctx.check_dates), fails)


def dq07_no_delisted_holdings(ctx: DQContext) -> RuleOutput:
    pos = _on_check_dates(ctx.positions, ctx).merge(ctx.instruments[["instrument_id", "status"]], on="instrument_id", how="left")
    fails = pos.loc[pos["status"] == "delisted", ["check_date", "portfolio_id", "instrument_id"]]
    return RuleOutput(per_date_count(pos["check_date"], ctx.check_dates), fails)


def dq08_symbol_unique(ctx: DQContext) -> RuleOutput:
    inst = ctx.instruments
    dup = inst[inst.duplicated(["exchange", "symbol"], keep=False)][["instrument_id", "exchange", "symbol"]]
    return broadcast_static(len(inst), dup, ctx.check_dates)


def dq09_code_map_disjoint(ctx: DQContext) -> RuleOutput:
    cm = ctx.code_map.merge(ctx.instruments[["instrument_id", "exchange"]], on="instrument_id", how="left")
    new_keys = set(zip(cm["exchange"], cm["new_symbol"]))
    current_keys = set(zip(ctx.instruments["exchange"], ctx.instruments["symbol"]))
    old_keys = list(zip(cm["exchange"], cm["old_symbol"]))
    bad = np.array([k in new_keys or k in current_keys for k in old_keys], dtype=bool)
    fails = cm.loc[bad, ["instrument_id", "exchange", "old_symbol", "new_symbol"]]
    return broadcast_static(len(cm), fails, ctx.check_dates)


def dq10_arrival_time(ctx: DQContext) -> RuleOutput:
    px = _on_check_dates(ctx.prices, ctx)
    px = px[px["load_type"] == "daily"]
    deadline = px["check_date"] + pd.Timedelta(ctx.params["arrival_deadline"] + ":00")
    fails = px.loc[px["ingested_at"] > deadline, ["check_date", "instrument_id", "ingested_at"]]
    return RuleOutput(per_date_count(px["check_date"], ctx.check_dates), fails)


def dq11_mv_change(ctx: DQContext) -> RuleOutput:
    m = ctx.positions.merge(ctx.prices[["trade_date", "instrument_id", "close"]], on=["trade_date", "instrument_id"], how="inner")
    mv = (m["quantity"] * m["close"]).groupby([m["portfolio_id"], m["trade_date"]]).sum().rename("mv").reset_index()
    mv = mv.sort_values(["portfolio_id", "trade_date"])
    mv["prev_mv"] = mv.groupby("portfolio_id")["mv"].shift()
    mv = mv[mv["trade_date"].isin(ctx.check_dates) & mv["prev_mv"].notna()].copy()
    mv["change"] = mv["mv"] / mv["prev_mv"] - 1
    fails = mv.loc[mv["change"].abs() >= ctx.params["mv_change_max"], ["trade_date", "portfolio_id", "prev_mv", "mv", "change"]]
    return RuleOutput(per_date_count(mv["trade_date"], ctx.check_dates), fails.rename(columns={"trade_date": "check_date"}))


def dq12_row_count_change(ctx: DQContext) -> RuleOutput:
    cal = pd.DatetimeIndex(np.sort(pd.to_datetime(ctx.calendar).to_numpy()))
    counts = ctx.prices["trade_date"].value_counts()
    if counts.empty:
        return RuleOutput(per_date_count(pd.Series([], dtype="datetime64[ns]"), ctx.check_dates), pd.DataFrame(columns=["check_date"]))
    cal = cal[(cal >= counts.index.min()) & (cal <= counts.index.max())]
    c = pd.DataFrame({"check_date": cal, "n": counts.reindex(cal, fill_value=0).to_numpy()})
    c["prev_n"] = c["n"].shift()
    c = c[c["check_date"].isin(ctx.check_dates) & (c["prev_n"] > 0)].copy()
    c["change"] = c["n"] / c["prev_n"] - 1
    fails = c.loc[c["change"].abs() >= ctx.params["row_count_change_max"], ["check_date", "prev_n", "n", "change"]]
    return RuleOutput(per_date_count(c["check_date"], ctx.check_dates), fails)


RULES: list[RuleSpec] = [
    RuleSpec("DQ01", "交易日行情不缺失", "完整性", "每只在市股票在每个交易日（上市后）都有行情行（含停牌补行）", dq01_price_complete),
    RuleSpec("DQ02", "持仓股票当日有行情或停牌标记", "完整性", "持仓中每只股票当日在 daily_price 中有记录", dq02_position_has_price),
    RuleSpec("DQ03", "收盘价大于 0", "准确性", "close > 0 且 adj_close > 0", dq03_close_positive),
    RuleSpec(
        "DQ04", "涨跌幅在涨跌停限制内", "准确性",
        "主板±10%、创业板±20%(2020-08-24前±10%)、科创板±20%、北交所±30%、主板ST±5%；上市前5日豁免；容差 max(0.1%, 0.01元)，除权日 max(0.5%, 0.01元)",
        dq04_within_price_limit,
    ),
    RuleSpec("DQ05", "复权因子单调", "准确性", "后复权因子 adj_close/close 随时间单调不减", dq05_adj_factor_monotone),
    RuleSpec("DQ06", "持仓为整手", "准确性", "沪深为100股整数倍；北交所≥100股的整数", dq06_lot_size),
    RuleSpec("DQ07", "持仓不含已退市股票", "一致性", "持仓股票在 instrument 中 status 不为 delisted", dq07_no_delisted_holdings),
    RuleSpec("DQ08", "symbol 交易所内唯一", "一致性", "instrument 中 (exchange, symbol) 不重复", dq08_symbol_unique),
    RuleSpec("DQ09", "代码映射新旧不重叠", "一致性", "旧代码不出现在新代码集合中，也不是任何现行代码", dq09_code_map_disjoint),
    RuleSpec("DQ10", "行情 17:00 前到达", "及时性", "仅评估每日增量(load_type=daily)；回灌数据记为 skip", dq10_arrival_time),
    RuleSpec("DQ11", "组合市值日环比 < 30%", "合理性", "各组合 sum(quantity×close) 与上一持仓日相比变化 < 30%", dq11_mv_change),
    RuleSpec("DQ12", "行情条数日环比 < 5%", "合理性", "daily_price 当日行数与上一交易日相比变化 < 5%", dq12_row_count_change),
]
