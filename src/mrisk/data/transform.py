"""Pure transforms from normalised source frames to daily_price rows."""

import numpy as np
import pandas as pd

BOARDS = ("main", "chinext", "star", "bj")


def infer_board(exchange: pd.Series, symbol: pd.Series) -> pd.Series:
    sym = symbol.astype(str)
    board = np.select(
        [
            exchange == "BJ",
            (exchange == "SH") & sym.str.startswith("688"),
            (exchange == "SZ") & sym.str.startswith(("300", "301")),
        ],
        ["bj", "star", "chinext"],
        default="main",
    )
    return pd.Series(board, index=exchange.index)


def instrument_id(exchange: pd.Series, symbol: pd.Series) -> pd.Series:
    return exchange.astype(str) + symbol.astype(str)


def apply_hfq(daily: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """daily[instrument_id, trade_date, close, ...] + factors[instrument_id, effective_date, hfq_factor]
    -> adds adj_close = close * factor in force on trade_date."""
    d = daily.assign(trade_date=daily["trade_date"].astype("datetime64[ns]")).sort_values("trade_date")
    f = factors.assign(effective_date=factors["effective_date"].astype("datetime64[ns]")).sort_values("effective_date")
    out = pd.merge_asof(
        d, f, left_on="trade_date", right_on="effective_date", by="instrument_id", direction="backward"
    )
    if out["hfq_factor"].isna().any():
        bad = out.loc[out["hfq_factor"].isna(), "instrument_id"].unique().tolist()
        raise ValueError(f"no hfq factor in force for {bad}")
    out["adj_close"] = out["close"] * out["hfq_factor"]
    return out.drop(columns=["effective_date", "hfq_factor"]).sort_values(["instrument_id", "trade_date"], ignore_index=True)


def drop_before_listing(daily: pd.DataFrame, list_dates: pd.DataFrame) -> pd.DataFrame:
    """Beijing names carry pre-listing NEEQ trades in the Sina history; keep only rows on/after list_date."""
    m = daily.merge(list_dates[["instrument_id", "list_date"]], on="instrument_id", how="left")
    keep = m["list_date"].isna() | (m["trade_date"] >= m["list_date"])
    return daily.loc[keep.to_numpy()].reset_index(drop=True)


def fill_suspended(daily: pd.DataFrame, calendar: pd.Series) -> pd.DataFrame:
    """Insert a row for every trading day strictly between an instrument's first and last observed
    dates that has no quote: close/adj_close carried forward, volume 0, is_suspended True.
    Gaps before the first or after the last observation are left for DQ01 to report."""
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values()
    bounds = daily.groupby("instrument_id")["trade_date"].agg(["min", "max"])
    lo = np.searchsorted(cal.values, bounds["min"].values, side="left")
    hi = np.searchsorted(cal.values, bounds["max"].values, side="right")
    counts = hi - lo
    idx = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)]) if len(counts) else np.array([], dtype=int)
    grid = pd.DataFrame({"instrument_id": np.repeat(bounds.index.to_numpy(), counts), "trade_date": cal.values[idx]})
    out = grid.merge(daily, on=["instrument_id", "trade_date"], how="left", indicator=True)
    missing = (out["_merge"] == "left_only").to_numpy()
    out = out.drop(columns="_merge")
    out[["close", "adj_close"]] = out.groupby("instrument_id")[["close", "adj_close"]].ffill()
    out["volume"] = out["volume"].fillna(0).astype("int64")
    observed_zero = ~missing & (out["volume"].to_numpy() == 0)
    out["is_suspended"] = missing | observed_zero
    return out


def build_daily_price(
    daily: pd.DataFrame, factors: pd.DataFrame, list_dates: pd.DataFrame, calendar: pd.Series
) -> pd.DataFrame:
    """daily[instrument_id, trade_date, close, volume] -> daily_price columns (without ingest metadata)."""
    px = drop_before_listing(daily, list_dates)
    px = apply_hfq(px, factors)
    px = fill_suspended(px, calendar)
    return px[["trade_date", "instrument_id", "close", "adj_close", "volume", "is_suspended"]]
