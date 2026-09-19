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
    Gaps before the first observation are left for DQ01; trailing gaps are handled by
    infer_trailing_suspensions."""
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


PRICE_COLS = ["trade_date", "instrument_id", "close", "adj_close", "volume", "is_suspended"]


def infer_trailing_suspensions(
    px: pd.DataFrame, observed: pd.DataFrame, calendar: pd.Series, end: pd.Timestamp,
    active_ids: set[str], min_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sina returns no row for a suspended day, so a series that stops before `end` is read as a suspension:
    append carried-forward suspended rows for each following trading day, but only while that day's quote
    coverage (instruments with a real quote / instruments already started) is >= min_coverage, stopping at
    the first day below it, so an unpublished or failed pull is never mistaken for market-wide suspension.
    -> (px with inferred rows, inferred rows)."""
    end = pd.Timestamp(end)
    last = px.sort_values(["instrument_id", "trade_date"]).groupby("instrument_id").tail(1)
    last = last[last["instrument_id"].isin(active_ids) & (last["trade_date"] < end)]
    if last.empty:
        return px, px.iloc[:0][PRICE_COLS]
    cal = pd.DatetimeIndex(pd.to_datetime(calendar)).sort_values()
    days = cal[(cal > last["trade_date"].min()) & (cal <= end)]
    first = np.sort(px.groupby("instrument_id")["trade_date"].min().to_numpy())
    started = np.searchsorted(first, days.to_numpy(), side="right")
    n_obs = observed.drop_duplicates(["instrument_id", "trade_date"])["trade_date"].value_counts().reindex(days, fill_value=0)
    ok = n_obs.to_numpy() >= min_coverage * np.maximum(started, 1)

    grid = last[["instrument_id", "trade_date", "close", "adj_close"]].rename(columns={"trade_date": "last"}).merge(
        pd.DataFrame({"trade_date": days, "ok": ok.astype(int)}), how="cross"
    )
    grid = grid[grid["trade_date"] > grid["last"]].sort_values(["instrument_id", "trade_date"])
    grid = grid[grid.groupby("instrument_id")["ok"].cummin().astype(bool)]
    inferred = grid.assign(volume=0, is_suspended=True)[PRICE_COLS].reset_index(drop=True)
    inferred["volume"] = inferred["volume"].astype("int64")
    return pd.concat([px[PRICE_COLS], inferred], ignore_index=True), inferred


def build_daily_price(
    daily: pd.DataFrame, factors: pd.DataFrame, instruments: pd.DataFrame, calendar: pd.Series,
    end: pd.Timestamp | None = None, min_coverage: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """daily[instrument_id, trade_date, close, volume] + instruments[instrument_id, list_date, status]
    -> (daily_price columns without ingest metadata, rows inferred as trailing suspensions).
    Trailing inference runs only when `end` is given."""
    px = drop_before_listing(daily, instruments)
    px = apply_hfq(px, factors)
    observed = px[["instrument_id", "trade_date"]]
    px = fill_suspended(px, calendar)[PRICE_COLS]
    if end is None:
        return px, px.iloc[:0]
    active = set(instruments.loc[instruments["status"] == "active", "instrument_id"])
    return infer_trailing_suspensions(px, observed, calendar, end, active, min_coverage)
