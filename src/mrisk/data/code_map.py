"""Resolve exchange symbols to stable instrument_ids across code switches (BJ 920 segment)."""

import pandas as pd


class UnresolvedSymbolError(ValueError):
    def __init__(self, rows: pd.DataFrame):
        self.rows = rows
        super().__init__(f"{len(rows)} rows could not be mapped to an instrument_id:\n{rows.head(10)}")


def resolve_instrument_id(
    df: pd.DataFrame, instruments: pd.DataFrame, code_map: pd.DataFrame
) -> pd.DataFrame:
    """df[exchange, symbol, trade_date, ...] -> same rows with instrument_id added.

    A current symbol resolves on any date (sources restate history under the new code).
    An old symbol resolves only before its effective_date. Anything else raises."""
    current = instruments[["instrument_id", "exchange", "symbol"]]
    out = df.merge(current, on=["exchange", "symbol"], how="left")

    old = code_map.merge(instruments[["instrument_id", "exchange"]], on="instrument_id")[
        ["exchange", "old_symbol", "instrument_id", "effective_date"]
    ].rename(columns={"old_symbol": "symbol", "instrument_id": "_old_id"})
    out = out.merge(old, on=["exchange", "symbol"], how="left")
    use_old = out["instrument_id"].isna() & out["_old_id"].notna() & (out["trade_date"] < out["effective_date"])
    out.loc[use_old, "instrument_id"] = out.loc[use_old, "_old_id"]
    out = out.drop(columns=["_old_id", "effective_date"])

    bad = out["instrument_id"].isna()
    if bad.any():
        raise UnresolvedSymbolError(out.loc[bad, list(df.columns)])
    return out
