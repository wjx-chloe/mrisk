"""Candidate stock pool and instrument master rows (pure)."""

import pandas as pd

from mrisk.data.transform import infer_board, instrument_id


def select_universe(
    listings: pd.DataFrame,
    sw_members: pd.DataFrame,
    industries: list[str],
    per_industry: int,
    listed_before: pd.Timestamp,
    must_include: list[str],
) -> pd.DataFrame:
    """Top `per_industry` names by SW index weight in each industry, listed before `listed_before`,
    non-ST, SH/SZ only; plus `must_include` instrument_ids. -> [instrument_id, exchange, symbol]."""
    lst = listings.assign(instrument_id=instrument_id(listings["exchange"], listings["symbol"]))
    m = sw_members.merge(lst, on="symbol")
    m = m[
        m["industry_l1"].isin(industries)
        & m["exchange"].isin(["SH", "SZ"])
        & (m["list_date"] < listed_before)
        & ~m["name"].str.contains("ST")
    ].copy()
    missing = set(industries) - set(m["industry_l1"])
    if missing:
        raise ValueError(f"no eligible members for industries {sorted(missing)}")
    m["sw_weight"] = pd.to_numeric(m["sw_weight"])
    top = m.sort_values("sw_weight", ascending=False).groupby("industry_l1").head(per_industry)
    forced = lst[lst["instrument_id"].isin(must_include)]
    if len(forced) != len(set(must_include)):
        raise ValueError(f"must_include not found in listings: {sorted(set(must_include) - set(forced['instrument_id']))}")
    out = pd.concat([top[["instrument_id", "exchange", "symbol"]], forced[["instrument_id", "exchange", "symbol"]]])
    return out.drop_duplicates("instrument_id").sort_values("instrument_id", ignore_index=True)


def build_instruments(
    universe: pd.DataFrame,
    listings: pd.DataFrame,
    sw_members: pd.DataFrame,
    delisted: pd.DataFrame,
    industry_override: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """-> instrument table rows. Every universe member must have a listing and an SW L1 industry.
    industry_override[instrument_id, industry_l1] only fills names SW index constituents do not cover
    (they exclude the Beijing exchange); overriding a covered name raises so stale overrides surface."""
    u = universe[["instrument_id", "exchange", "symbol"]]
    m = u.merge(listings[["exchange", "symbol", "name", "list_date"]], on=["exchange", "symbol"], how="left")
    m = m.merge(sw_members.drop_duplicates("symbol")[["symbol", "industry_l1"]], on="symbol", how="left")
    if industry_override is not None and not industry_override.empty:
        ov = industry_override.set_index("instrument_id")["industry_l1"]
        clash = m["instrument_id"].isin(ov.index) & m["industry_l1"].notna()
        if clash.any():
            raise ValueError(f"industry override for SW-covered names: {m.loc[clash, 'instrument_id'].tolist()}")
        fill = m["industry_l1"].isna()
        m.loc[fill, "industry_l1"] = m.loc[fill, "instrument_id"].map(ov)
    for col in ("name", "industry_l1"):
        if m[col].isna().any():
            raise ValueError(f"{col} missing for {m.loc[m[col].isna(), 'instrument_id'].tolist()}")
    dl = set(zip(delisted["exchange"], delisted["symbol"]))
    m["status"] = ["delisted" if k in dl else "active" for k in zip(m["exchange"], m["symbol"])]
    m["board"] = infer_board(m["exchange"], m["symbol"])
    return m[["instrument_id", "exchange", "symbol", "name", "industry_l1", "board", "list_date", "status"]]
