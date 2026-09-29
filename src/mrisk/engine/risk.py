"""One as-of date, all risk measures, all levels (SPEC §5.2). Pure: frames in, frames out."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mrisk.engine.returns import returns_window
from mrisk.engine.valuation import exposure_matrix, market_value
from mrisk.engine.var import hs_es, hs_var, param_var

RESULT_COLUMNS = ["level", "level_key", "confidence", "market_value", "var_hs", "var_param", "es_hs", "es_tail_n"]


@dataclass(frozen=True)
class RiskResult:
    table: pd.DataFrame  # RESULT_COLUMNS, one row per (risk unit, confidence)
    suspended_days: pd.Series  # per held instrument: suspended days inside the return window


def compute_risk(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    instruments: pd.DataFrame,
    asof,
    portfolio_id: str,
    confidence,
    lookback: int,
    return_type: str = "simple",
) -> RiskResult:
    """positions[instrument_id, quantity] held at the close of `asof`;
    prices[trade_date, instrument_id, close, adj_close, is_suspended] covering the window (extra rows are ignored);
    instruments[instrument_id, industry_l1].
    Rows are ordered portfolio, industries, instruments, and by confidence within each unit.
    VaR is not additive across levels and nothing here assumes it is. es_tail_n = 0 marks ES = VaR fallback."""
    asof = pd.Timestamp(asof)
    conf = np.sort(np.atleast_1d(np.asarray(confidence, dtype=float)))
    held = sorted(positions["instrument_id"].unique())
    px = prices[prices["instrument_id"].isin(held) & (prices["trade_date"] <= asof)]
    dates = np.sort(px["trade_date"].unique())[-(lookback + 1) :]
    wide = px[px["trade_date"].isin(dates)].pivot(index="trade_date", columns="instrument_id")
    wide = wide.reindex(columns=pd.MultiIndex.from_product([wide.columns.levels[0], held]))

    R, suspended_days = returns_window(wide["adj_close"], wide["is_suspended"], asof, lookback, return_type)
    mv = market_value(positions, wide["close"].loc[asof], instruments.set_index("instrument_id")["industry_l1"])
    E, units = exposure_matrix(mv, portfolio_id)

    Rm = R[mv["instrument_id"]].to_numpy()
    pnl = Rm @ E
    var = hs_var(pnl, conf)
    es, n_tail = hs_es(pnl, var)
    pvar = param_var(Rm, E, conf)

    C = len(conf)
    table = units.loc[units.index.repeat(C)].reset_index(drop=True)
    table.insert(2, "confidence", np.tile(conf, len(units)))
    # (C, K) -> unit-major rows
    table["var_hs"] = var.T.ravel()
    table["var_param"] = pvar.T.ravel()
    table["es_hs"] = es.T.ravel()
    table["es_tail_n"] = n_tail.T.ravel()
    return RiskResult(table[RESULT_COLUMNS], suspended_days)
