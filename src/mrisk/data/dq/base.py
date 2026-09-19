from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DQContext:
    """Everything a rule may look at. All frames use the db column names; dates are datetime64."""

    check_dates: pd.DatetimeIndex
    calendar: pd.Series
    instruments: pd.DataFrame
    code_map: pd.DataFrame
    prices: pd.DataFrame
    positions: pd.DataFrame
    params: dict = field(default_factory=dict)


@dataclass
class RuleOutput:
    """checked: rows examined per check_date; failures: one row per failing item, with a check_date column."""

    checked: pd.Series
    failures: pd.DataFrame


RuleFn = Callable[[DQContext], RuleOutput]


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    name: str
    category: str
    description: str
    fn: RuleFn


def per_date_count(dates: pd.Series, check_dates: pd.DatetimeIndex) -> pd.Series:
    return dates.value_counts().reindex(check_dates, fill_value=0).astype(int)


def broadcast_static(n_checked: int, failures: pd.DataFrame, check_dates: pd.DatetimeIndex) -> RuleOutput:
    """Rules on reference tables have no date of their own; report them on every check date."""
    checked = pd.Series(n_checked, index=check_dates, dtype=int)
    fails = failures.merge(pd.DataFrame({"check_date": check_dates}), how="cross")
    return RuleOutput(checked, fails)
