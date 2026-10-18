"""Regression on the numbers the docs quote, against a database built by `mrisk demo` (the full 497 position days of
config.demo.yaml) or the real one. The path comes from MRISK_REGRESSION_DB; marked slow, so the pre-commit hook
skips it. Run: MRISK_REGRESSION_DB=data/demo.db uv run pytest -m slow"""

import os
from pathlib import Path

import pandas as pd
import pytest

from mrisk.db import schema as s
from mrisk.db.repo import read_frame, readonly_engine

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def engine():
    path = os.environ.get("MRISK_REGRESSION_DB")
    if not path:
        pytest.skip("MRISK_REGRESSION_DB not set (path of a database built by mrisk demo)")
    if not Path(path).exists():
        pytest.fail(f"MRISK_REGRESSION_DB={path} does not exist")
    return readonly_engine(f"sqlite:///{Path(path).resolve().as_posix()}")


def test_position_days(engine):
    pos = read_frame(engine, s.position, columns=["trade_date"])
    assert pos["trade_date"].nunique() == 497
    assert (pos["trade_date"].min(), pos["trade_date"].max()) == (pd.Timestamp("2024-09-02"), pd.Timestamp("2026-09-18"))


def test_stress_2015_loss_share_max(engine):
    """docs/week3b-stress-limits.md §5.3: the 2015 scenario loses at most 41.51% of market value over 497 days."""
    r = read_frame(engine, s.stress_result, scenario_id="HIST_2015", level="portfolio", columns=["trade_date", "market_value", "pnl"])
    assert len(r) == 497
    assert round(float((-r["pnl"] / r["market_value"]).max()), 4) == 0.4151


def test_limit_check_status_counts(engine):
    """docs/week3b-stress-limits.md §5: 497 days x 5 rules = 2,485 rows; L_STRESS_2015 ok 459, warn 38."""
    lc = read_frame(engine, s.limit_check, columns=["rule_id", "status"])
    assert len(lc) == 2485
    assert lc["status"].value_counts().to_dict() == {"ok": 2055, "warn": 383, "breach": 47}
    assert lc.loc[lc["rule_id"] == "L_STRESS_2015", "status"].value_counts().to_dict() == {"ok": 459, "warn": 38}
