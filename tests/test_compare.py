from pathlib import Path

import pandas as pd
import pytest

from mrisk.db import schema as s
from mrisk.db.compare import compare_databases, identical
from mrisk.db.repo import get_engine, init_db, upsert


def _db(path: Path, rules: pd.DataFrame, checks: pd.DataFrame):
    e = get_engine(f"sqlite:///{path.as_posix()}")
    init_db(e)
    upsert(e, s.limit_rule, rules)
    upsert(e, s.limit_check, checks)
    return e


RULES = pd.DataFrame({
    "rule_id": ["R1"], "portfolio_id": ["P001"], "metric": ["var_hs_99"], "scope": ["portfolio"], "operator": ["<="],
    "threshold": [1.0], "threshold_type": ["absolute"], "warn_ratio": [0.9], "is_active": [True],
})


def _checks(values, days, detail=None):
    return pd.DataFrame({
        "trade_date": pd.to_datetime(days), "rule_id": "R1", "actual_value": values, "threshold": 1.0,
        "warn_threshold": 0.9, "status": "ok", "detail": [detail or {"a": 1, "b": [1, 2]}] * len(days),
    })


def test_compare_databases_counts_differences(tmp_path):
    a = _db(tmp_path / "a.db", RULES, _checks([0.5, 0.6, 0.7], ["2025-01-02", "2025-01-03", "2025-01-06"]))
    b = _db(tmp_path / "b.db", RULES, _checks([0.5, 0.65, 0.7, 0.8], ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]))
    rep = compare_databases(a, b, ["limit_rule", "limit_check"]).set_index("table")
    assert rep.loc["limit_rule", ["diff_rows", "only_a", "only_b"]].tolist() == [0, 0, 0]
    lc = rep.loc["limit_check"]
    assert (lc["rows_a"], lc["rows_b"], lc["only_a"], lc["only_b"], lc["diff_rows"], lc["diff_cols"]) == (3, 4, 0, 1, 1, "actual_value")
    assert lc["max_abs_diff"] == pytest.approx(0.05)
    assert not identical(rep.reset_index())


def test_compare_databases_ignores_run_metadata_and_json_key_order(tmp_path):
    a = _db(tmp_path / "a.db", RULES, _checks([0.5], ["2025-01-02"], {"a": 1, "b": 2}))
    b = _db(tmp_path / "b.db", RULES, _checks([0.5], ["2025-01-02"], {"b": 2, "a": 1}))
    for e, run in ((a, "risk-1"), (b, "risk-2")):
        upsert(e, s.risk_result, pd.DataFrame({
            "trade_date": [pd.Timestamp("2025-01-02")], "portfolio_id": "P001", "level": "portfolio", "level_key": "P001",
            "confidence": 0.99, "market_value": 1e6, "var_hs": 1.5e4, "var_param": float("nan"), "es_hs": 2e4, "run_id": run,
        }))
    rep = compare_databases(a, b)
    assert identical(rep), rep


def test_compare_databases_takes_the_latest_dq_run(tmp_path):
    def dq(path, runs):
        e = get_engine(f"sqlite:///{path.as_posix()}")
        init_db(e)
        upsert(e, s.dq_rule, pd.DataFrame({"rule_id": ["DQ01"], "name": ["n"], "category": ["c"], "severity": ["warn"], "is_active": [True]}))
        upsert(e, s.dq_result, pd.DataFrame([
            {"run_id": run, "check_date": pd.Timestamp("2025-01-02"), "rule_id": "DQ01", "status": status, "n_checked": 1,
             "n_failed": int(status == "fail"), "detail_json": None, "created_at": pd.Timestamp(at)}
            for run, status, at in runs
        ]))
        return e
    a = dq(tmp_path / "a.db", [("r1", "fail", "2025-01-02 17:00"), ("r2", "pass", "2025-01-02 18:00")])
    b = dq(tmp_path / "b.db", [("r9", "pass", "2025-02-01 09:00")])
    rep = compare_databases(a, b, ["dq_result"]).iloc[0]
    assert (rep["rows_a"], rep["superseded_a"], rep["superseded_b"], rep["diff_rows"]) == (1, 1, 0, 0)
