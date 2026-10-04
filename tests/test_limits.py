import pandas as pd
import pytest

from mrisk.limits.engine import CHECK_COLUMNS, evaluate, metric_rows


def rule(rule_id, metric, scope, threshold, op="<=", ttype="pct_of_mv", warn=0.9, active=True):
    return {"rule_id": rule_id, "portfolio_id": "P", "metric": metric, "scope": scope, "operator": op,
            "threshold": threshold, "threshold_type": ttype, "warn_ratio": warn, "is_active": active}


MV = 1_000_000.0
METRICS = pd.concat(
    [
        metric_rows("var_hs_99", "portfolio", ["P"], [40_000.0], MV),  # 4.0% exactly
        metric_rows("es_hs_99", "portfolio", ["P"], [63_000.0], MV),  # 6.3% = 90% of 7%
        metric_rows("concentration_instrument", "instrument", ["A", "B", "C"], [114_600.0, 85_000.0, 60_000.0], MV),
        metric_rows("concentration_industry", "industry", ["银行", "电子"], [250_000.0, 310_000.0], MV),
        metric_rows("stress_loss:S15", "portfolio", ["P"], [180_000.0], MV),
    ],
    ignore_index=True,
)


def check(rules, missing=None):
    out = evaluate(pd.DataFrame(rules), METRICS, missing)
    assert list(out.columns) == CHECK_COLUMNS
    return out.set_index("rule_id")


def test_boundaries_with_both_operators():
    out = check([
        rule("le", "var_hs_99", "portfolio", 0.04),  # 4% <= 4%: no breach, but past the 3.6% warn line -> warn
        rule("lt", "var_hs_99", "portfolio", 0.04, op="<"),  # 4% < 4% fails: breach
        # the 4% value sitting exactly on a warn line (0.5 x 8%, exact in binary floating point)
        rule("warn_edge", "var_hs_99", "portfolio", 0.08, warn=0.5),  # '<=': on the line is not past it -> ok
        rule("warn_edge_lt", "var_hs_99", "portfolio", 0.08, op="<", warn=0.5),  # '<': the line itself is past -> warn
        rule("warn", "es_hs_99", "portfolio", 0.068),  # warn line 6.12% < 6.3% <= 6.8% -> warn
        rule("ok", "es_hs_99", "portfolio", 0.08),  # 6.3% <= warn line 7.2% -> ok
    ])
    assert out["status"].to_dict() == {
        "le": "warn", "lt": "breach", "warn_edge": "ok", "warn_edge_lt": "warn", "warn": "warn", "ok": "ok",
    }
    assert out.at["le", "actual_value"] == pytest.approx(0.04)
    assert out.at["warn", "warn_threshold"] == pytest.approx(0.0612)


def test_wildcard_reports_the_max_and_members_past_warn():
    out = check([
        rule("inst", "concentration_instrument", "instrument:*", 0.10, warn=0.8),  # warn line 8%
        rule("ind", "concentration_industry", "industry:*", 0.30, warn=0.8),
    ])
    assert out.at["inst", "status"] == "breach" and out.at["inst", "actual_value"] == pytest.approx(0.1146)
    assert out.at["inst", "detail"] == {"members": [{"key": "A", "value": pytest.approx(0.1146)}, {"key": "B", "value": pytest.approx(0.085)}]}
    assert out.at["ind", "status"] == "breach" and out.at["ind", "actual_value"] == pytest.approx(0.31)
    assert [m["key"] for m in out.at["ind", "detail"]["members"]] == ["电子", "银行"]


def test_named_member_and_absent_member():
    out = check([
        rule("b", "concentration_instrument", "instrument:B", 0.10, warn=0.8),  # 8.5% -> warn
        rule("z", "concentration_instrument", "instrument:Z", 0.10, warn=0.8),  # not held -> 0 -> ok
    ])
    assert out.at["b", "status"] == "warn" and out.at["z", "status"] == "ok" and out.at["z", "actual_value"] == 0.0


def test_absolute_threshold_and_stress_loss():
    out = check([
        rule("abs", "stress_loss:S15", "portfolio", 200_000.0, ttype="absolute", warn=0.8),  # 180k > 160k -> warn
        rule("pct", "stress_loss:S15", "portfolio", 0.15, warn=0.8),  # 18% > 15% -> breach
    ])
    assert out.at["abs", "status"] == "warn" and out.at["abs", "actual_value"] == 180_000.0
    assert out.at["pct", "status"] == "breach"


def test_missing_input_gives_no_data_per_rule():
    why = {"missing": "risk_result", "reason": "blocked: price in window ...: ['DQ03@2025-05-21']"}
    out = check(
        [rule("v", "var_hs_99", "portfolio", 0.04), rule("c", "concentration_instrument", "instrument:*", 0.10, warn=0.8)],
        missing={"var_hs_99": why},
    )
    assert out.at["v", "status"] == "no_data" and pd.isna(out.at["v", "actual_value"]) and out.at["v", "detail"] == why
    assert out.at["v", "threshold"] == 0.04  # thresholds are still shown
    assert out.at["c", "status"] == "breach"  # rules with their inputs are evaluated as usual


def test_inactive_rules_are_skipped_and_unknown_metrics_raise():
    out = check([rule("off", "var_hs_99", "portfolio", 0.01, active=False), rule("on", "var_hs_99", "portfolio", 0.05)])
    assert out.index.tolist() == ["on"]
    with pytest.raises(ValueError, match="no values for metric"):
        check([rule("x", "es_hs_95", "portfolio", 0.05)])
