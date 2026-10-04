"""Limit rule evaluation (SPEC §3.7, §6.2). Pure: rules + metric values in, one check per rule out.

Metric values come as rows [metric, scope, key, value, base]: the checked number is value / base for
threshold_type pct_of_mv and value for absolute. Losses (VaR, ES, stress loss) are positive numbers.
A metric whose input is missing is listed in `missing` and every rule on it gets status no_data."""

import numpy as np
import pandas as pd

CHECK_COLUMNS = ["rule_id", "actual_value", "threshold", "warn_threshold", "status", "detail"]
CONCENTRATION = ("concentration_instrument", "concentration_industry")


def _violates(x, threshold, operator: str):
    if operator == "<=":
        return x > threshold
    if operator == "<":
        return x >= threshold
    raise ValueError(f"unknown operator {operator!r}")


def evaluate(rules: pd.DataFrame, metrics: pd.DataFrame, missing: dict[str, dict] | None = None) -> pd.DataFrame:
    """rules: limit_rule rows (inactive ones are skipped). metrics: [metric, scope, key, value, base].
    missing: metric -> detail explaining why its input is absent. -> CHECK_COLUMNS, one row per active rule.
    breach if the checked value violates threshold under the rule's operator; else warn if it violates
    warn_ratio x threshold; else ok. A wildcard scope (industry:* / instrument:*) checks every member, reports the
    largest value and lists the members past the warn threshold in detail."""
    missing = missing or {}
    out = []
    for r in rules[rules["is_active"].astype(bool)].itertuples(index=False):
        thr = float(r.threshold)
        warn = float(r.warn_ratio) * thr
        level, _, key = str(r.scope).partition(":")
        row = {"rule_id": r.rule_id, "threshold": thr, "warn_threshold": warn, "actual_value": None, "detail": None}
        if r.metric in missing:
            out.append({**row, "status": "no_data", "detail": missing[r.metric]})
            continue
        m = metrics[(metrics["metric"] == r.metric) & (metrics["scope"] == level)]
        if m.empty and r.metric not in CONCENTRATION:
            raise ValueError(f"rule {r.rule_id}: no values for metric {r.metric} at scope {level}")
        if r.threshold_type == "pct_of_mv":
            vals = m["value"].to_numpy(dtype=float) / m["base"].to_numpy(dtype=float)
        elif r.threshold_type == "absolute":
            vals = m["value"].to_numpy(dtype=float)
        else:
            raise ValueError(f"unknown threshold_type {r.threshold_type!r}")
        vals = pd.Series(vals, index=m["key"].to_numpy())
        if key == "*":
            actual = float(vals.max()) if len(vals) else 0.0
            over = vals[_violates(vals, warn, r.operator)].sort_values(ascending=False)
            row["detail"] = {"members": [{"key": k, "value": float(v)} for k, v in over.items()]}
        elif level == "portfolio" or key == "":
            if len(vals) != 1:
                raise ValueError(f"rule {r.rule_id}: expected one portfolio value for {r.metric}, got {len(vals)}")
            actual = float(vals.iloc[0])
        elif key in vals.index:
            actual = float(vals[key])
        elif r.metric in CONCENTRATION:
            actual = 0.0  # not held: no concentration
        else:
            out.append({**row, "status": "no_data", "detail": {"reason": f"no {r.metric} value for {r.scope}"}})
            continue
        status = "breach" if _violates(actual, thr, r.operator) else "warn" if _violates(actual, warn, r.operator) else "ok"
        out.append({**row, "actual_value": actual, "status": status})
    return pd.DataFrame(out, columns=CHECK_COLUMNS)


def metric_rows(metric: str, scope: str, keys, values, base) -> pd.DataFrame:
    """Helper to build metric rows with one base (e.g. the portfolio market value) for every key."""
    keys = list(keys)
    return pd.DataFrame(
        {"metric": metric, "scope": scope, "key": keys, "value": np.asarray(values, dtype=float), "base": float(base)}
    )
