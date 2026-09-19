import json

import pandas as pd

from mrisk.data.dq.base import DQContext, RuleSpec

MAX_DETAIL_ROWS = 20


def _detail(fails: pd.DataFrame) -> dict | None:
    if fails.empty:
        return None
    sample = fails.drop(columns="check_date").head(MAX_DETAIL_ROWS)
    return {"n_failed": len(fails), "sample": json.loads(sample.to_json(orient="records", date_format="iso", force_ascii=False))}


def run_rules(ctx: DQContext, rules: list[RuleSpec], blocking: set[str]) -> pd.DataFrame:
    """-> one row per (check_date, rule_id): status, n_checked, n_failed, detail_json, severity."""
    rows = []
    for spec in rules:
        out = spec.fn(ctx)
        n_failed = out.failures["check_date"].value_counts().reindex(ctx.check_dates, fill_value=0)
        fails_by_date = dict(tuple(out.failures.groupby("check_date"))) if not out.failures.empty else {}
        for d in ctx.check_dates:
            n_chk, n_bad = int(out.checked.get(d, 0)), int(n_failed.get(d, 0))
            status = "fail" if n_bad else ("skip" if n_chk == 0 else "pass")
            rows.append(
                {
                    "check_date": d,
                    "rule_id": spec.rule_id,
                    "status": status,
                    "n_checked": n_chk,
                    "n_failed": n_bad,
                    "detail_json": _detail(fails_by_date[d]) if d in fails_by_date else None,
                    "severity": "blocking" if spec.rule_id in blocking else "warn",
                }
            )
    return pd.DataFrame(rows)


def blocked_dates(results: pd.DataFrame) -> pd.DatetimeIndex:
    hit = results[(results["status"] == "fail") & (results["severity"] == "blocking")]
    return pd.DatetimeIndex(sorted(hit["check_date"].unique()))
