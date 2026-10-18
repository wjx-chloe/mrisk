"""Row-by-row comparison of two mrisk databases (the replay check of docs/week4-plan.md §3.1): every table of the
schema, matched on its primary key, run metadata left out. Values are compared exactly (floats bit for bit, NaN
equal to NaN, JSON as parsed objects), so the same code on the same input should report no difference at all."""

import json

import numpy as np
import pandas as pd
from sqlalchemy import JSON, BigInteger, Engine, Float, Integer, select

from mrisk.db.schema import metadata

# run metadata: who / when produced the row, not what the row says
RUN_METADATA = ("run_id", "created_at", "ingested_at")


def _canon(v):
    """JSON values -> one canonical string, so dict key order does not count as a difference."""
    return json.dumps(v, sort_keys=True, ensure_ascii=False) if isinstance(v, (dict, list)) else v


def _equal(a: pd.Series, b: pd.Series) -> np.ndarray:
    both_na = a.isna().to_numpy() & b.isna().to_numpy()
    with np.errstate(invalid="ignore"):
        eq = (a.to_numpy(dtype=object) == b.to_numpy(dtype=object))
    return np.asarray(eq, dtype=bool) | both_na


def compare_table(a: Engine, b: Engine, name: str) -> dict:
    """Counts for one table: rows in a and b, keys only in a / only in b, matched rows that differ in any compared
    column, the columns that differ, and the largest absolute difference over the numeric ones. Where run metadata
    is part of the primary key (dq_result keeps every DQ run), only the latest run of each key is compared; the rows
    of older runs are counted in superseded_a / superseded_b and not compared."""
    table = metadata.tables[name]
    key = [c.name for c in table.primary_key.columns if c.name not in RUN_METADATA]
    cols = [c.name for c in table.columns if c.name not in RUN_METADATA and c.name not in key]
    numeric = {c.name for c in table.columns if isinstance(c.type, (Float, Integer, BigInteger))}
    js = [c.name for c in table.columns if c.name in cols and isinstance(c.type, JSON)]
    stamp = ["created_at"] if "created_at" in table.columns and len(key) < len(table.primary_key.columns) else []
    frames, superseded = [], []
    for eng in (a, b):
        with eng.connect() as conn:
            df = pd.DataFrame(conn.execute(select(*[table.c[k] for k in key + cols + stamp])).all(), columns=key + cols + stamp)
        n = len(df)
        if stamp:  # several runs of the same key (dq_result): keep the latest run, as the gate and the dashboard do
            df = df[df["created_at"] == df.groupby(key)["created_at"].transform("max")].drop(columns=stamp)
        superseded.append(n - len(df))
        if df.duplicated(key).any():
            raise ValueError(f"{name}: key {key} is not unique without run metadata")
        for c in js:
            df[c] = df[c].map(_canon)
        frames.append(df)
    fa, fb = frames
    m = fa.merge(fb, on=key, how="outer", suffixes=("_a", "_b"), indicator=True)
    both = m[m["_merge"] == "both"]
    differs = np.zeros(len(both), dtype=bool)
    diff_cols, max_abs = [], 0.0
    for c in cols:
        eq = _equal(both[f"{c}_a"], both[f"{c}_b"])
        if (~eq).any():
            diff_cols.append(c)
            differs |= ~eq
            if c in numeric:
                d = (pd.to_numeric(both[f"{c}_a"]) - pd.to_numeric(both[f"{c}_b"])).abs()
                max_abs = max(max_abs, float(d[~eq].max()))
    return {
        "table": name, "rows_a": len(fa), "rows_b": len(fb), "superseded_a": superseded[0], "superseded_b": superseded[1],
        "only_a": int((m["_merge"] == "left_only").sum()), "only_b": int((m["_merge"] == "right_only").sum()),
        "diff_rows": int(differs.sum()), "diff_cols": ",".join(diff_cols), "max_abs_diff": max_abs,
    }


def compare_databases(a: Engine, b: Engine, tables: list[str] | None = None) -> pd.DataFrame:
    """compare_table for every table of the schema (or the ones given), in schema order."""
    return pd.DataFrame([compare_table(a, b, t) for t in tables or list(metadata.tables)])


def identical(report: pd.DataFrame) -> bool:
    return bool((report[["only_a", "only_b", "diff_rows"]] == 0).all().all() and (report["rows_a"] == report["rows_b"]).all())
