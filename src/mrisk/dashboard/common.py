"""What every page shares: the context the entry script puts in session state, status labels and number formats."""

from dataclasses import dataclass

import pandas as pd
import streamlit as st
from sqlalchemy import Engine

# status always shown with an icon and a word, never colour alone
LIMIT_LABEL = {"breach": "🔴 突破", "warn": "🟡 预警", "ok": "🟢 正常", "no_data": "⚪ 无数据"}
ZONE_LABEL = {"green": "🟢 绿", "yellow": "🟡 黄", "red": "🔴 红"}
GATE_LABEL = {"ok": "🟢 通过", "no_dq": "🟡 窗口内有未做 DQ 的日期", "blocked": "🔴 阻断", "unknown": "⚪ 无法判断"}
DQ_LABEL = {"pass": "🟢 通过", "fail": "🔴 失败", "skip": "⚪ 跳过"}
BREACH_BG = "background-color: rgba(208, 59, 59, 0.18)"
WARN_BG = "background-color: rgba(250, 178, 25, 0.18)"
FOCUS_BG = "background-color: rgba(42, 120, 214, 0.16)"  # rows a limit link asks to highlight (not a status colour)


@dataclass(frozen=True)
class Ctx:
    engine: Engine
    cfg: dict
    pid: str
    date: pd.Timestamp
    conf: float
    complete: pd.DatetimeIndex  # position dates with complete results, oldest first


def ctx() -> Ctx:
    return st.session_state["ctx"]


def adopt_query_param(param: str, key: str, parse=str, valid=None):
    """Puts ?param= of the URL into the widget state `key` when the URL brings a value this page has not seen yet (a
    page_link, a pasted link, a reload), before the widget is created. A value already seen is left alone, so the
    widget can be changed afterwards while the URL still carries the old value. Unparsable or invalid values are
    ignored. -> the parsed value when adopted, else None."""
    raw = st.query_params.get(param)
    seen = st.session_state.setdefault("_qp_seen", {})
    if raw == seen.get(param):
        return None
    seen[param] = raw
    if raw is None:
        return None
    try:
        v = parse(raw)
    except (TypeError, ValueError):
        return None
    if valid is not None and v not in valid:
        return None
    st.session_state[key] = v
    return v


JUMP_CONTEXT = ("from", "inst", "ind")  # what a limit link adds on top of the widgets' own params


def write_query_param(param: str, raw: str) -> None:
    """Puts the widget's value into the URL (so the page can be reloaded or shared as it is) and marks it as seen.
    When the user moved the widget away from the value the URL carried, the limit link's context no longer applies to
    what is shown, so it is dropped."""
    if st.query_params.get(param) not in (None, raw):
        for k in JUMP_CONTEXT:
            st.query_params.pop(k, None)
    st.query_params[param] = raw
    st.session_state.setdefault("_qp_seen", {})[param] = raw


def focus_from_query(param: str) -> list[str]:
    """Comma-separated keys of ?param= (the members a limit link asks to highlight)."""
    raw = st.query_params.get(param)
    return [k for k in raw.split(",") if k] if raw else []


def jumped_from(page: str, where=None) -> None:
    """One line on a page reached from a limit rule's link (?from=rule_id), in `where` (a placeholder at the top of
    the page, filled after the page's widgets have written their params) or inline."""
    rid = st.query_params.get("from")
    if rid:
        (where or st).info(f"从总览限额 **{rid}** 跳转到{page}。", icon="↪️")


def wan(x: float) -> str:
    """元 -> 万元 with separators."""
    return f"{x / 1e4:,.1f} 万"


def pct(x: float, digits: int = 2) -> str:
    return f"{x * 100:.{digits}f}%"


def conf_label(c: float) -> str:
    return f"{c:.0%}"


def highlight_status(df: pd.DataFrame, col: str = "status"):
    """Styler: breach rows red, warn rows amber (SPEC §6.3 red highlight). df keeps the raw status in `col`."""
    def row_style(r):
        bg = BREACH_BG if r[col] == "breach" else WARN_BG if r[col] == "warn" else ""
        return [bg] * len(r)
    return df.style.apply(row_style, axis=1)


def status_dataframe(df: pd.DataFrame, col: str = "status") -> None:
    """st.dataframe with highlight_status rows; the raw status column drives the colours but is not shown.
    column_order hides it: st.dataframe ignores Styler.hide."""
    st.dataframe(highlight_status(df, col), hide_index=True, width="stretch", column_order=[c for c in df.columns if c != col])
