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
