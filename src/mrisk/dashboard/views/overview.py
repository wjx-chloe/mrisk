import pandas as pd
import streamlit as st

from mrisk.dashboard import queries as q
from mrisk.dashboard.common import LIMIT_LABEL, conf_label, ctx, highlight_status, pct, wan

c = ctx()
st.header(f"总览 · {c.date:%Y-%m-%d}")

ov = q.overview(c.engine, c.pid, c.date, c.conf, c.complete)
today = ov.loc["today"]
prev = ov.loc["prev"] if "prev" in ov.index else None
prev_day = c.complete[c.complete < c.date]
st.caption(f"置信度 {conf_label(c.conf)} · 持有 1 天 · 变动对比上一个有完整结果的日期" + (f"（{prev_day[-1]:%Y-%m-%d}）" if len(prev_day) else "（无）"))

items = [
    ("组合市值", "market_value", "normal"),
    (f"VaR 历史模拟 {conf_label(c.conf)}", "var_hs", "inverse"),
    (f"VaR 参数法 {conf_label(c.conf)}", "var_param", "inverse"),
    (f"ES 历史模拟 {conf_label(c.conf)}", "es_hs", "inverse"),
]
for col, (label, k, colour) in zip(st.columns(4), items):
    delta = None if prev is None else f"{(today[k] - prev[k]) / 1e4:+,.1f} 万"
    col.metric(label, wan(today[k]), delta, delta_color=colour)
    if k != "market_value":
        col.caption(f"占市值 {pct(today[k] / today['market_value'])}")

st.subheader("限额状态")
lim = q.limits_today(c.engine, c.pid, c.date)


def _value(r) -> str:
    if pd.isna(r.actual_value):
        return "—"
    return pct(r.actual_value) if r.threshold_type == "pct_of_mv" else f"{r.actual_value:,.0f}"


def _run(r) -> str:
    if r.status not in ("warn", "breach"):
        return ""
    return "今日新触发" if r.new else f"已持续 {r.run_days} 天"


table = pd.DataFrame({
    "规则": lim["rule_id"],
    "状态": lim["status"].map(LIMIT_LABEL),
    "检查值": [_value(r) for r in lim.itertuples()],
    "突破线": [pct(t) if tt == "pct_of_mv" else f"{t:,.0f}" for t, tt in zip(lim["threshold"], lim["threshold_type"])],
    "预警线": [pct(t) if tt == "pct_of_mv" else f"{t:,.0f}" for t, tt in zip(lim["warn_threshold"], lim["threshold_type"])],
    "持续": [_run(r) for r in lim.itertuples()],
    "status": lim["status"],
})
st.dataframe(highlight_status(table).hide(["status"], axis="columns"), hide_index=True, width="stretch")
counts = lim["status"].value_counts()
st.caption(" · ".join(f"{LIMIT_LABEL[k]} {counts.get(k, 0)}" for k in ("breach", "warn", "ok", "no_data")))
