import pandas as pd
import streamlit as st

from mrisk.dashboard import charts
from mrisk.dashboard import queries as q
from mrisk.dashboard.common import LIMIT_LABEL, ctx, pct, status_dataframe

c = ctx()
st.header(f"限额 · {c.date:%Y-%m-%d}")


def fmt(v, kind) -> str:
    if pd.isna(v):
        return "—"
    return pct(v) if kind == "pct_of_mv" else f"{v:,.0f}"


st.subheader("当日检查")
lim = q.limits_today(c.engine, c.pid, c.date)
new = lim[lim["new"]]
cont = lim[lim["status"].isin(["warn", "breach"]) & ~lim["new"]]
a, b, d = st.columns(3)
a.metric("今日新触发", len(new), help="当日进入预警或突破、前一个持仓日不是同一状态的规则")
b.metric("持续中", len(cont), help="预警或突破已持续 2 天以上；无数据的日子中断计数")
d.metric("突破", int((lim["status"] == "breach").sum()))
table = pd.DataFrame({
    "规则": lim["rule_id"],
    "指标": lim["metric"],
    "范围": lim["scope"],
    "状态": lim["status"].map(LIMIT_LABEL),
    "检查值": [fmt(v, k) for v, k in zip(lim["actual_value"], lim["threshold_type"])],
    "突破线": [fmt(v, k) for v, k in zip(lim["threshold"], lim["threshold_type"])],
    "预警线": [fmt(v, k) for v, k in zip(lim["warn_threshold"], lim["threshold_type"])],
    "触发": ["今日新触发" if n else (f"已持续 {r} 天" if s in ("warn", "breach") else "") for n, r, s in zip(lim["new"], lim["run_days"], lim["status"])],
    "status": lim["status"],
})
status_dataframe(table)
for r in lim.itertuples():
    det = r.detail if isinstance(r.detail, dict) else None
    if det and det.get("members"):
        with st.expander(f"{r.rule_id}：超过预警线的成员"):
            st.dataframe(pd.DataFrame(det["members"]).assign(value=lambda x: x["value"].map(pct)).rename(columns={"key": "成员", "value": "检查值"}), hide_index=True)
    elif r.status == "no_data" and det:
        st.caption(f"{r.rule_id} 无数据：{det}")

st.subheader("历史突破记录")
eps = q.limit_episodes(c.engine, c.pid, c.date, "breach")
kinds = lim.set_index("rule_id")["threshold_type"]
if eps.empty:
    st.caption("截至该日没有突破记录。")
else:
    st.dataframe(pd.DataFrame({
        "规则": eps["rule_id"], "开始": eps["start"].dt.date, "结束": eps["end"].dt.date, "天数": eps["days"],
        "最高检查值": [fmt(v, kinds.get(r, "pct_of_mv")) for v, r in zip(eps["peak"], eps["rule_id"])],
    }), hide_index=True, width="stretch")
with st.expander("历史预警记录"):
    w = q.limit_episodes(c.engine, c.pid, c.date, "warn")
    st.dataframe(pd.DataFrame({"规则": w["rule_id"], "开始": w["start"].dt.date, "结束": w["end"].dt.date, "天数": w["days"],
                               "最高检查值": [fmt(v, kinds.get(r, "pct_of_mv")) for v, r in zip(w["peak"], w["rule_id"])]}),
                 hide_index=True, width="stretch")

st.subheader("规则历史")
rid = st.selectbox("规则", lim["rule_id"].tolist(), key="limit_rule")
hist = q.limit_checks(c.engine, c.pid, end=c.date)
hist = hist[hist["rule_id"] == rid].sort_values("trade_date")
r = lim.set_index("rule_id").loc[rid]
st.plotly_chart(charts.limit_history(hist, r["threshold"], r["warn_threshold"], f"{rid} 检查值",
                                     ".1%" if r["threshold_type"] == "pct_of_mv" else ",.0f"), width="stretch")

st.subheader("规则列表")
rules = q.active_rules(c.engine, c.pid)
st.dataframe(pd.DataFrame({
    "规则": rules["rule_id"], "指标": rules["metric"], "范围": rules["scope"], "比较": rules["operator"],
    "突破线": [fmt(v, k) for v, k in zip(rules["threshold"], rules["threshold_type"])],
    "预警比例": [pct(v, 0) for v in rules["warn_ratio"]], "阈值类型": rules["threshold_type"],
}), hide_index=True, width="stretch")
