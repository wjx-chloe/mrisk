import pandas as pd
import streamlit as st

from mrisk.dashboard import charts
from mrisk.dashboard import queries as q
from mrisk.dashboard.common import conf_label, ctx, wan

c = ctx()
st.header(f"归因 · {c.date:%Y-%m-%d}")

METRICS = {"var_hs": "VaR 历史模拟", "var_param": "VaR 参数法", "es_hs": "ES 历史模拟"}
EFFECTS = {"quantity": "持仓数量", "revaluation_scale": "重估：规模", "revaluation_drift": "重估：权重漂移", "window": "窗口滚动"}
metric = st.radio("指标", list(METRICS), format_func=METRICS.get, horizontal=True, key="attr_metric")

port, inst = q.attribution(c.engine, c.pid, c.date, c.conf, metric)
if port.empty:
    st.info("该日没有归因结果：需要上一交易日和当日的风险窗口、持仓都齐全（组合首日或被 DQ 拦下时没有）。")
    st.stop()

today = q.overview(c.engine, c.pid, c.date, c.conf, c.complete).loc["today", metric]
total = port["total"]
start = today - total
effects = {EFFECTS[k]: port.get(k, 0.0) for k in EFFECTS}
st.caption(
    f"{METRICS[metric]}（{conf_label(c.conf)}）从上一交易日的 {wan(start)} 变为 {wan(today)}，变化 {total / 1e4:+,.1f} 万。"
    "四项之和等于总变化。"
)
st.plotly_chart(charts.waterfall(effects, f"{METRICS[metric]} 日变化分解（元）"), width="stretch")

a, b, d, e = st.columns(4)
for col, (name, v) in zip((a, b, d, e), effects.items()):
    col.metric(name, f"{v / 1e4:+,.1f} 万")

st.subheader("个股贡献（按总变化绝对值排序，前 10）")
top = inst.head(10)
st.dataframe(pd.DataFrame({
    "代码": top["instrument_id"], "名称": top["name"], "行业": top["industry"],
    **{EFFECTS[k]: top[k].round(0) for k in EFFECTS if k in top},
    "合计": top["total"].round(0),
}), hide_index=True, width="stretch")
st.caption("个股贡献按 Euler 分解（各股之和等于组合）。")
