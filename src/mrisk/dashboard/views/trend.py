import pandas as pd
import streamlit as st

from mrisk.dashboard import charts
from mrisk.dashboard import queries as q
from mrisk.dashboard.common import conf_label, ctx

c = ctx()
st.header("趋势")

RANGES = {"近 1 月": 1, "近 3 月": 3, "近半年": 6, "近 1 年": 12}
span = st.radio("区间", list(RANGES), index=1, horizontal=True, key="trend_range")
start = c.date - pd.DateOffset(months=RANGES[span])
df = q.portfolio_risk(c.engine, c.pid, c.conf, start, c.date)
st.caption(f"{df.index.min():%Y-%m-%d} 至 {c.date:%Y-%m-%d}，{len(df)} 个持仓日，置信度 {conf_label(c.conf)}")

labels = {"var_hs": "VaR 历史模拟", "var_param": "VaR 参数法", "es_hs": "ES 历史模拟"}
st.plotly_chart(charts.lines(df, labels, f"VaR / ES（{conf_label(c.conf)}，元）", "元", ",.0f"), width="stretch")
st.plotly_chart(charts.lines(df, {"market_value": "组合市值"}, "组合市值（元）", "元", ",.0f"), width="stretch")
# risk relative to size, on its own chart rather than a second axis
pct_df = df[list(labels)].div(df["market_value"], axis=0)
st.plotly_chart(charts.lines(pct_df, labels, "VaR / ES 占市值", y_fmt=".2%"), width="stretch")

with st.expander("数据表"):
    st.dataframe(df.rename(columns={"market_value": "市值", **labels}).rename_axis("日期"), width="stretch")
