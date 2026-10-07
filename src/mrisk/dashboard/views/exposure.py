import pandas as pd
import streamlit as st

from mrisk.dashboard import charts
from mrisk.dashboard import queries as q
from mrisk.dashboard.common import ctx, pct, wan

c = ctx()
st.header(f"敞口 · {c.date:%Y-%m-%d}")

ind, inst = q.exposure(c.engine, c.pid, c.date, c.conf)
a, b, d = st.columns(3)
a.metric("持仓只数", len(inst))
b.metric("行业数", len(ind))
d.metric("前十大占比", pct(inst["weight"].head(10).sum(), 1))

st.plotly_chart(charts.hbar(ind["industry"], ind["weight"], "行业分布（占组合市值）"), width="stretch")

st.subheader("个股 top 10")
top = inst.head(10)
st.dataframe(pd.DataFrame({
    "代码": top["instrument_id"], "名称": top["name"], "行业": top["industry"],
    "市值": [wan(v) for v in top["market_value"]], "占比": [pct(w) for w in top["weight"]],
}), hide_index=True, width="stretch")

with st.expander("全部行业"):
    st.dataframe(pd.DataFrame({"行业": ind["industry"], "市值": [wan(v) for v in ind["market_value"]],
                               "占比": [pct(w) for w in ind["weight"]]}), hide_index=True, width="stretch")
