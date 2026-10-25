import pandas as pd
import streamlit as st

from mrisk.dashboard import charts
from mrisk.dashboard import queries as q
from mrisk.dashboard.common import FOCUS_BG, ctx, focus_from_query, jumped_from, pct, wan

c = ctx()
st.header(f"敞口 · {c.date:%Y-%m-%d}")
jumped_from("敞口页")
focus_inst, focus_ind = focus_from_query("inst"), focus_from_query("ind")

ind, inst = q.exposure(c.engine, c.pid, c.date, c.conf)
a, b, d = st.columns(3)
a.metric("持仓只数", len(inst))
b.metric("行业数", len(ind))
d.metric("前十大占比", pct(inst["weight"].head(10).sum(), 1))

if focus_ind:
    shown = ind[ind["industry"].isin(focus_ind)]
    st.caption("高亮行业：" + "、".join(f"{r.industry} {pct(r.weight)}" for r in shown.itertuples()))
st.plotly_chart(charts.hbar(ind["industry"], ind["weight"], "行业分布（占组合市值）", highlight=focus_ind), width="stretch")


def _focus_rows(df: pd.DataFrame, keys: list[str], col: str):
    return df.style.apply(lambda r: [FOCUS_BG if r[col] in keys else ""] * len(r), axis=1)


st.subheader("个股 top 10")
top = inst.head(10)
extra = inst[inst["instrument_id"].isin(focus_inst) & ~inst["instrument_id"].isin(top["instrument_id"])]
top = pd.concat([top, extra])  # a highlighted name outside the top 10 is shown below it
if focus_inst:
    st.caption("高亮个股：" + "、".join(f"{r.instrument_id} {r.name} {pct(r.weight)}" for r in inst[inst["instrument_id"].isin(focus_inst)].itertuples()))
st.dataframe(_focus_rows(pd.DataFrame({
    "代码": top["instrument_id"], "名称": top["name"], "行业": top["industry"],
    "市值": [wan(v) for v in top["market_value"]], "占比": [pct(w) for w in top["weight"]],
}), focus_inst, "代码"), hide_index=True, width="stretch")

with st.expander("全部行业", expanded=bool(focus_ind)):
    st.dataframe(_focus_rows(pd.DataFrame({"行业": ind["industry"], "市值": [wan(v) for v in ind["market_value"]],
                                           "占比": [pct(w) for w in ind["weight"]]}), focus_ind, "行业"), hide_index=True, width="stretch")
