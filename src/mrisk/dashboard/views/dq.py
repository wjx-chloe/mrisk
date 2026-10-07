import pandas as pd
import streamlit as st

from mrisk.dashboard import queries as q
from mrisk.dashboard.common import DQ_LABEL, GATE_LABEL, ctx

c = ctx()
st.header(f"数据质量 · {c.date:%Y-%m-%d}")

dq = q.dq_today(c.engine, c.date)
fails = dq[dq["status"] == "fail"]
a, b, d = st.columns(3)
a.metric("规则", len(dq))
b.metric("失败", len(fails))
d.metric("其中 blocking", int((fails["severity"] == "blocking").sum()))
st.dataframe(pd.DataFrame({
    "规则": dq["rule_id"], "名称": dq["name"], "类别": dq["category"], "级别": dq["severity"],
    "结果": dq["status"].map(DQ_LABEL).fillna("⚪ 未运行"),
    "检查数": dq["n_checked"].astype("Int64"), "失败数": dq["n_failed"].astype("Int64"),
}).style.apply(lambda r: ["background-color: rgba(208, 59, 59, 0.18)"] * len(r) if r["结果"] == DQ_LABEL["fail"] else [""] * len(r), axis=1),
    hide_index=True, width="stretch")
for r in fails.itertuples():
    with st.expander(f"{r.rule_id} 失败明细"):
        st.json(r.detail_json)


@st.cache_data(ttl=600, show_spinner="检查门禁……")
def _gate(_engine, _cfg, url: str, d: pd.Timestamp) -> pd.DataFrame:
    return q.gate_today(_engine, _cfg, d)


st.subheader("门禁状态")
st.caption("每类结果依赖的行情和持仓日期上，blocking 规则是否失败。历史情景取各情景中最差的一个。")
g = _gate(c.engine, c.cfg, str(c.engine.url), c.date)
st.dataframe(pd.DataFrame({"结果": g["结果"], "门禁": g["status"].map(GATE_LABEL), "原因": g["reason"]}), hide_index=True, width="stretch")
