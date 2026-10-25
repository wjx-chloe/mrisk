import pandas as pd
import streamlit as st

from mrisk.dashboard import charts
from mrisk.dashboard import queries as q
from mrisk.dashboard import transforms as tf
from mrisk.dashboard.common import FOCUS_BG, adopt_query_param, ctx, jumped_from, pct, wan, write_query_param

c = ctx()
st.header(f"压力测试 · {c.date:%Y-%m-%d}")
banner = st.empty()
st.caption("损失以占组合市值表示，正数为亏损。历史情景：基准日收盘到结束日收盘，买入持有；β 传导：因子冲击按近 250 日的 β 传导到个股；其余不变：只冲击指定的行业或个股。")

sr = q.stress(c.engine, c.pid, c.date)
if sr.empty:
    st.info("该日没有压测结果（可能被 DQ 门禁拦下）。")
    st.stop()

labels = {r.scenario_id: f"{r.shock} · {r.mode}" for r in sr.itertuples()}
adopt_query_param("scenario", "stress_scenario", str, labels)  # a limit link names its scenario
if st.session_state.get("stress_scenario") not in labels:
    st.session_state["stress_scenario"] = sr["scenario_id"].iloc[0]
chosen = st.session_state["stress_scenario"]  # the drill-down's scenario, highlighted in the ranking

st.subheader("重大损失情景")
n = st.slider("显示前 N 个", 1, len(sr), min(10, len(sr)), key="stress_top_n")
top = sr.head(n)
st.dataframe(pd.DataFrame({
    "排名": range(1, len(top) + 1),
    "冲击": top["shock"],
    "模式": top["mode"],
    "损失占市值": [pct(v) for v in top["loss_pct"]],
    "损失": [wan(-v) for v in top["pnl"]],
    "最大单日损失": [pct(v) if pd.notna(v) else "—" for v in top["max_daily_pct"]],
    "其中代理贡献": [pct(v) if m == tf.MODE_HIST else "—" for v, m in zip(top["proxy_pct"], top["mode"])],
    "情景 ID": top["scenario_id"],
}).style.apply(lambda r: [FOCUS_BG if r["情景 ID"] == chosen else ""] * len(r), axis=1),
    hide_index=True, width="stretch")
st.caption("同一个冲击的 β 传导版和其余不变版是同一冲击的两种算法，不是两个情景；下表并排对比。高亮行是下方「情景下钻」所选的情景。")

st.subheader("β 传导 vs 其余不变")
pair = tf.side_by_side(sr)
if pair.empty:
    st.caption("没有同时有两个版本的冲击。")
else:
    st.dataframe(pd.DataFrame({
        "冲击": pair["shock"],
        "β 传导（基于近 250 日相关性）": [pct(v) for v in pair["beta"]],
        "其余不变": [pct(v) for v in pair["raw"]],
        "差（β − 其余不变）": [pct(v) for v in pair["gap"]],
    }), hide_index=True, width="stretch")
    st.caption(
        "β 传导是近 250 日相关结构下的条件期望 E[r | 因子 = 冲击]，不是预测；其余不变只算被冲击行业的持仓，"
        "等于行业权重 × 冲击。损失为负表示组合在该冲击下盈利（见 docs/week3b-stress-limits.md §5.5 b）。"
    )

st.subheader("情景下钻")
sid = st.selectbox("情景", list(labels), format_func=labels.get, key="stress_scenario")
write_query_param("scenario", sid)
jumped_from(f"压力测试页：{labels[sid]} 已在下方「情景下钻」选中，排名表中高亮", banner)
ind, inst = q.stress_detail(c.engine, c.pid, c.date, sid)
st.plotly_chart(charts.hbar(ind["industry"], ind["loss_pct"], f"按行业的损失（占组合市值）：{labels[sid]}", ".2%"), width="stretch")

proxied = inst["proxied"]
st.dataframe(pd.DataFrame({
    "代码": inst["instrument_id"], "名称": inst["name"], "行业": inst["industry_l1"],
    "市值": [wan(v) for v in inst["market_value"]],
    "损失占组合市值": ["—" if p else pct(v) for v, p in zip(inst["loss_pct"], proxied)],
    "个股情景收益": ["—" if p else pct(v) for v, p in zip(inst["own_return"], proxied)],
    "说明": ["代理路径：不提供单股结果" if p else "" for p in proxied],
}).style.apply(lambda r: ["color: #898781"] * len(r) if r["说明"] else [""] * len(r), axis=1), hide_index=True, width="stretch")
if proxied.any():
    st.caption(
        f"{int(proxied.sum())} 只股票在该情景期间停牌或未上市，走同行业代理路径。代理平均无偏，但单只股票误差很大"
        "（复牌日残差标准差约 8%，docs/week3b-stress-limits.md §5.5 a），所以不提供单股数字；它们的损失仍计入行业和组合合计。"
    )
