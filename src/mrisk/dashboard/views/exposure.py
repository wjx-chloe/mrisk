import pandas as pd
import streamlit as st

from mrisk.dashboard import charts
from mrisk.dashboard import queries as q
from mrisk.dashboard import transforms as tf
from mrisk.dashboard.common import FOCUS_BG, conf_label, ctx, focus_from_query, jumped_from, pct, wan

c = ctx()
st.header(f"敞口 · {c.date:%Y-%m-%d}")
jumped_from("敞口页")
focus_inst, focus_ind = focus_from_query("inst"), focus_from_query("ind")

ind, inst = q.exposure(c.engine, c.pid, c.date, c.conf)
a, b, d = st.columns(3)
a.metric("持仓只数", len(inst))
b.metric("行业数", len(ind))
d.metric("前十大占比", pct(inst["weight"].head(10).sum(), 1))

# ES risk contribution: where the portfolio's tail losses come from, next to where its money is
T = c.cfg["lookback_days"]
n_tail = tf.tail_days(c.conf, T)
widest = min(c.cfg["confidence"])  # the confidence with the most tail days
if c.conf != widest:
    st.warning(f"尾部仅约 {n_tail} 个场景，贡献随单日剧烈变化，建议以 {conf_label(widest)}（尾部约 {tf.tail_days(widest, T)} 天）为准。", icon="⚠️")
has_contrib = inst["es_contrib"].notna().all()
if not has_contrib:
    st.info("该日没有 ES 风险贡献：升级旧库时，库里存的风险数与重算结果不一致，没有补算（见 `mrisk upgrade-db` 的日志）。"
            "对该日重跑 `uv run mrisk risk --date …` 即可。")

if focus_ind:
    shown = ind[ind["industry"].isin(focus_ind)]
    st.caption("高亮行业：" + "、".join(f"{r.industry} {pct(r.weight)}" for r in shown.itertuples()))
st.plotly_chart(charts.hbar_pair(ind["industry"], ind["weight"], ind["es_share"], ("市值占比", f"ES 贡献占比（{conf_label(c.conf)}）"),
                                 "行业：市值占比 vs ES 风险贡献占比", highlight=focus_ind), width="stretch")
st.caption(
    f"ES 风险贡献（历史模拟 {conf_label(c.conf)}，Euler 分解）：组合在近 {T} 日里亏损最重的约 {n_tail} 个场景日"
    "（组合损益低于 −VaR 的那些天，ES 就是它们的平均损失）上，该行业或个股的平均损失。各行业之和、各股之和都等于组合 ES。"
    "贡献占比高于市值占比，说明它在组合最坏的日子里亏得比它的仓位更多；为负表示它在这些天平均盈利，对组合有对冲作用。"
    "VaR 不做同类分解：历史模拟 VaR 只取决于分位数插值的两个场景日，太不稳定。"
)


def _focus_rows(df: pd.DataFrame, keys: list[str], col: str):
    return df.style.apply(lambda r: [FOCUS_BG if r[col] in keys else ""] * len(r), axis=1)


def _names_table(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "代码": rows["instrument_id"], "名称": rows["name"], "行业": rows["industry"],
        "市值": [wan(v) for v in rows["market_value"]], "市值占比": [pct(w) for w in rows["weight"]],
        "ES 贡献": ["—" if pd.isna(v) else wan(v) for v in rows["es_contrib"]],
        "ES 贡献占比": ["—" if pd.isna(v) else pct(v) for v in rows["es_share"]],
    })


st.subheader("个股 top 10（按市值）")
top = inst.head(10)
extra = inst[inst["instrument_id"].isin(focus_inst) & ~inst["instrument_id"].isin(top["instrument_id"])]
top = pd.concat([top, extra])  # a highlighted name outside the top 10 is shown below it
if focus_inst:
    st.caption("高亮个股：" + "、".join(f"{r.instrument_id} {r.name} {pct(r.weight)}" for r in inst[inst["instrument_id"].isin(focus_inst)].itertuples()))
st.dataframe(_focus_rows(_names_table(top), focus_inst, "代码"), hide_index=True, width="stretch")

with st.expander("全部个股（按 ES 贡献排序）", expanded=False):
    st.dataframe(_focus_rows(_names_table(inst.sort_values("es_contrib", ascending=False)), focus_inst, "代码"),
                 hide_index=True, width="stretch")

with st.expander("全部行业", expanded=bool(focus_ind)):
    st.dataframe(_focus_rows(pd.DataFrame({
        "行业": ind["industry"], "市值": [wan(v) for v in ind["market_value"]], "市值占比": [pct(w) for w in ind["weight"]],
        "ES 贡献": ["—" if pd.isna(v) else wan(v) for v in ind["es_contrib"]],
        "ES 贡献占比": ["—" if pd.isna(v) else pct(v) for v in ind["es_share"]],
    }), focus_ind, "行业"), hide_index=True, width="stretch")
