import pandas as pd
import streamlit as st

from mrisk.dashboard import queries as q
from mrisk.dashboard.common import GATE_LABEL, LIMIT_LABEL, conf_label, ctx, pct, status_dataframe, wan

c = ctx()
st.header(f"总览 · {c.date:%Y-%m-%d}")


def _day(t) -> str:
    return "—" if t is None else f"{t:%Y-%m-%d}"


ds = q.data_status(c.engine, c.cfg, c.date)
dq_text = f"🔴 {ds['dq_fail']} 条失败（blocking {ds['dq_fail_blocking']}）" if ds["dq_fail"] else "🟢 无失败"
st.markdown(
    f"**数据状态** · 行情截至 {_day(ds['price_end'])} · 持仓截至 {_day(ds['position_end'])} · 来源 {ds['source']} · "
    f"门禁 {GATE_LABEL[ds['gate']]} · DQ {ds['dq_run']} 条 {dq_text}",
    help="日期取自库里的行情表和持仓表；来源取自配置的 data.source（库本身不记录行情来源）；门禁是所选日期 VaR / ES "
         "依赖的行情和持仓上 blocking 规则的判断；DQ 是所选日期各规则最近一次运行的结果。明细见「数据质量」页。",
)
if ds["gate"] != "ok":
    st.caption(f"门禁原因：{ds['gate_reason']}")

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
status_dataframe(table)
counts = lim["status"].value_counts()
st.caption(" · ".join(f"{LIMIT_LABEL[k]} {counts.get(k, 0)}" for k in ("breach", "warn", "ok", "no_data")))
