import pandas as pd
import streamlit as st

from mrisk.dashboard import charts
from mrisk.dashboard import queries as q
from mrisk.dashboard import transforms as tf
from mrisk.dashboard.common import ZONE_LABEL, conf_label, ctx
from mrisk.engine.backtest import CRITICAL_5PCT, zone_bounds

c = ctx()
W = c.cfg["backtest_window_days"]
p = 1 - c.conf
st.header(f"回测 · 截至 {c.date:%Y-%m-%d}")
st.caption(
    f"t 日的 VaR 对比 t → t+1 的假设损益，例外 = 损益 < −VaR。滚动 {W} 个观测，截至该日实现的损益。"
    f"置信度 {conf_label(c.conf)}，期望例外 {W * p:.1f} 次。"
)

br, obs = q.backtest(c.engine, c.cfg, c.date, c.conf)
if br.empty:
    st.info(f"截至该日没有回测结果：观测不足 {W} 个（组合刚开始），或窗口内有被 DQ 拦下的缺失观测。")
    st.stop()

green, yellow = zone_bounds(p, W)
st.caption(f"红绿灯边界（二项分布规则，{W} 个观测，p = {p:.2f}）：≤ {green} 绿，{green + 1}–{yellow} 黄，≥ {yellow + 1} 红")

NAMES = {"hs": "历史模拟", "param": "参数法"}
for col, m in zip(st.columns(len(br)), br.index):
    r = br.loc[m]
    col.metric(f"{NAMES[m]} 例外次数", int(r["n_exceptions"]), help=f"{int(r['n_obs'])} 个观测")
    col.markdown(f"红绿灯：**{ZONE_LABEL.get(r['zone'], '—')}**")

st.dataframe(pd.DataFrame({
    "方法": [NAMES[m] for m in br.index],
    "观测": br["n_obs"].astype(int).to_numpy(),
    "例外": br["n_exceptions"].astype(int).to_numpy(),
    "红绿灯": br["zone"].map(ZONE_LABEL).to_numpy(),
    "Kupiec LR": br["lr_pof"].round(3).to_numpy(),
    "Kupiec p 值": br["p_value_pof"].round(4).to_numpy(),
    "Kupiec（5%）": ["拒绝" if v > CRITICAL_5PCT else "不拒绝" for v in br["lr_pof"]],
    "独立性 LR": br["lr_ind"].round(3).to_numpy(),
    "独立性 p 值": br["p_value_ind"].round(4).to_numpy(),
    "独立性（5%）": ["拒绝" if v > CRITICAL_5PCT else "不拒绝" for v in br["lr_ind"]],
}), hide_index=True, width="stretch")

method = st.radio("图表方法", list(br.index), format_func=NAMES.get, horizontal=True, key="bt_method")
col = f"var_{method}"
st.plotly_chart(charts.backtest(obs, col, f"假设损益与 −VaR（{NAMES[method]}，{conf_label(c.conf)}）"), width="stretch")
hit = (obs["pnl"] < -obs[col]).to_numpy()
st.plotly_chart(charts.exception_calendar(tf.calendar_grid(pd.DatetimeIndex(obs["pnl_date"]), hit), "例外日历（按损益实现日）"), width="stretch")
with st.expander("例外明细"):
    ex = obs[hit]
    st.dataframe(pd.DataFrame({"持仓日": ex["position_date"].dt.date, "实现日": ex["pnl_date"].dt.date,
                               "损益": ex["pnl"].round(0), "VaR": ex[col].round(0)}), hide_index=True, width="stretch")
