"""mrisk dashboard entry: `uv run streamlit run src/mrisk/dashboard/app.py`.
Reads the database named in the config (MRISK_CONFIG, default config.yaml) read-only; never writes."""

import streamlit as st

from mrisk.config import database_url, load_config
from mrisk.dashboard import queries as q
from mrisk.dashboard.common import Ctx, conf_label

st.set_page_config(page_title="mrisk 风险看板", page_icon="📉", layout="wide")


@st.cache_resource
def _engine(url: str):
    return q.readonly_engine(url)


cfg = load_config()
engine = _engine(database_url(cfg))
pid = cfg["portfolio"]["portfolio_id"]
complete = q.complete_dates(engine, cfg)

st.sidebar.title("mrisk")
st.sidebar.caption(f"组合 {pid} · {cfg['portfolio']['name']}")
if complete.empty:
    st.error("库里还没有完整的风险结果（组合层 VaR 的每个置信度 + 每条 active 限额的检查）。先跑 mrisk risk / limits。")
    st.stop()

options = list(complete[::-1])
date = st.sidebar.selectbox("日期", options, index=0, format_func=lambda d: d.strftime("%Y-%m-%d"), key="date",
                            help="只列出有完整风险结果的持仓日：组合层 VaR 在每个置信度上都有，且每条 active 限额都有检查行")
conf = st.sidebar.radio("置信度", sorted(cfg["confidence"], reverse=True), format_func=conf_label, key="conf")

note = q.banner(engine, cfg, complete)
if note:
    (st.error if note["level"] == "error" else st.warning)(note["text"])

st.session_state["ctx"] = Ctx(engine, cfg, pid, date, conf, complete)

pages = [
    st.Page("views/overview.py", title="总览", icon="🏠", default=True),
    st.Page("views/trend.py", title="趋势", icon="📈"),
    st.Page("views/exposure.py", title="敞口", icon="🧭"),
    st.Page("views/backtest.py", title="回测", icon="🎯"),
    st.Page("views/attribution.py", title="归因", icon="🧩"),
    st.Page("views/stress.py", title="压力测试", icon="🌪️"),
    st.Page("views/limits.py", title="限额", icon="🚦"),
    st.Page("views/dq.py", title="数据质量", icon="🧪"),
]
st.navigation(pages).run()
