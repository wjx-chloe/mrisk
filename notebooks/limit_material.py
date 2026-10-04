"""Material for setting the week-3 VaR/ES limits. Read-only: reads risk_result, positions and prices from the db,
prints markdown tables and writes figures to docs/figures/. Does not touch the engine or write to the db.

    uv run --with matplotlib python notebooks/limit_material.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from mrisk.config import database_url, load_config
from mrisk.db import schema as s
from mrisk.db.repo import get_engine, read_frame
from mrisk.engine.returns import returns_window
from mrisk.engine.risk import compute_risk
from mrisk.engine.valuation import market_value
from mrisk.jobs import etl

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
EVENTS = [pd.Timestamp("2024-10-09"), pd.Timestamp("2025-04-07")]
LIMITS = {"var": 0.03, "es": 0.04}  # SPEC §6.2 presets, % of MV
QS = [0.50, 0.75, 0.90, 0.95]

# reference palette (dataviz skill), light mode
BLUE, ORANGE, INK, INK2, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"


def pct(x, d=2):
    return f"{x:.{d}%}"


def load(engine, cfg):
    rr = read_frame(engine, s.risk_result, level="portfolio")
    w = rr.pivot(index="trade_date", columns="confidence", values=["market_value", "var_hs", "var_param", "es_hs"])
    mv = w["market_value"][0.99]
    out = pd.DataFrame(
        {
            "var99": w["var_hs"][0.99] / mv,
            "es99": w["es_hs"][0.99] / mv,
            "var95": w["var_hs"][0.95] / mv,
            "es95": w["es_hs"][0.95] / mv,
            "pvar99": w["var_param"][0.99] / mv,
        }
    )
    pos = read_frame(engine, s.position, out.index.min(), out.index.max(), portfolio_id=cfg["portfolio"]["portfolio_id"])
    cal = etl.trading_days(engine)
    start = cal[cal.get_loc(out.index.min()) - cfg["lookback_days"]]
    prices = read_frame(engine, s.daily_price, start, out.index.max(), instrument_id=sorted(pos["instrument_id"].unique()))
    inst = read_frame(engine, s.instrument, date_col="list_date", columns=["instrument_id", "industry_l1"])
    return out, pos, prices, inst, cal


def window_flags(dates: pd.DatetimeIndex, cal: pd.DatetimeIndex, lookback: int) -> pd.DataFrame:
    """Whether each event's return is inside the lookback returns ending on t (cal index i_t-lookback+1 .. i_t)."""
    i = cal.get_indexer(dates)
    return pd.DataFrame(
        {e.date().isoformat(): (i >= cal.get_loc(e)) & (i <= cal.get_loc(e) + lookback - 1) for e in EVENTS}, index=dates
    )


def tail_composition(dates, pos, prices, inst, lookback, kind):
    """Per as-of date: the 3 worst window P&L days (they set VaR99 and ES99 when T = 250) and each event's rank."""
    wide = prices.pivot(index="trade_date", columns="instrument_id")
    ind = inst.set_index("instrument_id")["industry_l1"]
    rows = []
    for t in dates:
        p = pos.loc[pos["trade_date"] == t, ["instrument_id", "quantity"]]
        ids = sorted(p["instrument_id"])
        R, _ = returns_window(wide["adj_close"][ids], wide["is_suspended"][ids], t, lookback, kind)
        mv = market_value(p, wide["close"].loc[t], ind)
        pnl = pd.Series(R[mv["instrument_id"]].to_numpy() @ mv["market_value"].to_numpy(), index=R.index)
        pnl = pnl / mv["market_value"].sum()
        order = pnl.sort_values()
        row = {"trade_date": t, "worst3": [d.date().isoformat() for d in order.index[:3]], "worst3_ret": order.iloc[:3].tolist()}
        for e in EVENTS:
            row[f"rank_{e.date()}"] = int(order.index.get_loc(e)) + 1 if e in order.index else None
            row[f"ret_{e.date()}"] = float(pnl[e]) if e in pnl.index else None
        rows.append(row)
    return pd.DataFrame(rows).set_index("trade_date")


def exit_cliff(r, pos, prices, inst, cal, cfg, event):
    """Around the day `event` leaves the window: observed ratios, and the pure window effect on the exit day
    (yesterday's positions valued on today's window vs yesterday's window)."""
    lookback = cfg["lookback_days"]
    x = cal[cal.get_loc(event) + lookback]  # first as-of date whose window no longer holds the event return
    prev = cal[cal.get_loc(x) - 1]
    p_prev = pos.loc[pos["trade_date"] == prev, ["instrument_id", "quantity"]]

    def ratios(asof):
        t = compute_risk(p_prev, prices, inst, asof, "P", [0.99], lookback, cfg["return_type"]).table
        port = t[t["level"] == "portfolio"].iloc[0]
        return port["var_hs"] / port["market_value"], port["es_hs"] / port["market_value"]

    v0, e0 = ratios(prev)
    v1, e1 = ratios(x)
    d = r[["var99", "es99"]].diff()
    near = r.loc[cal[cal.get_loc(x) - 3] : cal[cal.get_loc(x) + 3], ["var99", "es99"]].assign(
        d_var99=d["var99"], d_es99=d["es99"]
    )
    rank_v = (d["var99"].abs() >= abs(d.at[x, "var99"])).sum()
    rank_e = (d["es99"].abs() >= abs(d.at[x, "es99"])).sum()
    return x, near, {"window_dv": v1 - v0, "window_de": e1 - e0, "rank_dv": int(rank_v), "rank_de": int(rank_e), "n": int(d["var99"].notna().sum())}


def figures(r, cal, lookback, exits):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt

    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )
    FIG.mkdir(parents=True, exist_ok=True)

    def style(ax):
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.yaxis.set_major_formatter(mt.PercentFormatter(1.0, decimals=1))

    # fig 1: one measure per panel, its preset limit, event entry/exit markers
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    for ax, col, color, lim, title in [
        (axes[0], "var99", BLUE, LIMITS["var"], "组合历史模拟 VaR99 / 市值"),
        (axes[1], "es99", ORANGE, LIMITS["es"], "组合历史模拟 ES99 / 市值"),
    ]:
        style(ax)
        ax.plot(r.index, r[col], color=color, lw=1.6)
        ax.axhline(lim, color=INK2, lw=1, ls=(0, (4, 3)))
        ax.annotate(f"预置限额 {lim:.0%}", (r.index[0], lim), xytext=(2, 4), textcoords="offset points", color=INK2, fontsize=8)
        for e, x in zip(EVENTS, exits):
            for d, lab in ((e, f"{e:%m-%d} 进入窗口"), (x, f"{e:%Y-%m-%d} 移出")):
                ax.axvline(d, color=MUTED, lw=0.9, ls=":")
                if ax is axes[0]:
                    ax.annotate(lab, (d, 0), xycoords=("data", "axes fraction"), xytext=(3, 4), textcoords="offset points", color=INK2, fontsize=7.5)
        ax.set_title(title, loc="left", color=INK, fontsize=10)
        ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(FIG / "limit_ratio_timeseries.png", dpi=150)
    plt.close(fig)

    # fig 2: small multiples around each exit, VaR99 and ES99 on one % axis
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, e, x in zip(axes, EVENTS, exits):
        style(ax)
        i = cal.get_loc(x)
        seg = r.loc[cal[i - 15] : cal[min(i + 15, len(cal) - 1)]]
        ax.plot(seg.index, seg["var99"], color=BLUE, lw=1.6, marker="o", ms=3)
        ax.plot(seg.index, seg["es99"], color=ORANGE, lw=1.6, marker="o", ms=3)
        ax.axvline(x, color=MUTED, lw=0.9, ls=":")
        ax.annotate("VaR99", (seg.index[-1], seg["var99"].iloc[-1]), xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=8, va="center")
        ax.annotate("ES99", (seg.index[-1], seg["es99"].iloc[-1]), xytext=(4, 0), textcoords="offset points", color=INK2, fontsize=8, va="center")
        ax.set_title(f"{e:%Y-%m-%d} 的收益于 {x:%Y-%m-%d} 移出窗口", loc="left", color=INK, fontsize=10)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d"))
        ax.set_ylim(0, None)
    axes[0].legend(["VaR99 / 市值", "ES99 / 市值"], frameon=False, loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "limit_exit_zoom.png", dpi=150)
    plt.close(fig)


def main() -> int:
    cfg = load_config()
    engine = get_engine(database_url(cfg))
    lookback = cfg["lookback_days"]
    r, pos, prices, inst, cal = load(engine, cfg)
    print(f"## 样本\n{len(r)} 个交易日，{r.index.min().date()} .. {r.index.max().date()}\n")

    q = r[["var99", "es99", "var95", "es95", "pvar99"]].quantile(QS)
    print("## 分位数（占市值）\n\n| 指标 | " + " | ".join(f"P{int(x * 100)}" for x in QS) + " | 最小 | 最大 |\n|---|" + "---|" * (len(QS) + 2))
    for col in q.columns:
        print(f"| {col} | " + " | ".join(pct(v) for v in q[col]) + f" | {pct(r[col].min())} | {pct(r[col].max())} |")

    flags = window_flags(r.index, cal, lookback)
    a, b = (flags[c] for c in flags.columns)
    before = r.index < EVENTS[0]
    group = np.select(
        [~a & ~b & before, a & ~b, a & b, ~a & b, ~a & ~b & ~before],
        ["都不在（事件前）", "仅 2024-10-09", "两者都在", "仅 2025-04-07", "都不在（两者都已移出）"],
        "?",
    )
    g = r.assign(group=group, over_var=r["var99"] > LIMITS["var"], over_es=r["es99"] > LIMITS["es"])
    print("\n## 超限是否集中在事件窗口内\n\n| 窗口内含 | 日期范围 | 天数 | VaR99>3% 天数（占比） | ES99>4% 天数（占比） | VaR99 中位数 | ES99 中位数 |\n|---|---|---|---|---|---|---|")
    for name in ["都不在（事件前）", "仅 2024-10-09", "两者都在", "仅 2025-04-07", "都不在（两者都已移出）"]:
        x = g[g["group"] == name]
        if x.empty:
            continue
        print(
            f"| {name} | {x.index.min().date()} .. {x.index.max().date()} | {len(x)} | {x['over_var'].sum()}（{x['over_var'].mean():.0%}） | "
            f"{x['over_es'].sum()}（{x['over_es'].mean():.0%}） | {pct(x['var99'].median())} | {pct(x['es99'].median())} |"
        )
    print(f"| 合计 | | {len(g)} | {g['over_var'].sum()}（{g['over_var'].mean():.0%}） | {g['over_es'].sum()}（{g['over_es'].mean():.0%}） | {pct(g['var99'].median())} | {pct(g['es99'].median())} |")

    tails = tail_composition(r.index, pos, prices, inst, lookback, cfg["return_type"])
    print("\n## 窗口尾部构成（每个 as-of 日按当日持仓重估的窗口损益，最差 3 天）\n")
    worst = tails["worst3"].explode().value_counts()
    print("| 日期 | 出现在最差 3 天的 as-of 天数 |\n|---|---|")
    for d, n in worst.head(8).items():
        print(f"| {d} | {n} |")
    for e in EVENTS:
        rk = tails[f"rank_{e.date()}"].dropna()
        ret = tails[f"ret_{e.date()}"].dropna()
        print(f"\n{e.date()}：在窗口内 {len(rk)} 天，在窗口中的损益排名 中位数 {rk.median():.0f}（1 = 最差），"
              f"排第 1 的 {int((rk == 1).sum())} 天，前 3 的 {int((rk <= 3).sum())} 天；当天组合收益（按各 as-of 持仓）{ret.min():.2%} .. {ret.max():.2%}")

    exits = []
    for e in EVENTS:
        x, near, eff = exit_cliff(r, pos, prices, inst, cal, cfg, e)
        exits.append(x)
        print(f"\n## {e.date()} 移出窗口：{x.date()}\n\n| 日期 | VaR99 | ES99 | ΔVaR99 | ΔES99 |\n|---|---|---|---|---|")
        for d, row in near.iterrows():
            print(f"| {d.date()}{' ←' if d == x else ''} | {pct(row['var99'])} | {pct(row['es99'])} | {row['d_var99']:+.2%} | {row['d_es99']:+.2%} |")
        print(
            f"\n持仓固定为前一日、只换窗口：ΔVaR99 {eff['window_dv']:+.2%}，ΔES99 {eff['window_de']:+.2%}。"
            f"当日 |ΔVaR99| 在 {eff['n']} 个日变动中排第 {eff['rank_dv']}，|ΔES99| 排第 {eff['rank_de']}。"
        )

    figures(r, cal, lookback, exits)
    print(f"\nfigures written to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
