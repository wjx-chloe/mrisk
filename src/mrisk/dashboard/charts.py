"""Plotly figures for the dashboard: frames in, go.Figure out. Colours follow one palette: categorical slots in a
fixed order for identity, a reserved status set for state (always beside a text label), muted ink for context.
Every chart has one y-axis; measures on different scales get separate charts."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
MUTED = "#898781"
NEUTRAL = "#c3c2b7"
FADED = 0.3  # opacity of the bars outside a highlight
ZONE_COLOR = {"green": STATUS["good"], "yellow": STATUS["warning"], "red": STATUS["critical"]}


def _layout(fig: go.Figure, title: str, y_title: str = "", y_fmt: str | None = None, height: int = 360) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left"},
        height=height,
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1},
    )
    fig.update_yaxes(title_text=y_title, tickformat=y_fmt, zeroline=True, zerolinewidth=1)
    return fig


def lines(df: pd.DataFrame, labels: dict[str, str], title: str, y_title: str = "", y_fmt: str | None = None) -> go.Figure:
    """One line per column in `labels` (column -> legend name), colours by slot in the order given."""
    fig = go.Figure()
    for i, (col, name) in enumerate(labels.items()):
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=name, mode="lines", line={"width": 2, "color": SERIES[i]},
                                 hovertemplate="%{y:" + (y_fmt or ",.0f") + "}<extra>" + name + "</extra>"))
    return _layout(fig, title, y_title, y_fmt)


def _emphasis(labels: list, highlight) -> float | list[float]:
    """Marker opacity: everything full, or with a highlight the highlighted labels full and the rest faded."""
    if not highlight:
        return 1.0
    return [1.0 if x in set(highlight) else FADED for x in labels]


def hbar(labels, values, title: str, x_fmt: str = ".1%", color: str = SERIES[0], text_fmt: str | None = None,
         highlight=None) -> go.Figure:
    """Horizontal bars, first item on top. Negative values extend left of the axis. highlight: labels kept at full
    strength while the other bars fade (a limit link's focus)."""
    labels, values = list(labels), np.asarray(values, dtype=float)
    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h", marker={"color": color, "cornerradius": 4, "opacity": _emphasis(labels, highlight)},
        texttemplate="%{x:" + (text_fmt or x_fmt) + "}", textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:" + x_fmt + "}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    lo, hi = min(0.0, float(values.min(initial=0.0))), max(0.0, float(values.max(initial=0.0)))
    pad = 0.18 * (hi - lo or 1.0)  # room for the outside labels on either side
    fig.update_xaxes(tickformat=x_fmt, range=[lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0)])
    return _layout(fig, title, height=max(240, 32 * len(labels) + 80))


def hbar_pair(labels, a, b, names: tuple[str, str], title: str, x_fmt: str = ".1%", highlight=None) -> go.Figure:
    """Two measures side by side per label (grouped horizontal bars, first label on top): slot 1 for a, slot 2 for b,
    same units on one axis. highlight fades the other labels' bars, as in hbar."""
    labels = list(labels)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    fig = go.Figure()
    for i, (v, name) in enumerate(zip((a, b), names)):
        fig.add_trace(go.Bar(
            y=labels, x=v, name=name, orientation="h",
            marker={"color": SERIES[i], "cornerradius": 3, "opacity": _emphasis(labels, highlight)},
            texttemplate="%{x:" + x_fmt + "}", textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:" + x_fmt + "}<extra>" + name + "</extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    fig.update_yaxes(autorange="reversed")
    both = np.concatenate([a, b])
    both = both[np.isfinite(both)]
    lo, hi = min(0.0, float(both.min(initial=0.0))), max(0.0, float(both.max(initial=0.0)))
    pad = 0.18 * (hi - lo or 1.0)
    fig.update_xaxes(tickformat=x_fmt, range=[lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0)])
    return _layout(fig, title, height=max(280, 52 * len(labels) + 100))


def backtest(obs: pd.DataFrame, var_col: str, title: str) -> go.Figure:
    """Daily hypothetical P&L (bars) against -VaR (line); exceptions (pnl < -VaR) marked and named in the legend."""
    x = obs["pnl_date"]
    hit = obs["pnl"] < -obs[var_col]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=obs["pnl"], name="假设损益", marker={"color": NEUTRAL}, hovertemplate="%{y:,.0f}<extra>假设损益</extra>"))
    fig.add_trace(go.Scatter(x=x, y=-obs[var_col], name="−VaR", mode="lines", line={"width": 2, "color": SERIES[0]},
                             hovertemplate="%{y:,.0f}<extra>−VaR</extra>"))
    fig.add_trace(go.Scatter(x=x[hit], y=obs["pnl"][hit], name=f"例外（{int(hit.sum())} 次）", mode="markers",
                             marker={"size": 9, "color": STATUS["critical"], "symbol": "x"}, hovertemplate="%{y:,.0f}<extra>例外</extra>"))
    return _layout(fig, title, "元", ",.0f")


def exception_calendar(grid: pd.DataFrame, title: str) -> go.Figure:
    """Weeks across, weekdays down; a cell per observation, exception cells in the critical colour."""
    wk = sorted(grid["week"].unique())
    z = np.full((5, len(wk)), np.nan)
    text = np.full((5, len(wk)), "", dtype=object)
    col = {w: i for i, w in enumerate(wk)}
    for r in grid.itertuples():
        z[r.weekday, col[r.week]] = 1.0 if r.hit else 0.0
        text[r.weekday, col[r.week]] = f"{r.date.date()}{' 例外' if r.hit else ''}"
    fig = go.Figure(go.Heatmap(
        z=z, x=[pd.Timestamp(w) for w in wk], y=["周一", "周二", "周三", "周四", "周五"], text=text,
        colorscale=[[0, "#e1e0d9"], [0.5, "#e1e0d9"], [0.5, STATUS["critical"]], [1, STATUS["critical"]]],
        zmin=0, zmax=1, showscale=False, xgap=2, ygap=2, hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title, height=220)


def waterfall(effects: dict[str, float], title: str) -> go.Figure:
    """The day-on-day change built up from its effects, ending in the total change. Starts from 0 rather than from
    yesterday's level, which would dwarf the steps. Rises and falls in two categorical colours."""
    names = [*effects, "总变化"]
    fig = go.Figure(go.Waterfall(
        x=names, y=[*effects.values(), 0.0], measure=[*["relative"] * len(effects), "total"],
        increasing={"marker": {"color": SERIES[1]}}, decreasing={"marker": {"color": SERIES[0]}},
        totals={"marker": {"color": MUTED}}, connector={"line": {"color": NEUTRAL, "width": 1}},
        text=[f"{v:+,.0f}" for v in effects.values()] + [f"{sum(effects.values()):+,.0f}"],
        textposition="outside", hovertemplate="%{x}: %{text}<extra></extra>",
    ))
    return _layout(fig, title, "元", ",.0f")


def limit_history(hist: pd.DataFrame, threshold: float, warn: float, title: str, fmt: str = ".1%") -> go.Figure:
    """A rule's checked value over time against its breach and warn lines; breach days marked."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["trade_date"], y=hist["actual_value"], name="检查值", mode="lines",
                             line={"width": 2, "color": SERIES[0]}, hovertemplate="%{y:" + fmt + "}<extra>检查值</extra>"))
    b = hist[hist["status"] == "breach"]
    fig.add_trace(go.Scatter(x=b["trade_date"], y=b["actual_value"], name=f"突破（{len(b)} 天）", mode="markers",
                             marker={"size": 8, "color": STATUS["critical"]}, hovertemplate="%{y:" + fmt + "}<extra>突破</extra>"))
    fig.add_hline(y=threshold, line={"color": STATUS["critical"], "width": 1}, annotation_text="突破线", annotation_position="top left")
    fig.add_hline(y=warn, line={"color": STATUS["warning"], "width": 1, "dash": "dash"}, annotation_text="预警线", annotation_position="bottom left")
    return _layout(fig, title, y_fmt=fmt)
