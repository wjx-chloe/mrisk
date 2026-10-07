"""Pure helpers behind the dashboard pages: frames in, frames out. No database, no Streamlit."""

import numpy as np
import pandas as pd

STATUS_RANK = {"ok": 0, "no_data": 1, "warn": 2, "breach": 3}
MODE_HIST, MODE_BETA, MODE_RAW = "历史", "β 传导", "其余不变"


def trailing_runs(status: pd.DataFrame) -> pd.Series:
    """status: (position dates x keys), the calendar of every position date up to the day of interest, NaN where a
    key has no row that day. -> per key, how many trailing rows carry the last row's value: 1 means it changed (or
    started) on the last day. A different status, a missing row or no_data in between all break the run."""
    if status.empty:
        return pd.Series(dtype=int)
    last = status.iloc[-1]
    same = status.eq(last, axis=1).to_numpy()[::-1]  # NaN never equals anything
    broken = np.cumsum(~same, axis=0) > 0
    return pd.Series((~broken).sum(axis=0), index=status.columns).where(last.notna(), 0).astype(int)


def segments(flag: pd.Series, values: pd.Series | None = None) -> pd.DataFrame:
    """flag: bool on consecutive position dates (index = dates). -> one row per run of True:
    [start, end, days] (+ peak = max of values over the run when given), oldest first."""
    f = flag.to_numpy(dtype=bool)
    edges = np.diff(np.concatenate([[0], f.astype(int), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    out = pd.DataFrame({"start": flag.index[starts], "end": flag.index[ends], "days": ends - starts + 1})
    if values is not None:
        v = values.to_numpy(dtype=float)
        out["peak"] = [np.nanmax(v[a : b + 1]) if np.isfinite(v[a : b + 1]).any() else np.nan for a, b in zip(starts, ends)]
    return out


def _pct(x: float) -> str:
    return f"{x * 100:+g}%"


def describe_scenario(scenario_type: str, definition: dict) -> tuple[str, str]:
    """-> (shock, mode). The shock names what is moved; the mode says how: 历史 (a replayed period), β 传导 (factor shock
    propagated by beta) or 其余不变 (raw shocks, nothing else moves). A beta scenario and its raw twin share the shock."""
    if scenario_type == "historical":
        return f"{definition['base']} → {definition['end']}", MODE_HIST
    if "raw" in definition:
        raw = definition["raw"]
        parts = []
        if "market" in raw:
            parts.append(f"全市场 {_pct(raw['market'])}")
        parts += [f"{k} {_pct(v)}" for k, v in raw.get("industry", {}).items()]
        if raw.get("instrument"):
            parts.append(f"{len(raw['instrument'])} 只个股")
        return "、".join(parts), MODE_RAW
    if "market" in definition:
        return f"全市场 {_pct(definition['market'])}", MODE_BETA
    (k, v), = definition["industry"].items()
    return f"{k} {_pct(v)}", MODE_BETA


def label_scenarios(scen: pd.DataFrame) -> pd.DataFrame:
    """scenario rows [scenario_id, type, definition] -> [scenario_id, type, shock, mode]."""
    lab = [describe_scenario(r.type, r.definition) for r in scen.itertuples()]
    return scen[["scenario_id", "type"]].assign(shock=[a for a, _ in lab], mode=[b for _, b in lab])


def side_by_side(port: pd.DataFrame) -> pd.DataFrame:
    """Portfolio stress rows [shock, mode, loss_pct] -> one row per shock that has both a β 传导 and a 其余不变 version:
    [shock, beta, raw, gap = beta - raw], in the order of the β losses, worst first."""
    p = port[port["mode"].isin([MODE_BETA, MODE_RAW])].pivot_table(index="shock", columns="mode", values="loss_pct", aggfunc="first")
    if MODE_BETA not in p or MODE_RAW not in p:
        return pd.DataFrame(columns=["shock", "beta", "raw", "gap"])
    p = p.dropna(subset=[MODE_BETA, MODE_RAW]).rename(columns={MODE_BETA: "beta", MODE_RAW: "raw"})
    p["gap"] = p["beta"] - p["raw"]
    return p.sort_values("beta", ascending=False).reset_index()[["shock", "beta", "raw", "gap"]]


def calendar_grid(dates: pd.DatetimeIndex, hit: np.ndarray) -> pd.DataFrame:
    """Observation dates and exception flags -> one row per date with [date, week (Monday of its week), weekday 0-4,
    hit] for a week x weekday calendar."""
    d = pd.DatetimeIndex(dates)
    return pd.DataFrame({"date": d, "week": d - pd.to_timedelta(d.weekday, unit="D"), "weekday": d.weekday, "hit": np.asarray(hit, dtype=bool)})
