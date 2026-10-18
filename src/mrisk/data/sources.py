"""Market data sources. AkshareSource wraps akshare and is the only class in the project that touches the network
(akshare is imported on the first call, never at import time). SnapshotSource replays an offline snapshot of
AkshareSource's raw cache (mrisk export-snapshot) through the same normalisation, with no network.

Prices come from akshare's Sina endpoints (stock_zh_a_daily): Eastmoney blocks this
host and Tencent does not cover the Beijing exchange.
"""

import contextlib
import datetime as dt
import logging
import os
import re
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def to_ns(s: pd.Series) -> pd.Series:
    """pandas 3 infers s/us resolutions per parse; merge_asof requires identical ones."""
    return pd.to_datetime(s, errors="coerce").astype("datetime64[ns]")


def clean_name(name: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(name)))


def norm_trade_calendar(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"trade_date": to_ns(raw["trade_date"])}).sort_values("trade_date", ignore_index=True)


def norm_listings(sh_main: pd.DataFrame, sh_star: pd.DataFrame, sz: pd.DataFrame, bj: pd.DataFrame) -> pd.DataFrame:
    parts = [
        pd.DataFrame({"exchange": "SH", "symbol": sh_main["证券代码"], "name": sh_main["证券简称"], "list_date": sh_main["上市日期"]}),
        pd.DataFrame({"exchange": "SH", "symbol": sh_star["证券代码"], "name": sh_star["证券简称"], "list_date": sh_star["上市日期"]}),
        pd.DataFrame({"exchange": "SZ", "symbol": sz["A股代码"], "name": sz["A股简称"], "list_date": sz["A股上市日期"]}),
        pd.DataFrame({"exchange": "BJ", "symbol": bj["证券代码"], "name": bj["证券简称"], "list_date": bj["上市日期"]}),
    ]
    out = pd.concat(parts, ignore_index=True)
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    out["name"] = out["name"].map(clean_name)
    out["list_date"] = to_ns(out["list_date"])
    return out


def norm_delisted(sh: pd.DataFrame, sz: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat(
        [
            pd.DataFrame({"exchange": "SH", "symbol": sh["公司代码"]}),
            pd.DataFrame({"exchange": "SZ", "symbol": sz["证券代码"]}),
        ],
        ignore_index=True,
    )
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    return out.drop_duplicates(ignore_index=True)


def norm_sw_members(components: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """components: SW L1 industry name -> index_component_sw frame."""
    parts = [
        pd.DataFrame({"symbol": df["证券代码"].astype(str).str.zfill(6), "industry_l1": name, "sw_weight": df["最新权重"]})
        for name, df in components.items()
    ]
    return pd.concat(parts, ignore_index=True)


def norm_daily(raw: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": to_ns(raw["date"]),
            "close": pd.to_numeric(raw["close"]).astype(float),
            "volume": pd.to_numeric(raw["volume"]).astype("int64"),
        }
    ).sort_values("trade_date", ignore_index=True)


def norm_hfq_factor(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {"effective_date": to_ns(raw["date"]), "hfq_factor": pd.to_numeric(raw["hfq_factor"]).astype(float)}
    )
    return out.sort_values("effective_date", ignore_index=True)


@contextlib.contextmanager
def _no_proxy(enabled: bool):
    if not enabled:
        yield
        return
    saved = {k: os.environ.get(k) for k in ("NO_PROXY", "no_proxy")}
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class AkshareSource:
    def __init__(
        self,
        cache_dir: str | Path,
        request_interval_sec: float = 1.0,
        max_retries: int = 4,
        bypass_proxy: bool = True,
        today: dt.date | None = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.interval = request_interval_sec
        self.max_retries = max_retries
        self.bypass_proxy = bypass_proxy
        self.today = (today or dt.date.today()).strftime("%Y%m%d")
        self._last_call = 0.0

    def _call(self, fn_name: str, **kwargs) -> pd.DataFrame:
        import akshare as ak

        fn = getattr(ak, fn_name)
        for attempt in range(self.max_retries + 1):
            wait = self._last_call + self.interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                with _no_proxy(self.bypass_proxy):
                    return fn(**kwargs)
            except Exception as e:
                if attempt == self.max_retries:
                    raise
                backoff = 2 ** (attempt + 1)
                log.warning("%s(%s) failed (%s: %s), retry in %ss", fn_name, kwargs, type(e).__name__, e, backoff)
                time.sleep(backoff)
            finally:
                self._last_call = time.monotonic()
        raise AssertionError("unreachable")

    def _cached(self, key: str, fetch: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        path = self.cache_dir / f"{key}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        df = fetch()
        df.to_parquet(path, index=False)
        return df

    def trade_calendar(self) -> pd.DataFrame:
        raw = self._cached(f"calendar_{self.today}", lambda: self._call("tool_trade_date_hist_sina").astype(str))
        return norm_trade_calendar(raw)

    def listings(self) -> pd.DataFrame:
        def get(key, fn, **kw):
            return self._cached(f"{key}_{self.today}", lambda: self._call(fn, **kw).astype(str))

        return norm_listings(
            get("list_sh_main", "stock_info_sh_name_code", symbol="主板A股"),
            get("list_sh_star", "stock_info_sh_name_code", symbol="科创板"),
            get("list_sz", "stock_info_sz_name_code", symbol="A股列表"),
            get("list_bj", "stock_info_bj_name_code"),
        )

    def delisted(self) -> pd.DataFrame:
        sh = self._cached(f"delist_sh_{self.today}", lambda: self._call("stock_info_sh_delist", symbol="全部").astype(str))
        sz = self._cached(
            f"delist_sz_{self.today}", lambda: self._call("stock_info_sz_delist", symbol="终止上市公司").astype(str)
        )
        return norm_delisted(sh, sz)

    def sw_l1_members(self) -> pd.DataFrame:
        first = self._cached(f"sw_l1_{self.today}", lambda: self._call("sw_index_first_info").astype(str))
        comps = {}
        for code, name in zip(first["行业代码"], first["行业名称"]):
            num = code.split(".")[0]
            comps[name] = self._cached(
                f"sw_comp_{num}_{self.today}", lambda num=num: self._call("index_component_sw", symbol=num).astype(str)
            )
        return norm_sw_members(comps)

    def daily_raw(self, exchange: str, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        s, e = pd.Timestamp(start).strftime("%Y%m%d"), pd.Timestamp(end).strftime("%Y%m%d")
        sina = f"{exchange.lower()}{symbol}"
        raw = self._cached(
            f"daily_{exchange}{symbol}_{s}_{e}",
            lambda: self._call("stock_zh_a_daily", symbol=sina, start_date=s, end_date=e, adjust=""),
        )
        return norm_daily(raw)

    def hfq_factor(self, exchange: str, symbol: str) -> pd.DataFrame:
        sina = f"{exchange.lower()}{symbol}"
        raw = self._cached(
            f"hfq_{exchange}{symbol}_{self.today}",
            lambda: self._call("stock_zh_a_daily", symbol=sina, adjust="hfq-factor").astype(str),
        )
        return norm_hfq_factor(raw)


class SnapshotSource:
    """Same six methods as AkshareSource, read from a snapshot directory (see mrisk.data.snapshot): one parquet per
    raw cache key with the fetch date dropped, the raw akshare frames unchanged, so the same norm_* functions run on
    the same input. A missing file is an error, never a network fetch."""

    def __init__(self, snapshot_dir: str | Path):
        self.dir = Path(snapshot_dir)
        if not (self.dir / "manifest.json").exists():
            raise FileNotFoundError(f"no snapshot in {self.dir} (manifest.json missing); run mrisk fetch-snapshot")

    def _read(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(self.dir / f"{key}.parquet")

    def trade_calendar(self) -> pd.DataFrame:
        return norm_trade_calendar(self._read("calendar"))

    def listings(self) -> pd.DataFrame:
        return norm_listings(self._read("list_sh_main"), self._read("list_sh_star"), self._read("list_sz"), self._read("list_bj"))

    def delisted(self) -> pd.DataFrame:
        return norm_delisted(self._read("delist_sh"), self._read("delist_sz"))

    def sw_l1_members(self) -> pd.DataFrame:
        first = self._read("sw_l1")
        return norm_sw_members({name: self._read(f"sw_comp_{code.split('.')[0]}") for code, name in zip(first["行业代码"], first["行业名称"])})

    def daily_raw(self, exchange: str, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        """The rows Sina returns for [start, end] (both inclusive)."""
        raw = self._read(f"daily_{exchange}{symbol}")
        d = to_ns(raw["date"])
        return norm_daily(raw[(d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))])

    def hfq_factor(self, exchange: str, symbol: str) -> pd.DataFrame:
        return norm_hfq_factor(self._read(f"hfq_{exchange}{symbol}"))
