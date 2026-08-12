"""
Historical futures positioning metrics from Binance's public data archive.

Why this module exists
----------------------
Everything analysed in this project so far — order-book walls, aggressor
volume, whale prints, CVD — is a transform of spot price and spot volume.
That family has now been searched exhaustively (1,728 combos at 15m, 638 at
4h) and it has no directional edge: the direction study found the best
drift-free signal at 0.08-0.12% against a 0.21% fee floor, with no sign
stability across splits.

Positioning data is a different kind of information. It says nothing about
what has traded; it says what leveraged participants are currently *holding*
and therefore what they may be forced to do. That matters because the largest
directional moves in crypto are liquidation cascades, and a cascade is
mechanically predictable in a way a price move is not: crowded leverage plus
an adverse move produces forced flow in a known direction.

Available series (5-minute resolution, one file per symbol per UTC day):

  sum_open_interest                  contracts outstanding
  sum_open_interest_value            notional outstanding
  count_long_short_ratio             ratio of accounts long vs short
  count_toptrader_long_short_ratio   same, restricted to the largest accounts
  sum_toptrader_long_short_ratio     top accounts by position size, not count
  sum_taker_long_short_vol_ratio     aggressor buy/sell volume on perps

The distinction between `count_*` and `sum_toptrader_*` is the useful one:
the first is the crowd, the second is size. When they diverge, one of the two
groups is usually wrong, and it is not normally the larger one.

Data begins around 2023 and is published daily with a lag of a day or two.
"""

from __future__ import annotations

import io
import logging
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = (
    "https://data.binance.vision/data/futures/um/daily/metrics/"
    "{symbol}/{symbol}-metrics-{day}.zip"
)

NUMERIC_COLUMNS = [
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]


def _to_binance_symbol(symbol: str) -> str:
    """'BTC/USDT' -> 'BTCUSDT'. Perps are USDT-quoted regardless of the spot pair."""
    return f"{symbol.split('/')[0]}USDT".upper()


class FuturesMetricsFetcher:
    """Downloads and caches daily futures positioning metrics."""

    def __init__(self, cache_dir: str = "data/metrics_cache", timeout: int = 30) -> None:
        self.cache_dir = cache_dir
        self.timeout = timeout
        os.makedirs(cache_dir, exist_ok=True)

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
        self.session.mount("https://", adapter)

    def _day_path(self, symbol: str, day: date) -> str:
        return os.path.join(
            self.cache_dir, f"{_to_binance_symbol(symbol)}_{day.isoformat()}.parquet",
        )

    def fetch_day(self, symbol: str, day: date) -> Optional[pd.DataFrame]:
        """One UTC day of 5-minute metrics. None when unavailable."""
        cache_path = self._day_path(symbol, day)
        if os.path.exists(cache_path):
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                os.remove(cache_path)      # corrupt cache entry, refetch

        sym = _to_binance_symbol(symbol)
        url = BASE_URL.format(symbol=sym, day=day.isoformat())
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                return None
            archive = zipfile.ZipFile(io.BytesIO(resp.content))
            df = pd.read_csv(archive.open(archive.namelist()[0]))
        except Exception as exc:
            logger.debug("Metrics fetch failed %s %s: %s", symbol, day, exc)
            return None

        if df.empty or "create_time" not in df.columns:
            return None

        df["create_time"] = pd.to_datetime(df["create_time"], utc=True)
        for col in NUMERIC_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        try:
            df.to_parquet(cache_path, index=False)
        except Exception as exc:
            logger.debug("Could not cache metrics for %s %s: %s", symbol, day, exc)
        return df

    def fetch_range(self, symbol: str, start: date, end: date,
                    max_workers: int = 16) -> pd.DataFrame:
        """Concatenated metrics for [start, end], indexed by timestamp."""
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        frames: List[pd.DataFrame] = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(self.fetch_day, symbol, d): d for d in days}
            for fut in as_completed(futures):
                df = fut.result()
                if df is not None and not df.empty:
                    frames.append(df)

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values("create_time").set_index("create_time")
        return out[~out.index.duplicated(keep="first")]

    def fetch_many(self, symbols: List[str], start: date, end: date,
                   max_workers: int = 16) -> Dict[str, pd.DataFrame]:
        """Metrics for several symbols. Symbols without data are omitted."""
        out: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            df = self.fetch_range(symbol, start, end, max_workers=max_workers)
            if df.empty:
                logger.warning("No futures metrics for %s", symbol)
                continue
            out[symbol] = df
            logger.info("%s: %d metric rows", symbol, len(df))
        return out


def align_to_bars(metrics: pd.DataFrame, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Align 5-minute metrics onto a bar index.

    Uses forward-fill onto each bar's timestamp, which takes the last metric
    reading at or before the bar's start. That is deliberately the reading
    that was already published when the bar opened — sampling the metric at
    the bar's close, or interpolating across it, would leak information the
    strategy could not have had.
    """
    if metrics.empty:
        return pd.DataFrame(index=bar_index)
    aligned = metrics.reindex(
        metrics.index.union(bar_index),
    ).sort_index().ffill().reindex(bar_index)
    return aligned[[c for c in NUMERIC_COLUMNS if c in aligned.columns]]
