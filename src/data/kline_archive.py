"""
Lightweight historical OHLCV + buy/sell-volume fetcher via Binance Vision's
kline archive, instead of raw aggTrades.

src.data.trade_tape downloads raw tick-level aggTrades and resamples them to
get the buy/sell aggressor volume split tape_signal.py needs — accurate, but
a single BTC day of raw ticks is ~6MB. Binance's kline (candle) files already
carry that same split via `taker_buy_quote_asset_volume` (the notional volume
where the taker was the buyer) without needing tick-level data at all: a
whole day of 4h klines is ~500 bytes. For a multi-year, multi-symbol backtest
that difference is the gap between a few minutes of downloading and
100+ GB — this module trades a small amount of precision (kline-level
buy/sell split vs true tick-level) for ~12,000x less data.

kline CSV columns (no header row):
  open_time, open, high, low, close, volume, close_time, quote_asset_volume,
  number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume,
  ignore
"""

from __future__ import annotations

import io
import logging
import os
import zipfile
from datetime import date
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

MONTHLY_URL = "https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{year:04d}-{month:02d}.zip"
DAILY_URL = "https://data.binance.vision/data/spot/daily/klines/{symbol}/{interval}/{symbol}-{interval}-{day}.zip"

_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume", "ignore",
]


def _to_binance_symbol(symbol: str) -> str:
    base = symbol.split("/")[0]
    return f"{base}USDT".upper()


def _infer_time_unit(sample: float) -> str:
    """Same vintage ambiguity as the aggTrades archive — ms in older dumps, us/ns in newer ones."""
    if sample > 1e17:
        return "ns"
    if sample > 1e14:
        return "us"
    return "ms"


def _parse_klines_csv(raw: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), header=None, names=_COLUMNS)
    if df.empty:
        return df
    unit = _infer_time_unit(float(df["open_time"].iloc[0]))
    df["open_time"] = pd.to_datetime(df["open_time"], unit=unit, utc=True)
    return df


def _klines_to_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Same notional-volume convention as src.data.trade_tape.resample_to_bars."""
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "buy_volume", "sell_volume"])
    out = pd.DataFrame({
        "open": df["open"].astype(float).to_numpy(),
        "high": df["high"].astype(float).to_numpy(),
        "low": df["low"].astype(float).to_numpy(),
        "close": df["close"].astype(float).to_numpy(),
        "volume": df["quote_asset_volume"].astype(float).to_numpy(),
        "buy_volume": df["taker_buy_quote_asset_volume"].astype(float).to_numpy(),
    }, index=pd.DatetimeIndex(df["open_time"].to_numpy()))
    out["sell_volume"] = (out["volume"] - out["buy_volume"]).clip(lower=0.0)
    return out.sort_index()


class KlineArchiveFetcher:
    """Downloads and caches monthly (falling back to daily for the current month) kline archives."""

    def __init__(self, cache_dir: str = "data/kline_cache", timeout: int = 20) -> None:
        self.cache_dir = cache_dir
        self.timeout = timeout
        os.makedirs(cache_dir, exist_ok=True)
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
        self.session.mount("https://", adapter)

    def _month_cache_path(self, symbol: str, year: int, month: int, interval: str) -> str:
        return os.path.join(self.cache_dir, f"{_to_binance_symbol(symbol)}_{interval}_{year:04d}-{month:02d}.parquet")

    def _get(self, url: str) -> Optional[bytes]:
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                return None
            return resp.content
        except Exception as exc:
            logger.debug("kline fetch failed %s: %s", url, exc)
            return None

    def fetch_month(self, symbol: str, year: int, month: int, interval: str = "4h") -> Optional[pd.DataFrame]:
        """Fetch one calendar month of klines for `symbol`. Returns None if unavailable (e.g. not listed yet)."""
        cache_path = self._month_cache_path(symbol, year, month, interval)
        if os.path.exists(cache_path):
            return pd.read_parquet(cache_path)

        sym = _to_binance_symbol(symbol)
        content = self._get(MONTHLY_URL.format(symbol=sym, interval=interval, year=year, month=month))
        if content is None:
            return None

        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                name = zf.namelist()[0]
                with zf.open(name) as f:
                    raw_df = _parse_klines_csv(f.read())
        except Exception as exc:
            logger.warning("kline parse failed %s %04d-%02d: %s", symbol, year, month, exc)
            return None

        bars = _klines_to_bars(raw_df)
        bars.to_parquet(cache_path)
        return bars

    def fetch_current_month_so_far(self, symbol: str, year: int, month: int, interval: str = "4h") -> Optional[pd.DataFrame]:
        """
        The monthly archive for an in-progress month doesn't exist yet —
        stitch together daily archives instead. Not cached (the month is
        still accumulating), so this re-downloads on every call; only meant
        for the most recent, still-open month.
        """
        from calendar import monthrange
        sym = _to_binance_symbol(symbol)
        _, days_in_month = monthrange(year, month)
        frames = []
        for day_num in range(1, days_in_month + 1):
            day = date(year, month, day_num)
            if day > date.today():
                break
            content = self._get(DAILY_URL.format(symbol=sym, interval=interval, day=day.isoformat()))
            if content is None:
                continue
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    name = zf.namelist()[0]
                    with zf.open(name) as f:
                        frames.append(_parse_klines_csv(f.read()))
            except Exception as exc:
                logger.debug("daily kline parse failed %s %s: %s", symbol, day, exc)
        if not frames:
            return None
        return _klines_to_bars(pd.concat(frames, ignore_index=True))

    def fetch_range(self, symbol: str, start: date, end: date, interval: str = "4h") -> pd.DataFrame:
        """Fetch and concatenate every month overlapping [start, end] (inclusive)."""
        frames: List[pd.DataFrame] = []
        year, month = start.year, start.month
        today = date.today()
        while (year, month) <= (end.year, end.month):
            is_current_month = (year, month) == (today.year, today.month)
            bars = (
                self.fetch_current_month_so_far(symbol, year, month, interval)
                if is_current_month
                else self.fetch_month(symbol, year, month, interval)
            )
            if bars is not None and not bars.empty:
                frames.append(bars)
            month += 1
            if month > 12:
                month = 1
                year += 1

        if not frames:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "buy_volume", "sell_volume"])
        combined = pd.concat(frames).sort_index()
        return combined[(combined.index.date >= start) & (combined.index.date <= end)]

    def earliest_available_month(self, symbol: str, search_start: date, search_end: date) -> Optional[date]:
        """
        Binary search for the first calendar month with any available data,
        within [search_start, search_end] — used to build the expanding
        universe's per-symbol listing date without probing every month.
        """
        sym = _to_binance_symbol(symbol)

        def _has_data(year: int, month: int) -> bool:
            url = MONTHLY_URL.format(symbol=sym, interval="4h", year=year, month=month)
            try:
                r = self.session.head(url, timeout=self.timeout)
                return r.status_code == 200
            except Exception:
                return False

        lo = search_start.year * 12 + search_start.month
        hi = search_end.year * 12 + search_end.month

        def month_index_to_ym(idx: int):
            y, m = divmod(idx, 12)
            if m == 0:
                y -= 1
                m = 12
            return y, m

        # If even the latest month has no data, the symbol isn't on Binance at all.
        if not _has_data(*month_index_to_ym(hi)):
            return None

        result = None
        while lo <= hi:
            mid = (lo + hi) // 2
            y, m = month_index_to_ym(mid)
            if _has_data(y, m):
                result = date(y, m, 1)
                hi = mid - 1
            else:
                lo = mid + 1
        return result
