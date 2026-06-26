"""
Historical Trade Tape Fetcher.

Pulls free historical tick (aggTrades) data from Binance's public data
archive (data.binance.vision) — daily CSV dumps, no API key required,
available for years of history. This is the only free source of deep
historical market-microstructure data: full historical L2 order-book
depth is not archived anywhere for free, only live snapshots are
available (see src/data/orderbook.py). Trade-tape data lets us backtest
a *proxy* of the live wall-absorption/repulsion signal using executed
aggressor volume instead of resting depth.

aggTrades CSV columns (no header row):
  agg_trade_id, price, quantity, first_trade_id, last_trade_id,
  transact_time_ms, is_buyer_maker, is_best_match

is_buyer_maker == True  -> taker was the seller (sell-aggressor, hits bid)
is_buyer_maker == False -> taker was the buyer  (buy-aggressor, hits ask)
"""

from __future__ import annotations

import io
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades/{symbol}/{symbol}-aggTrades-{day}.zip"

_COLUMNS = [
    "agg_trade_id", "price", "quantity", "first_trade_id",
    "last_trade_id", "transact_time", "is_buyer_maker", "is_best_match",
]


def _to_binance_symbol(symbol: str) -> str:
    """
    'BTC/USDT' -> 'BTCUSDT' (Binance Vision uses no separator). The live
    universe is pulled from coinbase/kucoin/kraken/okx/gateio and many of
    those list pairs as e.g. 'BTC/USD' — Binance has no such pair, so the
    quote is always normalised to USDT for the archive lookup regardless
    of what quote currency the live exchange used.
    """
    base = symbol.split("/")[0]
    return f"{base}USDT".upper()


def _infer_time_unit(sample: float) -> str:
    """
    Binance Vision switched aggTrades timestamps from milliseconds to
    microseconds partway through their archive (older dumps: ms ~1e12,
    newer dumps: us ~1e15-1e16). Infer per-file from magnitude rather than
    assuming a fixed unit, or dates silently explode by 1000x.
    """
    if sample > 1e17:
        return "ns"
    if sample > 1e14:
        return "us"
    return "ms"


class TradeTapeFetcher:
    """Downloads and caches daily historical trade-tick dumps."""

    def __init__(self, cache_dir: str = "data/tape_cache", timeout: int = 30) -> None:
        self.cache_dir = cache_dir
        self.timeout = timeout
        import os
        os.makedirs(cache_dir, exist_ok=True)

        # Pooled, thread-safe session — fetch_many hits this concurrently
        # from many threads, and requests' default per-host pool (10) would
        # bottleneck well before our thread count does.
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32)
        self.session.mount("https://", adapter)

    def _day_path(self, symbol: str, day: date) -> str:
        import os
        return os.path.join(self.cache_dir, f"{_to_binance_symbol(symbol)}_{day.isoformat()}.parquet")

    def fetch_day(self, symbol: str, day: date) -> Optional[pd.DataFrame]:
        """Fetch one UTC day of aggTrades for `symbol`. Returns None if unavailable."""
        cache_path = self._day_path(symbol, day)
        import os
        if os.path.exists(cache_path):
            return pd.read_parquet(cache_path)

        sym = _to_binance_symbol(symbol)
        url = BASE_URL.format(symbol=sym, day=day.isoformat())
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                logger.debug("No tape data for %s %s (HTTP %d)", symbol, day, resp.status_code)
                return None

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                name = zf.namelist()[0]
                with zf.open(name) as f:
                    df = pd.read_csv(f, header=None, names=_COLUMNS)
        except Exception as exc:
            logger.warning("Tape fetch failed %s %s: %s", symbol, day, exc)
            return None

        unit = _infer_time_unit(float(df["transact_time"].iloc[0]))
        df["transact_time"] = pd.to_datetime(df["transact_time"], unit=unit, utc=True)
        df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
        df.to_parquet(cache_path)
        return df

    def fetch_range(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Fetch and concatenate all days in [start, end] (inclusive)."""
        frames: List[pd.DataFrame] = []
        day = start
        while day <= end:
            df = self.fetch_day(symbol, day)
            if df is not None:
                frames.append(df)
            day += timedelta(days=1)

        if not frames:
            return pd.DataFrame(columns=_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def _fetch_day_bars(self, symbol: str, day: date, timeframe: str) -> Optional[pd.DataFrame]:
        """Fetch one day of ticks and immediately collapse to bars, so the
        (potentially multi-million-row) raw trade frame is freed right
        away instead of accumulating across an entire backtest range."""
        trades = self.fetch_day(symbol, day)
        if trades is None or trades.empty:
            return None
        return resample_to_bars(trades, timeframe=timeframe)

    def fetch_many_bars(
        self,
        symbols: List[str],
        start: date,
        end: date,
        timeframe: str = "4h",
        max_workers: int = 16,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch [start, end] for every symbol concurrently and return resampled
        bars per symbol — every (symbol, day) pair is one HTTP+unzip+resample
        task, since fetch_day's caching makes each independent. This is
        purely I/O-bound (static CDN archive), so threading turns what would
        be hours of sequential downloads for a large universe into a single
        bounded-concurrency pass.

        Bars rather than raw trades are accumulated across the whole range:
        a year of full tick data for ~75 symbols held in memory at once
        (rather than discarded per-day after resampling) is enough to OOM —
        4h bars divide evenly into a UTC day, so resampling day-by-day before
        concatenating is equivalent to resampling the full concatenated
        series.
        """
        days: List[date] = []
        d = start
        while d <= end:
            days.append(d)
            d += timedelta(days=1)

        tasks = [(symbol, day) for symbol in symbols for day in days]
        bars_by_symbol: Dict[str, List[pd.DataFrame]] = {s: [] for s in symbols}

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_task = {
                pool.submit(self._fetch_day_bars, symbol, day, timeframe): (symbol, day)
                for symbol, day in tasks
            }
            for future in as_completed(future_to_task):
                symbol, day = future_to_task[future]
                try:
                    bars = future.result()
                except Exception as exc:
                    logger.warning("Tape fetch failed %s %s: %s", symbol, day, exc)
                    continue
                if bars is not None and not bars.empty:
                    bars_by_symbol[symbol].append(bars)

        result = {}
        for symbol, frames in bars_by_symbol.items():
            if not frames:
                result[symbol] = pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume", "buy_volume", "sell_volume"]
                )
                continue
            result[symbol] = pd.concat(frames).sort_index()
        return result


def dedupe_by_binance_symbol(symbols: List[str]) -> List[str]:
    """
    Collapse symbols that normalise to the same Binance ticker (e.g.
    'BTC/USD' and 'BTC/USDT' both -> 'BTCUSDT') to a single representative,
    preferring the USDT-quoted spelling. Needed because the live universe
    pulls the same base asset under different quote currencies across its
    five exchanges, but they'd fetch identical tape data here.
    """
    chosen: Dict[str, str] = {}
    for symbol in symbols:
        key = _to_binance_symbol(symbol)
        if key not in chosen or symbol.upper().endswith("/USDT"):
            chosen[key] = symbol
    return list(chosen.values())


def resample_to_bars(trades: pd.DataFrame, timeframe: str = "4h") -> pd.DataFrame:
    """
    Resample raw tick trades into OHLCV bars, split into buy-aggressor and
    sell-aggressor volume (the tape-based stand-in for order-book pressure).
    """
    if trades.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "buy_volume", "sell_volume"])

    df = trades.set_index("transact_time").sort_index()
    notional = df["price"] * df["quantity"]
    buy_notional  = notional.where(~df["is_buyer_maker"], 0.0)   # taker bought
    sell_notional = notional.where(df["is_buyer_maker"], 0.0)    # taker sold

    bars = df["price"].resample(timeframe).ohlc()
    bars["volume"]      = notional.resample(timeframe).sum()
    bars["buy_volume"]  = buy_notional.resample(timeframe).sum()
    bars["sell_volume"] = sell_notional.resample(timeframe).sum()
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    return bars
