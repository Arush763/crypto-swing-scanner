"""
Multi-exchange OHLCV data fetcher built on ccxt.

Responsibilities:
  - Fetch OHLCV candles from Binance, Bybit, Coinbase, Kraken
  - Apply liquidity pre-filters (volume, market cap)
  - Cache responses locally to avoid redundant API calls within a scan cycle
  - Provide a unified DataFrame schema regardless of exchange source
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ccxt
import pandas as pd

from src.config.config import (
    EXCHANGES,
    MIN_DAILY_VOLUME_USD,
    MIN_HISTORY_DAYS,
    OHLCV_LIMIT,
    SCAN_TIMEFRAME,
    BTC_SYMBOL,
)

logger = logging.getLogger(__name__)

# Column names used throughout the system
OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


class ExchangeClient:
    """Thin wrapper around a single ccxt exchange instance."""

    def __init__(self, exchange_id: str, api_key: str = "", secret: str = "") -> None:
        self.exchange_id = exchange_id
        options: Dict = {"enableRateLimit": True}
        if api_key:
            options["apiKey"] = api_key
            options["secret"] = secret

        exchange_class = getattr(ccxt, exchange_id)
        self.client: ccxt.Exchange = exchange_class(options)

    def load_markets(self) -> None:
        self.client.load_markets()

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Return a clean OHLCV DataFrame sorted by timestamp ascending."""
        try:
            raw = self.client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except ccxt.BadSymbol:
            logger.debug("%s: symbol %s not found on %s", symbol, symbol, self.exchange_id)
            return pd.DataFrame(columns=OHLCV_COLS)
        except ccxt.NetworkError as exc:
            logger.warning("%s fetch failed (network): %s", symbol, exc)
            return pd.DataFrame(columns=OHLCV_COLS)
        except ccxt.ExchangeError as exc:
            logger.warning("%s fetch failed (exchange): %s", symbol, exc)
            return pd.DataFrame(columns=OHLCV_COLS)

        if not raw:
            return pd.DataFrame(columns=OHLCV_COLS)

        df = pd.DataFrame(raw, columns=OHLCV_COLS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df[["open", "high", "low", "close", "volume"]] = df[
            ["open", "high", "low", "close", "volume"]
        ].astype(float)
        return df

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        try:
            return self.client.fetch_ticker(symbol)
        except Exception as exc:
            logger.debug("Ticker fetch failed for %s: %s", symbol, exc)
            return None

    def get_tradeable_symbols(self) -> List[str]:
        """Return USDT-quoted spot symbols available on this exchange."""
        try:
            markets = self.client.load_markets()
        except Exception as exc:
            logger.error("Failed to load markets for %s: %s", self.exchange_id, exc)
            return []

        symbols = []
        for symbol, market in markets.items():
            if (
                market.get("active", False)
                and market.get("spot", False)
                and market.get("quote", "") in ("USDT", "USD")
            ):
                symbols.append(symbol)
        return symbols


class MarketDataFetcher:
    """
    Orchestrates data collection across multiple exchanges.

    For each scan cycle it:
      1. Collects all tradeable symbols from every exchange.
      2. Applies a quick volume filter from ticker data.
      3. Fetches full OHLCV history for assets that pass the filter.
      4. Returns a dict mapping symbol -> OHLCV DataFrame.
    """

    def __init__(
        self,
        exchange_ids: List[str] = EXCHANGES,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.cache_dir = cache_dir or Path("data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.clients: Dict[str, ExchangeClient] = {}
        for eid in exchange_ids:
            try:
                self.clients[eid] = ExchangeClient(eid)
                logger.info("Initialised exchange: %s", eid)
            except Exception as exc:
                logger.error("Could not init exchange %s: %s", eid, exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_universe(
        self,
        min_volume: float = MIN_DAILY_VOLUME_USD,
        timeframe: str = SCAN_TIMEFRAME,
        limit: int = OHLCV_LIMIT,
    ) -> Dict[str, pd.DataFrame]:
        """
        Return {symbol: ohlcv_df} for all assets passing the liquidity filter.
        Symbols are normalised to the first exchange that carries them.
        """
        candidate_symbols = self._collect_candidate_symbols(min_volume)
        logger.info("Fetching OHLCV for %d candidate symbols", len(candidate_symbols))

        result: Dict[str, pd.DataFrame] = {}
        for symbol, exchange_id in candidate_symbols.items():
            df = self._fetch_with_cache(symbol, exchange_id, timeframe, limit)
            if self._is_sufficient(df):
                result[symbol] = df
            time.sleep(0.05)  # gentle rate-limit padding

        logger.info("Universe built: %d assets", len(result))
        return result

    def fetch_single(
        self,
        symbol: str,
        timeframe: str = SCAN_TIMEFRAME,
        limit: int = OHLCV_LIMIT,
    ) -> pd.DataFrame:
        """Fetch OHLCV for a single symbol, trying each exchange in order."""
        for exchange_id, client in self.clients.items():
            df = client.fetch_ohlcv(symbol, timeframe, limit)
            if not df.empty:
                return df
        return pd.DataFrame(columns=OHLCV_COLS)

    def fetch_btc(self, timeframe: str = SCAN_TIMEFRAME, limit: int = OHLCV_LIMIT) -> pd.DataFrame:
        return self.fetch_single(BTC_SYMBOL, timeframe, limit)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_candidate_symbols(self, min_volume: float) -> Dict[str, str]:
        """
        Returns {normalised_symbol: exchange_id} for symbols whose 24h
        volume in USD exceeds min_volume. If the same symbol is on multiple
        exchanges the one with highest volume wins.
        """
        volume_map: Dict[str, Tuple[float, str]] = {}  # symbol -> (volume, exchange)

        for exchange_id, client in self.clients.items():
            logger.info("Loading tickers from %s…", exchange_id)
            try:
                tickers = client.client.fetch_tickers()
            except Exception as exc:
                logger.warning("Could not fetch tickers from %s: %s", exchange_id, exc)
                continue

            for symbol, ticker in tickers.items():
                # Keep only USDT/USD pairs with sufficient volume
                if not (symbol.endswith("/USDT") or symbol.endswith("/USD")):
                    continue
                quote_vol = ticker.get("quoteVolume") or 0.0
                if quote_vol < min_volume:
                    continue
                existing_vol, _ = volume_map.get(symbol, (0.0, ""))
                if quote_vol > existing_vol:
                    volume_map[symbol] = (quote_vol, exchange_id)

        return {sym: exc for sym, (_, exc) in volume_map.items()}

    def _fetch_with_cache(
        self, symbol: str, exchange_id: str, timeframe: str, limit: int
    ) -> pd.DataFrame:
        safe_name = symbol.replace("/", "_")
        cache_file = self.cache_dir / f"{exchange_id}_{safe_name}_{timeframe}.parquet"

        # Use cache if it was written within the last hour
        if cache_file.exists():
            age_seconds = time.time() - cache_file.stat().st_mtime
            if age_seconds < 3600:
                try:
                    return pd.read_parquet(cache_file)
                except Exception:
                    pass

        client = self.clients.get(exchange_id)
        if client is None:
            return pd.DataFrame(columns=OHLCV_COLS)

        df = client.fetch_ohlcv(symbol, timeframe, limit)
        if not df.empty:
            try:
                df.to_parquet(cache_file, index=False)
            except Exception as exc:
                logger.debug("Cache write failed for %s: %s", symbol, exc)
        return df

    @staticmethod
    def _is_sufficient(df: pd.DataFrame) -> bool:
        """Require at least MIN_HISTORY_DAYS rows of clean data."""
        return len(df) >= MIN_HISTORY_DAYS
