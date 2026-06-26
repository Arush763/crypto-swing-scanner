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

# Minutes per timeframe, used to resample onto a timeframe an exchange
# doesn't natively support (e.g. coinbase has no "4h" candle — only 2h/6h).
_TIMEFRAME_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720,
    "1d": 1440,
}


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

    def _resolve_timeframe(self, timeframe: str, limit: int) -> Tuple[str, int, Optional[str]]:
        """
        If `timeframe` isn't natively supported, pick the largest supported
        timeframe that evenly divides it and fetch at that resolution instead.
        Returns (fetch_timeframe, fetch_limit, resample_rule); resample_rule
        is None when no resampling is needed.
        """
        supported = getattr(self.client, "timeframes", None) or {}
        if not supported or timeframe in supported:
            return timeframe, limit, None

        target_minutes = _TIMEFRAME_MINUTES.get(timeframe)
        if target_minutes is None:
            return timeframe, limit, None  # unknown timeframe, let ccxt raise naturally

        candidates = [
            (tf, _TIMEFRAME_MINUTES[tf]) for tf in supported
            if tf in _TIMEFRAME_MINUTES
            and _TIMEFRAME_MINUTES[tf] < target_minutes
            and target_minutes % _TIMEFRAME_MINUTES[tf] == 0
        ]
        if not candidates:
            return timeframe, limit, None

        base_tf, base_minutes = max(candidates, key=lambda x: x[1])
        ratio = target_minutes // base_minutes
        logger.debug(
            "%s has no native %s candle — resampling from %s",
            self.exchange_id, timeframe, base_tf,
        )
        return base_tf, limit * ratio, f"{target_minutes}min"

    @staticmethod
    def _resample(df: pd.DataFrame, rule: str, limit: int) -> pd.DataFrame:
        if df.empty:
            return df
        out = (
            df.set_index("timestamp")
            .resample(rule, label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
            .reset_index()
        )
        return out.tail(limit).reset_index(drop=True)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """Return a clean OHLCV DataFrame sorted by timestamp ascending."""
        fetch_timeframe, fetch_limit, resample_rule = self._resolve_timeframe(timeframe, limit)
        try:
            raw = self.client.fetch_ohlcv(symbol, timeframe=fetch_timeframe, limit=fetch_limit)
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

        if resample_rule:
            df = self._resample(df, resample_rule, limit)
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

        # Populated by fetch_universe(): symbol -> exchange_id it was fetched
        # from (the exchange with the highest 24h volume for that symbol).
        # Downstream live data (order book, trade flow) must query this same
        # exchange, since other exchanges may not even list the symbol.
        self.symbol_exchange: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_universe_symbols(self, min_volume: float = MIN_DAILY_VOLUME_USD) -> List[str]:
        """
        Return just the symbols passing the liquidity filter (ticker data
        only — no OHLCV fetch), for callers that need the universe's
        membership without the cost of pulling candle history for it.
        """
        return list(self._collect_candidate_symbols(min_volume).keys())

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
        self.symbol_exchange = {}
        for symbol, exchange_id in candidate_symbols.items():
            df = self._fetch_with_cache(symbol, exchange_id, timeframe, limit)
            if self._is_sufficient(df):
                result[symbol] = df
                self.symbol_exchange[symbol] = exchange_id
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
