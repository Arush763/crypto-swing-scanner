"""
Asset universe manager.

Holds the current snapshot of all assets that passed liquidity filters,
their OHLCV data, and reference data (BTC returns) used in relative-
strength calculations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AssetData:
    """Container for one asset's market data."""

    symbol: str
    ohlcv: pd.DataFrame           # columns: timestamp, open, high, low, close, volume
    exchange: str = ""
    market_cap: float = 0.0       # USD, if available from exchange metadata

    @property
    def close(self) -> pd.Series:
        return self.ohlcv["close"]

    @property
    def volume(self) -> pd.Series:
        return self.ohlcv["volume"]

    @property
    def high(self) -> pd.Series:
        return self.ohlcv["high"]

    @property
    def low(self) -> pd.Series:
        return self.ohlcv["low"]

    @property
    def latest_close(self) -> float:
        return float(self.ohlcv["close"].iloc[-1])

    @property
    def latest_volume(self) -> float:
        return float(self.ohlcv["volume"].iloc[-1])


@dataclass
class Universe:
    """
    Snapshot of the scannable asset universe.

    Assets are keyed by their normalised symbol (e.g. 'BTC/USDT').
    Reference data (BTC, market-wide) is stored separately.
    """

    assets: Dict[str, AssetData] = field(default_factory=dict)
    btc: Optional[pd.DataFrame] = None          # BTC OHLCV used for RS calculations
    scan_timestamp: Optional[pd.Timestamp] = None

    def add(self, symbol: str, ohlcv: pd.DataFrame, exchange: str = "") -> None:
        self.assets[symbol] = AssetData(symbol=symbol, ohlcv=ohlcv, exchange=exchange)

    def __len__(self) -> int:
        return len(self.assets)

    def symbols(self):
        return list(self.assets.keys())

    def get(self, symbol: str) -> Optional[AssetData]:
        return self.assets.get(symbol)

    @property
    def btc_returns(self) -> Optional[pd.Series]:
        """Daily log-returns for BTC, used in relative-strength computation."""
        if self.btc is None or self.btc.empty:
            return None
        return self.btc["close"].pct_change()
