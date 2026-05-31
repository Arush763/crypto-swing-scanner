"""
Central configuration for the crypto swing-trading scanner.
All thresholds, weights, and parameters are defined here for easy tuning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Score weights (must sum to 1.0)
# ---------------------------------------------------------------------------
SCORE_WEIGHTS: Dict[str, float] = {
    "trend": 0.30,
    "momentum": 0.30,
    "liquidity": 0.20,
    "smart_money": 0.20,
}

# ---------------------------------------------------------------------------
# Liquidity filters — assets that fail these are excluded from the universe
# ---------------------------------------------------------------------------
MIN_DAILY_VOLUME_USD: float = 5_000_000      # $5M minimum daily volume
MIN_MARKET_CAP_USD: float = 50_000_000       # $50M minimum market cap
MIN_HISTORY_DAYS: int = 60                   # Minimum candle history required

# ---------------------------------------------------------------------------
# EMA periods (bar-count lookbacks — same meaning on any timeframe)
# ---------------------------------------------------------------------------
EMA_SHORT: int = 20
EMA_MID: int   = 50
EMA_LONG: int  = 200

# ---------------------------------------------------------------------------
# Momentum lookback periods (days)
# ---------------------------------------------------------------------------
MOMENTUM_PERIODS: List[int] = [7, 14, 30]

# ---------------------------------------------------------------------------
# Breakout detection
# ---------------------------------------------------------------------------
BREAKOUT_RESISTANCE_LOOKBACK: int = 20       # Bars to look back for swing highs
BREAKOUT_VOLUME_MULTIPLIER: float = 2.0      # Volume must be 2x 30-day average

# ---------------------------------------------------------------------------
# Volatility / squeeze
# ---------------------------------------------------------------------------
BB_PERIOD: int = 20
BB_STD: float = 2.0
ATR_PERIOD: int = 14
ATR_PERCENTILE_LOOKBACK: int = 252           # 1 year of daily bars
HV_PERIOD: int = 20                          # Historical volatility window
SQUEEZE_PERCENTILE_THRESHOLD: float = 20.0  # Below this percentile = squeeze

# ---------------------------------------------------------------------------
# Signal thresholds
# ---------------------------------------------------------------------------
SIGNAL_SCORE_THRESHOLD: float = 80.0        # Minimum score to generate an alert
STRONG_SIGNAL_SCORE: float = 90.0           # Score considered a strong signal

# ---------------------------------------------------------------------------
# Exit parameters
# ---------------------------------------------------------------------------
ATR_TRAILING_STOP_MULTIPLIER: float = 2.0   # 2 ATR trailing stop

# ---------------------------------------------------------------------------
# Supported exchanges (ccxt ids)
# ---------------------------------------------------------------------------
EXCHANGES: List[str] = ["binance", "bybit", "coinbase", "kraken"]

# ---------------------------------------------------------------------------
# Scan settings
# ---------------------------------------------------------------------------
SCAN_TIMEFRAME: str = "4h"                  # 4-hour candles for swing trading
OHLCV_LIMIT: int = 2000                     # Candles to fetch per asset

# ---------------------------------------------------------------------------
# Reference assets used in relative-strength calculations
# ---------------------------------------------------------------------------
BTC_SYMBOL: str = "BTC/USDT"
ETH_SYMBOL: str = "ETH/USDT"

# ---------------------------------------------------------------------------
# Volume consistency window
# ---------------------------------------------------------------------------
VOLUME_CONSISTENCY_WINDOW: int = 30

# ---------------------------------------------------------------------------
# Retest detection
# ---------------------------------------------------------------------------
RETEST_TOLERANCE_PCT: float = 0.03          # 3% tolerance around breakout level

# ---------------------------------------------------------------------------
# Backtesting defaults
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    timeframe: str = "4h"
    ema_short: int = EMA_SHORT
    ema_mid: int   = EMA_MID
    ema_long: int  = EMA_LONG
    volume_multiplier: float = BREAKOUT_VOLUME_MULTIPLIER
    score_threshold: float = SIGNAL_SCORE_THRESHOLD
    momentum_periods: List[int] = field(default_factory=lambda: list(MOMENTUM_PERIODS))
    initial_capital: float = 10_000.0
    risk_per_trade_pct: float = 0.02
    commission_pct: float = 0.001
    btc_regime_filter: bool = True          # Only enter when BTC > 200 EMA


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
DASHBOARD_TOP_N: int = 20                   # Rows shown in leaderboards
DASHBOARD_REFRESH_SECONDS: int = 300        # Auto-refresh interval

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = "INFO"
LOG_FILE: str = "logs/scanner.log"
