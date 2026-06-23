"""
Central configuration for the crypto swing-trading scanner.
All thresholds, weights, and parameters are defined here for easy tuning.
"""

from dataclasses import dataclass
from typing import Dict, List


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
MIN_DAILY_VOLUME_USD: float = 3_000_000      # $3M minimum daily volume
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
# General volume-expansion default (consumed by indicators/volume.py)
# ---------------------------------------------------------------------------
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
#
# binance and bybit are geo-blocked from this environment (451 / CloudFront
# 403), and binanceus's real volume has migrated away — its USDT/USD pairs
# come in near-zero, so it never contributes assets. okx and gateio are
# free, geo-accessible, and have genuinely deep spot liquidity. bitget was
# evaluated and excluded: its huge "volume" figures are dominated by
# tokenized-stock proxy pairs (RNVDA, RTSLA, ...) with implausible/wash
# volume, not real crypto liquidity.
# ---------------------------------------------------------------------------
EXCHANGES: List[str] = ["coinbase", "kucoin", "kraken", "okx", "gateio"]

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
# Order-book wall signal (sole signal source — absorption vs. repulsion of
# large resting orders, tracked across consecutive scan cycles)
# ---------------------------------------------------------------------------
WALL_SAME_LEVEL_TOLERANCE_PCT: float = 0.01   # Walls within 1% are the "same" wall across cycles
WALL_SHRINK_THRESHOLD: float = 0.5            # Wall is absorbed once size drops >=50% (or vanishes)
WALL_SIGNAL_BONUS: float = 15.0               # Points added to composite score on a confirmed setup

# ---------------------------------------------------------------------------
# Live order-flow corroboration — a wall shrinking in the book is ambiguous
# (it could've been eaten by real volume, or just cancelled/spoofed). Each
# scan cycle, we pull executed trades since the prior cycle for the same
# symbol and require aggressor (taker) volume to corroborate the wall
# classification, mirroring the tape-based backtest proxy in
# src/modules/tape_signal.py.
# ---------------------------------------------------------------------------
FLOW_TRADE_LIMIT: int = 500                   # Max trades fetched per cycle per symbol
FLOW_DOMINANCE_RATIO: float = 1.2             # Aggressor side must lead the other by this ratio to count as "dominant"

# ---------------------------------------------------------------------------
# Tape-based backtest signal — a historical proxy for the live OB wall
# signal. Full historical L2 order-book depth isn't archived anywhere for
# free, so backtesting instead uses free historical trade-tick data
# (src/data/trade_tape.py) and infers absorption/repulsion from clustered
# aggressor (taker buy/sell) volume around a recent swing level.
# ---------------------------------------------------------------------------
TAPE_LEVEL_LOOKBACK: int = 20          # Bars to look back for the swing high/low "level"
TAPE_PROXIMITY_PCT: float = 0.03      # Price must be within 3% of the level to count as "tested"
TAPE_VOLUME_SPIKE_MULT: float = 2.0   # Aggressor volume must be >= this x the rolling average
TAPE_SIGNAL_BONUS: float = 15.0       # Points added to composite score on a confirmed setup

# ---------------------------------------------------------------------------
# Tape backtest defaults
# ---------------------------------------------------------------------------
@dataclass
class TapeBacktestConfig:
    timeframe: str = "4h"
    ema_short: int = EMA_SHORT
    ema_mid: int   = EMA_MID
    ema_long: int  = EMA_LONG
    initial_capital: float = 10_000.0
    risk_per_trade_pct: float = 0.02
    commission_pct: float = 0.001
    btc_regime_filter: bool = True          # Only enter when BTC > 50/200 EMA


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
