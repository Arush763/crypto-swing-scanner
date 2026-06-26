"""
Central configuration for the crypto swing-trading scanner.
All thresholds, weights, and parameters are defined here for easy tuning.
"""

from dataclasses import dataclass
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
WALL_ICEBERG_VOLUME_MULT: float = 3.0         # Iceberg: wall holds its size but >=3x its size traded through it

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
# ML signal filter (src/modules/signal_filter.py) — a year-long backtest
# showed ask_absorption setups have ~no real edge on their own (and get
# *worse* with stronger breakouts/buy-margins, a buy-the-top pattern), while
# bid_repulsion is consistently profitable. Rather than hard-dropping
# ask_absorption, every setup of either type is scored by a model trained on
# real per-signal trade outcomes and only passed through above this
# probability threshold. Retrain via scripts/train_signal_filter.py.
# ---------------------------------------------------------------------------
ML_FILTER_MODEL_PATH: str = "data/models/signal_filter.joblib"
ML_FILTER_THRESHOLD: float = 0.5

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

# Liquidity sweep (stop-hunt reclaim)
TAPE_SWEEP_WINDOW: int = 1             # Bars allowed between the wick-break and the reclaim
TAPE_SWEEP_VOLUME_MULT: float = 2.0    # Sweep bar's volume must be >= this x the rolling average

# Climax exhaustion (capitulation reversal)
TAPE_CLIMAX_WINDOW: int = 3            # Bars allowed between the climax bar and the reclaim
TAPE_CLIMAX_VOLUME_MULT: float = 3.0   # Climax bar's volume must be >= this x the rolling average
TAPE_CLIMAX_WIDE_MULT: float = 1.5     # Climax bar's range must be >= this x the rolling average range

# VWAP / mean-reversion fade
TAPE_VWAP_WINDOW: int = 20             # Bars in the rolling volume-weighted average price
TAPE_VWAP_STRETCH_PCT: float = 0.03   # Price must be this far below VWAP to count as "stretched"
TAPE_VWAP_VOLUME_MULT: float = 1.5     # Stretch bar's volume must be >= this x the rolling average

# Momentum-ignition continuation (trend-following, not mean-reversion — bets
# the breakout keeps running rather than that a level holds or bounces)
TAPE_BREAKOUT_MARGIN_PCT: float = 0.01      # Close must clear resistance by at least this much
TAPE_BREAKOUT_VOLUME_MULT: float = 2.0      # Breakout bar's volume must be >= this x the rolling average
TAPE_BREAKOUT_DOMINANCE_RATIO: float = 1.5  # Buy volume must be >= this x sell volume on the breakout bar

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

    # Core detection thresholds — previously hardcoded to detect_tape_signals's
    # own module-level defaults regardless of this config; exposed here so a
    # backtest can tune trade frequency (looser proximity/volume = more setups).
    lookback: int = TAPE_LEVEL_LOOKBACK
    proximity_pct: float = TAPE_PROXIMITY_PCT
    volume_spike_mult: float = TAPE_VOLUME_SPIKE_MULT
    bonus_pts: float = TAPE_SIGNAL_BONUS

    # Order-flow signal variants — see src.modules.tape_signal for what each
    # one changes; defaults reproduce the original single-pass signal.
    two_phase_absorption: bool = False
    two_phase_window: int = 5
    two_phase_narrow_mult: float = 0.7
    cvd_filter: bool = False
    cvd_window: int = 5
    stacked_bars: int = 1
    enable_ask_absorption: bool = True
    enable_bid_repulsion: bool = True
    enable_liquidity_sweep: bool = False
    sweep_window: int = TAPE_SWEEP_WINDOW
    sweep_volume_mult: float = TAPE_SWEEP_VOLUME_MULT
    enable_climax_exhaustion: bool = False
    climax_window: int = TAPE_CLIMAX_WINDOW
    climax_volume_mult: float = TAPE_CLIMAX_VOLUME_MULT
    climax_wide_mult: float = TAPE_CLIMAX_WIDE_MULT
    enable_delta_divergence: bool = False
    enable_vwap_fade: bool = False
    vwap_window: int = TAPE_VWAP_WINDOW
    vwap_stretch_pct: float = TAPE_VWAP_STRETCH_PCT
    vwap_volume_mult: float = TAPE_VWAP_VOLUME_MULT

    # Exit logic — previously hardcoded to the global ATR_TRAILING_STOP_MULTIPLIER
    # regardless of this config; exposed here so the exit can be tuned per backtest.
    # `momentum_atr_trailing_stop_mult` lets momentum_breakout trades use a
    # wider trailing stop than every other (mean-reversion) event type,
    # which needs a much tighter one — mixing signal families that want
    # opposite exit styles under one shared multiplier understates both.
    # Defaults to the same value as atr_trailing_stop_mult (no behaviour
    # change) unless explicitly overridden.
    atr_trailing_stop_mult: float = ATR_TRAILING_STOP_MULTIPLIER
    momentum_atr_trailing_stop_mult: Optional[float] = None

    # Per-symbol daily-trend regime filter — an alternative/addition to the
    # market-wide BTC-only regime filter, gating entries on that symbol's own
    # daily EMA trend instead of (or alongside) BTC's.
    enable_daily_trend_filter: bool = False
    daily_trend_ema_period: int = 20

    # Momentum-ignition continuation — a trend-following entry, structurally
    # different from the mean-reversion family above (absorption, repulsion,
    # climax, vwap_fade all bet on a hold/bounce; this bets the move runs).
    enable_momentum_breakout: bool = False
    breakout_margin_pct: float = TAPE_BREAKOUT_MARGIN_PCT
    breakout_volume_mult: float = TAPE_BREAKOUT_VOLUME_MULT
    breakout_dominance_ratio: float = TAPE_BREAKOUT_DOMINANCE_RATIO

    # Post-hoc quality filter — every detect_tape_signals() call already
    # computes bonus_score per setup (current-bar-only inputs, same
    # decide-at-close/fill-at-next-open timing the rest of the engine already
    # relies on, so this adds no new lookahead) but the engine never used it
    # to gate entries; every is_setup=True signal was taken regardless of
    # strength. This filters out the weakest setups instead of adding more
    # signal types.
    min_bonus_score: float = 0.0

    # Post-loss cooldown — after a symbol's trade closes at a loss, skip new
    # entries on that same symbol for this many bars. Uses only that symbol's
    # own already-closed trade history (sequential, no lookahead) on the
    # theory that whatever made the level fail once (a still-trending market
    # against the setup, a genuinely bad level) is likely to still be true
    # for a little while after.
    cooldown_bars_after_loss: int = 0

    # Hard per-trade loss cap, independent of the ATR trailing stop. The
    # trailing stop's width scales with that symbol's own recent volatility
    # (trade_atr_mult * ATR), so a high-volatility symbol can ride a single
    # trade much further underwater before the trailing stop ever triggers
    # (e.g. FIL/USDT lost -14.37% on one trade in the 5-year walk-forward,
    # well past any reasonable per-symbol cumulative-loss circuit breaker's
    # own per-trade resolution). This forces an exit the moment unrealized
    # loss reaches this percentage, regardless of ATR. `None` disables it
    # (default — matches prior behaviour exactly).
    max_single_trade_loss_pct: Optional[float] = None


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
