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
# Majors-only universe restriction
# ---------------------------------------------------------------------------
# A $3M-daily-volume filter admits assets whose bid/ask spread is 25bps or
# worse. Round-tripping one of those costs ~0.32% against a strategy whose
# measured gross edge is ~0.24%/trade — the position is underwater the
# instant it opens, and no amount of signal quality recovers that.
#
# Restricting to majors is what makes frequent trading arguable at all: on
# BTC/ETH the spread term is ~0.5bps rather than 25bps, so cost is dominated
# by fees, which post-only entries can halve. See src/execution/costs.py.
#
# Set MAJORS_ONLY = False to restore the old wide-universe behaviour.
MAJORS_ONLY: bool = True

# Base assets permitted when MAJORS_ONLY is on. Deliberately conservative —
# membership is by sustained real book depth, not market-cap ranking, since
# it's depth that determines what a round trip actually costs.
MAJOR_BASES: List[str] = [
    "BTC", "ETH",                                    # mega tier
    "SOL", "XRP", "BNB", "ADA", "DOGE", "AVAX",      # large tier
    "LINK", "DOT", "LTC", "BCH", "TRX", "UNI",
    "ATOM", "XLM", "NEAR", "APT", "ARB", "OP", "SUI",
]

# Volume floor applied to majors. Far higher than MIN_DAILY_VOLUME_USD
# because the point is depth, not mere listing — a major with only $3M of
# daily turnover on a given venue is a thin book wearing a familiar ticker.
MAJORS_MIN_DAILY_VOLUME_USD: float = 25_000_000

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
    btc_regime_filter: bool = True          # Only enter when BTC > 50/200 EMA

    # ---- Trading costs -------------------------------------------------
    # This replaces the old `commission_pct` field, which was declared here
    # but referenced nowhere in the engine — every backtest this repo has
    # produced was gross of fees, spread, and slippage. See
    # src/execution/costs.py for the decomposition and why it matters far
    # more at high frequency than at 4h.
    #
    # `apply_costs=False` reproduces the old (gross) behaviour, which is
    # useful only for comparing against historical results — never for
    # judging whether a strategy is tradeable.
    apply_costs: bool = True
    venue: str = "okx"
    entry_is_maker: bool = False
    exit_is_maker: bool = False
    # Notional per trade used for the market-impact term. Impact is
    # negligible for small orders on majors, which is why this defaults low;
    # raise it to see how the edge degrades as size grows.
    order_size_usd: float = 1_000.0
    impact_coefficient: float = 10.0

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
# Flow-concentration gate (src/modules/flow_gate.py)
# ---------------------------------------------------------------------------
# Suppresses entries taken while volume is dominated by a few very large
# prints. Measured over 90 days across 12 majors, a high whale share precedes
# SMALLER subsequent moves — monotonically across deciles, in both train and
# test, at every timeframe tested. Applied as a filter to the existing tape
# signal it moved gross P&L from -0.052%/trade (gate shut) to +0.224% (gate
# open) and win rate from 27.9% to 39.1%.
#
# Note this runs opposite to the usual intuition that whale activity signals
# an imminent move. The reading that fits the data: concentrated prints are
# liquidity being consumed (a block crossing, a liquidation absorbed), which
# is the move ending rather than starting.
#
# DISABLED after full-year validation. The filter looked net-positive at 90d
# (+0.059%/trade) and 200d (+0.125%), but on 365 days across 12 majors it went
# NEGATIVE: -0.099%/trade on 221 out-of-sample trades, gross t=1.63, net
# t=-1.22. The earlier samples were inside their own confidence intervals the
# whole time — at 200d the net CI was [-0.162%, +0.411%], which contains the
# year's result.
#
# What survives is discrimination, not profit: gate-open trades still average
# +0.108% gross vs -0.061% when shut (a +0.169% swing, 33.8% vs 28.4% win
# rate) across 616 trades. So whale_share is a real signal and a poor
# money-maker — filtering a losing strategy more selectively does not make it
# a winning one. Left in the codebase, off by default. See
# scripts/study_gate_as_filter.py to re-check.
FLOW_GATE_ENABLED: bool = False

# Percentile of a symbol's OWN whale-share history above which entries are
# suppressed. Per-symbol because whale share differs ~5x across majors (90d
# 30th percentile: BTC 0.374, LINK 0.082) — a fixed cutoff would suppress
# every BTC signal and no LINK signal.
#   0.30 = aggressive (only the broadest-participation 30% of readings trade)
#   0.50 = moderate, the default
FLOW_GATE_MAX_PERCENTILE: float = 0.50

# ---------------------------------------------------------------------------
# Execution / auto-trading
# ---------------------------------------------------------------------------
# Venue used for both cost modelling and live order routing. OKX is the
# default because it's geo-accessible from where this scanner runs (unlike
# binance/bybit — see EXCHANGES above), has deep major-pair books, and
# offers a demo-trading mode for validating the executor without capital.
#
# Must be a key in src/execution/venues.VENUES — that registry, not this
# string, is what knows whether the venue is spot or perp and whether it can
# hold a short. Setting a venue here that cannot express a strategy's direction
# does not degrade gracefully; it refuses, by design.
EXECUTION_VENUE: str = "okx"

# Coinbase target for scripts/run_coinbase_trader.py.
#
# Defaults to the CFTC-regulated Coinbase Derivatives Exchange (CDE) futures,
# reached through the same ccxt client as Advanced Trade. This is the only
# Coinbase product that is BOTH shortable AND open to US persons — the strategy
# is short-only, Coinbase's spot venues have no borrow (and cost ~1.8-2.4% per
# round trip against a +0.655% edge), and Coinbase International's perps are
# closed to US persons.
#
# The contracts are nano-sized and dated; the furthest series (Dec 2030) is the
# "perp" product and is what src/execution/venues.resolve_contract picks.
COINBASE_VENUE: str = "coinbasederivatives"

# Post-only (maker) entries. This is the single largest cost lever available:
# it drops the entry leg from taker fee + half-spread to just the maker fee.
# The tradeoff is fill uncertainty — a post-only order that never gets
# crossed simply doesn't trade, so some signals are missed rather than
# entered late. At high frequency that tradeoff strongly favours post-only.
ENTRY_POST_ONLY: bool = True

# How long to leave an unfilled post-only entry resting before abandoning the
# signal. Past this the setup that justified the entry has usually decayed.
ENTRY_LIMIT_TIMEOUT_SECONDS: int = 90

# ---- Hard risk limits (enforced by src/execution/executor.py) -------------
# These are circuit breakers, not suggestions. The executor refuses to open a
# position that would violate any of them, and the daily-loss breaker halts
# all new entries until the next UTC day once tripped.
MAX_CONCURRENT_POSITIONS: int = 3
# Per-position notional ceiling. Raised from $500 to $800 for the Coinbase CDE
# venue: contracts are indivisible, and one nano BTC contract is ~$770, so a
# $500 cap made BTC, XRP, BNB, LINK and SOL literally unsizeable — the runner
# would refuse the five largest-notional (and therefore cheapest per fee)
# contracts and trade only DOGE. This is a real increase in per-position risk on
# a strategy that carries no stop; lower it and you lose symbols, in order of
# contract size. See src/execution/contracts.size_in_contracts.
MAX_POSITION_USD: float = 800.0
MAX_DAILY_LOSS_USD: float = 100.0           # Halt trading for the day at this realised loss
MAX_DAILY_TRADES: int = 60                  # Runaway-loop guard
RISK_PER_TRADE_PCT: float = 0.01            # Fraction of equity risked per trade

# Live trading is opt-in and requires BOTH this flag and API credentials in
# the environment. Default is paper: orders are simulated against the live
# book, nothing is sent to the exchange. Never flip this to True without
# having watched the paper log first.
LIVE_TRADING_ENABLED: bool = False

# ---------------------------------------------------------------------------
# High-frequency scan loop
# ---------------------------------------------------------------------------
# The wall signal reads order-book state changes between consecutive scans.
# Sampling that every 4h (the old GitHub Actions cron) means almost every
# absorption/repulsion event is born and dies unobserved between snapshots —
# a structural mismatch between the signal's timescale and the observation
# rate, and a large part of why so few signals ever fired.
HFT_SCAN_INTERVAL_SECONDS: int = 60
HFT_SCAN_TIMEFRAME: str = "5m"
HFT_OHLCV_LIMIT: int = 500
# Lower than SIGNAL_SCORE_THRESHOLD (80): over 40 logged scan cycles the top
# composite score was 59-79, so an 80 gate fired roughly once a week. The
# quality bar at high frequency is enforced by the cost check and the ML
# filter, not by a score threshold the scoring function rarely reaches.
HFT_SCORE_THRESHOLD: float = 62.0

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
