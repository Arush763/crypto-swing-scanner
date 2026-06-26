"""
Tape-Based Backtesting Engine.

Simulates entries off the historical tape-signal proxy (src.modules.tape_signal)
against resampled trade-tick bars (src.data.trade_tape) — there is no free
historical L2 order-book archive to replay the live wall_signal.py against,
so this is the closest backtestable approximation of that strategy's edge.

Exit rule: 2 ATR trailing stop (mirrors the live system's alternative exit).

Performance metrics via metrics.py:
  total return, CAGR, Sharpe, Sortino, Calmar, max drawdown + duration,
  win rate, profit factor, expectancy, payoff ratio, consecutive losses,
  recovery factor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config.config import TapeBacktestConfig, ATR_TRAILING_STOP_MULTIPLIER
from src.indicators.trend import ema
from src.indicators.volatility import atr as compute_atr
from src.modules.tape_signal import detect_tape_signals
from src.modules.signal_filter import SignalFilter
from src.backtesting.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    symbol: str
    entry_bar: int
    entry_price: float
    exit_bar: int = -1
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_pct: float = 0.0
    holding_bars: int = 0
    is_open: bool = True
    signal_event: str = ""   # "ask_absorption" | "bid_repulsion"
    entry_time: Optional[pd.Timestamp] = None   # real timestamp, for ordering trades across symbols
    risk_pct: Optional[float] = None   # per-trade risk fraction override (e.g. confidence-scaled sizing); None = use the backtest's default


def simulate_exit(
    bars: pd.DataFrame,
    atr_ser: pd.Series,
    signal_bar_idx: int,
    atr_mult: float = ATR_TRAILING_STOP_MULTIPLIER,
) -> Optional[float]:
    """
    Simulate the ATR trailing-stop exit for a single hypothetical entry
    taken off the setup at `signal_bar_idx` (entry fills at the next bar's
    open, mirroring `_run_single_asset`). Used both by the live trade
    simulation there and by the ML signal-filter trainer, which needs a
    per-signal outcome label independent of the one-trade-at-a-time state
    machine (so overlapping setups each get their own label).

    `atr_mult` defaults to the global ATR_TRAILING_STOP_MULTIPLIER (existing
    behaviour for current callers); pass a backtest config's own
    atr_trailing_stop_mult to keep training labels consistent with whatever
    exit width that backtest actually uses.

    Returns None if there isn't enough data left to enter.
    """
    close = bars["close"]
    open_ = bars["open"]
    n = len(close)
    entry_bar = signal_bar_idx + 1
    if entry_bar >= n - 1:
        return None

    entry_price = float(open_.iloc[entry_bar])
    highest_close = entry_price
    trailing_stop = entry_price - atr_mult * float(atr_ser.iloc[signal_bar_idx])

    for i in range(entry_bar, n - 1):
        price = float(close.iloc[i])
        current_atr = float(atr_ser.iloc[i])
        if price > highest_close:
            highest_close = price
            trailing_stop = highest_close - atr_mult * current_atr
        if price < trailing_stop:
            exit_price = float(open_.iloc[i + 1])
            return (exit_price - entry_price) / entry_price * 100

    return (float(close.iloc[-1]) - entry_price) / entry_price * 100


def _daily_trend_bull(bars: pd.DataFrame, ema_period: int) -> np.ndarray:
    """
    Per-symbol daily-trend filter: resamples this asset's own close to daily
    bars, judges trend by close vs. EMA, then forward-fills that *prior*
    day's verdict across the current day's intraday bars (shifted by one day
    so today's still-forming daily close is never used before it exists).

    Alternative to the market-wide BTC-only regime filter (_build_btc_regime),
    which applies one proxy's trend to every symbol regardless of how that
    symbol itself is actually behaving.
    """
    daily_close = bars["close"].resample("1D").last()
    daily_bull = (daily_close > ema(daily_close, ema_period)).shift(1)
    aligned = daily_bull.reindex(bars.index, method="ffill")
    return aligned.fillna(False).values.astype(bool)


@dataclass
class _PositionState:
    """
    Mutable one-trade-at-a-time state for a single symbol, carried across
    calls to `_step_bar_range` — extracted out of `_run_single_asset` so the
    walk-forward engine (src.backtesting.walk_forward) can process a
    symbol's history in weekly chunks, persisting this same state across
    week boundaries, instead of every call starting flat from bar 0.
    """
    in_trade: bool = False
    entry_price: float = 0.0
    entry_bar: int = 0
    trailing_stop: float = 0.0
    highest_close: float = 0.0
    entry_event: str = ""
    trade_atr_mult: float = 0.0
    cooldown_until: int = -1
    entry_risk_pct: Optional[float] = None


def _step_bar_range(
    symbol: str,
    bars: pd.DataFrame,
    atr_ser: pd.Series,
    signals: pd.DataFrame,
    cfg: TapeBacktestConfig,
    momentum_atr_mult: float,
    state: _PositionState,
    lo: int,
    hi: int,
    btc_in_bull: Optional[np.ndarray] = None,
    daily_trend_bull: Optional[np.ndarray] = None,
) -> List[Trade]:
    """
    Process bar indices [lo, hi) — mutates `state` in place so a caller can
    resume from exactly where this left off (later bars, same open
    position/cooldown state). `hi` is clamped to len(close)-1 since entries
    fill at the *next* bar's open and the tail (open) end-of-data trade is
    handled by the caller, not here.
    """
    close = bars["close"]
    open_ = bars["open"]
    n = len(close)

    trades: List[Trade] = []
    for i in range(lo, min(hi, n - 1)):
        price       = float(close.iloc[i])
        current_atr = float(atr_ser.iloc[i])

        if state.in_trade:
            if price > state.highest_close:
                state.highest_close = price
                state.trailing_stop = state.highest_close - state.trade_atr_mult * current_atr

            unrealized_pct = (price - state.entry_price) / state.entry_price * 100
            hard_cap_hit = (
                cfg.max_single_trade_loss_pct is not None
                and unrealized_pct <= -cfg.max_single_trade_loss_pct
            )

            if price < state.trailing_stop or hard_cap_hit:
                exit_price = float(open_.iloc[i + 1])
                pnl_pct = (exit_price - state.entry_price) / state.entry_price * 100
                exit_reason = "max_single_trade_loss" if hard_cap_hit and not (price < state.trailing_stop) else "atr_trailing_stop"
                trades.append(Trade(
                    symbol=symbol,
                    entry_bar=state.entry_bar,
                    entry_price=state.entry_price,
                    exit_bar=i + 1,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    pnl_pct=pnl_pct,
                    holding_bars=i + 1 - state.entry_bar,
                    is_open=False,
                    signal_event=state.entry_event,
                    entry_time=bars.index[state.entry_bar],
                    risk_pct=state.entry_risk_pct,
                ))
                if cfg.cooldown_bars_after_loss > 0 and pnl_pct < 0:
                    state.cooldown_until = (i + 1) + cfg.cooldown_bars_after_loss
                state.in_trade = False
            continue

        if cfg.cooldown_bars_after_loss > 0 and i < state.cooldown_until:
            continue

        if cfg.btc_regime_filter and btc_in_bull is not None:
            if i < len(btc_in_bull) and not btc_in_bull[i]:
                continue

        if cfg.enable_daily_trend_filter and daily_trend_bull is not None:
            if i < len(daily_trend_bull) and not daily_trend_bull[i]:
                continue

        if not bool(signals["is_setup"].iloc[i]):
            continue

        if float(signals["bonus_score"].iloc[i]) < cfg.min_bonus_score:
            continue

        state.entry_price   = float(open_.iloc[i + 1])
        state.entry_bar     = i + 1
        state.entry_event   = str(signals["event"].iloc[i])
        state.trade_atr_mult = momentum_atr_mult if state.entry_event == "momentum_breakout" else cfg.atr_trailing_stop_mult
        state.highest_close = state.entry_price
        state.trailing_stop = state.entry_price - state.trade_atr_mult * current_atr
        state.entry_risk_pct = float(signals["risk_pct"].iloc[i]) if "risk_pct" in signals.columns else None
        state.in_trade      = True

    return trades


def _run_single_asset(
    symbol: str,
    bars: pd.DataFrame,
    cfg: TapeBacktestConfig,
    btc_in_bull: Optional[np.ndarray] = None,
    ml_filter: Optional["SignalFilter"] = None,
) -> List[Trade]:
    """
    Simulate tape-signal -> entry -> exit for one asset's bar history.

    Entry: BTC regime OK (optional) + a confirmed tape-signal setup.
    Exit:  2 ATR trailing stop.
    Next-bar open used for fill price (realistic).
    """
    close = bars["close"]
    high  = bars["high"]
    low   = bars["low"]

    daily_trend_bull = _daily_trend_bull(bars, cfg.daily_trend_ema_period) if cfg.enable_daily_trend_filter else None

    atr_ser = compute_atr(high, low, close)
    signals = detect_tape_signals(
        bars,
        lookback=cfg.lookback,
        proximity_pct=cfg.proximity_pct,
        volume_spike_mult=cfg.volume_spike_mult,
        bonus_pts=cfg.bonus_pts,
        ml_filter=ml_filter,
        two_phase_absorption=cfg.two_phase_absorption,
        two_phase_window=cfg.two_phase_window,
        two_phase_narrow_mult=cfg.two_phase_narrow_mult,
        cvd_filter=cfg.cvd_filter,
        cvd_window=cfg.cvd_window,
        stacked_bars=cfg.stacked_bars,
        enable_ask_absorption=cfg.enable_ask_absorption,
        enable_bid_repulsion=cfg.enable_bid_repulsion,
        enable_liquidity_sweep=cfg.enable_liquidity_sweep,
        sweep_window=cfg.sweep_window,
        sweep_volume_mult=cfg.sweep_volume_mult,
        enable_climax_exhaustion=cfg.enable_climax_exhaustion,
        climax_window=cfg.climax_window,
        climax_volume_mult=cfg.climax_volume_mult,
        climax_wide_mult=cfg.climax_wide_mult,
        enable_delta_divergence=cfg.enable_delta_divergence,
        enable_vwap_fade=cfg.enable_vwap_fade,
        vwap_window=cfg.vwap_window,
        vwap_stretch_pct=cfg.vwap_stretch_pct,
        vwap_volume_mult=cfg.vwap_volume_mult,
        enable_momentum_breakout=cfg.enable_momentum_breakout,
        breakout_margin_pct=cfg.breakout_margin_pct,
        breakout_volume_mult=cfg.breakout_volume_mult,
        breakout_dominance_ratio=cfg.breakout_dominance_ratio,
    )

    momentum_atr_mult = (
        cfg.momentum_atr_trailing_stop_mult
        if cfg.momentum_atr_trailing_stop_mult is not None
        else cfg.atr_trailing_stop_mult
    )

    state = _PositionState()
    warm_up = max(cfg.ema_long, 60)
    trades = _step_bar_range(
        symbol, bars, atr_ser, signals, cfg, momentum_atr_mult, state,
        warm_up, len(close), btc_in_bull, daily_trend_bull,
    )

    if state.in_trade and len(close) > 0:
        price = float(close.iloc[-1])
        trades.append(Trade(
            symbol=symbol,
            entry_bar=state.entry_bar,
            entry_price=state.entry_price,
            exit_bar=len(close) - 1,
            exit_price=price,
            exit_reason="end_of_data",
            pnl_pct=(price - state.entry_price) / state.entry_price * 100,
            holding_bars=len(close) - 1 - state.entry_bar,
            is_open=True,
            signal_event=state.entry_event,
            entry_time=bars.index[state.entry_bar],
            risk_pct=state.entry_risk_pct,
        ))

    return trades


def exclude_dominant_trades(trades: List[Trade], max_profit_share: float = 1 / 3) -> List[Trade]:
    """
    Strip out any single trade whose own profit exceeds `max_profit_share`
    of the strategy's total gross profit — a result carried by one or two
    outlier trades isn't evidence of a repeatable edge, it's a lottery
    ticket. Each trade's share is judged against the gross profit of the
    *original* trade list, computed once — not recursively re-derived from
    an ever-shrinking remainder, which would cascade and strip almost
    everything from a small, naturally lopsided sample (a handful of big
    winners alongside many small ones is a normal profile, not in itself
    evidence of overfitting).
    """
    winners = [t for t in trades if t.pnl_pct > 0]
    gross_profit = sum(t.pnl_pct for t in winners)
    if gross_profit <= 0:
        return list(trades)
    threshold = gross_profit * max_profit_share
    return [t for t in trades if not (t.pnl_pct > 0 and t.pnl_pct > threshold)]


def _build_equity_curve(
    trades: List[Trade], initial_capital: float, risk_per_trade: float
) -> pd.Series:
    """
    Fixed-fractional position sizing: risk `risk_per_trade` of current equity
    per trade, unless a trade carries its own `risk_pct` override (e.g.
    confidence-scaled sizing from the walk-forward's ML filter), in which
    case that trade's own fraction is used instead.
    """
    equity = [initial_capital]
    capital = initial_capital
    for t in trades:
        trade_risk = t.risk_pct if t.risk_pct is not None else risk_per_trade
        position_value = capital * trade_risk
        gain = position_value * (t.pnl_pct / 100)
        capital = max(0.01, capital + gain)
        equity.append(capital)
    return pd.Series(equity, dtype=float)


@dataclass
class BacktestResult:
    trades:           List[Trade]
    equity_curve:     pd.Series
    metrics:          dict
    config:           TapeBacktestConfig
    per_symbol_stats: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def total_return_pct(self) -> float:
        return self.metrics.get("total_return_pct", 0.0)
    @property
    def win_rate(self) -> float:
        return self.metrics.get("win_rate_pct", 0.0)
    @property
    def sharpe_ratio(self) -> float:
        return self.metrics.get("sharpe_ratio", 0.0)
    @property
    def max_drawdown_pct(self) -> float:
        return self.metrics.get("max_drawdown_pct", 0.0)
    @property
    def profit_factor(self):
        return self.metrics.get("profit_factor", 0.0)
    @property
    def avg_return_pct(self) -> float:
        return self.metrics.get("expectancy_pct", 0.0)
    @property
    def avg_holding_bars(self) -> float:
        completed = [t for t in self.trades if not t.is_open]
        if not completed:
            return 0.0
        return float(np.mean([t.holding_bars for t in completed]))
    @property
    def num_trades(self) -> int:
        return len(self.trades)
    @property
    def sortino_ratio(self) -> float:
        return self.metrics.get("sortino_ratio", 0.0)
    @property
    def calmar_ratio(self) -> float:
        return self.metrics.get("calmar_ratio", 0.0)


def _build_btc_regime(universe: Dict[str, pd.DataFrame], cfg: TapeBacktestConfig) -> Optional[np.ndarray]:
    btc_bars = universe.get("BTC/USDT")
    if btc_bars is None:
        btc_bars = universe.get("BTC/USD")
    if btc_bars is None:
        logger.warning("BTC not found in universe — regime filter disabled")
        return None

    btc_close = btc_bars["close"]
    btc_e50   = ema(btc_close, cfg.ema_mid)
    btc_e200  = ema(btc_close, cfg.ema_long)
    in_bull   = ((btc_close > btc_e50) & (btc_close > btc_e200)).values.astype(bool)
    bull_pct  = in_bull.mean() * 100
    logger.info("BTC regime filter: %.1f%% of bars in bull market", bull_pct)
    return in_bull


def run_backtest(
    universe: Dict[str, pd.DataFrame],
    cfg: Optional[TapeBacktestConfig] = None,
    ml_filter: Optional[SignalFilter] = None,
) -> BacktestResult:
    """
    Run the tape-signal backtest across all assets in the universe.

    `universe` maps symbol -> resampled bars (see src.data.trade_tape.resample_to_bars),
    with columns: open, high, low, close, volume, buy_volume, sell_volume.

    `ml_filter` defaults to loading the trained model at its default path
    (current behaviour). Pass an explicitly untrained/disabled SignalFilter
    to evaluate a signal variant on its own merits, independent of a model
    trained on the original signal's feature distribution.
    """
    cfg = cfg or TapeBacktestConfig()
    all_trades: List[Trade] = []
    per_symbol: List[dict] = []
    ml_filter = ml_filter if ml_filter is not None else SignalFilter()

    btc_in_bull = _build_btc_regime(universe, cfg) if cfg.btc_regime_filter else None

    for symbol, bars in universe.items():
        if len(bars) < 100:
            continue
        try:
            asset_btc = None
            if btc_in_bull is not None:
                n = len(bars)
                btc_n = len(btc_in_bull)
                if btc_n >= n:
                    asset_btc = btc_in_bull[-n:]
                else:
                    asset_btc = np.concatenate([np.ones(n - btc_n, dtype=bool), btc_in_bull])

            trades = _run_single_asset(symbol, bars, cfg, btc_in_bull=asset_btc, ml_filter=ml_filter)
            all_trades.extend(trades)

            completed = [t for t in trades if not t.is_open]
            if completed:
                pnls = [t.pnl_pct for t in completed]
                per_symbol.append({
                    "symbol": symbol,
                    "trades": len(trades),
                    "win_rate": round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
                    "avg_pnl_pct": round(float(np.mean(pnls)), 2),
                    "total_pnl_pct": round(float(sum(pnls)), 2),
                })
        except Exception as exc:
            logger.warning("Backtest failed for %s: %s", symbol, exc)

    # Trades were appended symbol-by-symbol above, not in the order they
    # actually occurred — sort by real entry timestamp before compounding,
    # since fixed-fractional position sizing depends on capital evolving in
    # the correct chronological sequence across the whole portfolio.
    completed_all = sorted((t for t in all_trades if not t.is_open), key=lambda t: t.entry_time)
    equity = _build_equity_curve(completed_all, cfg.initial_capital, cfg.risk_per_trade_pct)
    pnls = [t.pnl_pct for t in completed_all]
    metrics = compute_all_metrics(equity, pnls) if pnls else {
        "total_return_pct": 0.0, "win_rate_pct": 0.0, "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0, "profit_factor": 0.0, "expectancy_pct": 0.0,
        "sortino_ratio": 0.0, "calmar_ratio": 0.0, "num_trades": 0,
    }

    ps_df = pd.DataFrame(per_symbol).sort_values("total_pnl_pct", ascending=False).reset_index(drop=True) if per_symbol else pd.DataFrame()

    logger.info(
        "Backtest complete: %d trades | WR=%.1f%% | Sharpe=%.2f | MDD=%.1f%%",
        len(all_trades), metrics.get("win_rate_pct", 0), metrics.get("sharpe_ratio", 0), metrics.get("max_drawdown_pct", 0),
    )
    return BacktestResult(trades=all_trades, equity_curve=equity, metrics=metrics, config=cfg, per_symbol_stats=ps_df)


def recompute_excluding_dominant_trades(
    result: BacktestResult, max_profit_share: float = 1 / 3
) -> BacktestResult:
    """
    Re-derive a BacktestResult's equity curve and metrics after stripping out
    any trade whose own profit exceeds `max_profit_share` of total gross
    profit (see exclude_dominant_trades) — an honesty check for whether a
    backtest's headline numbers depend on a small number of outlier trades
    rather than a broad, repeatable edge. `result.trades` (incl. open ones)
    is filtered down to completed trades first, same as run_backtest does.
    """
    completed = sorted((t for t in result.trades if not t.is_open), key=lambda t: t.entry_time)
    filtered = exclude_dominant_trades(completed, max_profit_share)

    equity = _build_equity_curve(filtered, result.config.initial_capital, result.config.risk_per_trade_pct)
    pnls = [t.pnl_pct for t in filtered]
    metrics = compute_all_metrics(equity, pnls) if pnls else {
        "total_return_pct": 0.0, "win_rate_pct": 0.0, "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0, "profit_factor": 0.0, "expectancy_pct": 0.0,
        "sortino_ratio": 0.0, "calmar_ratio": 0.0, "num_trades": 0,
    }
    return BacktestResult(trades=filtered, equity_curve=equity, metrics=metrics, config=result.config)
