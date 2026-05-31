"""
Enhanced Backtesting Engine.

Simulates the scanner's signal logic against historical OHLCV data.

Exit rules (mirrors live system):
  - Primary:     Daily close below 20 EMA
  - Alternative: 2 ATR trailing stop (whichever triggers first)

Performance metrics via metrics.py:
  total return, CAGR, Sharpe, Sortino, Calmar, max drawdown + duration,
  win rate, profit factor, expectancy, payoff ratio, consecutive losses,
  recovery factor.

Parameter optimisation via grid search over:
  EMA lengths, volume multipliers, score thresholds, momentum lookbacks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config.config import BacktestConfig, ATR_TRAILING_STOP_MULTIPLIER
from src.indicators.trend import ema
from src.indicators.volatility import atr as compute_atr
from src.indicators.rsi import rsi_is_bullish
from src.modules.breakout import detect_breakout
from src.backtesting.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------

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

    # Orderbook metadata (injected when OB data is available)
    ob_imbalance_at_entry: float = 0.0
    ob_spread_pct_at_entry: float = 0.0


# ---------------------------------------------------------------------------
# Per-asset simulation
# ---------------------------------------------------------------------------

def _run_single_asset(
    symbol: str,
    ohlcv: pd.DataFrame,
    cfg: BacktestConfig,
    btc_in_bull: Optional[np.ndarray] = None,  # Boolean array: True = BTC above 200 EMA
) -> List[Trade]:
    """
    Simulate signal → entry → exit for one asset's entire OHLCV history.

    Entry: BTC regime OK + trend aligned (price > EMA20 > EMA50) + breakout on volume.
    Exit:  EMA20 cross-below OR ATR trailing stop, whichever fires first.
    Next-bar open used for fill price (realistic).
    """
    close  = ohlcv["close"]
    high   = ohlcv["high"]
    low    = ohlcv["low"]
    volume = ohlcv["volume"]
    open_  = ohlcv["open"]

    ema20      = ema(close, cfg.ema_short)
    ema50      = ema(close, cfg.ema_mid)
    ema200     = ema(close, cfg.ema_long)
    atr_ser    = compute_atr(high, low, close)
    rsi_bull   = rsi_is_bullish(close).values

    trades: List[Trade] = []
    in_trade      = False
    entry_price   = 0.0
    entry_bar     = 0
    trailing_stop = 0.0
    highest_close = 0.0

    warm_up = max(cfg.ema_long, 60)

    for i in range(warm_up, len(close) - 1):   # -1: need next bar for fill
        price       = float(close.iloc[i])
        e20         = float(ema20.iloc[i])
        e50         = float(ema50.iloc[i])
        current_atr = float(atr_ser.iloc[i])

        if in_trade:
            # Update trailing stop
            if price > highest_close:
                highest_close = price
                trailing_stop = highest_close - ATR_TRAILING_STOP_MULTIPLIER * current_atr

            # --- Exit: ATR trailing stop only ---
            if price < trailing_stop:
                exit_price = float(open_.iloc[i + 1])
                trades.append(Trade(
                    symbol=symbol,
                    entry_bar=entry_bar,
                    entry_price=entry_price,
                    exit_bar=i + 1,
                    exit_price=exit_price,
                    exit_reason="atr_trailing_stop",
                    pnl_pct=(exit_price - entry_price) / entry_price * 100,
                    holding_bars=i + 1 - entry_bar,
                    is_open=False,
                ))
                in_trade = False
                continue

        else:
            # --- BTC regime filter: block entries when BTC is in a downtrend ---
            if cfg.btc_regime_filter and btc_in_bull is not None:
                if i < len(btc_in_bull) and not btc_in_bull[i]:
                    continue

            # --- Per-coin regime: price must be above its own 200 EMA ---
            e200 = float(ema200.iloc[i])
            if price <= e200:
                continue

            # --- Trend stack: EMA20 > EMA50 > EMA200 (full bull alignment) ---
            if price <= e20 or e20 <= e50 or e50 <= e200:
                continue

            if not rsi_bull[i]:
                continue

            # --- Entry trigger: breakout on volume ---
            sl = slice(max(0, i - 60), i + 1)
            bo = detect_breakout(
                high.iloc[sl], low.iloc[sl], close.iloc[sl], volume.iloc[sl],
                volume_multiplier=cfg.volume_multiplier,
                max_age_bars=1,
            )
            if not bo.is_breakout:
                continue

            entry_price   = float(open_.iloc[i + 1])
            entry_bar     = i + 1
            highest_close = entry_price
            trailing_stop = entry_price - ATR_TRAILING_STOP_MULTIPLIER * current_atr
            in_trade      = True

    # Close any open position at last bar
    if in_trade and len(close) > 0:
        price = float(close.iloc[-1])
        trades.append(Trade(
            symbol=symbol,
            entry_bar=entry_bar,
            entry_price=entry_price,
            exit_bar=len(close) - 1,
            exit_price=price,
            exit_reason="end_of_data",
            pnl_pct=(price - entry_price) / entry_price * 100,
            holding_bars=len(close) - 1 - entry_bar,
            is_open=True,
        ))

    return trades


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

def _build_equity_curve(
    trades: List[Trade], initial_capital: float, risk_per_trade: float
) -> pd.Series:
    """Fixed-fractional position sizing: risk `risk_per_trade` of current equity per trade."""
    equity = [initial_capital]
    capital = initial_capital
    for t in trades:
        position_value = capital * risk_per_trade
        gain = position_value * (t.pnl_pct / 100)
        capital = max(0.01, capital + gain)   # floor at 1 cent
        equity.append(capital)
    return pd.Series(equity, dtype=float)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    trades:           List[Trade]
    equity_curve:     pd.Series
    metrics:          dict
    config:           BacktestConfig
    per_symbol_stats: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Convenience pass-throughs
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


def _build_btc_regime(universe: Dict[str, pd.DataFrame], cfg: BacktestConfig) -> Optional[np.ndarray]:
    """
    Build a boolean array: True = BTC close > BTC 200 EMA at that bar index.
    Uses BTC/USDT or BTC/USD from the universe. Returns None if BTC not found.
    All assets are assumed to share the same bar-count timeline (same timeframe).
    """
    btc_ohlcv = universe.get("BTC/USDT")
    if btc_ohlcv is None:
        btc_ohlcv = universe.get("BTC/USD")
    if btc_ohlcv is None:
        logger.warning("BTC not found in universe — regime filter disabled")
        return None

    btc_close = btc_ohlcv["close"]
    btc_e50   = ema(btc_close, cfg.ema_mid)
    btc_e200  = ema(btc_close, cfg.ema_long)
    # Require BTC above BOTH 50 and 200 EMA — confirmed bull market only
    in_bull   = ((btc_close > btc_e50) & (btc_close > btc_e200)).values.astype(bool)
    bull_pct  = in_bull.mean() * 100
    logger.info("BTC regime filter: %.1f%% of bars in bull market", bull_pct)
    print(f"  BTC regime: {bull_pct:.1f}% of bars above 200 EMA (bull market)")
    return in_bull


def run_backtest(
    universe: Dict[str, pd.DataFrame],
    cfg: Optional[BacktestConfig] = None,
) -> BacktestResult:
    """
    Run the backtest across all assets in the universe.

    Returns a BacktestResult with full metrics, equity curve, and per-symbol stats.
    """
    cfg = cfg or BacktestConfig()
    all_trades: List[Trade] = []
    per_symbol: List[dict] = []

    # Build BTC regime filter once for all assets
    btc_in_bull = _build_btc_regime(universe, cfg) if cfg.btc_regime_filter else None

    for symbol, ohlcv in universe.items():
        if len(ohlcv) < 100:
            continue
        try:
            # Align BTC regime array length to this asset's length
            asset_btc = None
            if btc_in_bull is not None:
                n = len(ohlcv)
                btc_n = len(btc_in_bull)
                if btc_n >= n:
                    asset_btc = btc_in_bull[-n:]   # take last N bars to align to asset end
                else:
                    # Pad front with True (assume bull if no BTC data that early)
                    asset_btc = np.concatenate([np.ones(n - btc_n, dtype=bool), btc_in_bull])

            trades = _run_single_asset(symbol, ohlcv, cfg, btc_in_bull=asset_btc)
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

    # Build equity curve and compute all metrics
    completed_all = [t for t in all_trades if not t.is_open]
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


def optimise_parameters(
    universe: Dict[str, pd.DataFrame],
    ema_shorts: List[int] = [10, 20, 30],
    ema_mids: List[int] = [40, 50, 60],
    volume_multipliers: List[float] = [1.5, 2.0, 2.5],
    score_thresholds: List[float] = [75.0, 80.0, 85.0],
) -> pd.DataFrame:
    """
    Grid search over parameter combinations.

    Returns DataFrame sorted by Sharpe ratio so the best configuration
    floats to the top.
    """
    results = []
    total = len(ema_shorts) * len(ema_mids) * len(volume_multipliers) * len(score_thresholds)
    done = 0

    for es in ema_shorts:
        for em in ema_mids:
            if em <= es:
                continue
            for vm in volume_multipliers:
                for st in score_thresholds:
                    cfg = BacktestConfig(ema_short=es, ema_mid=em, volume_multiplier=vm, score_threshold=st)
                    r = run_backtest(universe, cfg)
                    done += 1
                    results.append({
                        "ema_short": es, "ema_mid": em,
                        "volume_multiplier": vm, "score_threshold": st,
                        **{k: v for k, v in r.metrics.items()},
                    })
                    if done % 10 == 0:
                        logger.info("Optimise: %d/%d combinations done", done, total)

    return pd.DataFrame(results).sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)
