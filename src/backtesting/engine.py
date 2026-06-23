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


def _run_single_asset(
    symbol: str,
    bars: pd.DataFrame,
    cfg: TapeBacktestConfig,
    btc_in_bull: Optional[np.ndarray] = None,
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
    open_ = bars["open"]

    atr_ser = compute_atr(high, low, close)
    signals = detect_tape_signals(bars)

    trades: List[Trade] = []
    in_trade      = False
    entry_price   = 0.0
    entry_bar     = 0
    trailing_stop = 0.0
    highest_close = 0.0
    entry_event   = ""

    warm_up = max(cfg.ema_long, 60)

    for i in range(warm_up, len(close) - 1):
        price       = float(close.iloc[i])
        current_atr = float(atr_ser.iloc[i])

        if in_trade:
            if price > highest_close:
                highest_close = price
                trailing_stop = highest_close - ATR_TRAILING_STOP_MULTIPLIER * current_atr

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
                    signal_event=entry_event,
                ))
                in_trade = False
            continue

        if cfg.btc_regime_filter and btc_in_bull is not None:
            if i < len(btc_in_bull) and not btc_in_bull[i]:
                continue

        if not bool(signals["is_setup"].iloc[i]):
            continue

        entry_price   = float(open_.iloc[i + 1])
        entry_bar     = i + 1
        entry_event   = str(signals["event"].iloc[i])
        highest_close = entry_price
        trailing_stop = entry_price - ATR_TRAILING_STOP_MULTIPLIER * current_atr
        in_trade      = True

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
            signal_event=entry_event,
        ))

    return trades


def _build_equity_curve(
    trades: List[Trade], initial_capital: float, risk_per_trade: float
) -> pd.Series:
    """Fixed-fractional position sizing: risk `risk_per_trade` of current equity per trade."""
    equity = [initial_capital]
    capital = initial_capital
    for t in trades:
        position_value = capital * risk_per_trade
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
) -> BacktestResult:
    """
    Run the tape-signal backtest across all assets in the universe.

    `universe` maps symbol -> resampled bars (see src.data.trade_tape.resample_to_bars),
    with columns: open, high, low, close, volume, buy_volume, sell_volume.
    """
    cfg = cfg or TapeBacktestConfig()
    all_trades: List[Trade] = []
    per_symbol: List[dict] = []

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

            trades = _run_single_asset(symbol, bars, cfg, btc_in_bull=asset_btc)
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
