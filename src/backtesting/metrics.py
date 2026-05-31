"""
Performance metrics library for the backtesting engine.

Each function is pure (takes arrays/series, returns a scalar) so they can
be used independently in optimisation sweeps or result reporting.
"""

from __future__ import annotations

from typing import List, Sequence
import numpy as np
import pandas as pd


def total_return(equity: pd.Series) -> float:
    """Total percentage return over the equity curve."""
    if len(equity) < 2 or float(equity.iloc[0]) == 0:
        return 0.0
    return float((equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0] * 100)


def cagr(equity: pd.Series, periods_per_year: int = 365) -> float:
    """Compound Annual Growth Rate assuming daily bars."""
    if len(equity) < 2 or float(equity.iloc[0]) == 0:
        return 0.0
    years = len(equity) / periods_per_year
    return float(((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100)


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative percentage."""
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(dd.min() * 100)


def max_drawdown_duration(equity: pd.Series) -> int:
    """Number of bars spent in the worst drawdown before recovery."""
    roll_max = equity.cummax()
    in_dd = equity < roll_max
    durations = []
    current = 0
    for val in in_dd:
        current = current + 1 if val else 0
        durations.append(current)
    return int(max(durations)) if durations else 0


def sharpe_ratio(equity: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 365) -> float:
    """Annualised Sharpe ratio."""
    returns = equity.pct_change().dropna()
    excess = returns - risk_free_rate / periods_per_year
    std = float(excess.std())
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(equity: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 365) -> float:
    """Annualised Sortino ratio (uses downside deviation only)."""
    returns = equity.pct_change().dropna()
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    downside_std = float(downside.std()) if len(downside) > 1 else 0.0
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def calmar_ratio(equity: pd.Series) -> float:
    """CAGR divided by absolute max drawdown."""
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return round(cagr(equity) / mdd, 3)


def win_rate(pnls: Sequence[float]) -> float:
    """Fraction of trades with positive P&L."""
    if not pnls:
        return 0.0
    winners = sum(1 for p in pnls if p > 0)
    return winners / len(pnls) * 100


def profit_factor(pnls: Sequence[float]) -> float:
    """Gross profit / gross loss."""
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    return round(gross_profit / gross_loss, 3) if gross_loss > 0 else float("inf")


def expectancy(pnls: Sequence[float]) -> float:
    """Average expected P&L per trade (in same units as pnls)."""
    return float(np.mean(pnls)) if pnls else 0.0


def avg_win(pnls: Sequence[float]) -> float:
    wins = [p for p in pnls if p > 0]
    return float(np.mean(wins)) if wins else 0.0


def avg_loss(pnls: Sequence[float]) -> float:
    losses = [p for p in pnls if p < 0]
    return float(np.mean(losses)) if losses else 0.0


def payoff_ratio(pnls: Sequence[float]) -> float:
    """Ratio of average win to average loss magnitude."""
    aw = avg_win(pnls)
    al = abs(avg_loss(pnls))
    return round(aw / al, 3) if al > 0 else 0.0


def consecutive_losses(pnls: Sequence[float]) -> int:
    """Maximum consecutive losing trades."""
    max_streak = current = 0
    for p in pnls:
        current = current + 1 if p < 0 else 0
        max_streak = max(max_streak, current)
    return max_streak


def recovery_factor(equity: pd.Series) -> float:
    """Total return / abs(max drawdown) — measures recovery speed."""
    mdd = abs(max_drawdown(equity))
    tr = total_return(equity)
    return round(tr / mdd, 3) if mdd > 0 else 0.0


def compute_all_metrics(equity: pd.Series, pnls: Sequence[float]) -> dict:
    """Return every metric in a single dict for display or storage."""
    return {
        "total_return_pct": round(total_return(equity), 2),
        "cagr_pct": round(cagr(equity), 2),
        "max_drawdown_pct": round(max_drawdown(equity), 2),
        "max_drawdown_duration_bars": max_drawdown_duration(equity),
        "sharpe_ratio": round(sharpe_ratio(equity), 3),
        "sortino_ratio": round(sortino_ratio(equity), 3),
        "calmar_ratio": calmar_ratio(equity),
        "recovery_factor": recovery_factor(equity),
        "win_rate_pct": round(win_rate(pnls), 1),
        "profit_factor": profit_factor(pnls),
        "expectancy_pct": round(expectancy(pnls), 3),
        "avg_win_pct": round(avg_win(pnls), 2),
        "avg_loss_pct": round(avg_loss(pnls), 2),
        "payoff_ratio": payoff_ratio(pnls),
        "max_consecutive_losses": consecutive_losses(pnls),
        "num_trades": len(pnls),
    }
