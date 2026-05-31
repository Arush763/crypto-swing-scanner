"""
Momentum indicators: n-day returns and relative-strength ratios.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from src.config.config import MOMENTUM_PERIODS


def nday_return(close: pd.Series, n: int) -> float:
    """Simple percentage return over the last n bars."""
    if len(close) <= n:
        return 0.0
    return float((close.iloc[-1] / close.iloc[-n - 1]) - 1.0)


def all_momentum_returns(close: pd.Series, periods: List[int] = MOMENTUM_PERIODS) -> Dict[int, float]:
    """Return dict of {period: return} for each lookback period."""
    return {p: nday_return(close, p) for p in periods}


def relative_strength_vs_benchmark(
    asset_close: pd.Series, benchmark_close: pd.Series, period: int = 30
) -> float:
    """
    Compute the ratio of asset return vs benchmark return over `period` bars.

    RS > 1.0 → asset outperforms benchmark.
    Returns 0.0 if data is insufficient.
    """
    if len(asset_close) <= period or len(benchmark_close) <= period:
        return 0.0

    asset_ret = nday_return(asset_close, period)
    bench_ret = nday_return(benchmark_close, period)

    # Avoid division by zero; treat flat benchmark as neutral
    if bench_ret == 0.0:
        return 1.0 + asset_ret

    return (1.0 + asset_ret) / (1.0 + bench_ret)


def momentum_score_raw(
    close: pd.Series,
    btc_close: Optional[pd.Series],
    market_close: Optional[pd.Series],
    periods: List[int] = MOMENTUM_PERIODS,
) -> Dict[str, float]:
    """
    Compute all raw momentum inputs.

    Returns:
      - per-period returns
      - rs_vs_btc
      - rs_vs_market
    """
    result: Dict[str, float] = {}
    for p in periods:
        result[f"return_{p}d"] = nday_return(close, p)

    result["rs_vs_btc"] = (
        relative_strength_vs_benchmark(close, btc_close, period=30)
        if btc_close is not None
        else 1.0
    )
    result["rs_vs_market"] = (
        relative_strength_vs_benchmark(close, market_close, period=30)
        if market_close is not None
        else 1.0
    )
    return result
