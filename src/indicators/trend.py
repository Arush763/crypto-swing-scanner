"""
Trend indicators: Exponential Moving Averages and derived signals.

All functions accept a pd.Series of prices and return pd.Series or scalar
values so they can be composed freely.
"""

from __future__ import annotations

import pandas as pd

from src.config.config import EMA_SHORT, EMA_MID, EMA_LONG


def ema(series: pd.Series, period: int) -> pd.Series:
    """Standard exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def ema_20(series: pd.Series) -> pd.Series:
    return ema(series, EMA_SHORT)


def ema_50(series: pd.Series) -> pd.Series:
    return ema(series, EMA_MID)


def ema_200(series: pd.Series) -> pd.Series:
    return ema(series, EMA_LONG)


def compute_trend_signals(close: pd.Series) -> dict:
    """
    Compute all EMA-based trend conditions for the latest bar.

    Returns a dict with five boolean conditions and the raw EMA values:
      - price_above_ema20
      - price_above_ema50
      - price_above_ema200
      - ema20_above_ema50  (golden-cross short)
      - ema50_above_ema200 (golden-cross long)
      - ema20, ema50, ema200 (latest values)
    """
    e20 = ema_20(close)
    e50 = ema_50(close)
    e200 = ema_200(close)

    latest_price = float(close.iloc[-1])
    latest_e20 = float(e20.iloc[-1])
    latest_e50 = float(e50.iloc[-1])
    latest_e200 = float(e200.iloc[-1])

    return {
        "price_above_ema20": latest_price > latest_e20,
        "price_above_ema50": latest_price > latest_e50,
        "price_above_ema200": latest_price > latest_e200,
        "ema20_above_ema50": latest_e20 > latest_e50,
        "ema50_above_ema200": latest_e50 > latest_e200,
        "ema20": latest_e20,
        "ema50": latest_e50,
        "ema200": latest_e200,
        "ema20_series": e20,
        "ema50_series": e50,
        "ema200_series": e200,
    }


def price_distance_from_ema(close: pd.Series, period: int) -> float:
    """Percentage distance of latest close from its EMA (positive = above)."""
    e = ema(close, period)
    return (float(close.iloc[-1]) - float(e.iloc[-1])) / float(e.iloc[-1])
