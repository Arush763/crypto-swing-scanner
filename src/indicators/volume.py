"""
Volume analysis indicators.

Covers: average volume, volume ratio, consistency score, and volume expansion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.config import VOLUME_CONSISTENCY_WINDOW, BREAKOUT_VOLUME_MULTIPLIER


def avg_volume(volume: pd.Series, period: int = 30) -> float:
    """Simple average volume over the last `period` bars."""
    return float(volume.iloc[-period:].mean())


def volume_ratio(volume: pd.Series, period: int = 30) -> float:
    """
    Ratio of the latest bar's volume to the period average.
    >1 = above average, >2 = strong expansion.
    """
    avg = avg_volume(volume, period)
    if avg == 0:
        return 0.0
    return float(volume.iloc[-1]) / avg


def volume_consistency_score(volume: pd.Series, window: int = VOLUME_CONSISTENCY_WINDOW) -> float:
    """
    Measures how consistently traded the asset is.

    Returns a 0-1 score: 1.0 = perfectly consistent daily volume (low CV),
    0.0 = extremely erratic volume (high CV).
    """
    recent = volume.iloc[-window:]
    if recent.empty or recent.mean() == 0:
        return 0.0
    cv = recent.std() / recent.mean()  # coefficient of variation
    # Invert and clip to [0, 1]
    return float(max(0.0, 1.0 - min(cv, 1.0)))


def is_volume_expansion(volume: pd.Series, multiplier: float = BREAKOUT_VOLUME_MULTIPLIER) -> bool:
    """True when today's volume exceeds `multiplier` × 30-day average."""
    return volume_ratio(volume) >= multiplier


def volume_trend_slope(volume: pd.Series, period: int = 20) -> float:
    """
    Linear regression slope of volume over `period` bars, normalised by mean.
    Positive = increasing volume trend.
    """
    recent = volume.iloc[-period:].values.astype(float)
    if len(recent) < 2:
        return 0.0
    x = np.arange(len(recent))
    slope = float(np.polyfit(x, recent, 1)[0])
    mean_vol = float(recent.mean())
    return slope / mean_vol if mean_vol > 0 else 0.0


def volume_signals(volume: pd.Series) -> dict:
    """Return all volume metrics for the latest bar."""
    return {
        "latest_volume": float(volume.iloc[-1]),
        "avg_volume_30d": avg_volume(volume, 30),
        "volume_ratio": volume_ratio(volume, 30),
        "volume_consistency": volume_consistency_score(volume),
        "is_expansion": is_volume_expansion(volume),
        "volume_slope_20d": volume_trend_slope(volume, 20),
    }
