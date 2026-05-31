"""
Trend Score (30% of final score).

Grades the quality of the current uptrend using EMA alignment.
Each condition contributes a weighted point value; bonus points are
awarded when all conditions align simultaneously (perfect stack).
"""

from __future__ import annotations

from src.indicators.trend import compute_trend_signals, price_distance_from_ema
from src.config.config import EMA_SHORT, EMA_MID, EMA_LONG

import pandas as pd


# Individual condition weights (must sum to ≤ 100; remainder reserved for bonus)
_CONDITION_WEIGHTS = {
    "price_above_ema20": 15,
    "price_above_ema50": 20,
    "price_above_ema200": 25,
    "ema20_above_ema50": 20,
    "ema50_above_ema200": 20,
}
_FULL_ALIGNMENT_BONUS = 0  # already maxes at 100; no separate bonus needed


def compute_trend_score(close: pd.Series) -> dict:
    """
    Compute the trend score for a given price series.

    Returns:
        score       : 0-100 float
        breakdown   : per-condition boolean flags and their point contributions
        signals     : raw EMA values and derived signals
    """
    signals = compute_trend_signals(close)
    breakdown = {}
    raw_score = 0.0

    for condition, weight in _CONDITION_WEIGHTS.items():
        passed = bool(signals[condition])
        points = weight if passed else 0
        breakdown[condition] = {"passed": passed, "points": points, "max": weight}
        raw_score += points

    # Distance bonus: add up to 5 extra points when price is comfortably
    # above the 50 EMA (not overextended, but not scraping the line).
    dist_50 = price_distance_from_ema(close, EMA_MID)
    if 0.01 <= dist_50 <= 0.10:          # 1%-10% above 50 EMA = sweet spot
        distance_bonus = min(5.0, dist_50 * 50)
        raw_score = min(100.0, raw_score + distance_bonus)
    else:
        distance_bonus = 0.0

    return {
        "score": round(float(raw_score), 2),
        "breakdown": breakdown,
        "signals": signals,
        "distance_bonus": distance_bonus,
    }
