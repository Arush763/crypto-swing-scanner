"""
Breakout Detection Module.

Identifies when price closes above a well-defined resistance level on
significantly elevated volume, signalling a potential trend expansion.

A valid breakout requires:
  1. A resistance level derived from recent swing highs.
  2. Daily close above that resistance.
  3. Volume ≥ BREAKOUT_VOLUME_MULTIPLIER × 30-day average volume.
  4. Price still above breakout level at close (no wick rejection).

The module produces a bonus score (0-10 pts) that the composite scorer adds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from src.config.config import (
    BREAKOUT_RESISTANCE_LOOKBACK,
    BREAKOUT_VOLUME_MULTIPLIER,
)
from src.indicators.volume import volume_ratio


@dataclass
class BreakoutResult:
    is_breakout: bool
    resistance_level: float
    breakout_bar_index: int          # Index in the OHLCV series (-1 = no breakout)
    volume_ratio_at_breakout: float
    bonus_score: float               # Points added to composite score
    age_bars: int                    # How many bars ago the breakout occurred (0 = latest bar)


def _swing_highs(high: pd.Series, lookback: int) -> pd.Series:
    """
    Return a boolean Series marking bars where `high` is the local maximum
    within a ±lookback/2 window on either side.
    """
    window = max(3, lookback // 4)
    rolling_max = high.rolling(window, center=True).max()
    return high == rolling_max


def resistance_level(high: pd.Series, lookback: int = BREAKOUT_RESISTANCE_LOOKBACK) -> float:
    """
    Identify the most recent dominant resistance level from swing highs.
    Returns the highest swing high within the lookback window, excluding
    the last bar (current candle).
    """
    window = high.iloc[-(lookback + 1):-1]   # exclude current bar
    if window.empty:
        return float("inf")
    return float(window.max())


def detect_breakout(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    lookback: int = BREAKOUT_RESISTANCE_LOOKBACK,
    volume_multiplier: float = BREAKOUT_VOLUME_MULTIPLIER,
    max_age_bars: int = 3,           # Look at last N bars for recent breakout
) -> BreakoutResult:
    """
    Check for a breakout on the last `max_age_bars` bars.

    Returns BreakoutResult with is_breakout=False if no breakout is found.
    """
    resistance = resistance_level(high, lookback)
    no_breakout = BreakoutResult(
        is_breakout=False,
        resistance_level=resistance,
        breakout_bar_index=-1,
        volume_ratio_at_breakout=0.0,
        bonus_score=0.0,
        age_bars=-1,
    )

    if resistance == float("inf"):
        return no_breakout

    vol_30d_avg = float(volume.iloc[-30:].mean())

    # Search recent bars from newest to oldest
    for age in range(min(max_age_bars, len(close))):
        idx = -(age + 1)
        c = float(close.iloc[idx])
        v = float(volume.iloc[idx])

        if c <= resistance:
            continue

        vol_ratio = v / vol_30d_avg if vol_30d_avg > 0 else 0.0
        if vol_ratio < volume_multiplier:
            continue

        # Valid breakout found — score decays with age
        age_decay = max(0.0, 1.0 - age * 0.25)   # -25% per bar of age
        vol_bonus = min(1.0, (vol_ratio - volume_multiplier) / volume_multiplier)
        bonus = round(10.0 * age_decay * (0.7 + 0.3 * vol_bonus), 2)

        return BreakoutResult(
            is_breakout=True,
            resistance_level=resistance,
            breakout_bar_index=idx,
            volume_ratio_at_breakout=round(vol_ratio, 2),
            bonus_score=bonus,
            age_bars=age,
        )

    return no_breakout
