"""
Retest Confirmation Module.

After a breakout, price often pulls back to test the former resistance
(now acting as support) before continuing higher.  Entering on a confirmed
retest reduces risk and improves reward-to-risk.

Detection sequence:
  1. A breakout is identified (resistance broken on volume).
  2. Price retraces back within RETEST_TOLERANCE_PCT of the resistance level.
  3. The retest holds — price does NOT close back below resistance.
  4. A bullish confirmation candle forms (close > open, above resistance).

A confirmed retest scores higher than a raw breakout and provides a
better-defined entry zone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd

from src.config.config import RETEST_TOLERANCE_PCT
from src.modules.breakout import BreakoutResult


@dataclass
class RetestResult:
    is_retest: bool
    retest_level: float
    confirmation_candle: bool
    entry_zone_low: float
    entry_zone_high: float
    bonus_score: float           # Points added to composite score (higher than breakout alone)


def detect_retest(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    breakout: BreakoutResult,
    tolerance_pct: float = RETEST_TOLERANCE_PCT,
    lookback_bars: int = 10,     # Bars after breakout to watch for retest
) -> RetestResult:
    """
    Identify whether a retest of the breakout level has occurred.

    Requires a prior BreakoutResult to know the resistance level.
    """
    no_retest = RetestResult(
        is_retest=False,
        retest_level=breakout.resistance_level,
        confirmation_candle=False,
        entry_zone_low=0.0,
        entry_zone_high=0.0,
        bonus_score=0.0,
    )

    if not breakout.is_breakout:
        return no_retest

    res = breakout.resistance_level
    upper_band = res * (1.0 + tolerance_pct)
    lower_band = res * (1.0 - tolerance_pct)

    # The retest window starts after the breakout bar
    breakout_abs_idx = len(close) + breakout.breakout_bar_index  # convert negative to positive
    search_start = breakout_abs_idx + 1
    search_end = min(len(close), search_start + lookback_bars)

    if search_start >= len(close):
        return no_retest

    for i in range(search_start, search_end):
        low_i = float(low.iloc[i])
        close_i = float(close.iloc[i])
        open_i = float(open_.iloc[i])

        # Price touched the retest zone
        if low_i <= upper_band:
            # Retest holds: did NOT close back below resistance
            if close_i >= lower_band:
                # Check for bullish confirmation candle
                is_confirmation = close_i > open_i and close_i > res

                entry_low = lower_band
                entry_high = res * 1.01  # 1% above resistance

                # Higher bonus for confirmed retest vs raw breakout
                bonus = 15.0 if is_confirmation else 10.0

                return RetestResult(
                    is_retest=True,
                    retest_level=res,
                    confirmation_candle=is_confirmation,
                    entry_zone_low=round(entry_low, 8),
                    entry_zone_high=round(entry_high, 8),
                    bonus_score=bonus,
                )

    return no_retest
