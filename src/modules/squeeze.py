"""
Volatility Compression (Squeeze) Detection Module.

Finds assets that have been consolidating in a tight range — compressing
volatility — and are now starting to expand.  Breakouts from compression
often lead to outsized moves.

Detection logic:
  1. Both BB width percentile and ATR percentile are below threshold
     (compression phase).
  2. A recent bar breaks above the upper Bollinger Band or a swing high
     (expansion begins).
  3. Volume expands on the breakout bar (optional but preferred).

A squeeze breakout earns a bonus that rewards the compression → expansion
setup without double-counting the breakout module.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from src.config.config import SQUEEZE_PERCENTILE_THRESHOLD, BREAKOUT_VOLUME_MULTIPLIER
from src.indicators.volatility import (
    bollinger_bands,
    bb_width_percentile,
    atr_percentile,
    atr,
)
from src.indicators.volume import volume_ratio


@dataclass
class SqueezeResult:
    in_squeeze: bool             # Currently in compression phase
    squeeze_breakout: bool       # Breaking out of compression right now
    bb_width_pct: float          # BB width percentile (low = compressed)
    atr_pct: float               # ATR percentile (low = compressed)
    squeeze_duration_bars: int   # How long the squeeze lasted before breakout
    bonus_score: float           # Points added to composite score


def _squeeze_duration(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    threshold: float,
    lookback: int = 60,
) -> int:
    """Count consecutive bars the asset was in squeeze before the current bar."""
    count = 0
    for i in range(2, min(lookback, len(close)) + 1):
        sl = close.iloc[:-i]
        sh = high.iloc[:-i]
        sl2 = low.iloc[:-i]
        bb_p = bb_width_percentile(sl)
        at_p = atr_percentile(sh, sl2, sl)
        if bb_p < threshold and at_p < threshold:
            count += 1
        else:
            break
    return count


def detect_squeeze(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    threshold_pct: float = SQUEEZE_PERCENTILE_THRESHOLD,
    require_volume_expansion: bool = True,
) -> SqueezeResult:
    """
    Detect whether an asset is in a squeeze or breaking out of one.
    """
    bb_pct = bb_width_percentile(close)
    at_pct = atr_percentile(high, low, close)

    currently_compressed = bb_pct < threshold_pct and at_pct < threshold_pct

    # Previous bar was in squeeze but current bar breaks above upper BB
    bb = bollinger_bands(close)
    upper_bb = float(bb["upper"].iloc[-1])
    latest_close = float(close.iloc[-1])
    prev_close = float(close.iloc[-2]) if len(close) >= 2 else latest_close

    # Check previous bar's compression state
    prev_bb_pct = bb_width_percentile(close.iloc[:-1]) if len(close) > 1 else 100.0
    prev_at_pct = atr_percentile(high.iloc[:-1], low.iloc[:-1], close.iloc[:-1]) if len(close) > 1 else 100.0
    was_compressed = prev_bb_pct < threshold_pct and prev_at_pct < threshold_pct

    # Breakout: previous bar was squeezed AND current bar breaks upper BB
    is_breakout = was_compressed and latest_close > upper_bb

    if require_volume_expansion and is_breakout:
        vol_ratio = volume_ratio(volume, 30)
        is_breakout = is_breakout and vol_ratio >= BREAKOUT_VOLUME_MULTIPLIER

    duration = _squeeze_duration(high, low, close, threshold_pct) if is_breakout else 0

    # Bonus: more points for longer squeeze → stronger expected move
    if is_breakout:
        duration_bonus = min(5.0, duration * 0.5)
        bonus = round(8.0 + duration_bonus, 2)
    elif currently_compressed:
        bonus = 3.0   # Small reward for being primed (anticipatory)
    else:
        bonus = 0.0

    return SqueezeResult(
        in_squeeze=currently_compressed,
        squeeze_breakout=is_breakout,
        bb_width_pct=round(bb_pct, 2),
        atr_pct=round(at_pct, 2),
        squeeze_duration_bars=duration,
        bonus_score=bonus,
    )
