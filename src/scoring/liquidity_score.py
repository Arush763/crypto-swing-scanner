"""
Liquidity Score (20% of final score).

Evaluates the tradeability and data quality of an asset.

Components:
  - Daily volume (absolute and relative to universe)
  - Volume consistency (low day-to-day variation = reliable fills)
  - Market cap tier
  - Exchange coverage (how many of our watched exchanges carry the asset)
"""

from __future__ import annotations

import math
from typing import Optional
import pandas as pd

from src.indicators.volume import volume_signals, volume_consistency_score
from src.config.config import MIN_DAILY_VOLUME_USD, MIN_MARKET_CAP_USD


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _volume_to_points(daily_volume_usd: float, max_points: float = 35.0) -> float:
    """
    Logarithmic scale: $5M → 0 pts, $500M → full points.
    Assets below MIN_DAILY_VOLUME_USD should already be filtered out.
    """
    if daily_volume_usd <= 0:
        return 0.0
    lo = math.log10(max(MIN_DAILY_VOLUME_USD, 1))
    hi = math.log10(500_000_000)
    val = math.log10(daily_volume_usd)
    ratio = min(1.0, max(0.0, (val - lo) / (hi - lo)))
    return max_points * ratio


def _market_cap_to_points(market_cap_usd: float, max_points: float = 25.0) -> float:
    """$50M → 0 pts, $10B → full points."""
    if market_cap_usd <= 0:
        return max_points * 0.3   # unknown mcap gets partial credit
    lo = math.log10(max(MIN_MARKET_CAP_USD, 1))
    hi = math.log10(10_000_000_000)
    val = math.log10(market_cap_usd)
    ratio = min(1.0, max(0.0, (val - lo) / (hi - lo)))
    return max_points * ratio


def _consistency_to_points(consistency_score: float, max_points: float = 25.0) -> float:
    """consistency_score is already 0-1 from volume_consistency_score."""
    return max_points * max(0.0, min(1.0, consistency_score))


def _exchange_coverage_to_points(exchange_count: int, max_points: float = 15.0) -> float:
    """More exchanges = better liquidity. 4 exchanges = full points."""
    return min(max_points, max_points * (exchange_count / 4.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_liquidity_score(
    volume: pd.Series,
    market_cap_usd: float = 0.0,
    exchange_count: int = 1,
) -> dict:
    """
    Compute the liquidity score for an asset.

    Args:
        volume          : pd.Series of daily volume in base asset units
                          (we use USD-normalised volume from the fetcher)
        market_cap_usd  : latest market cap in USD (0 = unknown)
        exchange_count  : number of our watched exchanges listing this asset

    Returns:
        score     : 0-100 float
        breakdown : per-component point contributions
    """
    vol_sigs = volume_signals(volume)
    daily_vol_usd = vol_sigs["avg_volume_30d"]  # use 30d avg for stability
    consistency = vol_sigs["volume_consistency"]

    vol_pts = _volume_to_points(daily_vol_usd)
    mcap_pts = _market_cap_to_points(market_cap_usd)
    cons_pts = _consistency_to_points(consistency)
    exc_pts = _exchange_coverage_to_points(exchange_count)

    score = min(100.0, vol_pts + mcap_pts + cons_pts + exc_pts)

    return {
        "score": round(float(score), 2),
        "breakdown": {
            "volume": {"value": daily_vol_usd, "points": round(vol_pts, 2), "max": 35},
            "market_cap": {"value": market_cap_usd, "points": round(mcap_pts, 2), "max": 25},
            "consistency": {"value": consistency, "points": round(cons_pts, 2), "max": 25},
            "exchange_coverage": {"value": exchange_count, "points": round(exc_pts, 2), "max": 15},
        },
        "signals": vol_sigs,
    }
