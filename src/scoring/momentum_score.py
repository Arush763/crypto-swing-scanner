"""
Momentum Score (30% of final score).

Combines n-day returns across multiple lookback windows with relative
strength versus Bitcoin and the broader crypto market.

Scoring approach:
  - Each return period is scored 0-20 using a sigmoid-like ramp.
  - RS vs BTC and RS vs market each contribute up to 10 points.
  - Total is normalised to 0-100.
"""

from __future__ import annotations

from typing import Optional
import pandas as pd

from src.indicators.momentum import momentum_score_raw
from src.config.config import MOMENTUM_PERIODS


# Points allocated per period (should sum to 60)
_PERIOD_MAX_POINTS = {7: 20, 14: 20, 30: 20}
_RS_MAX_POINTS = 20  # 10 vs BTC + 10 vs market


def _return_to_points(ret: float, max_points: float) -> float:
    """
    Map a return value to [0, max_points].

    Thresholds (tuned for daily swing trading):
      < 0%     →  0 pts          (negative momentum)
      0-5%     →  0-50% of max  (weak momentum)
      5-15%    →  50-100% of max (solid momentum)
      > 15%    →  100% of max   (strong momentum, capped)
    """
    if ret <= 0.0:
        return 0.0
    if ret < 0.05:
        return max_points * (ret / 0.05) * 0.5
    if ret < 0.15:
        return max_points * (0.5 + ((ret - 0.05) / 0.10) * 0.5)
    return float(max_points)


def _rs_to_points(rs: float, max_points: float) -> float:
    """
    Map relative-strength ratio to [0, max_points].

    RS == 1.0  →  half points  (market-neutral)
    RS == 1.5  →  full points  (50% stronger than benchmark)
    RS < 0.8   →  0 points
    """
    if rs < 0.8:
        return 0.0
    if rs < 1.0:
        return max_points * ((rs - 0.8) / 0.2) * 0.5
    if rs < 1.5:
        return max_points * (0.5 + ((rs - 1.0) / 0.5) * 0.5)
    return float(max_points)


def compute_momentum_score(
    close: pd.Series,
    btc_close: Optional[pd.Series] = None,
    market_close: Optional[pd.Series] = None,
) -> dict:
    """
    Compute the momentum score.

    Returns:
        score     : 0-100 float
        breakdown : per-metric contribution details
        raw       : raw computed values (returns, RS ratios)
    """
    raw = momentum_score_raw(close, btc_close, market_close, MOMENTUM_PERIODS)
    breakdown = {}
    total_points = 0.0

    for period in MOMENTUM_PERIODS:
        ret = raw.get(f"return_{period}d", 0.0)
        max_pts = _PERIOD_MAX_POINTS.get(period, 20)
        pts = _return_to_points(ret, max_pts)
        breakdown[f"return_{period}d"] = {
            "value": ret,
            "points": round(pts, 2),
            "max": max_pts,
        }
        total_points += pts

    rs_btc = raw.get("rs_vs_btc", 1.0)
    rs_mkt = raw.get("rs_vs_market", 1.0)
    rs_btc_pts = _rs_to_points(rs_btc, _RS_MAX_POINTS / 2)
    rs_mkt_pts = _rs_to_points(rs_mkt, _RS_MAX_POINTS / 2)

    breakdown["rs_vs_btc"] = {"value": rs_btc, "points": round(rs_btc_pts, 2), "max": _RS_MAX_POINTS / 2}
    breakdown["rs_vs_market"] = {"value": rs_mkt, "points": round(rs_mkt_pts, 2), "max": _RS_MAX_POINTS / 2}
    total_points += rs_btc_pts + rs_mkt_pts

    score = min(100.0, total_points)

    return {
        "score": round(float(score), 2),
        "breakdown": breakdown,
        "raw": raw,
    }
