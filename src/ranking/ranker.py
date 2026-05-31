"""
Relative Strength Ranking.

After the scanner scores every asset, this module produces leaderboards
across multiple dimensions:
  - Overall final score
  - Momentum rank
  - Trend quality rank
  - Volume growth rank
  - Liquidity rank

All ranks are percentile-based (100 = best in universe, 0 = worst).
"""

from __future__ import annotations

from typing import Dict, List
import pandas as pd

from src.scoring.composite import ScoreResult


def rank_results(scores: List[ScoreResult]) -> pd.DataFrame:
    """
    Build a comprehensive leaderboard DataFrame from a list of ScoreResults.

    Columns include per-category scores and percentile ranks.
    """
    if not scores:
        return pd.DataFrame()

    rows = []
    for s in scores:
        # Extract volume ratio from liquidity signals if available
        vol_ratio = 0.0
        if s.liquidity_detail and "signals" in s.liquidity_detail:
            vol_ratio = s.liquidity_detail["signals"].get("volume_ratio", 0.0)

        # 30-day return for momentum ranking
        ret_30d = 0.0
        if s.momentum_detail and "raw" in s.momentum_detail:
            ret_30d = s.momentum_detail["raw"].get("return_30d", 0.0)

        rows.append({
            "symbol": s.symbol,
            "final_score": s.final_score,
            "trend_score": s.trend_score,
            "momentum_score": s.momentum_score,
            "liquidity_score": s.liquidity_score,
            "smart_money_score": s.smart_money_score,
            "is_breakout": s.is_breakout,
            "is_retest": s.is_retest,
            "is_squeeze": s.is_squeeze,
            "breakout_bonus": s.breakout_bonus,
            "retest_bonus": s.retest_bonus,
            "squeeze_bonus": s.squeeze_bonus,
            "latest_price": s.latest_price,
            "atr": s.atr,
            "volume_ratio": vol_ratio,
            "return_30d": ret_30d,
        })

    df = pd.DataFrame(rows)

    # Add percentile rank columns (0-100, higher = better)
    rank_cols = {
        "final_score": "rank_overall",
        "momentum_score": "rank_momentum",
        "trend_score": "rank_trend",
        "volume_ratio": "rank_volume_growth",
        "liquidity_score": "rank_liquidity",
    }
    for col, rank_col in rank_cols.items():
        df[rank_col] = df[col].rank(pct=True) * 100

    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-based ranking
    return df


def top_n(df: pd.DataFrame, n: int, sort_by: str = "final_score") -> pd.DataFrame:
    """Return the top N rows sorted by the given column."""
    if sort_by not in df.columns:
        return df.head(n)
    return df.nlargest(n, sort_by).reset_index(drop=True)


def filter_breakouts(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_breakout"]].sort_values("final_score", ascending=False).reset_index(drop=True)


def filter_retests(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_retest"]].sort_values("final_score", ascending=False).reset_index(drop=True)


def filter_squeezes(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_squeeze"]].sort_values("final_score", ascending=False).reset_index(drop=True)


def leaderboard_summary(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return a dict of named leaderboard sub-tables."""
    return {
        "top_overall": top_n(df, 20, "final_score"),
        "top_momentum": top_n(df, 20, "momentum_score"),
        "top_trend": top_n(df, 20, "trend_score"),
        "top_volume_growth": top_n(df, 20, "volume_ratio"),
        "breakouts": filter_breakouts(df),
        "retests": filter_retests(df),
        "squeezes": filter_squeezes(df),
    }
