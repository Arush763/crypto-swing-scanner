"""
ML Signal Filter — gates wall-signal setups (both ask_absorption and
bid_repulsion) by a trained win-probability model instead of a fixed
boolean rule.

Why: a year-long backtest showed ask_absorption has ~no real edge (42%
forward-return win rate, and counter-intuitively gets *worse* with stronger
breakouts/buy-margins — a buy-the-top pattern, not real absorption), while
bid_repulsion is consistently profitable. Rather than hard-dropping
ask_absorption, this model learns from the same setup features which
individual occurrences (of either event type) are actually worth taking,
trained on real per-signal trade outcomes (src.backtesting.engine.simulate_exit).

Feature schema is intentionally small and defined identically for both
regimes, since the live wall_signal.py (order book + live trade flow) and
the backtest's tape_signal.py (resampled trade-tick bars) observe different
underlying data:
  - is_ask              : 1.0 for ask_absorption, 0.0 for bid_repulsion
  - distance_pct         : how close price was to the level when tested
  - dominance_margin      : buy_volume / sell_volume on the confirmation bar/cycle
  - move_strength_pct    : signed % move past the level in the trade's favor
  - volume_ratio          : confirmation-bar volume vs. its recent baseline
                            (neutral 1.0 live, where no baseline is tracked yet)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "is_ask",
    "distance_pct",
    "dominance_margin",
    "move_strength_pct",
    "volume_ratio",
]

DEFAULT_MODEL_PATH = Path("data/models/signal_filter.joblib")


class SignalFilter:
    """
    Thin wrapper around a persisted scikit-learn pipeline (StandardScaler +
    LogisticRegression) that scores a setup's win probability from
    FEATURE_NAMES. Falls back to "always pass" (probability 1.0) if no
    model has been trained yet, so the system degrades to its prior
    threshold-only behaviour rather than blocking all signals.
    """

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._pipeline = None
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            logger.warning(
                "No signal-filter model at %s — filter disabled (all setups pass)",
                self.model_path,
            )
            return
        import joblib
        self._pipeline = joblib.load(self.model_path)

    @property
    def is_trained(self) -> bool:
        return self._pipeline is not None

    def score(self, features: Dict[str, float]) -> float:
        """Return predicted win probability in [0, 1]. 1.0 if no model loaded."""
        if self._pipeline is None:
            return 1.0
        row = pd.DataFrame([[features.get(name, 0.0) for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
        return float(self._pipeline.predict_proba(row)[0, 1])

    def score_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """Vectorised scoring for backtesting — features_df columns must match FEATURE_NAMES."""
        if self._pipeline is None:
            return np.ones(len(features_df))
        return self._pipeline.predict_proba(features_df[FEATURE_NAMES])[:, 1]

    def passes(self, features: Dict[str, float], threshold: float) -> bool:
        return self.score(features) >= threshold


def tape_features(bars: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    Vectorised FEATURE_NAMES computation aligned to `bars.index`, covering
    both event types at once (the caller selects rows by event mask). Mirrors
    the quantities src.modules.tape_signal.detect_tape_signals already
    computes internally, factored out so training and inference share one
    definition.
    """
    close = bars["close"]
    high = bars["high"]
    low = bars["low"]
    volume = bars["volume"]
    buy_volume = bars["buy_volume"]
    sell_volume = bars["sell_volume"]

    resistance = high.rolling(lookback).max().shift(1)
    support = low.rolling(lookback).min().shift(1)
    avg_volume = volume.rolling(lookback).mean().shift(1)
    prev_close = close.shift(1)

    dist_to_res = (resistance - prev_close) / resistance
    dist_to_sup = (prev_close - support) / support

    sell_safe = sell_volume.where(sell_volume > 0, 1e-9)
    buy_safe = buy_volume.where(buy_volume > 0, 1e-9)
    dominance_margin = buy_volume / sell_safe

    breakout_strength = (close - resistance) / resistance * 100
    bounce_strength = (close - prev_close) / prev_close * 100

    volume_ratio = volume / avg_volume.where(avg_volume > 0, 1e-9)

    return pd.DataFrame({
        "dist_to_res": dist_to_res,
        "dist_to_sup": dist_to_sup,
        "dominance_margin": dominance_margin,
        "sell_over_buy": sell_volume / buy_safe,
        "breakout_strength": breakout_strength,
        "bounce_strength": bounce_strength,
        "volume_ratio": volume_ratio,
    })


def select_tape_features(raw: pd.DataFrame, is_ask_mask: pd.Series) -> pd.DataFrame:
    """Collapse tape_features()'s per-event columns into the shared FEATURE_NAMES schema."""
    is_ask = is_ask_mask.astype(float)
    distance_pct = raw["dist_to_res"].where(is_ask_mask, raw["dist_to_sup"])
    dominance_margin = raw["dominance_margin"].where(is_ask_mask, raw["sell_over_buy"])
    move_strength_pct = raw["breakout_strength"].where(is_ask_mask, raw["bounce_strength"])

    return pd.DataFrame({
        "is_ask": is_ask,
        "distance_pct": distance_pct,
        "dominance_margin": dominance_margin,
        "move_strength_pct": move_strength_pct,
        "volume_ratio": raw["volume_ratio"],
    })


def live_features(
    is_ask: bool,
    distance_pct: float,
    move_strength_pct: float,
    buy_volume_usd: float,
    sell_volume_usd: float,
) -> Dict[str, float]:
    """
    Best-effort feature mapping from live order-book/flow state onto the same
    FEATURE_NAMES schema. volume_ratio has no live equivalent yet (no rolling
    baseline is tracked for OB/flow data), so it's held at a neutral 1.0 —
    the model was trained with that baseline present, so this is a known
    approximation, not an exact replay of the backtest features.
    """
    sell_safe = sell_volume_usd if sell_volume_usd > 0 else 1e-9
    return {
        "is_ask": 1.0 if is_ask else 0.0,
        "distance_pct": distance_pct,
        "dominance_margin": buy_volume_usd / sell_safe,
        "move_strength_pct": move_strength_pct,
        "volume_ratio": 1.0,
    }
