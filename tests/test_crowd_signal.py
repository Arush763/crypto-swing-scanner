"""
Tests for the crowd-short signal.

This is the one validated edge in the project, so the properties that matter
are the ones that would silently change *which* signal is being traded:
percentile ranking rather than absolute level, refusal to fire on an
unvalidated data source, and refusal to fire on too little history.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modules.crowd_signal import (
    CrowdShortSignal,
    DEFAULT_SHORT_PERCENTILE,
    HOLD_HOURS,
)


def _history(values) -> pd.Series:
    idx = pd.date_range("2026-01-01", periods=len(values), freq="1h", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


@pytest.fixture
def signal():
    # Binance is the validated source, so these tests exercise the signal
    # logic rather than the source gate (which has its own tests below).
    return CrowdShortSignal(source="binance", min_history=100)


class _Snap:
    def __init__(self, ratio, history):
        self.long_short_ratio = ratio
        self.history = history


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def test_fires_when_crowd_is_at_an_extreme(signal):
    v = signal.evaluate("BTC/USDT", 5.0, _history(np.linspace(1.0, 2.0, 500)))
    assert v.fired
    assert v.percentile == pytest.approx(1.0)


def test_silent_when_crowd_is_normal(signal):
    v = signal.evaluate("BTC/USDT", 1.5, _history(np.linspace(1.0, 2.0, 500)))
    assert not v.fired
    assert "not extreme" in v.reason


def test_ranks_within_symbol_not_against_an_absolute_level(signal):
    """
    Ratio levels differ ~2x across symbols and venues. A reading of 2.5 must
    be able to fire for a symbol that normally sits near 1.0 and stay silent
    for one that normally sits near 3.0 — a global threshold would trade some
    symbols constantly and others never.
    """
    low = _history(np.linspace(0.8, 1.2, 500))
    high = _history(np.linspace(2.8, 3.6, 500))

    assert signal.evaluate("A/USDT", 2.5, low).fired
    assert not signal.evaluate("B/USDT", 2.5, high).fired


def test_current_reading_is_not_part_of_its_own_distribution(signal):
    """
    The caller supplies history excluding the current value. Verify the
    percentile reflects that — if the reading were included, an extreme value
    would be pulled toward the middle of its own window.
    """
    hist = _history(np.full(500, 1.0))
    v = signal.evaluate("BTC/USDT", 9.0, hist)
    assert v.fired
    assert v.observations == 500


def test_signal_is_always_short(signal):
    snaps = {"BTC/USDT": _Snap(5.0, _history(np.linspace(1.0, 2.0, 500)))}
    signals, _ = signal.generate(snaps)
    assert len(signals) == 1
    assert signals[0].direction == -1
    assert signals[0].is_short
    assert signals[0].hold_hours == HOLD_HOURS


def test_low_ratio_never_produces_a_long(signal):
    """
    The mirror trade loses -0.240%/trade. A crowd at its least-long extreme
    must produce no signal at all, not a long.
    """
    snaps = {"BTC/USDT": _Snap(0.1, _history(np.linspace(1.0, 2.0, 500)))}
    signals, verdicts = signal.generate(snaps)
    assert signals == []
    assert not verdicts[0].fired


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def test_refuses_on_insufficient_history(signal):
    v = signal.evaluate("BTC/USDT", 5.0, _history(np.linspace(1.0, 2.0, 50)))
    assert not v.fired
    assert "observations" in v.reason


def test_refuses_on_missing_ratio(signal):
    assert not signal.evaluate("BTC/USDT", float("nan"), _history([1.0] * 500)).fired
    assert not signal.evaluate("BTC/USDT", 0.0, _history([1.0] * 500)).fired


def test_refuses_unvalidated_source_symbol_pairs():
    """
    OKX's ratio tracks Binance's for BTC/ETH/SOL but is weak or inverted
    elsewhere (AVAX rank correlation -0.41). Trading those live would fire at
    different — sometimes opposite — times to the backtest.
    """
    okx = CrowdShortSignal(source="okx", min_history=100)
    hist = _history(np.linspace(1.0, 2.0, 500))

    assert okx.evaluate("BTC/USDT", 5.0, hist).fired
    assert okx.evaluate("ETH/USDT", 5.0, hist).fired
    assert okx.evaluate("SOL/USDT", 5.0, hist).fired

    for symbol in ("AVAX/USDT", "LINK/USDT", "DOGE/USDT"):
        v = okx.evaluate(symbol, 5.0, hist)
        assert not v.fired
        assert "not validated" in v.reason


def test_source_gate_can_be_disabled_explicitly():
    okx = CrowdShortSignal(source="okx", min_history=100, require_validated_source=False)
    assert okx.evaluate("AVAX/USDT", 5.0, _history(np.linspace(1.0, 2.0, 500))).fired


def test_unknown_source_is_refused_by_default():
    other = CrowdShortSignal(source="bitget", min_history=100)
    assert not other.evaluate("BTC/USDT", 5.0, _history(np.linspace(1.0, 2.0, 500))).fired


def test_percentile_threshold_matches_the_validated_setting():
    assert DEFAULT_SHORT_PERCENTILE == 0.90
    assert HOLD_HOURS == 16
