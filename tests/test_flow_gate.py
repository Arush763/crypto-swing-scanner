"""
Tests for the live flow-concentration gate.

The gate suppresses real trades, so the properties that matter most are the
ones that decide whether it fails open or closed, and whether it judges each
symbol against the right distribution.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.orderbook import FlowSignals
from src.modules.flow_gate import FlowConcentrationGate


@pytest.fixture
def gate(tmp_path):
    return FlowConcentrationGate(
        state_path=str(tmp_path / "flow_gate.json"),
        max_percentile=0.50,
        min_observations=10,
    )


def _warm(gate, symbol, values):
    for v in values:
        gate.observe(symbol, v)


# ---------------------------------------------------------------------------
# Warm-up behaviour
# ---------------------------------------------------------------------------

def test_gate_fails_open_while_warming_up(gate):
    """
    A gate that blocked everything during warm-up would be indistinguishable
    from the strategy having stopped working — so it must fail permissive.
    """
    verdict = gate.check("BTC/USDT", 0.9)
    assert verdict.is_open
    assert verdict.warming_up


def test_warm_up_ends_after_enough_observations(gate):
    _warm(gate, "BTC/USDT", [0.2] * 10)
    verdict = gate.check("BTC/USDT", 0.2)
    assert not verdict.warming_up


# ---------------------------------------------------------------------------
# Core judgement
# ---------------------------------------------------------------------------

def test_high_concentration_closes_the_gate(gate):
    _warm(gate, "BTC/USDT", np.linspace(0.1, 0.5, 40))
    verdict = gate.check("BTC/USDT", 0.95)
    assert not verdict.is_open
    assert "concentrated" in verdict.reason


def test_broad_participation_keeps_the_gate_open(gate):
    _warm(gate, "BTC/USDT", np.linspace(0.1, 0.5, 40))
    verdict = gate.check("BTC/USDT", 0.05)
    assert verdict.is_open


def test_judgement_is_per_symbol(gate):
    """
    Whale share differs ~5x across majors, so the same absolute reading must
    be able to pass for one symbol and fail for another. A single global
    cutoff would suppress every BTC signal and no LINK signal.
    """
    _warm(gate, "BTC/USDT", np.linspace(0.30, 0.50, 40))    # naturally high
    _warm(gate, "LINK/USDT", np.linspace(0.02, 0.12, 40))   # naturally low

    reading = 0.35
    assert gate.check("BTC/USDT", reading).is_open
    assert not gate.check("LINK/USDT", reading).is_open


def test_current_reading_is_not_judged_against_itself(gate):
    """
    The observation under test must not already be in the distribution it is
    compared to, or every value drifts toward its own window's middle.
    """
    _warm(gate, "BTC/USDT", [0.1] * 40)
    before = len(gate.history["BTC/USDT"])
    verdict = gate.check("BTC/USDT", 0.99)
    assert not verdict.is_open
    assert len(gate.history["BTC/USDT"]) == before + 1


def test_observe_can_be_suppressed(gate):
    _warm(gate, "BTC/USDT", [0.1] * 40)
    before = len(gate.history["BTC/USDT"])
    gate.check("BTC/USDT", 0.5, observe=False)
    assert len(gate.history["BTC/USDT"]) == before


def test_non_finite_readings_are_ignored(gate):
    gate.observe("BTC/USDT", float("nan"))
    gate.observe("BTC/USDT", float("inf"))
    assert len(gate.history.get("BTC/USDT", [])) == 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_state_survives_restart(tmp_path):
    path = str(tmp_path / "flow_gate.json")
    first = FlowConcentrationGate(state_path=path, min_observations=10)
    _warm(first, "BTC/USDT", np.linspace(0.1, 0.5, 40))
    first.save()

    # The live scanner is a fresh process each run; without persistence the
    # gate would never leave warm-up.
    second = FlowConcentrationGate(state_path=path, min_observations=10)
    assert len(second.history["BTC/USDT"]) == 40
    assert not second.check("BTC/USDT", 0.95).is_open


def test_corrupt_state_does_not_crash(tmp_path):
    path = tmp_path / "flow_gate.json"
    path.write_text("{broken", encoding="utf-8")
    gate = FlowConcentrationGate(state_path=str(path))
    assert gate.history == {}


def test_history_is_bounded(tmp_path):
    gate = FlowConcentrationGate(state_path=None, window=50, min_observations=10)
    _warm(gate, "BTC/USDT", np.random.default_rng(0).random(500))
    assert len(gate.history["BTC/USDT"]) == 50


# ---------------------------------------------------------------------------
# FlowSignals.whale_share
# ---------------------------------------------------------------------------

def test_whale_share_is_a_fraction_of_total_volume():
    f = FlowSignals("BTC/USDT", buy_volume_usd=700_000,
                    sell_volume_usd=300_000, whale_volume_usd=250_000)
    assert f.total_volume_usd == pytest.approx(1_000_000)
    assert f.whale_share == pytest.approx(0.25)


def test_whale_share_is_zero_without_volume():
    assert FlowSignals("BTC/USDT").whale_share == 0.0


# ---------------------------------------------------------------------------
# Small-sample guard
# ---------------------------------------------------------------------------

def test_too_few_prints_is_discarded_not_acted_on(gate):
    """
    Whale share was validated on 15m bars carrying thousands of prints. A
    live window with 5 trades makes "top 1% of prints" mean "the largest
    single trade" — observed at 67.8% on BTC and 100% on a 1-trade window.
    Such a reading must neither block a trade nor enter the history.
    """
    _warm(gate, "BTC/USDT", np.linspace(0.1, 0.5, 40))
    before = len(gate.history["BTC/USDT"])

    verdict = gate.check("BTC/USDT", 0.99, trade_count=5)

    assert verdict.is_open, "must fail permissive on an unreliable reading"
    assert verdict.unreliable
    assert len(gate.history["BTC/USDT"]) == before, "must not poison the distribution"


def test_sufficient_prints_are_judged_normally(gate):
    _warm(gate, "BTC/USDT", np.linspace(0.1, 0.5, 40))
    verdict = gate.check("BTC/USDT", 0.99, trade_count=500)
    assert not verdict.is_open
    assert not verdict.unreliable
