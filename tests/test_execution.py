"""
Tests for the execution layer: risk limits, position accounting, and the
lifecycle monitor.

These carry more weight than the scanner tests. A scoring bug produces a bad
suggestion a human can ignore; a bug in here moves real money on a 60-second
loop with nobody watching.
"""

from __future__ import annotations

import json

import pytest

from src.execution.monitor import Mark, PositionMonitor
from src.execution.position import Position, PositionStore
from src.execution.risk import RiskLimits, RiskManager, RiskState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return PositionStore(path=str(tmp_path / "positions.json"))


@pytest.fixture
def risk(tmp_path):
    return RiskManager(
        limits=RiskLimits(
            max_concurrent_positions=2,
            max_position_usd=500.0,
            max_daily_loss_usd=100.0,
            max_daily_trades=5,
            risk_per_trade_pct=0.01,
        ),
        state_path=str(tmp_path / "risk.json"),
    )


def _position(symbol="BTC/USDT", entry=100.0, qty=1.0, sl=95.0, tp1=105.0, tp2=110.0):
    return Position(
        symbol=symbol, side="long", entry_price=entry, quantity=qty,
        stop_loss=sl, take_profit_1=tp1, take_profit_2=tp2,
    )


# ---------------------------------------------------------------------------
# Risk limits
# ---------------------------------------------------------------------------

def test_blocks_second_position_in_same_symbol(risk):
    """Pyramiding is the most common way an entry loop exceeds its budget."""
    ok, reason = risk.can_open("BTC/USDT", 100.0, open_count=1, already_open_symbol=True)
    assert not ok
    assert "already holding" in reason


def test_blocks_at_max_concurrent_positions(risk):
    ok, reason = risk.can_open("ETH/USDT", 100.0, open_count=2)
    assert not ok
    assert "max concurrent" in reason


def test_blocks_oversized_position(risk):
    ok, reason = risk.can_open("BTC/USDT", 501.0, open_count=0)
    assert not ok
    assert "exceeds cap" in reason


def test_blocks_at_daily_trade_cap(risk):
    for _ in range(5):
        risk.record_open()
    ok, reason = risk.can_open("BTC/USDT", 100.0, open_count=0)
    assert not ok
    assert "daily trade cap" in reason


def test_daily_loss_breaker_trips_and_blocks_new_entries(risk):
    assert risk.record_close(-60.0) is None
    ok, _ = risk.can_open("BTC/USDT", 100.0, open_count=0)
    assert ok

    halt = risk.record_close(-50.0)          # cumulative -110 vs -100 limit
    assert halt is not None

    ok, reason = risk.can_open("BTC/USDT", 100.0, open_count=0)
    assert not ok
    assert "halted" in reason


def test_kill_switch_blocks_everything_until_released(risk):
    risk.engage_kill_switch("testing")
    ok, reason = risk.can_open("BTC/USDT", 10.0, open_count=0)
    assert not ok
    assert "kill switch" in reason

    risk.release_kill_switch()
    ok, _ = risk.can_open("BTC/USDT", 10.0, open_count=0)
    assert ok


def test_risk_state_survives_restart(tmp_path):
    path = str(tmp_path / "risk.json")
    first = RiskManager(limits=RiskLimits(max_daily_loss_usd=100.0), state_path=path)
    first.record_close(-150.0)
    assert first.state.halted

    # A limit that forgets itself on restart is not a limit.
    second = RiskManager(limits=RiskLimits(max_daily_loss_usd=100.0), state_path=path)
    assert second.state.halted
    ok, _ = second.can_open("BTC/USDT", 10.0, open_count=0)
    assert not ok


def test_new_utc_day_clears_loss_halt_but_not_kill_switch(risk):
    risk.record_close(-150.0)
    assert risk.state.halted

    risk.state.date = "2000-01-01"           # force a day roll
    risk.state.roll_if_new_day()
    assert not risk.state.halted
    assert risk.state.realised_pnl_usd == 0.0

    risk.engage_kill_switch()
    risk.state.date = "2000-01-01"
    risk.state.roll_if_new_day()
    assert risk.state.halted, "kill switch must survive the day boundary"


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

@pytest.fixture
def uncapped_risk(tmp_path):
    """
    Sizing fixture with the notional cap lifted out of the way, so these
    tests exercise the risk-based arithmetic rather than the clamp (which
    test_size_is_clamped_by_the_notional_cap covers separately).
    """
    return RiskManager(
        limits=RiskLimits(max_position_usd=1_000_000.0, risk_per_trade_pct=0.01),
        state_path=str(tmp_path / "risk.json"),
    )


def test_size_scales_inversely_with_stop_distance(uncapped_risk):
    tight_qty, _ = uncapped_risk.position_size(10_000, entry_price=100.0, stop_price=99.0)
    wide_qty, _ = uncapped_risk.position_size(10_000, entry_price=100.0, stop_price=90.0)
    assert tight_qty > wide_qty


def test_size_risks_the_configured_fraction_of_equity(uncapped_risk):
    qty, _ = uncapped_risk.position_size(10_000, entry_price=100.0, stop_price=99.0)
    loss_at_stop = qty * (100.0 - 99.0)
    assert loss_at_stop == pytest.approx(10_000 * 0.01)


def test_size_is_clamped_by_the_notional_cap(risk):
    _, notional = risk.position_size(1_000_000, entry_price=100.0, stop_price=99.9)
    assert notional <= risk.limits.max_position_usd


def test_invalid_stop_yields_no_position(risk):
    assert risk.position_size(10_000, 100.0, 100.0) == (0.0, 0.0)
    assert risk.position_size(10_000, 100.0, 105.0) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Position accounting
# ---------------------------------------------------------------------------

def test_partial_exit_reduces_remaining_and_books_pnl():
    p = _position()
    net = p.record_partial_exit(price=105.0, quantity=0.5, fee_usd=0.05)
    assert p.remaining_quantity == pytest.approx(0.5)
    assert net == pytest.approx(0.5 * 5.0 - 0.05)
    assert p.realised_pnl_usd == pytest.approx(net)
    assert p.is_open


def test_close_books_fees_into_realised_pnl():
    p = _position()
    p.close(price=110.0, reason="tp2", fee_usd=0.10)
    assert not p.is_open
    assert p.realised_pnl_usd == pytest.approx(10.0 - 0.10)
    assert p.fees_paid_usd == pytest.approx(0.10)


def test_store_round_trips_through_disk(tmp_path):
    path = str(tmp_path / "positions.json")
    first = PositionStore(path=path)
    first.add(_position())

    second = PositionStore(path=path)
    restored = second.get("BTC/USDT")
    assert restored is not None
    assert restored.entry_price == 100.0
    assert restored.take_profit_2 == 110.0


def test_store_survives_a_corrupt_file(tmp_path):
    path = tmp_path / "positions.json"
    path.write_text("{not valid json", encoding="utf-8")
    store = PositionStore(path=str(path))
    assert store.open_count == 0


# ---------------------------------------------------------------------------
# Lifecycle monitor
# ---------------------------------------------------------------------------

class _RecordingNotifier:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append(name)
        return record


def test_stop_hit_closes_position_and_alerts(store):
    store.add(_position())
    notifier = _RecordingNotifier()
    monitor = PositionMonitor(store, notifier=notifier)

    events = monitor.check({"BTC/USDT": Mark(last=94.0, low=94.0, high=96.0)})

    assert len(events) == 1
    assert events[0].kind == "stop"
    assert events[0].closed
    assert store.open_count == 0
    assert "send_stop_hit" in notifier.calls


def test_tp1_scales_out_and_moves_stop_to_breakeven(store):
    store.add(_position())
    monitor = PositionMonitor(store, notifier=_RecordingNotifier())

    events = monitor.check({"BTC/USDT": Mark(last=106.0, low=104.0, high=106.0)})

    assert [e.kind for e in events] == ["tp1"]
    position = store.get("BTC/USDT")
    assert position is not None
    assert position.tp1_hit
    assert position.remaining_quantity == pytest.approx(0.5)
    assert position.stop_loss == pytest.approx(position.entry_price)


def test_a_single_cycle_can_carry_through_both_targets(store):
    store.add(_position())
    monitor = PositionMonitor(store, notifier=_RecordingNotifier())

    events = monitor.check({"BTC/USDT": Mark(last=111.0, low=100.0, high=111.0)})

    assert [e.kind for e in events] == ["tp1", "tp2"]
    assert store.open_count == 0


def test_stop_wins_when_a_cycle_spans_both_stop_and_target(store):
    """
    A 60s interval whose range covers both the stop and the target is
    genuinely ambiguous. Resolving it in favour of the target is how a paper
    log ends up describing an account nobody has.
    """
    store.add(_position())
    monitor = PositionMonitor(store, notifier=_RecordingNotifier())

    events = monitor.check({"BTC/USDT": Mark(last=100.0, low=94.0, high=111.0)})

    assert [e.kind for e in events] == ["stop"]
    assert store.open_count == 0


def test_missing_mark_leaves_position_untouched(store):
    store.add(_position())
    monitor = PositionMonitor(store, notifier=_RecordingNotifier())

    events = monitor.check({"ETH/USDT": Mark(last=1.0)})

    assert events == []
    assert store.open_count == 1


def test_exit_is_charged_a_fee(store):
    store.add(_position())
    monitor = PositionMonitor(store, notifier=_RecordingNotifier())

    monitor.check({"BTC/USDT": Mark(last=94.0, low=94.0, high=94.0)})

    closed = store.closed_positions[-1]
    assert closed.fees_paid_usd > 0
    # Net loss must be worse than the raw price move, never better.
    assert closed.realised_pnl_usd < (95.0 - 100.0) * 1.0 + 1e-9


def test_force_close_flattens_everything(store):
    store.add(_position("BTC/USDT"))
    store.add(_position("ETH/USDT", entry=50.0, sl=45.0, tp1=55.0, tp2=60.0))
    monitor = PositionMonitor(store, notifier=_RecordingNotifier())

    events = monitor.force_close_all({
        "BTC/USDT": Mark(last=101.0),
        "ETH/USDT": Mark(last=51.0),
    }, reason="kill_switch")

    assert len(events) == 2
    assert store.open_count == 0
