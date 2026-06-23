"""Unit tests for the order-book wall signal module (the sole signal source)."""

import pytest

from src.data.orderbook import OrderBookSignals, WALL_SCAN_PCT
from src.modules.wall_signal import WallTracker


def make_ob_signals(
    wall_bid_price=0.0, wall_bid_size_usd=0.0,
    wall_ask_price=0.0, wall_ask_size_usd=0.0,
) -> OrderBookSignals:
    return OrderBookSignals(
        symbol="TEST/USDT",
        timestamp=0.0,
        wall_bid_price=wall_bid_price,
        wall_bid_size_usd=wall_bid_size_usd,
        wall_ask_price=wall_ask_price,
        wall_ask_size_usd=wall_ask_size_usd,
    )


class TestWallTracker:
    def test_first_cycle_never_fires(self):
        tracker = WallTracker()
        ob = make_ob_signals(wall_ask_price=103.0, wall_ask_size_usd=10_000.0)
        result = tracker.update("TEST/USDT", price=99.0, ob_signals=ob)
        assert not result.is_setup

    def test_no_setup_without_ob_signals(self):
        tracker = WallTracker()
        result = tracker.update("TEST/USDT", price=99.0, ob_signals=None)
        assert not result.is_setup

    def test_no_setup_with_invalid_price(self):
        tracker = WallTracker()
        ob = make_ob_signals(wall_ask_price=103.0, wall_ask_size_usd=10_000.0)
        result = tracker.update("TEST/USDT", price=0.0, ob_signals=ob)
        assert not result.is_setup

    def test_ask_wall_absorption_detected(self):
        tracker = WallTracker()
        # Cycle 1: price testing a large ask wall just above it
        ob1 = make_ob_signals(wall_ask_price=103.0, wall_ask_size_usd=10_000.0)
        tracker.update("TEST/USDT", price=99.0, ob_signals=ob1)

        # Cycle 2: price has pushed through the level, wall has collapsed
        ob2 = make_ob_signals(wall_ask_price=0.0, wall_ask_size_usd=0.0)
        result = tracker.update("TEST/USDT", price=104.0, ob_signals=ob2)

        assert result.is_setup
        assert result.event == "ask_absorption"
        assert result.wall_side == "ask"
        assert result.wall_price == 103.0
        assert result.bonus_score > 0

    def test_no_absorption_when_wall_holds_size(self):
        tracker = WallTracker()
        ob1 = make_ob_signals(wall_ask_price=103.0, wall_ask_size_usd=10_000.0)
        tracker.update("TEST/USDT", price=99.0, ob_signals=ob1)

        # Wall still at the same level with the same size — not absorbed
        ob2 = make_ob_signals(wall_ask_price=103.0, wall_ask_size_usd=9_800.0)
        result = tracker.update("TEST/USDT", price=104.0, ob_signals=ob2)

        assert not result.is_setup

    def test_no_absorption_when_price_never_tested_wall(self):
        tracker = WallTracker()
        # Wall far away (>5%) — price wasn't actually testing it
        ob1 = make_ob_signals(wall_ask_price=120.0, wall_ask_size_usd=10_000.0)
        tracker.update("TEST/USDT", price=99.0, ob_signals=ob1)

        ob2 = make_ob_signals(wall_ask_price=0.0, wall_ask_size_usd=0.0)
        result = tracker.update("TEST/USDT", price=121.0, ob_signals=ob2)

        assert not result.is_setup

    def test_bid_wall_repulsion_detected(self):
        tracker = WallTracker()
        # Cycle 1: price testing a large bid wall just below it
        ob1 = make_ob_signals(wall_bid_price=99.0, wall_bid_size_usd=10_000.0)
        tracker.update("TEST/USDT", price=101.0, ob_signals=ob1)

        # Cycle 2: price bounced away, wall held its size
        ob2 = make_ob_signals(wall_bid_price=99.0, wall_bid_size_usd=9_800.0)
        result = tracker.update("TEST/USDT", price=103.0, ob_signals=ob2)

        assert result.is_setup
        assert result.event == "bid_repulsion"
        assert result.wall_side == "bid"
        assert result.wall_price == 99.0
        assert result.bonus_score > 0

    def test_no_repulsion_when_wall_disappears(self):
        tracker = WallTracker()
        ob1 = make_ob_signals(wall_bid_price=99.0, wall_bid_size_usd=10_000.0)
        tracker.update("TEST/USDT", price=101.0, ob_signals=ob1)

        # Wall vanished — support was eaten, not defended
        ob2 = make_ob_signals(wall_bid_price=0.0, wall_bid_size_usd=0.0)
        result = tracker.update("TEST/USDT", price=103.0, ob_signals=ob2)

        assert not result.is_setup

    def test_no_repulsion_when_price_keeps_falling(self):
        tracker = WallTracker()
        ob1 = make_ob_signals(wall_bid_price=99.0, wall_bid_size_usd=10_000.0)
        tracker.update("TEST/USDT", price=101.0, ob_signals=ob1)

        # Price didn't bounce — kept dropping toward/through the wall
        ob2 = make_ob_signals(wall_bid_price=99.0, wall_bid_size_usd=10_000.0)
        result = tracker.update("TEST/USDT", price=100.0, ob_signals=ob2)

        assert not result.is_setup

    def test_bonus_scales_with_absorption_strength(self):
        tracker_weak = WallTracker()
        tracker_strong = WallTracker()

        ob1 = make_ob_signals(wall_ask_price=103.0, wall_ask_size_usd=10_000.0)
        tracker_weak.update("TEST/USDT", price=99.0, ob_signals=ob1)
        tracker_strong.update("TEST/USDT", price=99.0, ob_signals=ob1)

        # Weak: wall shrinks exactly to the threshold (50%)
        ob_weak = make_ob_signals(wall_ask_price=103.0, wall_ask_size_usd=5_000.0)
        weak = tracker_weak.update("TEST/USDT", price=104.0, ob_signals=ob_weak)

        # Strong: wall fully disappears
        ob_strong = make_ob_signals(wall_ask_price=0.0, wall_ask_size_usd=0.0)
        strong = tracker_strong.update("TEST/USDT", price=104.0, ob_signals=ob_strong)

        assert weak.is_setup and strong.is_setup
        assert strong.bonus_score >= weak.bonus_score

    def test_tracks_multiple_symbols_independently(self):
        tracker = WallTracker()
        ob_a1 = make_ob_signals(wall_ask_price=103.0, wall_ask_size_usd=10_000.0)
        ob_b1 = make_ob_signals(wall_ask_price=50.0, wall_ask_size_usd=10_000.0)
        tracker.update("AAA/USDT", price=99.0, ob_signals=ob_a1)
        tracker.update("BBB/USDT", price=48.0, ob_signals=ob_b1)

        # Only AAA absorbs its wall; BBB's wall holds
        ob_a2 = make_ob_signals(wall_ask_price=0.0, wall_ask_size_usd=0.0)
        ob_b2 = make_ob_signals(wall_ask_price=50.0, wall_ask_size_usd=10_000.0)
        result_a = tracker.update("AAA/USDT", price=104.0, ob_signals=ob_a2)
        result_b = tracker.update("BBB/USDT", price=51.0, ob_signals=ob_b2)

        assert result_a.is_setup
        assert not result_b.is_setup
