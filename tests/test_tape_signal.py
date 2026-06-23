"""Unit tests for the tape-based backtest signal (historical proxy for the live wall signal)."""

import pandas as pd
import pytest

from src.modules.tape_signal import detect_tape_signals

LOOKBACK = 5


def _bar(open_, high, low, close, buy, sell):
    return {
        "open": open_, "high": high, "low": low, "close": close,
        "volume": buy + sell, "buy_volume": buy, "sell_volume": sell,
    }


def test_ask_absorption_detected():
    rows = [
        _bar(95, 100, 94, 95, 1000, 1000),   # 0
        _bar(95, 100, 94, 95, 1000, 1000),   # 1
        _bar(95, 100, 94, 95, 1000, 1000),   # 2
        _bar(95, 100, 94, 95, 1000, 1000),   # 3
        _bar(95, 99.5, 94, 95, 1000, 1000),  # 4 — resistance ~100 from rows 0-4
        _bar(96, 99.8, 95, 99, 500, 2500),   # 5 — testing resistance, heavy sell tape
        _bar(100, 101.5, 99, 101, 3000, 1000),  # 6 — pushes through, buyers dominant
        _bar(101, 102, 100, 101, 1000, 1000),   # 7 — padding
    ]
    bars = pd.DataFrame(rows)
    result = detect_tape_signals(bars, lookback=LOOKBACK)

    assert bool(result["is_setup"].iloc[6])
    assert result["event"].iloc[6] == "ask_absorption"
    assert result["level_price"].iloc[6] == pytest.approx(100.0)
    assert result["bonus_score"].iloc[6] > 0


def test_bid_repulsion_detected():
    rows = [
        _bar(105, 150, 100, 105, 1000, 1000),   # 0
        _bar(105, 150, 100, 105, 1000, 1000),   # 1
        _bar(105, 150, 100, 105, 1000, 1000),   # 2
        _bar(105, 150, 100, 105, 1000, 1000),   # 3
        _bar(105, 150, 100, 105, 1000, 1000),   # 4 — support ~100 from rows 0-4
        _bar(103, 150, 101, 102, 500, 2500),    # 5 — testing support, heavy sell tape
        _bar(102, 150, 101, 103, 1500, 1000),   # 6 — bounces away, support holds
        _bar(103, 150, 102, 104, 1000, 1000),   # 7 — padding
    ]
    bars = pd.DataFrame(rows)
    result = detect_tape_signals(bars, lookback=LOOKBACK)

    assert bool(result["is_setup"].iloc[6])
    assert result["event"].iloc[6] == "bid_repulsion"
    assert result["level_price"].iloc[6] == pytest.approx(100.0)
    assert result["bonus_score"].iloc[6] > 0


def test_no_setup_on_flat_market():
    rows = [_bar(100, 101, 99, 100, 1000, 1000) for _ in range(10)]
    bars = pd.DataFrame(rows)
    result = detect_tape_signals(bars, lookback=LOOKBACK)
    assert not result["is_setup"].any()


def test_no_absorption_without_volume_spike():
    rows = [
        _bar(95, 100, 94, 95, 1000, 1000),
        _bar(95, 100, 94, 95, 1000, 1000),
        _bar(95, 100, 94, 95, 1000, 1000),
        _bar(95, 100, 94, 95, 1000, 1000),
        _bar(95, 99.5, 94, 95, 1000, 1000),
        _bar(96, 99.8, 95, 99, 1000, 1000),     # no sell-side spike here
        _bar(100, 101.5, 99, 101, 3000, 1000),
        _bar(101, 102, 100, 101, 1000, 1000),
    ]
    bars = pd.DataFrame(rows)
    result = detect_tape_signals(bars, lookback=LOOKBACK)
    assert not bool(result["is_setup"].iloc[6])


def test_output_aligned_to_input_length():
    rows = [_bar(100, 101, 99, 100, 1000, 1000) for _ in range(12)]
    bars = pd.DataFrame(rows)
    result = detect_tape_signals(bars, lookback=LOOKBACK)
    assert len(result) == len(bars)
    assert list(result.index) == list(bars.index)
