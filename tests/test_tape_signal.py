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


def test_liquidity_sweep_detected():
    rows = [
        _bar(100, 100, 94, 98, 1000, 1000),
        _bar(100, 100, 94, 98, 1000, 1000),
        _bar(100, 100, 94, 98, 1000, 1000),
        _bar(100, 100, 94, 98, 1000, 1000),
        _bar(100, 100, 94, 98, 1000, 1000),
        _bar(94, 96, 93, 95, 2000, 3000),   # wicks below support (94), closes back above
        _bar(95, 97, 94, 96, 1000, 1000),
    ]
    bars = pd.DataFrame(rows)
    result = detect_tape_signals(bars, lookback=LOOKBACK, enable_liquidity_sweep=True)
    assert bool(result["is_setup"].iloc[5])
    assert result["event"].iloc[5] == "liquidity_sweep"
    assert result["level_price"].iloc[5] == pytest.approx(94.0)
    assert result["bonus_score"].iloc[5] > 0


def test_liquidity_sweep_off_by_default():
    rows = [
        _bar(100, 100, 94, 98, 1000, 1000),
        _bar(100, 100, 94, 98, 1000, 1000),
        _bar(100, 100, 94, 98, 1000, 1000),
        _bar(100, 100, 94, 98, 1000, 1000),
        _bar(100, 100, 94, 98, 1000, 1000),
        _bar(94, 96, 93, 95, 2000, 3000),
        _bar(95, 97, 94, 96, 1000, 1000),
    ]
    bars = pd.DataFrame(rows)
    result = detect_tape_signals(bars, lookback=LOOKBACK)
    assert not bool(result["is_setup"].iloc[5])


def test_climax_exhaustion_detected():
    rows = [
        _bar(100, 101, 99, 100, 1000, 1000),
        _bar(100, 101, 99, 100, 1000, 1000),
        _bar(100, 101, 99, 100, 1000, 1000),
        _bar(100, 101, 99, 100, 1000, 1000),
        _bar(100, 101, 99, 100, 1000, 1000),
        _bar(99, 100, 90, 92, 1000, 6000),   # climax bar: wide range, extreme volume, new low, red
        _bar(95, 103, 94, 102, 3000, 500),   # reclaims the climax bar's high
    ]
    bars = pd.DataFrame(rows)
    result = detect_tape_signals(
        bars, lookback=LOOKBACK, enable_climax_exhaustion=True,
        enable_ask_absorption=False, enable_bid_repulsion=False,
    )
    assert bool(result["is_setup"].iloc[6])
    assert result["event"].iloc[6] == "climax_exhaustion"
    assert result["bonus_score"].iloc[6] > 0


def test_delta_divergence_detected():
    rows = [
        _bar(100, 101, 99, 99, 1000, 1500),
        _bar(99, 100, 97, 97, 1000, 1800),
        _bar(97, 98, 95, 95, 1000, 1200),
        _bar(95, 96, 93, 93, 1000, 1100),
        _bar(94, 95, 92, 93, 1000, 1050),
        _bar(93, 95, 90, 94, 900, 850),      # new price low, but delta less negative than cvd's recent min
        _bar(94, 97, 93, 96, 1500, 800),
    ]
    bars = pd.DataFrame(rows)
    result = detect_tape_signals(
        bars, lookback=LOOKBACK, enable_delta_divergence=True,
        enable_ask_absorption=False, enable_bid_repulsion=False,
    )
    assert bool(result["is_setup"].iloc[5])
    assert result["event"].iloc[5] == "delta_divergence"
    assert result["bonus_score"].iloc[5] > 0


def test_momentum_breakout_detected():
    rows = [_bar(100, 101, 99, 100, 500, 500) for _ in range(5)]
    rows.append(_bar(100, 108, 100, 107, 3000, 1000))   # clears resistance by >1%, volume spike, buyers strongly dominant
    bars = pd.DataFrame(rows)
    result = detect_tape_signals(
        bars, lookback=LOOKBACK, enable_momentum_breakout=True,
        enable_ask_absorption=False, enable_bid_repulsion=False,
    )
    assert bool(result["is_setup"].iloc[5])
    assert result["event"].iloc[5] == "momentum_breakout"
    assert result["level_price"].iloc[5] == pytest.approx(101.0)
    assert result["bonus_score"].iloc[5] > 0


def test_momentum_breakout_off_by_default():
    rows = [_bar(100, 101, 99, 100, 500, 500) for _ in range(5)]
    rows.append(_bar(100, 108, 100, 107, 3000, 1000))
    bars = pd.DataFrame(rows)
    result = detect_tape_signals(bars, lookback=LOOKBACK, enable_ask_absorption=False, enable_bid_repulsion=False)
    assert not bool(result["is_setup"].iloc[5])


def test_vwap_fade_detected():
    rows = [_bar(110, 111, 109, 110, 250, 250) for _ in range(8)]
    rows.append(_bar(110, 111, 84, 85, 250, 250))     # dip
    rows.append(_bar(85, 92, 84, 90, 600, 400))        # turns up, well below the rolling VWAP, elevated volume
    bars = pd.DataFrame(rows)
    result = detect_tape_signals(
        bars, lookback=LOOKBACK, enable_vwap_fade=True, vwap_window=10,
        enable_ask_absorption=False, enable_bid_repulsion=False,
    )
    assert bool(result["is_setup"].iloc[9])
    assert result["event"].iloc[9] == "vwap_fade"
    assert result["bonus_score"].iloc[9] > 0
