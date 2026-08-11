"""
Tests for the trading cost model and its integration into the backtester.

The central property under test is that `Trade.pnl_pct` is net of costs —
this repo previously reported gross returns everywhere, and these tests exist
to stop that regressing silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtesting import engine
from src.backtesting.engine import _run_single_asset, apply_cost_model, Trade, simulate_exit
from src.config.config import TapeBacktestConfig
from src.execution.costs import (
    CostModel,
    FEE_SCHEDULES,
    cost_model_for,
    impact_bps,
    liquidity_tier,
)


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

def test_majors_classify_into_tight_spread_tiers():
    assert liquidity_tier("BTC/USDT") == "mega"
    assert liquidity_tier("ETH/USD") == "mega"
    assert liquidity_tier("SOL/USDT") == "large"
    assert liquidity_tier("RAIN/USDT") == "mid"


# ---------------------------------------------------------------------------
# Fee / spread / impact decomposition
# ---------------------------------------------------------------------------

def test_maker_legs_skip_the_spread():
    """
    The whole argument for post-only entries: a maker leg pays only the maker
    fee, while a taker leg additionally crosses half the spread.
    """
    model = CostModel(venue="okx", liquidity_tier="mid")
    taker = model.leg_cost_pct(is_maker=False)
    maker = model.leg_cost_pct(is_maker=True)

    schedule = FEE_SCHEDULES["okx"]
    assert maker == pytest.approx(schedule.maker_pct)
    assert taker > maker
    # The gap is exactly the half-spread for the tier (6bps = 0.06%).
    assert taker - maker == pytest.approx(schedule.taker_pct - schedule.maker_pct + 0.06)


def test_thinner_tiers_cost_more_to_take():
    mega = cost_model_for("BTC/USDT").round_trip_pct()
    mid = cost_model_for("RAIN/USDT").round_trip_pct()
    assert mid > mega


def test_maker_cost_is_tier_independent():
    """A resting order never crosses the spread, so tier shouldn't matter."""
    mega = cost_model_for("BTC/USDT", entry_is_maker=True, exit_is_maker=True).round_trip_pct()
    mid = cost_model_for("RAIN/USDT", entry_is_maker=True, exit_is_maker=True).round_trip_pct()
    assert mega == pytest.approx(mid)


def test_impact_grows_with_square_root_of_participation():
    small = impact_bps(1_000, 1_000_000)
    big = impact_bps(4_000, 1_000_000)
    # 4x the size is 2x the impact under a square-root law, not 4x.
    assert big == pytest.approx(2 * small)


def test_impact_is_zero_without_a_volume_reference():
    assert impact_bps(1_000, 0) == 0.0
    assert impact_bps(0, 1_000_000) == 0.0


def test_measured_spread_overrides_the_tier_default():
    model = CostModel(venue="okx", liquidity_tier="small")
    default = model.round_trip_pct()
    measured = model.round_trip_pct(entry_half_spread_pct=0.001)
    assert measured < default


# ---------------------------------------------------------------------------
# Integration with the backtest engine
# ---------------------------------------------------------------------------

def _flat_bars(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    price = np.full(n, 100.0)
    return pd.DataFrame(
        {"open": price, "high": price, "low": price, "close": price,
         "volume": np.full(n, 1_000.0)},
        index=idx,
    )


def test_apply_cost_model_makes_pnl_net_and_preserves_gross():
    bars = _flat_bars()
    cfg = TapeBacktestConfig()
    trade = Trade(symbol="BTC/USDT", entry_bar=10, entry_price=100.0, pnl_pct=1.0)

    apply_cost_model([trade], bars, cfg)

    assert trade.gross_pnl_pct == pytest.approx(1.0)
    assert trade.cost_pct > 0
    assert trade.pnl_pct == pytest.approx(1.0 - trade.cost_pct)


def test_apply_costs_false_reproduces_gross_behaviour():
    bars = _flat_bars()
    cfg = TapeBacktestConfig(apply_costs=False)
    trade = Trade(symbol="BTC/USDT", entry_bar=10, entry_price=100.0, pnl_pct=1.0)

    apply_cost_model([trade], bars, cfg)

    assert trade.cost_pct == 0.0
    assert trade.pnl_pct == pytest.approx(1.0)


def test_a_thin_symbol_is_charged_more_than_a_major():
    bars = _flat_bars()
    cfg = TapeBacktestConfig()
    major = Trade(symbol="BTC/USDT", entry_bar=10, entry_price=100.0, pnl_pct=1.0)
    thin = Trade(symbol="RAIN/USDT", entry_bar=10, entry_price=100.0, pnl_pct=1.0)

    apply_cost_model([major, thin], bars, cfg)

    assert thin.cost_pct > major.cost_pct


def test_costs_are_not_double_charged_on_the_open_tail_trade(monkeypatch):
    """
    _run_single_asset appends a still-open trade after _step_bar_range has
    already costed the closed ones. Charging the whole list again there would
    bill the closed trades twice.
    """
    n = 120
    idx = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    price = np.full(n, 100.0)
    # A steady climb so the entry never trips the trailing stop and the trade
    # is still open at end of data.
    close = price + np.arange(n) * 0.5
    bars = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": np.full(n, 1_000.0)},
        index=idx,
    )

    is_setup = pd.Series(False, index=idx)
    is_setup.iloc[100] = True
    signals = pd.DataFrame({
        "is_setup": is_setup,
        "event": pd.Series("bid_repulsion", index=idx),
        "level_price": pd.Series(0.0, index=idx),
        "bonus_score": pd.Series(10.0, index=idx),
    })
    monkeypatch.setattr(engine, "detect_tape_signals", lambda *a, **k: signals)

    cfg = TapeBacktestConfig(ema_long=5, btc_regime_filter=False)
    trades = _run_single_asset("BTC/USDT", bars, cfg)

    assert len(trades) == 1
    tail = trades[0]
    assert tail.is_open
    expected = cost_model_for("BTC/USDT", venue=cfg.venue).round_trip_pct(
        order_size_usd=cfg.order_size_usd,
        reference_volume_usd=1_000.0 * float(bars["close"].iloc[tail.entry_bar]),
    )
    assert tail.cost_pct == pytest.approx(expected)
    assert tail.pnl_pct == pytest.approx(tail.gross_pnl_pct - expected)


# ---------------------------------------------------------------------------
# ML label consistency
# ---------------------------------------------------------------------------

def test_simulate_exit_subtracts_cost_from_the_training_label():
    """
    Labels drive `win = pnl > 0`. A setup that gains less than the round-trip
    cost must not be labelled a winner, or the filter learns to pick trades
    that lose money net.
    """
    n = 60
    idx = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    close = np.full(n, 100.0)
    # Entry fills at open of bar 11 (the bar after the signal), so the gain
    # has to start at bar 12 to actually be captured by the trade.
    close[12:] = 100.1          # +0.1% — real, but smaller than any real fee
    bars = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": np.full(n, 1_000.0)},
        index=idx,
    )
    atr_ser = pd.Series(np.full(n, 5.0), index=idx)

    gross = simulate_exit(bars, atr_ser, signal_bar_idx=10)
    net = simulate_exit(bars, atr_ser, signal_bar_idx=10, cost_pct=0.2)

    assert gross > 0            # looks like a winner before costs
    assert net < 0              # is actually a loser after them
    assert net == pytest.approx(gross - 0.2)
