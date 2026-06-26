"""Unit tests for the tape backtest engine's regime-filter helpers."""

import numpy as np
import pandas as pd
import pytest

import src.backtesting.engine as engine
from src.backtesting.engine import _daily_trend_bull, _run_single_asset, exclude_dominant_trades, Trade
from src.config.config import TapeBacktestConfig


def test_daily_trend_bull_detects_uptrend_after_warmup():
    idx = pd.date_range("2024-01-01", periods=6 * 10, freq="4h", tz="UTC")
    closes = np.linspace(100, 160, len(idx))
    bars = pd.DataFrame({"close": closes}, index=idx)

    result = _daily_trend_bull(bars, ema_period=3)

    assert result.dtype == bool
    assert len(result) == len(bars)
    assert not result[:6].any()   # no completed prior daily bar yet
    assert result[-6:].all()      # clearly bullish once the trend is established


def test_daily_trend_bull_no_lookahead_within_a_day():
    idx = pd.date_range("2024-01-01", periods=6 * 5, freq="4h", tz="UTC")
    closes = np.concatenate([
        np.full(6 * 4, 100.0),     # flat for 4 days
        np.linspace(100, 200, 6),  # day 5 spikes intraday
    ])
    bars = pd.DataFrame({"close": closes}, index=idx)

    result = _daily_trend_bull(bars, ema_period=3)

    # Day 5's intraday spike can't be seen until day 6 — every bar on day 5
    # itself must reflect only the (flat) trend known before that day opened.
    assert not result[-6:].any()


def _make_cooldown_bars():
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    open_ = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    close = np.full(n, 100.0)

    # bar 61: closes hard down, triggering the ATR trailing stop on the next bar
    close[61] = 90.0
    open_[62] = 85.0   # exit fill for the loss

    bars = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)

    # is_setup at bars 60 (first entry) and 64 (second entry, inside any cooldown window)
    is_setup = pd.Series(False, index=idx)
    is_setup.iloc[60] = True
    is_setup.iloc[64] = True
    signals = pd.DataFrame({
        "is_setup": is_setup,
        "event": pd.Series("bid_repulsion", index=idx),
        "level_price": pd.Series(0.0, index=idx),
        "bonus_score": pd.Series(10.0, index=idx),
    })
    return bars, signals


def test_cooldown_suppresses_reentry_after_a_loss(monkeypatch):
    bars, signals = _make_cooldown_bars()
    monkeypatch.setattr(engine, "detect_tape_signals", lambda *a, **k: signals)

    cfg = TapeBacktestConfig(ema_long=5, cooldown_bars_after_loss=5, btc_regime_filter=False)
    trades = _run_single_asset("TEST", bars, cfg)

    assert len(trades) == 1
    assert trades[0].pnl_pct < 0


def test_no_cooldown_allows_reentry_after_a_loss(monkeypatch):
    bars, signals = _make_cooldown_bars()
    monkeypatch.setattr(engine, "detect_tape_signals", lambda *a, **k: signals)

    cfg = TapeBacktestConfig(ema_long=5, cooldown_bars_after_loss=0, btc_regime_filter=False)
    trades = _run_single_asset("TEST", bars, cfg)

    assert len(trades) == 2


def _make_max_loss_bars():
    n = 80
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    open_ = np.full(n, 100.0)
    high = np.full(n, 100.0)
    low = np.full(n, 100.0)
    close = np.full(n, 100.0)

    # Huge pre-entry volatility inflates ATR so the ATR-based trailing stop
    # sits far below price and never triggers on its own -- isolating the
    # hard per-trade cap's own behaviour from the ATR stop.
    high[:60] = 300.0
    low[:60] = 0.0

    # entry at bar 61 (signal at bar 60, entry_price = open_[61] = 100),
    # then a controlled decline: -3%, -6%, -9%, -12% -- a 10% hard cap
    # should fire once unrealized loss reaches -12% (the first close past
    # -10%), filling at the next bar's open.
    close[61] = 100.0
    close[62] = 97.0
    close[63] = 94.0
    close[64] = 91.0
    close[65] = 88.0
    open_[66] = 85.0

    bars = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
    is_setup = pd.Series(False, index=idx)
    is_setup.iloc[60] = True
    signals = pd.DataFrame({
        "is_setup": is_setup,
        "event": pd.Series("bid_repulsion", index=idx),
        "level_price": pd.Series(0.0, index=idx),
        "bonus_score": pd.Series(10.0, index=idx),
    })
    return bars, signals


def test_max_single_trade_loss_caps_a_wide_atr_trade(monkeypatch):
    bars, signals = _make_max_loss_bars()
    monkeypatch.setattr(engine, "detect_tape_signals", lambda *a, **k: signals)

    cfg = TapeBacktestConfig(ema_long=5, btc_regime_filter=False, max_single_trade_loss_pct=10.0)
    trades = _run_single_asset("TEST", bars, cfg)

    assert len(trades) == 1
    assert trades[0].exit_reason == "max_single_trade_loss"
    assert trades[0].exit_bar == 66
    assert trades[0].pnl_pct == pytest.approx(-15.0)


def test_no_max_single_trade_loss_lets_the_wide_atr_stop_ride(monkeypatch):
    bars, signals = _make_max_loss_bars()
    monkeypatch.setattr(engine, "detect_tape_signals", lambda *a, **k: signals)

    cfg = TapeBacktestConfig(ema_long=5, btc_regime_filter=False, max_single_trade_loss_pct=None)
    trades = _run_single_asset("TEST", bars, cfg)

    # ATR stop is deliberately too wide to trigger in this scenario -- with
    # the hard cap disabled, the trade should still be open at end of data.
    assert len(trades) == 1
    assert trades[0].is_open is True


def _trade(pnl_pct):
    return Trade(symbol="X", entry_bar=0, entry_price=100.0, pnl_pct=pnl_pct, is_open=False)


def test_exclude_dominant_trades_strips_lone_outlier():
    # one trade is most of the gross profit -> must be removed
    trades = [_trade(500.0), _trade(10.0), _trade(8.0), _trade(-5.0), _trade(-3.0)]
    result = exclude_dominant_trades(trades, max_profit_share=1 / 3)
    assert all(t.pnl_pct != 500.0 for t in result)
    assert len(result) == 4


def test_exclude_dominant_trades_judges_share_against_original_total_only():
    # 200 is well under 1/3 of the *original* total (718) even though it's
    # the biggest trade left after 500 is removed -- no recursive re-check
    # against a shrinking remainder, which would otherwise cascade and wipe
    # out a normal "few big winners, many small ones" profile entirely.
    trades = [_trade(500.0), _trade(200.0), _trade(10.0), _trade(8.0)]
    result = exclude_dominant_trades(trades, max_profit_share=1 / 3)
    assert all(t.pnl_pct != 500.0 for t in result)
    assert any(t.pnl_pct == 200.0 for t in result)
    assert len(result) == 3


def test_exclude_dominant_trades_no_op_when_profits_are_even():
    trades = [_trade(10.0), _trade(10.0), _trade(10.0), _trade(-4.0)]
    result = exclude_dominant_trades(trades, max_profit_share=1 / 3)
    assert len(result) == len(trades)


def test_exclude_dominant_trades_handles_no_winners():
    trades = [_trade(-5.0), _trade(-2.0)]
    result = exclude_dominant_trades(trades, max_profit_share=1 / 3)
    assert len(result) == 2
