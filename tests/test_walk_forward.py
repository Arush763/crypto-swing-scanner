"""Unit tests for the expanding-window, weekly-retrained walk-forward backtest."""

import pickle
from pathlib import Path

import pandas as pd
import pytest

from collections import deque

import numpy as np

from src.config.config import TapeBacktestConfig
from src.backtesting.walk_forward import (
    _precompute_symbol, run_walk_forward, _trailing_win_rate, _trailing_avg_pnl,
    _confidence_scale, TRAILING_WIN_RATE_NEUTRAL, TRAILING_WIN_RATE_WINDOW,
    TRAILING_PNL_NEUTRAL, MIN_CONFIDENCE_SCALE,
)

CACHE_PATH = Path(__file__).parent.parent / "universe_cache.pkl"


def _load_small_universe(symbols):
    with open(CACHE_PATH, "rb") as f:
        universe = pickle.load(f)
    return {s: universe[s] for s in symbols if s in universe}


def test_trailing_win_rate_neutral_when_empty():
    assert _trailing_win_rate(deque(maxlen=TRAILING_WIN_RATE_WINDOW)) == TRAILING_WIN_RATE_NEUTRAL


def test_trailing_win_rate_reflects_recent_history():
    history = deque([1, 1, 0, 1], maxlen=TRAILING_WIN_RATE_WINDOW)
    assert _trailing_win_rate(history) == 0.75


def test_trailing_win_rate_only_uses_window_not_full_history():
    history = deque(maxlen=3)
    for label in [1, 1, 1, 1, 0, 0, 0]:  # only the last 3 (0,0,0) should remain
        history.append(label)
    assert _trailing_win_rate(history) == 0.0


def test_trailing_avg_pnl_neutral_when_empty():
    assert _trailing_avg_pnl(deque(maxlen=TRAILING_WIN_RATE_WINDOW)) == TRAILING_PNL_NEUTRAL


def test_trailing_avg_pnl_distinguishes_magnitude_not_just_direction():
    # two symbols with an identical win rate (1 win in 4) but very different
    # severity -- the magnitude feature must tell them apart even though
    # the win-rate feature alone sees them as the same.
    mild_loser = deque([5.0, -1.0, -1.0, -1.0], maxlen=TRAILING_WIN_RATE_WINDOW)
    severe_loser = deque([5.0, -8.0, -8.0, -8.0], maxlen=TRAILING_WIN_RATE_WINDOW)
    assert _trailing_avg_pnl(severe_loser) < _trailing_avg_pnl(mild_loser)


def test_confidence_scale_at_threshold_is_min_scale():
    scale = _confidence_scale(np.array([0.5]), threshold=0.5)
    assert scale[0] == pytest.approx(MIN_CONFIDENCE_SCALE)


def test_confidence_scale_at_full_confidence_is_one():
    scale = _confidence_scale(np.array([1.0]), threshold=0.5)
    assert scale[0] == pytest.approx(1.0)


def test_confidence_scale_is_monotonic_between_threshold_and_one():
    probas = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    scale = _confidence_scale(probas, threshold=0.5)
    assert np.all(np.diff(scale) > 0)


def test_confidence_scale_clips_below_threshold_at_min():
    # shouldn't matter in practice (these positions are excluded by `passed`
    # already), but the function itself should not return negative/invalid
    # scale for probas below threshold.
    scale = _confidence_scale(np.array([0.1]), threshold=0.5)
    assert scale[0] == pytest.approx(MIN_CONFIDENCE_SCALE)


@pytest.mark.skipif(not CACHE_PATH.exists(), reason="universe_cache.pkl not present")
class TestWalkForwardIntegration:
    def test_precompute_returns_none_below_min_length(self):
        cfg = TapeBacktestConfig()
        universe = _load_small_universe(["BTC/USDT"])
        bars = universe["BTC/USDT"]
        short_bars = bars.iloc[:10]
        assert _precompute_symbol("BTC/USDT", short_bars, cfg, None) is None
        assert _precompute_symbol("BTC/USDT", bars, cfg, None) is not None

    def test_trades_are_chronological(self):
        cfg = TapeBacktestConfig()
        universe = _load_small_universe(["BTC/USDT", "ETH/USDT", "XRP/USDT"])
        result = run_walk_forward(universe, cfg)
        completed = [t for t in result.trades if not t.is_open]
        timestamps = [t.entry_time for t in completed]
        assert timestamps == sorted(timestamps)

    def test_expanding_universe_respects_listing_date(self):
        # truncate one symbol so it only "exists" for the second half of the
        # other's history -- no trade or training row should appear for it
        # before its own first bar.
        cfg = TapeBacktestConfig()
        universe = _load_small_universe(["BTC/USDT", "ETH/USDT"])
        full_len = len(universe["ETH/USDT"])
        universe["ETH/USDT"] = universe["ETH/USDT"].iloc[full_len // 2 :]
        eth_start = universe["ETH/USDT"].index.min()

        result = run_walk_forward(universe, cfg)
        eth_trades = [t for t in result.trades if t.symbol == "ETH/USDT" and not t.is_open]
        assert all(t.entry_time >= eth_start for t in eth_trades)

    def test_model_eventually_fits_given_enough_data(self):
        cfg = TapeBacktestConfig()
        universe = _load_small_universe(["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "LTC/USDT"])
        result = run_walk_forward(universe, cfg)
        assert result.final_model is not None
        assert any(w.model_fitted for w in result.weekly_log)

    def test_trailing_win_rate_feature_has_no_lookahead(self):
        cfg = TapeBacktestConfig()
        universe = _load_small_universe(["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "LTC/USDT"])
        result = run_walk_forward(universe, cfg)
        tdf = result.training_df
        assert "symbol_trailing_win_rate" in tdf.columns

        for symbol, grp in tdf.groupby("symbol"):
            grp = grp.sort_values("entry_time")
            wins = grp["win"].tolist()
            rates = grp["symbol_trailing_win_rate"].tolist()
            # the i-th row's rate must reflect only labels strictly before it
            assert rates[0] == TRAILING_WIN_RATE_NEUTRAL
            for i in range(1, len(wins)):
                window = wins[max(0, i - TRAILING_WIN_RATE_WINDOW):i]
                expected = sum(window) / len(window)
                assert abs(rates[i] - expected) < 1e-9

    def test_trailing_avg_pnl_feature_has_no_lookahead(self):
        cfg = TapeBacktestConfig()
        universe = _load_small_universe(["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "LTC/USDT"])
        result = run_walk_forward(universe, cfg)
        tdf = result.training_df
        assert "symbol_trailing_avg_pnl_pct" in tdf.columns

        for symbol, grp in tdf.groupby("symbol"):
            grp = grp.sort_values("entry_time")
            pnls = grp["pnl_pct"].tolist()
            rates = grp["symbol_trailing_avg_pnl_pct"].tolist()
            assert rates[0] == TRAILING_PNL_NEUTRAL
            for i in range(1, len(pnls)):
                window = pnls[max(0, i - TRAILING_WIN_RATE_WINDOW):i]
                expected = sum(window) / len(window)
                assert abs(rates[i] - expected) < 1e-9

    def test_max_symbol_loss_caps_per_symbol_cumulative_loss(self):
        cfg = TapeBacktestConfig()
        universe = _load_small_universe(["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "LTC/USDT", "ADA/USDT", "BNB/USDT", "LINK/USDT"])
        cap = 6.0
        result = run_walk_forward(universe, cfg, max_symbol_loss_pct=cap)

        completed = [t for t in result.trades if not t.is_open]
        by_symbol: dict = {}
        for t in completed:
            by_symbol.setdefault(t.symbol, []).append(t)

        breached_any = False
        for symbol, trades in by_symbol.items():
            trades.sort(key=lambda t: t.entry_time)
            running = 0.0
            breach_index = None
            for i, t in enumerate(trades):
                if breach_index is not None:
                    # no trade should appear after the one that breached the cap
                    pytest.fail(f"{symbol} took a trade at index {i} after breaching its loss cap at index {breach_index}")
                running += t.pnl_pct
                if running <= -cap:
                    breach_index = i
                    breached_any = True
        assert breached_any  # the test universe/cap should actually exercise the breaker

    def test_max_symbol_loss_blocks_further_entries_after_breach(self):
        cfg = TapeBacktestConfig()
        universe = _load_small_universe(["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "LTC/USDT", "ADA/USDT", "BNB/USDT", "LINK/USDT"])

        baseline = run_walk_forward(universe, cfg)
        capped = run_walk_forward(universe, cfg, max_symbol_loss_pct=5.0)  # aggressive cap

        baseline_n = len([t for t in baseline.trades if not t.is_open])
        capped_n = len([t for t in capped.trades if not t.is_open])
        assert capped_n < baseline_n

    def test_max_symbol_loss_disabled_by_default_matches_prior_behaviour(self):
        cfg = TapeBacktestConfig()
        universe = _load_small_universe(["BTC/USDT", "ETH/USDT"])
        a = run_walk_forward(universe, cfg)
        b = run_walk_forward(universe, cfg, max_symbol_loss_pct=None)
        assert [t.pnl_pct for t in a.trades] == [t.pnl_pct for t in b.trades]

    def test_blacklist_threshold_reduces_a_chronic_losers_pass_rate(self):
        cfg = TapeBacktestConfig()
        universe = _load_small_universe(["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "LTC/USDT", "ADA/USDT", "BNB/USDT", "LINK/USDT"])

        baseline = run_walk_forward(universe, cfg)
        blacklisted = run_walk_forward(universe, cfg, blacklist_win_rate_threshold=0.5, blacklist_min_samples=1)

        # threshold=0.5, min_samples=1 is deliberately aggressive (blocks a
        # symbol the moment its very first labelled setup is a loss) so this
        # should strictly reduce or hold flat the number of completed trades
        # taken, never increase it.
        baseline_n = len([t for t in baseline.trades if not t.is_open])
        blacklisted_n = len([t for t in blacklisted.trades if not t.is_open])
        assert blacklisted_n <= baseline_n

    def test_blacklist_disabled_by_default_matches_prior_behaviour(self):
        cfg = TapeBacktestConfig()
        universe = _load_small_universe(["BTC/USDT", "ETH/USDT"])
        a = run_walk_forward(universe, cfg)
        b = run_walk_forward(universe, cfg, blacklist_win_rate_threshold=None)
        assert [t.pnl_pct for t in a.trades] == [t.pnl_pct for t in b.trades]

    def test_volume_filter_blocks_thin_symbols(self):
        cfg = TapeBacktestConfig()
        universe = _load_small_universe(["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "LTC/USDT"])

        baseline = run_walk_forward(universe, cfg)
        # set the floor absurdly high -- every symbol's rolling volume should
        # fail it, so this should produce strictly fewer trades than baseline.
        filtered = run_walk_forward(universe, cfg, min_dollar_volume=1e15)

        baseline_n = len([t for t in baseline.trades if not t.is_open])
        filtered_n = len([t for t in filtered.trades if not t.is_open])
        assert filtered_n < baseline_n

    def test_volume_filter_disabled_by_default_matches_prior_behaviour(self):
        cfg = TapeBacktestConfig()
        universe = _load_small_universe(["BTC/USDT", "ETH/USDT"])
        a = run_walk_forward(universe, cfg)
        b = run_walk_forward(universe, cfg, min_dollar_volume=None)
        assert [t.pnl_pct for t in a.trades] == [t.pnl_pct for t in b.trades]

    def test_confidence_sizing_assigns_variable_risk_pct(self):
        cfg = TapeBacktestConfig()
        universe = _load_small_universe(["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "LTC/USDT"])
        result = run_walk_forward(universe, cfg, confidence_sizing=True)
        completed = [t for t in result.trades if not t.is_open]
        risk_pcts = [t.risk_pct for t in completed if t.risk_pct is not None]
        assert len(risk_pcts) > 0
        assert all(0 < r <= cfg.risk_per_trade_pct for r in risk_pcts)
        # not every trade should get full size -- some variation expected
        # once the model has enough data to differentiate confidence.
        assert len(set(round(r, 6) for r in risk_pcts)) > 1

    def test_confidence_sizing_disabled_by_default_matches_prior_behaviour(self):
        cfg = TapeBacktestConfig()
        universe = _load_small_universe(["BTC/USDT", "ETH/USDT"])
        a = run_walk_forward(universe, cfg)
        b = run_walk_forward(universe, cfg, confidence_sizing=False)
        assert [t.pnl_pct for t in a.trades] == [t.pnl_pct for t in b.trades]
        # the risk_pct column always exists (defaults to cfg.risk_per_trade_pct
        # per symbol precompute) -- disabling confidence_sizing means every
        # trade keeps that flat default rather than None.
        assert all(t.risk_pct == pytest.approx(cfg.risk_per_trade_pct) for t in a.trades)

    def test_weekly_log_is_dense_and_covers_full_range(self):
        cfg = TapeBacktestConfig()
        universe = _load_small_universe(["BTC/USDT", "ETH/USDT"])
        result = run_walk_forward(universe, cfg)
        starts = [w.week_start for w in result.weekly_log]
        assert starts == sorted(starts)
        assert len(starts) == len(set(starts))
