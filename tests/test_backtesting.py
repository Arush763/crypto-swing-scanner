"""Unit tests for the enhanced backtesting engine and metrics library."""

import numpy as np
import pandas as pd
import pytest

from src.backtesting.metrics import (
    total_return, cagr, max_drawdown, sharpe_ratio, sortino_ratio,
    win_rate, profit_factor, expectancy, payoff_ratio,
    consecutive_losses, recovery_factor, compute_all_metrics,
)
from src.backtesting.engine import (
    run_backtest, _run_single_asset, _build_equity_curve,
    BacktestResult, Trade,
)
from src.config.config import BacktestConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_ohlcv(n=200, trend="up"):
    rng = np.random.default_rng(42)
    if trend == "up":
        close = pd.Series(np.linspace(100, 300, n) + rng.normal(0, 2, n))
    elif trend == "down":
        close = pd.Series(np.linspace(300, 100, n) + rng.normal(0, 2, n))
    else:
        close = pd.Series(150 + rng.normal(0, 5, n))
    close = close.clip(lower=1)
    spread = close * 0.01
    volume = pd.Series(rng.uniform(5000, 15000, n))
    return pd.DataFrame({
        "open":   close.shift(1).fillna(close.iloc[0]),
        "high":   close + spread,
        "low":    close - spread,
        "close":  close,
        "volume": volume,
    })


@pytest.fixture
def up_ohlcv():  return make_ohlcv(200, "up")
@pytest.fixture
def dn_ohlcv():  return make_ohlcv(200, "down")


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_total_return_positive(self):
        eq = pd.Series([100.0, 150.0])
        assert abs(total_return(eq) - 50.0) < 1e-6

    def test_total_return_negative(self):
        eq = pd.Series([100.0, 80.0])
        assert abs(total_return(eq) - (-20.0)) < 1e-6

    def test_max_drawdown_flat(self):
        eq = pd.Series([100.0, 100.0, 100.0])
        assert max_drawdown(eq) == 0.0

    def test_max_drawdown_known(self):
        eq = pd.Series([100.0, 80.0, 90.0])
        assert abs(max_drawdown(eq) - (-20.0)) < 1e-6

    def test_sharpe_ratio_zero_on_flat(self):
        eq = pd.Series([100.0] * 100)
        assert sharpe_ratio(eq) == 0.0

    def test_sortino_positive_on_uptrend(self):
        eq = pd.Series(np.linspace(100, 200, 100))
        assert sortino_ratio(eq) >= 0

    def test_win_rate_all_winners(self):
        assert win_rate([1.0, 2.0, 3.0]) == 100.0

    def test_win_rate_half(self):
        assert win_rate([1.0, -1.0]) == 50.0

    def test_profit_factor_all_winners(self):
        pf = profit_factor([1.0, 2.0, 3.0])
        assert pf == float("inf")

    def test_profit_factor_mixed(self):
        pf = profit_factor([2.0, -1.0])
        assert abs(pf - 2.0) < 1e-6

    def test_expectancy_correct(self):
        assert abs(expectancy([10.0, -5.0]) - 2.5) < 1e-6

    def test_payoff_ratio_correct(self):
        pr = payoff_ratio([10.0, -5.0])
        assert abs(pr - 2.0) < 1e-6

    def test_consecutive_losses(self):
        pnls = [1.0, -1.0, -1.0, -1.0, 1.0, -1.0]
        assert consecutive_losses(pnls) == 3

    def test_recovery_factor_positive(self):
        eq = pd.Series([100.0, 80.0, 120.0])
        rf = recovery_factor(eq)
        assert rf > 0

    def test_compute_all_metrics_keys(self):
        eq = pd.Series(np.linspace(100, 150, 50))
        pnls = [1.0, -0.5, 2.0, -1.0, 3.0]
        result = compute_all_metrics(eq, pnls)
        required = {
            "total_return_pct", "cagr_pct", "max_drawdown_pct",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "win_rate_pct", "profit_factor", "expectancy_pct",
            "payoff_ratio", "max_consecutive_losses", "num_trades",
        }
        assert required.issubset(result.keys())


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------

class TestBacktestEngine:
    def test_run_single_asset_returns_list(self, up_ohlcv):
        cfg = BacktestConfig()
        trades = _run_single_asset("TEST/USDT", up_ohlcv, cfg)
        assert isinstance(trades, list)

    def test_trades_have_required_fields(self, up_ohlcv):
        cfg = BacktestConfig()
        trades = _run_single_asset("TEST/USDT", up_ohlcv, cfg)
        for t in trades:
            assert hasattr(t, "pnl_pct")
            assert hasattr(t, "entry_price")
            assert hasattr(t, "exit_reason")
            assert hasattr(t, "holding_bars")

    def test_equity_curve_starts_at_capital(self):
        trades = [Trade("X", 0, 100.0, 1, 110.0, "ema", 10.0, 1, False)]
        eq = _build_equity_curve(trades, 10_000.0, 0.02)
        assert float(eq.iloc[0]) == 10_000.0

    def test_equity_curve_grows_on_winning_trades(self):
        trades = [
            Trade("X", 0, 100.0, 5, 110.0,  "ema", 10.0, 5, False),
            Trade("X", 6, 110.0, 10, 121.0, "ema", 10.0, 4, False),
        ]
        eq = _build_equity_curve(trades, 10_000.0, 0.02)
        assert float(eq.iloc[-1]) > 10_000.0

    def test_equity_curve_shrinks_on_losing_trades(self):
        trades = [
            Trade("X", 0, 100.0, 5, 90.0, "stop", -10.0, 5, False),
            Trade("X", 6, 100.0, 10, 90.0, "stop", -10.0, 4, False),
        ]
        eq = _build_equity_curve(trades, 10_000.0, 0.02)
        assert float(eq.iloc[-1]) < 10_000.0

    def test_run_backtest_returns_result(self, up_ohlcv, dn_ohlcv):
        universe = {"UP/USDT": up_ohlcv, "DN/USDT": dn_ohlcv}
        result = run_backtest(universe, BacktestConfig())
        assert isinstance(result, BacktestResult)
        assert "total_return_pct" in result.metrics
        assert "sharpe_ratio" in result.metrics

    def test_backtest_metrics_in_sane_range(self, up_ohlcv):
        result = run_backtest({"UP/USDT": up_ohlcv}, BacktestConfig())
        assert -100 <= result.metrics.get("total_return_pct", 0) <= 10_000
        assert 0.0 <= result.metrics.get("win_rate_pct", 0) <= 100.0

    def test_uptrend_better_than_downtrend(self, up_ohlcv, dn_ohlcv):
        up_result = run_backtest({"UP/USDT": up_ohlcv}, BacktestConfig())
        dn_result = run_backtest({"DN/USDT": dn_ohlcv}, BacktestConfig())
        assert up_result.metrics.get("total_return_pct", 0) >= dn_result.metrics.get("total_return_pct", 0)

    def test_per_symbol_stats_present(self, up_ohlcv):
        result = run_backtest({"UP/USDT": up_ohlcv}, BacktestConfig())
        if not result.per_symbol_stats.empty:
            assert "symbol" in result.per_symbol_stats.columns
            assert "win_rate" in result.per_symbol_stats.columns

    def test_empty_universe_returns_zero_metrics(self):
        result = run_backtest({})
        assert result.metrics.get("num_trades", 0) == 0
