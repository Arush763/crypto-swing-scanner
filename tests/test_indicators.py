"""Unit tests for all technical indicator functions."""

import numpy as np
import pandas as pd
import pytest

from src.indicators.trend import (
    ema,
    ema_20,
    ema_50,
    compute_trend_signals,
    price_distance_from_ema,
)
from src.indicators.momentum import (
    nday_return,
    all_momentum_returns,
    relative_strength_vs_benchmark,
)
from src.indicators.volatility import (
    true_range,
    atr,
    atr_percentile,
    bollinger_bands,
    bb_width_percentile,
    is_in_squeeze,
    volatility_signals,
)
from src.indicators.volume import (
    avg_volume,
    volume_ratio,
    volume_consistency_score,
    is_volume_expansion,
    volume_trend_slope,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trending_up_series() -> pd.Series:
    """100-bar steadily rising price series."""
    return pd.Series(np.linspace(100, 200, 100))


@pytest.fixture
def flat_series() -> pd.Series:
    return pd.Series(np.ones(100) * 100.0)


@pytest.fixture
def noisy_series() -> pd.Series:
    rng = np.random.default_rng(42)
    prices = 100 + np.cumsum(rng.standard_normal(200))
    return pd.Series(prices)


@pytest.fixture
def ohlcv_df(trending_up_series) -> pd.DataFrame:
    n = len(trending_up_series)
    close = trending_up_series
    spread = close * 0.01
    high = close + spread
    low = close - spread
    volume = pd.Series(np.random.default_rng(0).uniform(1000, 5000, n))
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# ---------------------------------------------------------------------------
# EMA tests
# ---------------------------------------------------------------------------

class TestEMA:
    def test_ema_length_preserved(self, trending_up_series):
        result = ema(trending_up_series, 20)
        assert len(result) == len(trending_up_series)

    def test_ema_rising_on_uptrend(self, trending_up_series):
        e = ema(trending_up_series, 20)
        assert float(e.iloc[-1]) > float(e.iloc[0])

    def test_ema_flat_on_flat(self, flat_series):
        e = ema(flat_series, 20)
        assert abs(float(e.iloc[-1]) - 100.0) < 1e-6

    def test_ema_20_uses_period_20(self, trending_up_series):
        assert ema_20(trending_up_series).equals(ema(trending_up_series, 20))

    def test_compute_trend_signals_all_bullish(self, trending_up_series):
        signals = compute_trend_signals(trending_up_series)
        assert signals["price_above_ema20"]
        assert signals["price_above_ema50"]
        assert signals["ema20_above_ema50"]

    def test_compute_trend_signals_keys(self, trending_up_series):
        signals = compute_trend_signals(trending_up_series)
        required = {"price_above_ema20", "price_above_ema50", "price_above_ema200",
                    "ema20_above_ema50", "ema50_above_ema200"}
        assert required.issubset(signals.keys())

    def test_price_distance_positive_on_uptrend(self, trending_up_series):
        dist = price_distance_from_ema(trending_up_series, 20)
        assert dist > 0


# ---------------------------------------------------------------------------
# Momentum tests
# ---------------------------------------------------------------------------

class TestMomentum:
    def test_nday_return_correct(self):
        prices = pd.Series([100.0, 110.0])
        assert abs(nday_return(prices, 1) - 0.10) < 1e-9

    def test_nday_return_insufficient_data(self):
        prices = pd.Series([100.0])
        assert nday_return(prices, 5) == 0.0

    def test_all_momentum_returns_keys(self, trending_up_series):
        result = all_momentum_returns(trending_up_series, [7, 14, 30])
        assert set(result.keys()) == {7, 14, 30}

    def test_positive_returns_on_uptrend(self, trending_up_series):
        result = all_momentum_returns(trending_up_series, [7, 14, 30])
        assert all(v > 0 for v in result.values())

    def test_relative_strength_outperforming(self):
        asset = pd.Series(np.linspace(100, 150, 50))   # +50%
        bench = pd.Series(np.linspace(100, 110, 50))   # +10%
        rs = relative_strength_vs_benchmark(asset, bench, period=30)
        assert rs > 1.0

    def test_relative_strength_underperforming(self):
        asset = pd.Series(np.linspace(100, 105, 50))   # +5%
        bench = pd.Series(np.linspace(100, 130, 50))   # +30%
        rs = relative_strength_vs_benchmark(asset, bench, period=30)
        assert rs < 1.0


# ---------------------------------------------------------------------------
# Volatility tests
# ---------------------------------------------------------------------------

class TestVolatility:
    def test_true_range_always_non_negative(self, ohlcv_df):
        tr = true_range(ohlcv_df["high"], ohlcv_df["low"], ohlcv_df["close"])
        assert (tr.dropna() >= 0).all()

    def test_atr_length(self, ohlcv_df):
        result = atr(ohlcv_df["high"], ohlcv_df["low"], ohlcv_df["close"])
        assert len(result) == len(ohlcv_df)

    def test_atr_positive(self, ohlcv_df):
        result = atr(ohlcv_df["high"], ohlcv_df["low"], ohlcv_df["close"])
        assert float(result.dropna().iloc[-1]) > 0

    def test_bollinger_bands_columns(self, trending_up_series):
        bb = bollinger_bands(trending_up_series)
        assert set(bb.columns) == {"mid", "upper", "lower", "width"}

    def test_bollinger_upper_above_lower(self, trending_up_series):
        bb = bollinger_bands(trending_up_series)
        assert (bb["upper"].dropna() > bb["lower"].dropna()).all()

    def test_atr_percentile_in_range(self, ohlcv_df):
        pct = atr_percentile(ohlcv_df["high"], ohlcv_df["low"], ohlcv_df["close"])
        assert 0.0 <= pct <= 100.0

    def test_bb_width_percentile_in_range(self, trending_up_series):
        pct = bb_width_percentile(trending_up_series)
        assert 0.0 <= pct <= 100.0

    def test_volatility_signals_keys(self, ohlcv_df):
        sigs = volatility_signals(ohlcv_df["high"], ohlcv_df["low"], ohlcv_df["close"])
        required = {"atr", "bb_width_percentile", "atr_percentile", "in_squeeze"}
        assert required.issubset(sigs.keys())


# ---------------------------------------------------------------------------
# Volume tests
# ---------------------------------------------------------------------------

class TestVolume:
    def test_avg_volume_correct(self):
        vol = pd.Series([100.0] * 30)
        assert avg_volume(vol, 30) == 100.0

    def test_volume_ratio_above_average(self):
        vol = pd.Series([100.0] * 29 + [300.0])
        assert volume_ratio(vol, 30) > 1.0

    def test_volume_consistency_perfect(self):
        vol = pd.Series([1000.0] * 30)
        # Perfect consistency → std = 0 → CV = 0 → score = 1.0
        score = volume_consistency_score(vol, 30)
        assert score == 1.0

    def test_is_volume_expansion_true(self):
        vol = pd.Series([100.0] * 29 + [250.0])
        assert is_volume_expansion(vol, multiplier=2.0)

    def test_is_volume_expansion_false(self):
        vol = pd.Series([100.0] * 30)
        assert not is_volume_expansion(vol, multiplier=2.0)

    def test_volume_trend_slope_rising(self):
        vol = pd.Series(np.linspace(100, 200, 30))
        slope = volume_trend_slope(vol, 20)
        assert slope > 0
