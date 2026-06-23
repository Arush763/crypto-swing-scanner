"""Unit tests for all scoring modules and the composite scorer."""

import numpy as np
import pandas as pd
import pytest

from src.scoring.trend_score import compute_trend_score
from src.scoring.momentum_score import compute_momentum_score
from src.scoring.liquidity_score import compute_liquidity_score
from src.scoring.smart_money_score import compute_smart_money_score
from src.scoring.composite import score_asset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def strong_uptrend(n=200) -> pd.DataFrame:
    """Strongly trending upward OHLCV."""
    close = pd.Series(np.linspace(100, 400, n))
    spread = close * 0.01
    volume = pd.Series(np.random.default_rng(1).uniform(5000, 10000, n))
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + spread,
        "low": close - spread,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def downtrend(n=200) -> pd.DataFrame:
    """Strongly declining OHLCV."""
    close = pd.Series(np.linspace(400, 100, n))
    spread = close * 0.01
    volume = pd.Series(np.random.default_rng(2).uniform(5000, 10000, n))
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + spread,
        "low": close - spread,
        "close": close,
        "volume": volume,
    })


# ---------------------------------------------------------------------------
# Trend Score
# ---------------------------------------------------------------------------

class TestTrendScore:
    def test_score_in_range(self, strong_uptrend):
        result = compute_trend_score(strong_uptrend["close"])
        assert 0 <= result["score"] <= 100

    def test_uptrend_scores_higher_than_downtrend(self, strong_uptrend, downtrend):
        up_score = compute_trend_score(strong_uptrend["close"])["score"]
        dn_score = compute_trend_score(downtrend["close"])["score"]
        assert up_score > dn_score

    def test_breakdown_keys_present(self, strong_uptrend):
        result = compute_trend_score(strong_uptrend["close"])
        assert "breakdown" in result
        assert "score" in result
        assert "signals" in result

    def test_all_conditions_pass_on_uptrend(self, strong_uptrend):
        result = compute_trend_score(strong_uptrend["close"])
        breakdown = result["breakdown"]
        assert breakdown["price_above_ema20"]["passed"]
        assert breakdown["price_above_ema50"]["passed"]


# ---------------------------------------------------------------------------
# Momentum Score
# ---------------------------------------------------------------------------

class TestMomentumScore:
    def test_score_in_range(self, strong_uptrend):
        result = compute_momentum_score(strong_uptrend["close"])
        assert 0 <= result["score"] <= 100

    def test_uptrend_higher_momentum(self, strong_uptrend, downtrend):
        up = compute_momentum_score(strong_uptrend["close"])["score"]
        dn = compute_momentum_score(downtrend["close"])["score"]
        assert up > dn

    def test_rs_keys_present(self, strong_uptrend):
        result = compute_momentum_score(strong_uptrend["close"])
        assert "rs_vs_btc" in result["breakdown"]
        assert "rs_vs_market" in result["breakdown"]


# ---------------------------------------------------------------------------
# Liquidity Score
# ---------------------------------------------------------------------------

class TestLiquidityScore:
    def test_score_in_range(self, strong_uptrend):
        result = compute_liquidity_score(strong_uptrend["volume"])
        assert 0 <= result["score"] <= 100

    def test_higher_volume_scores_higher(self):
        low_vol = pd.Series([5_000.0] * 60)
        high_vol = pd.Series([100_000_000.0] * 60)
        low_score = compute_liquidity_score(low_vol)["score"]
        high_score = compute_liquidity_score(high_vol)["score"]
        assert high_score > low_score

    def test_exchange_coverage_bonus(self, strong_uptrend):
        one_ex = compute_liquidity_score(strong_uptrend["volume"], exchange_count=1)["score"]
        four_ex = compute_liquidity_score(strong_uptrend["volume"], exchange_count=4)["score"]
        assert four_ex > one_ex


# ---------------------------------------------------------------------------
# Smart Money Score
# ---------------------------------------------------------------------------

class TestSmartMoneyScore:
    def test_score_in_range(self, strong_uptrend):
        df = strong_uptrend
        result = compute_smart_money_score(
            df["open"], df["high"], df["low"], df["close"], df["volume"]
        )
        assert 0 <= result["score"] <= 100

    def test_breakdown_keys(self, strong_uptrend):
        df = strong_uptrend
        result = compute_smart_money_score(
            df["open"], df["high"], df["low"], df["close"], df["volume"]
        )
        assert "obv_trend" in result["breakdown"]
        assert "chaikin_money_flow" in result["breakdown"]

    def test_source_is_ohlcv_proxy_without_provider(self, strong_uptrend):
        df = strong_uptrend
        result = compute_smart_money_score(
            df["open"], df["high"], df["low"], df["close"], df["volume"]
        )
        assert result["source"] == "ohlcv_proxy"


# ---------------------------------------------------------------------------
# Composite Scorer
# ---------------------------------------------------------------------------

class TestCompositeScorer:
    def test_final_score_in_range(self, strong_uptrend):
        result = score_asset("TEST/USDT", strong_uptrend)
        assert 0 <= result.final_score <= 100

    def test_bonuses_increase_score(self, strong_uptrend):
        base = score_asset("TEST/USDT", strong_uptrend)
        with_bonus = score_asset("TEST/USDT", strong_uptrend, wall_bonus=10.0)
        assert with_bonus.final_score >= base.final_score

    def test_score_capped_at_100(self, strong_uptrend):
        result = score_asset(
            "TEST/USDT", strong_uptrend,
            wall_bonus=150.0,
        )
        assert result.final_score <= 100.0

    def test_uptrend_scores_higher_than_downtrend(self, strong_uptrend, downtrend):
        up = score_asset("UP/USDT", strong_uptrend).final_score
        dn = score_asset("DN/USDT", downtrend).final_score
        assert up > dn
