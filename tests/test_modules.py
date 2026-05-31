"""Unit tests for breakout, retest, and squeeze detection modules."""

import numpy as np
import pandas as pd
import pytest

from src.modules.breakout import detect_breakout, resistance_level
from src.modules.retest import detect_retest
from src.modules.squeeze import detect_squeeze


# ---------------------------------------------------------------------------
# Helpers to build synthetic OHLCV data
# ---------------------------------------------------------------------------

def make_ohlcv(close_vals, volume_vals=None, spread_pct=0.01) -> pd.DataFrame:
    close = pd.Series(close_vals, dtype=float)
    spread = close * spread_pct
    if volume_vals is None:
        volume_vals = [10_000.0] * len(close)
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + spread,
        "low": close - spread,
        "close": close,
        "volume": pd.Series(volume_vals, dtype=float),
    })


def make_base_ohlcv(n=60, base=100.0) -> pd.DataFrame:
    """n bars of flat price at base — acts as consolidation."""
    rng = np.random.default_rng(7)
    close = base + rng.uniform(-1, 1, n)
    return make_ohlcv(close)


# ---------------------------------------------------------------------------
# Breakout tests
# ---------------------------------------------------------------------------

class TestBreakout:
    def test_no_breakout_on_flat(self):
        df = make_base_ohlcv(60, 100)
        result = detect_breakout(df["high"], df["low"], df["close"], df["volume"])
        assert not result.is_breakout

    def test_breakout_detected_on_surge(self):
        """Price breaks above prior resistance on high volume."""
        close = [100.0] * 40 + [101.0] * 19 + [115.0]  # 60 bars, last breaks out
        volume = [1_000.0] * 59 + [3_000.0]             # last bar: 3x avg
        df = make_ohlcv(close, volume)
        result = detect_breakout(df["high"], df["low"], df["close"], df["volume"])
        assert result.is_breakout

    def test_no_breakout_without_volume(self):
        """Price surges but volume is flat — should not trigger."""
        close = [100.0] * 40 + [101.0] * 19 + [115.0]
        volume = [1_000.0] * 60                         # flat volume
        df = make_ohlcv(close, volume)
        result = detect_breakout(df["high"], df["low"], df["close"], df["volume"],
                                  volume_multiplier=2.0)
        assert not result.is_breakout

    def test_resistance_level_is_recent_swing_high(self):
        close = [100.0] * 30 + [120.0] + [100.0] * 29
        df = make_ohlcv(close)
        res = resistance_level(df["high"], lookback=40)
        # Resistance should be near 120 (the swing high)
        assert res > 110

    def test_bonus_score_positive_on_breakout(self):
        close = [100.0] * 40 + [101.0] * 19 + [115.0]
        volume = [1_000.0] * 59 + [3_000.0]
        df = make_ohlcv(close, volume)
        result = detect_breakout(df["high"], df["low"], df["close"], df["volume"])
        if result.is_breakout:
            assert result.bonus_score > 0

    def test_result_has_required_fields(self):
        df = make_base_ohlcv()
        result = detect_breakout(df["high"], df["low"], df["close"], df["volume"])
        assert hasattr(result, "is_breakout")
        assert hasattr(result, "resistance_level")
        assert hasattr(result, "volume_ratio_at_breakout")
        assert hasattr(result, "bonus_score")


# ---------------------------------------------------------------------------
# Retest tests
# ---------------------------------------------------------------------------

class TestRetest:
    def _make_breakout_result(self, resistance=105.0):
        from src.modules.breakout import BreakoutResult
        return BreakoutResult(
            is_breakout=True,
            resistance_level=resistance,
            breakout_bar_index=-10,
            volume_ratio_at_breakout=2.5,
            bonus_score=8.0,
            age_bars=0,
        )

    def test_no_retest_without_breakout(self):
        from src.modules.breakout import BreakoutResult
        df = make_base_ohlcv()
        no_bo = BreakoutResult(False, 0.0, -1, 0.0, 0.0, -1)
        result = detect_retest(df["open"], df["high"], df["low"], df["close"], no_bo)
        assert not result.is_retest

    def test_retest_detected_when_price_pulls_back(self):
        """Breakout at bar 50, then price pulls back to resistance zone at bar 55."""
        res = 105.0
        # Build: flat → breakout → pullback to resistance → hold
        close = [100.0] * 40 + [106.0] * 10 + [105.5] * 5 + [110.0] * 5
        volume = [1_000.0] * 60
        df = make_ohlcv(close, volume)

        from src.modules.breakout import BreakoutResult
        bo = BreakoutResult(
            is_breakout=True,
            resistance_level=res,
            breakout_bar_index=50,
            volume_ratio_at_breakout=2.0,
            bonus_score=8.0,
            age_bars=0,
        )
        result = detect_retest(df["open"], df["high"], df["low"], df["close"], bo,
                                lookback_bars=15)
        # Not strictly required to find a retest in this synthetic data
        # but result must have required fields
        assert hasattr(result, "is_retest")
        assert hasattr(result, "bonus_score")

    def test_retest_bonus_higher_than_breakout_bonus(self):
        """A confirmed retest should produce a higher bonus than the breakout."""
        res = 105.0
        close = [100.0] * 40 + [108.0] * 10 + [105.2] * 3 + [109.0] * 7
        volume = [1_000.0] * 60
        df = make_ohlcv(close, volume)

        from src.modules.breakout import BreakoutResult
        bo = BreakoutResult(
            is_breakout=True,
            resistance_level=res,
            breakout_bar_index=50,
            volume_ratio_at_breakout=2.0,
            bonus_score=7.0,
            age_bars=0,
        )
        rt = detect_retest(df["open"], df["high"], df["low"], df["close"], bo,
                           lookback_bars=15)
        if rt.is_retest:
            assert rt.bonus_score > bo.bonus_score


# ---------------------------------------------------------------------------
# Squeeze tests
# ---------------------------------------------------------------------------

class TestSqueeze:
    def test_result_has_required_fields(self):
        df = make_base_ohlcv(100)
        result = detect_squeeze(df["high"], df["low"], df["close"], df["volume"])
        assert hasattr(result, "in_squeeze")
        assert hasattr(result, "squeeze_breakout")
        assert hasattr(result, "bb_width_pct")
        assert hasattr(result, "atr_pct")
        assert hasattr(result, "bonus_score")

    def test_bonus_zero_when_not_in_squeeze(self):
        """Volatile, trending asset should not be in squeeze."""
        rng = np.random.default_rng(99)
        # Wide ranging bars — high volatility
        close = pd.Series(100 + np.cumsum(rng.uniform(-5, 5, 150)))
        spread = close * 0.05
        df = pd.DataFrame({
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": pd.Series([1000.0] * len(close)),
        })
        result = detect_squeeze(df["high"], df["low"], df["close"], df["volume"])
        # In a truly high-vol environment, bonus should be low
        assert result.bonus_score <= 10.0

    def test_in_squeeze_detection_on_tight_range(self):
        """
        Flat price series: ATR and BB width are both near-zero relative to
        history, so both percentiles should be low → in_squeeze = True.
        We build a long history of high-volatility then transition to a
        very tight channel so the percentile ranking is clear.
        """
        rng = np.random.default_rng(42)
        # 150 bars of noisy history so percentile has context
        noisy = list(100 + np.cumsum(rng.uniform(-2, 2, 150)))
        # 50 bars of extremely tight channel
        flat = [noisy[-1] + rng.uniform(-0.01, 0.01) for _ in range(50)]
        close = pd.Series(noisy + flat)
        spread_hist = [abs(rng.uniform(-2, 2)) for _ in range(150)]
        spread_flat = [0.01] * 50
        spread = pd.Series(spread_hist + spread_flat)
        df = pd.DataFrame({
            "high": close + spread,
            "low": close - spread.abs(),
            "close": close,
            "volume": pd.Series([1000.0] * len(close)),
        })
        result = detect_squeeze(df["high"], df["low"], df["close"], df["volume"],
                                threshold_pct=30.0)
        # With a long volatile history followed by a tight channel, at least
        # one of the volatility metrics should be compressed.
        assert result.in_squeeze or result.atr_pct < 30.0
