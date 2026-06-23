"""Unit tests for resampling raw tick trades into buy/sell-split OHLCV bars."""

import pandas as pd
import pytest

from src.data.trade_tape import resample_to_bars, _infer_time_unit


class TestInferTimeUnit:
    def test_milliseconds_detected(self):
        # ~2024-01-01 in ms epoch (13 digits)
        assert _infer_time_unit(1_704_067_200_000) == "ms"

    def test_microseconds_detected(self):
        # Binance switched aggTrades dumps to microsecond timestamps (16 digits)
        assert _infer_time_unit(1_781_481_600_142_209) == "us"

    def test_nanoseconds_detected(self):
        assert _infer_time_unit(1_781_481_600_142_209_000) == "ns"


def _trades(rows):
    """rows: list of (timestamp_str, price, qty, is_buyer_maker)."""
    df = pd.DataFrame(rows, columns=["transact_time", "price", "quantity", "is_buyer_maker"])
    df["transact_time"] = pd.to_datetime(df["transact_time"], utc=True)
    return df


class TestResampleToBars:
    def test_empty_trades_returns_empty_bars(self):
        bars = resample_to_bars(pd.DataFrame(columns=["transact_time", "price", "quantity", "is_buyer_maker"]))
        assert bars.empty

    def test_ohlc_correct_within_one_bar(self):
        trades = _trades([
            ("2024-01-01 00:00:00", 100.0, 1.0, False),
            ("2024-01-01 00:30:00", 105.0, 1.0, False),
            ("2024-01-01 01:30:00", 95.0, 1.0, True),
            ("2024-01-01 02:00:00", 102.0, 1.0, False),
        ])
        bars = resample_to_bars(trades, timeframe="4h")
        assert len(bars) == 1
        row = bars.iloc[0]
        assert row["open"] == 100.0
        assert row["high"] == 105.0
        assert row["low"] == 95.0
        assert row["close"] == 102.0

    def test_buy_sell_volume_split(self):
        trades = _trades([
            ("2024-01-01 00:00:00", 100.0, 2.0, False),   # taker buy -> buy_volume
            ("2024-01-01 00:10:00", 100.0, 3.0, True),    # taker sell -> sell_volume
        ])
        bars = resample_to_bars(trades, timeframe="4h")
        row = bars.iloc[0]
        assert row["buy_volume"] == pytest.approx(200.0)
        assert row["sell_volume"] == pytest.approx(300.0)
        assert row["volume"] == pytest.approx(500.0)

    def test_multiple_bars_split_by_timeframe(self):
        trades = _trades([
            ("2024-01-01 00:00:00", 100.0, 1.0, False),
            ("2024-01-01 05:00:00", 110.0, 1.0, False),
        ])
        bars = resample_to_bars(trades, timeframe="4h")
        assert len(bars) >= 2
