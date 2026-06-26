"""Unit tests for the lightweight kline-archive historical data fetcher."""

import pandas as pd
import pytest

from src.data.kline_archive import _klines_to_bars, _parse_klines_csv


def _raw_kline_row(open_time_ms, o, h, l, c, quote_vol, taker_buy_quote_vol):
    return (
        f"{open_time_ms},{o},{h},{l},{c},1.0,{open_time_ms + 14399999},"
        f"{quote_vol},10,1.0,{taker_buy_quote_vol},0"
    ).encode()


def test_klines_to_bars_preserves_values_and_index():
    # regression test: an earlier version built the DataFrame from a dict of
    # Series (each still carrying its own default RangeIndex) plus an
    # explicit `index=` kwarg -- pandas reindex-aligns each Series against
    # that index rather than just attaching it, and since the Series' own
    # index (0, 1, 2...) never matches real timestamps, every column came
    # out all-NaN.
    raw = b"\n".join([
        _raw_kline_row(1717200000000, 100.0, 101.0, 99.0, 100.5, 1000.0, 600.0),
        _raw_kline_row(1717214400000, 100.5, 102.0, 100.0, 101.5, 2000.0, 1200.0),
    ])
    df = _parse_klines_csv(raw)
    bars = _klines_to_bars(df)

    assert not bars.isna().any().any()
    assert isinstance(bars.index, pd.DatetimeIndex)
    assert list(bars["open"]) == [100.0, 100.5]
    assert list(bars["close"]) == [100.5, 101.5]
    assert list(bars["volume"]) == [1000.0, 2000.0]
    assert list(bars["buy_volume"]) == [600.0, 1200.0]
    assert list(bars["sell_volume"]) == pytest.approx([400.0, 800.0])


def test_klines_to_bars_empty_input():
    df = pd.DataFrame(columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "ignore",
    ])
    bars = _klines_to_bars(df)
    assert bars.empty
    assert list(bars.columns) == ["open", "high", "low", "close", "volume", "buy_volume", "sell_volume"]
