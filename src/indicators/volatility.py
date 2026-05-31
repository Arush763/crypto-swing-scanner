"""
Volatility indicators: ATR, Bollinger Bands, Historical Volatility.

Used by the squeeze/compression detection module and stop-loss calculations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.config import (
    ATR_PERIOD,
    ATR_PERCENTILE_LOOKBACK,
    BB_PERIOD,
    BB_STD,
    HV_PERIOD,
    SQUEEZE_PERCENTILE_THRESHOLD,
)


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    tr = true_range(high, low, close)
    return tr.ewm(span=period, adjust=False).mean()


def atr_latest(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> float:
    return float(atr(high, low, close, period).iloc[-1])


def atr_percentile(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = ATR_PERIOD,
    lookback: int = ATR_PERCENTILE_LOOKBACK,
) -> float:
    """
    Percentile rank of current ATR relative to its own history.
    0 = lowest ever, 100 = highest ever (over `lookback` bars).
    """
    atr_series = atr(high, low, close, period).dropna()
    if len(atr_series) < 2:
        return 50.0
    window = atr_series.iloc[-lookback:]
    current = float(atr_series.iloc[-1])
    pct = float((window < current).mean() * 100)
    return pct


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------

def bollinger_bands(
    close: pd.Series, period: int = BB_PERIOD, num_std: float = BB_STD
) -> pd.DataFrame:
    """Return DataFrame with columns: mid, upper, lower, width."""
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid  # normalised width
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "width": width})


def bb_width_percentile(
    close: pd.Series,
    period: int = BB_PERIOD,
    num_std: float = BB_STD,
    lookback: int = ATR_PERCENTILE_LOOKBACK,
) -> float:
    """
    Percentile rank of current BB width vs its own history.
    Low percentile = compression (squeeze). High = expansion.
    """
    bb = bollinger_bands(close, period, num_std)
    width_series = bb["width"].dropna()
    if len(width_series) < 2:
        return 50.0
    window = width_series.iloc[-lookback:]
    current = float(width_series.iloc[-1])
    return float((window < current).mean() * 100)


# ---------------------------------------------------------------------------
# Historical Volatility
# ---------------------------------------------------------------------------

def historical_volatility(close: pd.Series, period: int = HV_PERIOD) -> pd.Series:
    """Annualised historical volatility (log-return std * sqrt(365))."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(period).std() * np.sqrt(365)


def hv_percentile(close: pd.Series, period: int = HV_PERIOD, lookback: int = ATR_PERCENTILE_LOOKBACK) -> float:
    hv = historical_volatility(close, period).dropna()
    if len(hv) < 2:
        return 50.0
    window = hv.iloc[-lookback:]
    current = float(hv.iloc[-1])
    return float((window < current).mean() * 100)


# ---------------------------------------------------------------------------
# Composite squeeze detection
# ---------------------------------------------------------------------------

def is_in_squeeze(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    threshold_pct: float = SQUEEZE_PERCENTILE_THRESHOLD,
) -> bool:
    """
    True when both BB width percentile and ATR percentile are below threshold,
    indicating compressed volatility likely to expand.
    """
    bb_pct = bb_width_percentile(close)
    atr_pct = atr_percentile(high, low, close)
    return bb_pct < threshold_pct and atr_pct < threshold_pct


def volatility_signals(high: pd.Series, low: pd.Series, close: pd.Series) -> dict:
    """Return all volatility metrics for the latest bar."""
    bb = bollinger_bands(close)
    latest_atr = atr_latest(high, low, close)
    latest_price = float(close.iloc[-1])

    return {
        "atr": latest_atr,
        "atr_pct_of_price": latest_atr / latest_price if latest_price > 0 else 0.0,
        "atr_percentile": atr_percentile(high, low, close),
        "bb_upper": float(bb["upper"].iloc[-1]),
        "bb_lower": float(bb["lower"].iloc[-1]),
        "bb_mid": float(bb["mid"].iloc[-1]),
        "bb_width": float(bb["width"].iloc[-1]),
        "bb_width_percentile": bb_width_percentile(close),
        "hv_percentile": hv_percentile(close),
        "in_squeeze": is_in_squeeze(high, low, close),
    }
