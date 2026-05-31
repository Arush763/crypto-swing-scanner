"""RSI and derived signals."""

from __future__ import annotations
import pandas as pd
import numpy as np


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def rsi_is_bullish(close: pd.Series, period: int = 14, threshold: float = 50.0) -> pd.Series:
    """True when RSI > threshold AND rising (current > previous bar)."""
    r = rsi(close, period)
    return (r > threshold) & (r > r.shift(1))
