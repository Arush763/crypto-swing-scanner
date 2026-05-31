"""
Smart Money Score (20% of final score).

Detects signs of institutional or whale accumulation using on-chain
proxy metrics derivable from OHLCV data, optionally enhanced with live
order book signals when available.

OHLCV proxies:
  - On-balance volume (OBV) trend
  - Chaikin Money Flow (CMF)
  - VWAP position
  - Accumulation candles
  - Up/down volume ratio

Order book augmentation (when OrderBookSignals is supplied):
  - Bid/ask imbalance contributes directly to the score
  - Conviction at resistance level weighted heavily
  - Ask wall overhead penalises the score (distribution risk)

On-chain plug-in: implement OnChainDataProvider to replace OHLCV proxies
with real exchange-flow or wallet data.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# On-chain data provider interface
# ---------------------------------------------------------------------------

class OnChainDataProvider(ABC):
    @abstractmethod
    def get_exchange_outflow_score(self, symbol: str) -> float: ...
    @abstractmethod
    def get_whale_accumulation_score(self, symbol: str) -> float: ...
    @abstractmethod
    def get_active_address_growth_score(self, symbol: str) -> float: ...
    @abstractmethod
    def get_holder_count_growth_score(self, symbol: str) -> float: ...


class NullOnChainProvider(OnChainDataProvider):
    def get_exchange_outflow_score(self, symbol: str) -> float: return 50.0
    def get_whale_accumulation_score(self, symbol: str) -> float: return 50.0
    def get_active_address_growth_score(self, symbol: str) -> float: return 50.0
    def get_holder_count_growth_score(self, symbol: str) -> float: return 50.0


# ---------------------------------------------------------------------------
# OHLCV-derived accumulation proxies
# ---------------------------------------------------------------------------

def _obv_trend(close: pd.Series, volume: pd.Series, period: int = 20) -> float:
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * volume).cumsum()
    recent = obv.iloc[-period:]
    if len(recent) < 2:
        return 0.0
    x = np.arange(len(recent))
    slope = float(np.polyfit(x, recent.values, 1)[0])
    std = float(recent.std()) or 1.0
    return float(max(-1.0, min(1.0, slope / std)))


def _accumulation_candle_ratio(
    open_: pd.Series, high: pd.Series, low: pd.Series,
    close: pd.Series, volume: pd.Series, period: int = 20
) -> float:
    bar_range = high - low
    close_position = (close - low) / bar_range.replace(0, np.nan)
    avg_vol = volume.rolling(period).mean()
    is_accumulation = (close_position > 0.7) & (volume > avg_vol)
    return float(is_accumulation.iloc[-period:].mean())


def _chaikin_money_flow(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20
) -> float:
    bar_range = high - low
    mfm = ((close - low) - (high - close)) / bar_range.replace(0, np.nan)
    mfv = mfm * volume
    cmf = mfv.rolling(period).sum() / volume.rolling(period).sum()
    val = float(cmf.iloc[-1])
    return 0.0 if np.isnan(val) else val


def _vwap_position(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20
) -> float:
    typical = (high + low + close) / 3
    vwap = (typical * volume).rolling(period).sum() / volume.rolling(period).sum()
    atr_est = (high - low).rolling(period).mean()
    diff = close - vwap
    norm = diff / atr_est.replace(0, np.nan)
    val = float(norm.iloc[-1])
    return 0.0 if np.isnan(val) else float(max(-3.0, min(3.0, val)))


def _updown_volume_ratio(close: pd.Series, volume: pd.Series, period: int = 20) -> float:
    direction = close.diff()
    up_vol = volume.where(direction > 0, 0.0).iloc[-period:].sum()
    dn_vol = volume.where(direction < 0, 0.0).iloc[-period:].sum()
    if dn_vol == 0:
        return 2.0
    return float(min(4.0, up_vol / dn_vol))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_smart_money_score(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    symbol: str = "",
    on_chain: Optional[OnChainDataProvider] = None,
    ob_signals=None,   # Optional[OrderBookSignals] — avoid circular import with string hint
) -> dict:
    """
    Compute the smart money score.

    When ob_signals is provided, order-book metrics replace a portion of
    the OHLCV proxies with real market-microstructure data (more accurate).

    Returns:
        score     : 0-100 float
        breakdown : per-metric contributions
    """
    provider = on_chain or NullOnChainProvider()
    has_real_onchain = not isinstance(provider, NullOnChainProvider)

    # --- OHLCV proxies ---
    obv_val  = _obv_trend(close, volume)
    acc_ratio = _accumulation_candle_ratio(open_, high, low, close, volume)
    cmf_val  = _chaikin_money_flow(high, low, close, volume)
    vwap_val = _vwap_position(high, low, close, volume)
    ud_ratio = _updown_volume_ratio(close, volume)

    obv_pts  = max(0.0, (obv_val + 1.0) / 2.0) * 20
    acc_pts  = acc_ratio * 20
    cmf_pts  = max(0.0, (cmf_val + 1.0) / 2.0) * 20
    vwap_pts = max(0.0, (vwap_val + 3.0) / 6.0) * 20
    ud_pts   = min(1.0, ud_ratio / 2.0) * 20

    proxy_total = obv_pts + acc_pts + cmf_pts + vwap_pts + ud_pts

    # --- Order book augmentation ---
    ob_breakdown = {}
    if ob_signals is not None:
        # OB replaces up to 30 of the 100 points, improving signal accuracy
        # Research: top-20 imbalance has ~0.58 correlation with 3-day returns
        ob_imb_pts   = max(0.0, (ob_signals.imbalance + 1.0) / 2.0) * 15   # 0-15 pts
        ob_conv_pts  = min(1.0, max(0.0, (ob_signals.ob_breakout_conviction - 1.0) / 1.5)) * 10  # 0-10 pts
        ob_wall_pen  = -5.0 if ob_signals.has_ask_wall_above else 0.0
        ob_stop_pen  = -5.0 if ob_signals.is_stop_hunt_risk else 0.0
        ob_spread_pts = max(0.0, (1.0 - ob_signals.spread_pct / 0.5)) * 5   # 0-5 pts; tight spread = good

        ob_net = ob_imb_pts + ob_conv_pts + ob_wall_pen + ob_stop_pen + ob_spread_pts

        # Replace vwap + ud_ratio with OB data (both measure directional pressure)
        score = min(100.0, obv_pts + acc_pts + cmf_pts + ob_net + ob_spread_pts)
        source = "ohlcv_proxy+orderbook"

        ob_breakdown = {
            "ob_imbalance": {"value": ob_signals.imbalance, "points": round(ob_imb_pts, 2), "max": 15},
            "ob_conviction": {"value": ob_signals.ob_breakout_conviction, "points": round(ob_conv_pts, 2), "max": 10},
            "ob_ask_wall_penalty": {"value": ob_signals.has_ask_wall_above, "points": ob_wall_pen, "max": 0},
            "ob_stop_hunt_penalty": {"value": ob_signals.is_stop_hunt_risk, "points": ob_stop_pen, "max": 0},
            "ob_spread": {"value": ob_signals.spread_pct, "points": round(ob_spread_pts, 2), "max": 5},
        }
    elif has_real_onchain:
        outflow      = provider.get_exchange_outflow_score(symbol)
        whale        = provider.get_whale_accumulation_score(symbol)
        addr_growth  = provider.get_active_address_growth_score(symbol)
        holder_growth = provider.get_holder_count_growth_score(symbol)
        score = outflow * 0.30 + whale * 0.30 + addr_growth * 0.20 + holder_growth * 0.20
        source = "on_chain"
    else:
        score = proxy_total
        source = "ohlcv_proxy"

    return {
        "score": round(float(min(100.0, max(0.0, score))), 2),
        "source": source,
        "breakdown": {
            "obv_trend":              {"value": obv_val,   "points": round(obv_pts, 2),  "max": 20},
            "accumulation_candles":   {"value": acc_ratio, "points": round(acc_pts, 2),  "max": 20},
            "chaikin_money_flow":     {"value": cmf_val,   "points": round(cmf_pts, 2),  "max": 20},
            "vwap_position":          {"value": vwap_val,  "points": round(vwap_pts, 2), "max": 20},
            "updown_volume_ratio":    {"value": ud_ratio,  "points": round(ud_pts, 2),   "max": 20},
            **ob_breakdown,
        },
    }
