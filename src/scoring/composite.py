"""
Composite scorer — combines Trend, Momentum, Liquidity, and Smart Money
into a single 0-100 score, with optional order book augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from src.config.config import SCORE_WEIGHTS
from src.scoring.trend_score import compute_trend_score
from src.scoring.momentum_score import compute_momentum_score
from src.scoring.liquidity_score import compute_liquidity_score
from src.scoring.smart_money_score import compute_smart_money_score, OnChainDataProvider


@dataclass
class ScoreResult:
    symbol: str
    trend_score: float
    momentum_score: float
    liquidity_score: float
    smart_money_score: float
    final_score: float

    wall_bonus: float = 0.0

    trend_detail:       dict = field(default_factory=dict)
    momentum_detail:    dict = field(default_factory=dict)
    liquidity_detail:   dict = field(default_factory=dict)
    smart_money_detail: dict = field(default_factory=dict)

    is_wall_signal: bool = False
    wall_event: str = "none"

    latest_price: float = 0.0
    atr:          float = 0.0

    # Order book metadata
    ob_imbalance:  float = 0.0
    ob_conviction: float = 0.0


def score_asset(
    symbol: str,
    ohlcv: pd.DataFrame,
    btc_close: Optional[pd.Series] = None,
    market_close: Optional[pd.Series] = None,
    market_cap_usd: float = 0.0,
    exchange_count: int = 1,
    on_chain: Optional[OnChainDataProvider] = None,
    ob_signals=None,          # Optional[OrderBookSignals]
    wall_bonus: float = 0.0,
    is_wall_signal: bool = False,
    wall_event: str = "none",
    atr: float = 0.0,
) -> ScoreResult:
    open_  = ohlcv["open"]
    high   = ohlcv["high"]
    low    = ohlcv["low"]
    close  = ohlcv["close"]
    volume = ohlcv["volume"]

    trend     = compute_trend_score(close)
    momentum  = compute_momentum_score(close, btc_close, market_close)
    liquidity = compute_liquidity_score(volume, market_cap_usd, exchange_count)
    smart     = compute_smart_money_score(open_, high, low, close, volume, symbol, on_chain, ob_signals)

    base = (
        trend["score"]    * SCORE_WEIGHTS["trend"]
        + momentum["score"] * SCORE_WEIGHTS["momentum"]
        + liquidity["score"] * SCORE_WEIGHTS["liquidity"]
        + smart["score"]   * SCORE_WEIGHTS["smart_money"]
    )
    final = min(100.0, base + wall_bonus)

    ob_imb  = ob_signals.imbalance        if ob_signals else 0.0
    ob_conv = ob_signals.ob_breakout_conviction if ob_signals else 0.0

    return ScoreResult(
        symbol=symbol,
        trend_score=trend["score"],
        momentum_score=momentum["score"],
        liquidity_score=liquidity["score"],
        smart_money_score=smart["score"],
        final_score=round(final, 2),
        wall_bonus=wall_bonus,
        trend_detail=trend,
        momentum_detail=momentum,
        liquidity_detail=liquidity,
        smart_money_detail=smart,
        is_wall_signal=is_wall_signal,
        wall_event=wall_event,
        latest_price=float(close.iloc[-1]),
        atr=atr,
        ob_imbalance=ob_imb,
        ob_conviction=ob_conv,
    )
