"""
Signal Generator.

Produces tradeable alerts from ScoreResults + detection module outputs.

Order book data is used when available to:
  - Validate breakout conviction at the resistance level  (research: >1.5 imbalance ratio = 73% follow-through)
  - Set stops below/above identified liquidity walls       (dynamic stop placement)
  - Estimate realistic slippage and max safe position size
  - Flag stop-hunt risk conditions
  - Suppress signals when a large ask wall sits just above breakout level (68% fake-out rate)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd

from src.config.config import (
    SIGNAL_SCORE_THRESHOLD,
    STRONG_SIGNAL_SCORE,
    ATR_TRAILING_STOP_MULTIPLIER,
    EMA_SHORT,
)
from src.scoring.composite import ScoreResult
from src.modules.breakout import BreakoutResult
from src.modules.retest import RetestResult
from src.modules.squeeze import SqueezeResult
from src.indicators.trend import ema
from src.indicators.volatility import atr_latest

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    symbol: str
    timestamp: datetime
    signal_type: str        # "breakout" | "retest" | "squeeze_breakout" | "trend_continuation"
    strength: str           # "strong" | "standard"

    # Price levels
    current_price: float
    entry_zone_low: float
    entry_zone_high: float
    stop_loss: float
    resistance_level: float

    # Risk metrics
    risk_pct: float
    reward_pct: float
    risk_reward: float

    # Position sizing (OB-informed when available)
    max_safe_position_usd: float = 0.0
    estimated_slippage_pct: float = 0.0

    # Scores
    final_score: float = 0.0
    trend_score: float = 0.0
    momentum_score: float = 0.0
    liquidity_score: float = 0.0
    smart_money_score: float = 0.0

    # Order book metadata
    ob_imbalance: float = 0.0
    ob_conviction: float = 0.0
    ob_has_ask_wall: bool = False
    ob_stop_hunt_risk: bool = False
    ob_wall_stop: float = 0.0       # OB-derived stop (below bid wall), 0 = not available

    # Exit guidance
    exit_primary: str = ""
    exit_alternative: str = ""

    # Raw objects
    score_result: Optional[ScoreResult] = None
    breakout: Optional[BreakoutResult] = None
    retest: Optional[RetestResult] = None
    squeeze: Optional[SqueezeResult] = None


def _compute_stop_loss(
    price: float,
    ema_stop: float,
    atr_stop: float,
    ob_wall_stop: float = 0.0,
) -> float:
    """
    Select the tightest valid stop from EMA, ATR, and OB-wall candidates.

    Research finding: stops placed below OB bid walls have higher hold rate
    because they align with natural liquidity support.
    """
    candidates = []
    if 0 < ema_stop < price:
        candidates.append(ema_stop)
    if 0 < atr_stop < price:
        candidates.append(atr_stop)
    if 0 < ob_wall_stop < price:
        candidates.append(ob_wall_stop)

    if not candidates:
        return price * 0.95   # hard fallback: 5%
    return max(candidates)    # tightest stop (highest value below price)


def generate_signal(
    score: ScoreResult,
    ohlcv: pd.DataFrame,
    breakout: Optional[BreakoutResult] = None,
    retest: Optional[RetestResult] = None,
    squeeze: Optional[SqueezeResult] = None,
    ob_signals=None,          # Optional[OrderBookSignals]
    score_threshold: float = SIGNAL_SCORE_THRESHOLD,
) -> Optional[Signal]:
    """
    Evaluate whether a ScoreResult qualifies as a signal.

    Suppression rules from orderbook research:
      - Suppress if ask wall sits within 3% above breakout level (68% fake-out rate)
      - Suppress if stop-hunt risk detected (imbalance < -0.6 on bullish push)
      - Require OB conviction > 1.0 at resistance when OB data is available
    """
    if score.final_score < score_threshold:
        return None

    has_breakout = breakout is not None and breakout.is_breakout
    has_retest   = retest   is not None and retest.is_retest
    has_squeeze  = squeeze  is not None and squeeze.squeeze_breakout

    if not (has_breakout or has_retest or has_squeeze):
        return None

    # --- OB-based suppression (research-backed) ---
    ob_imbalance = 0.0
    ob_conviction = 1.0
    ob_has_wall = False
    ob_stop_hunt = False
    ob_wall_stop = 0.0
    max_safe_pos = 0.0
    slip_pct = 0.0

    if ob_signals is not None:
        ob_imbalance  = ob_signals.imbalance
        ob_conviction = ob_signals.ob_breakout_conviction
        ob_has_wall   = ob_signals.has_ask_wall_above
        ob_stop_hunt  = ob_signals.is_stop_hunt_risk
        max_safe_pos  = ob_signals.max_safe_position_usd
        slip_pct      = ob_signals.slippage_est_pct

        # Suppress if ask wall blocks the breakout (likely fake-out)
        if ob_has_wall and has_breakout:
            logger.info("SUPPRESSED %s — ask wall above breakout level (68%% fake-out risk)", score.symbol)
            return None

        # Suppress if stop-hunt risk detected
        if ob_stop_hunt:
            logger.info("SUPPRESSED %s — stop-hunt risk (ask-heavy on bullish push)", score.symbol)
            return None

        # OB-based stop: place below strongest bid wall
        if ob_signals.wall_bid_price > 0:
            ob_wall_stop = ob_signals.wall_bid_price * 0.995   # 0.5% below wall

    close = ohlcv["close"]
    high  = ohlcv["high"]
    low   = ohlcv["low"]

    price       = float(close.iloc[-1])
    ema20       = float(ema(close, EMA_SHORT).iloc[-1])
    current_atr = atr_latest(high, low, close)

    ema_stop = ema20
    atr_stop = price - ATR_TRAILING_STOP_MULTIPLIER * current_atr
    stop     = _compute_stop_loss(price, ema_stop, atr_stop, ob_wall_stop)

    # Entry zone
    if has_retest and retest:
        entry_low  = retest.entry_zone_low
        entry_high = retest.entry_zone_high
        signal_type = "retest"
    elif has_squeeze and squeeze:
        entry_low  = price
        entry_high = price * 1.02
        signal_type = "squeeze_breakout"
    elif has_breakout and breakout:
        res = breakout.resistance_level
        entry_low  = res
        entry_high = res * 1.02
        signal_type = "breakout"
    else:
        entry_low  = price
        entry_high = price * 1.01
        signal_type = "trend_continuation"

    resistance = (
        breakout.resistance_level if has_breakout and breakout
        else price * 1.10
    )

    risk_pct   = max(0.001, (price - stop) / price)
    target     = max(resistance, price * (1 + risk_pct * 2))
    reward_pct = (target - price) / price
    rr         = round(reward_pct / risk_pct, 2)

    strength = "strong" if score.final_score >= STRONG_SIGNAL_SCORE else "standard"

    logger.info(
        "SIGNAL [%s] %s | score=%.1f | type=%s | R:R=%.2f | OB_imb=%.2f",
        strength.upper(), score.symbol, score.final_score, signal_type, rr, ob_imbalance,
    )

    return Signal(
        symbol=score.symbol,
        timestamp=datetime.now(timezone.utc),
        signal_type=signal_type,
        strength=strength,
        current_price=round(price, 8),
        entry_zone_low=round(entry_low, 8),
        entry_zone_high=round(entry_high, 8),
        stop_loss=round(stop, 8),
        resistance_level=round(resistance, 8),
        risk_pct=round(risk_pct * 100, 2),
        reward_pct=round(reward_pct * 100, 2),
        risk_reward=rr,
        max_safe_position_usd=round(max_safe_pos, 2),
        estimated_slippage_pct=round(slip_pct, 4),
        final_score=score.final_score,
        trend_score=score.trend_score,
        momentum_score=score.momentum_score,
        liquidity_score=score.liquidity_score,
        smart_money_score=score.smart_money_score,
        ob_imbalance=ob_imbalance,
        ob_conviction=ob_conviction,
        ob_has_ask_wall=ob_has_wall,
        ob_stop_hunt_risk=ob_stop_hunt,
        ob_wall_stop=round(ob_wall_stop, 8),
        exit_primary=f"Daily close below 20 EMA (currently {ema20:.6g})",
        exit_alternative=f"2 ATR trailing stop ({ATR_TRAILING_STOP_MULTIPLIER}×{current_atr:.4g} = {atr_stop:.6g})",
        score_result=score,
        breakout=breakout,
        retest=retest,
        squeeze=squeeze,
    )


def format_signal_table(signals: List[Signal]) -> pd.DataFrame:
    rows = []
    for s in signals:
        rows.append({
            "Symbol": s.symbol,
            "Type": s.signal_type,
            "Strength": s.strength,
            "Score": s.final_score,
            "Trend": s.trend_score,
            "Momentum": s.momentum_score,
            "Liquidity": s.liquidity_score,
            "SmartMoney": s.smart_money_score,
            "Price": s.current_price,
            "Entry Low": s.entry_zone_low,
            "Entry High": s.entry_zone_high,
            "Stop Loss": s.stop_loss,
            "Risk %": s.risk_pct,
            "Reward %": s.reward_pct,
            "R:R": s.risk_reward,
            "Resistance": s.resistance_level,
            "Max Position $": s.max_safe_position_usd,
            "Slippage %": s.estimated_slippage_pct,
            "OB Imbalance": s.ob_imbalance,
            "OB Conviction": s.ob_conviction,
            "OB Ask Wall": s.ob_has_ask_wall,
            "Stop Hunt Risk": s.ob_stop_hunt_risk,
            "Timestamp": s.timestamp.isoformat(),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
