"""
Signal Generator.

Produces tradeable alerts purely from order-book wall behaviour — the only
signal source. Order book data is used to:
  - Detect whether a large resting order got absorbed (continuation) or
    repelled price (bounce) — see src.modules.wall_signal
  - Set stops below/above identified liquidity walls (dynamic stop placement)
  - Estimate realistic slippage and max safe position size
  - Flag stop-hunt risk conditions and suppress the signal
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
    ENTRY_POST_ONLY,
    EXECUTION_VENUE,
)
from src.execution.costs import cost_model_for
from src.scoring.composite import ScoreResult
from src.modules.wall_signal import WallSignalResult
from src.indicators.trend import ema
from src.indicators.volatility import atr_latest

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    symbol: str
    timestamp: datetime
    signal_type: str        # "ob_wall_ask_absorption" | "ob_wall_bid_repulsion"
    strength: str           # "strong" | "standard"

    # Price levels
    current_price: float
    entry_zone_low: float
    entry_zone_high: float
    stop_loss: float
    resistance_level: float

    # Profit targets. TP1 is a scale-out at 1R (risk-multiple parity with the
    # stop); TP2 is the structural target — the nearest real resistance, or
    # 2R if the book shows nothing overhead. These used to be computed and
    # discarded, which is why alerts carried a stop but no target.
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0

    # Risk metrics
    risk_pct: float = 0.0
    reward_pct: float = 0.0
    risk_reward: float = 0.0

    # Cost awareness — a target that doesn't clear the round-trip cost is not
    # a target, it's a donation. `net_reward_pct` is reward after costs.
    est_cost_pct: float = 0.0
    net_reward_pct: float = 0.0
    clears_costs: bool = True

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
    wall_signal: Optional[WallSignalResult] = None


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
    wall_signal: Optional[WallSignalResult] = None,
    ob_signals=None,          # Optional[OrderBookSignals]
    score_threshold: float = SIGNAL_SCORE_THRESHOLD,
) -> Optional[Signal]:
    """
    Evaluate whether a ScoreResult qualifies as a signal.

    wall_signal is the only entry path: it fires when a large resting order
    in the book gets absorbed (price eats through it) or repels price
    (bounces off it), both bullish cases (see src.modules.wall_signal).

    Suppression: stop-hunt risk detected (imbalance < -0.6 on bullish push).
    """
    if score.final_score < score_threshold:
        return None

    if wall_signal is None or not wall_signal.is_setup:
        return None

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

    # Entry zone — buy now, up to a small extension above price
    entry_low   = price
    entry_high  = price * (1.02 if wall_signal.event == "ask_absorption" else 1.01)
    signal_type = f"ob_wall_{wall_signal.event}"

    resistance = (
        ob_signals.wall_ask_price
        if (ob_signals is not None and ob_signals.wall_ask_price > 0)
        else price * 1.10
    )

    risk_pct   = max(0.001, (price - stop) / price)

    # TP1: 1R — scale out where reward first equals the risk taken.
    # TP2: the structural target, i.e. the nearest genuine resistance in the
    #      book, floored at 2R so a wall sitting almost on top of price
    #      doesn't produce a target that isn't worth trading toward.
    tp1    = price * (1 + risk_pct)
    tp2    = max(resistance, price * (1 + risk_pct * 2))
    reward_pct = (tp2 - price) / price
    rr         = round(reward_pct / risk_pct, 2)

    # Charge the modelled round trip against the target. A signal whose full
    # target barely clears its own execution cost is not tradeable, however
    # good the setup looks — this is the check that was missing entirely.
    cost_model = cost_model_for(
        score.symbol,
        venue=EXECUTION_VENUE,
        entry_is_maker=ENTRY_POST_ONLY,
        exit_is_maker=False,     # exits are taker: a stop must actually fill
    )
    est_cost_pct = cost_model.round_trip_pct(
        entry_half_spread_pct=(slip_pct / 2) if slip_pct > 0 else None,
    )
    net_reward_pct = reward_pct * 100 - est_cost_pct
    clears_costs = net_reward_pct > est_cost_pct   # target must beat cost by 2x

    if not clears_costs:
        logger.info(
            "SUPPRESSED %s — target %.2f%% doesn't clear round-trip cost %.2f%% with margin",
            score.symbol, reward_pct * 100, est_cost_pct,
        )
        return None

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
        take_profit_1=round(tp1, 8),
        take_profit_2=round(tp2, 8),
        risk_pct=round(risk_pct * 100, 2),
        reward_pct=round(reward_pct * 100, 2),
        risk_reward=rr,
        est_cost_pct=round(est_cost_pct, 4),
        net_reward_pct=round(net_reward_pct, 2),
        clears_costs=clears_costs,
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
        wall_signal=wall_signal,
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
            "TP1": s.take_profit_1,
            "TP2": s.take_profit_2,
            "Stop Loss": s.stop_loss,
            "Risk %": s.risk_pct,
            "Reward %": s.reward_pct,
            "R:R": s.risk_reward,
            "Cost %": s.est_cost_pct,
            "Net Reward %": s.net_reward_pct,
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
