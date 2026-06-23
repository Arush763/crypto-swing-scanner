"""
Tape-Based Wall Signal — historical proxy for the live order-book wall signal.

src.modules.wall_signal classifies absorption/repulsion from resting L2
depth, which is only ever observable live (no free historical archive of
full order-book depth exists). For backtesting, this module infers the same
concept from free historical trade-tick data instead: a burst of aggressor
(taker) volume clustering at a recent swing level is treated as the
"wall" being tested, and judged by what price does next:

  - ask_absorption : price was testing a swing-high level, a burst of
                      sell-side aggressor volume hit it (supply being sold
                      into), and price still pushed through with buyers
                      remaining dominant -> the supply got absorbed.
  - bid_repulsion   : price was testing a swing-low level, a burst of
                      sell-side aggressor volume hit it, and price bounced
                      away without breaking the level -> selling got
                      absorbed by resting demand and rejected.

This is a proxy, not the live signal itself — useful only for backtesting
the strategy's edge, since live trading uses the L2-based wall_signal.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config.config import (
    TAPE_LEVEL_LOOKBACK,
    TAPE_PROXIMITY_PCT,
    TAPE_VOLUME_SPIKE_MULT,
    TAPE_SIGNAL_BONUS,
)


@dataclass
class TapeSignalResult:
    is_setup: bool
    event: str          # "ask_absorption" | "bid_repulsion" | "none"
    level_price: float
    bonus_score: float


def detect_tape_signals(
    bars: pd.DataFrame,
    lookback: int = TAPE_LEVEL_LOOKBACK,
    proximity_pct: float = TAPE_PROXIMITY_PCT,
    volume_spike_mult: float = TAPE_VOLUME_SPIKE_MULT,
    bonus_pts: float = TAPE_SIGNAL_BONUS,
) -> pd.DataFrame:
    """
    Vectorised over the whole bar series (fast enough for backtesting many
    assets/days). `bars` must have columns: open, high, low, close, volume,
    buy_volume, sell_volume (see src.data.trade_tape.resample_to_bars).

    Returns a DataFrame aligned to `bars.index` with columns:
    is_setup, event, level_price, bonus_score.
    """
    close = bars["close"]
    high  = bars["high"]
    low   = bars["low"]
    volume     = bars["volume"]
    buy_volume = bars["buy_volume"]
    sell_volume = bars["sell_volume"]

    # Level known *before* the current bar — excludes it to avoid lookahead
    resistance = high.rolling(lookback).max().shift(1)
    support    = low.rolling(lookback).min().shift(1)
    avg_volume = volume.rolling(lookback).mean().shift(1)

    prev_close = close.shift(1)
    prev_sell  = sell_volume.shift(1)

    sell_spike = prev_sell >= (avg_volume * volume_spike_mult / 2.0)

    # --- Ask absorption: tested resistance from below, heavy sell-side
    # tape hit it, price still closed through with buyers still dominant ---
    dist_to_res = (resistance - prev_close) / resistance
    tested_res  = (dist_to_res >= 0) & (dist_to_res <= proximity_pct)
    pushed_through = close >= resistance * 0.999
    buy_dominant = buy_volume > sell_volume
    ask_absorption = tested_res & sell_spike & pushed_through & buy_dominant & resistance.notna()

    # --- Bid repulsion: tested support from above, heavy sell-side tape
    # hit it, price bounced away without breaking the level ---
    dist_to_sup  = (prev_close - support) / support
    tested_sup   = (dist_to_sup >= 0) & (dist_to_sup <= proximity_pct)
    bounced      = close > prev_close
    held_support = close > support
    bid_repulsion = tested_sup & sell_spike & bounced & held_support & support.notna() & ~ask_absorption

    sell_volume_safe = sell_volume.where(sell_volume > 0, 1.0)
    abs_strength = ((buy_volume - sell_volume) / sell_volume_safe).clip(lower=0, upper=1).fillna(0)
    rep_strength = (((close - prev_close) / prev_close) / proximity_pct).clip(lower=0, upper=1).fillna(0)

    bonus_absorption = bonus_pts * (0.6 + 0.4 * abs_strength)
    bonus_repulsion   = bonus_pts * (0.6 + 0.4 * rep_strength)

    is_setup = ask_absorption | bid_repulsion
    event = pd.Series("none", index=bars.index)
    event[ask_absorption] = "ask_absorption"
    event[bid_repulsion] = "bid_repulsion"

    level_price = pd.Series(0.0, index=bars.index)
    level_price[ask_absorption] = resistance[ask_absorption]
    level_price[bid_repulsion] = support[bid_repulsion]

    bonus_score = pd.Series(0.0, index=bars.index)
    bonus_score[ask_absorption] = bonus_absorption[ask_absorption]
    bonus_score[bid_repulsion] = bonus_repulsion[bid_repulsion]

    return pd.DataFrame({
        "is_setup": is_setup.fillna(False),
        "event": event,
        "level_price": level_price.round(8),
        "bonus_score": bonus_score.round(2),
    })
