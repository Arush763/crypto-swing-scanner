"""
Order-Book Wall Signal — the sole signal source.

Detects large resting orders ("walls") in the order book and classifies,
across consecutive scan cycles, whether price is absorbing a wall (eating
through it -> continuation) or being repelled by it (bouncing off ->
rejection). A wall is treated as the same instance across cycles if its
price stays within WALL_SAME_LEVEL_TOLERANCE_PCT of where it was last seen.

This bot only trades long, so only the bullish cases generate a signal:
  - ask_absorption : price pushed through a resting ask wall while its size
                      collapsed -> sellers got run over, expect continuation up.
  - bid_repulsion   : price approached a resting bid wall, failed to break it,
                      and bounced away while the wall held its size -> buyers
                      defended support, expect a bounce.

The mirror cases (bid wall eaten through -> breakdown; ask wall holding and
rejecting price) are bearish and intentionally do not produce a signal.

Requires live order-book data and at least one prior scan cycle's snapshot
per symbol; without either, no setup is reported.

Each classification is also corroborated against executed trade flow since
the prior cycle (src.data.orderbook.FlowSignals) where available — a wall
shrinking in the book is ambiguous (traded through vs. cancelled/spoofed),
and only executed aggressor volume can tell the two apart. This mirrors the
tape-based backtest proxy in src/modules/tape_signal.py. Flow data is
optional (None if the trade fetch failed or this is the first cycle for a
symbol) so a missing flow signal degrades to OB-only classification rather
than blocking signals outright.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Dict, Optional, Union

logger = logging.getLogger(__name__)

from src.config.config import (
    WALL_SAME_LEVEL_TOLERANCE_PCT,
    WALL_SHRINK_THRESHOLD,
    WALL_SIGNAL_BONUS,
)
from src.data.orderbook import FlowSignals, OrderBookSignals, WALL_SCAN_PCT


@dataclass
class _WallSnapshot:
    price: float
    wall_bid_price: float
    wall_bid_size: float
    wall_ask_price: float
    wall_ask_size: float


@dataclass
class WallSignalResult:
    is_setup: bool
    event: str                  # "ask_absorption" | "bid_repulsion" | "none"
    wall_side: str               # "ask" | "bid" | ""
    wall_price: float
    wall_size_usd: float
    distance_pct: float          # How close price was to the wall when it was tested
    bonus_score: float           # Points added to composite score


_NO_SETUP = WallSignalResult(
    is_setup=False, event="none", wall_side="", wall_price=0.0,
    wall_size_usd=0.0, distance_pct=0.0, bonus_score=0.0,
)


def _same_level(a: float, b: float, tolerance_pct: float = WALL_SAME_LEVEL_TOLERANCE_PCT) -> bool:
    if a <= 0 or b <= 0:
        return False
    return abs(a - b) / b <= tolerance_pct


def _bonus(strength: float) -> float:
    return round(WALL_SIGNAL_BONUS * (0.6 + 0.4 * min(1.0, max(0.0, strength))), 2)


def _classify(
    prev: _WallSnapshot, curr: _WallSnapshot, flow: Optional[FlowSignals] = None
) -> WallSignalResult:
    # --- Ask wall (resistance) absorption: price was testing it, now past it,
    # and the wall that was defending it has shrunk or vanished ---
    if prev.wall_ask_price > 0:
        distance_prev = (prev.wall_ask_price - prev.price) / prev.wall_ask_price
        tested_it = 0 <= distance_prev <= WALL_SCAN_PCT
        pushed_through = curr.price >= prev.wall_ask_price * (1 - WALL_SAME_LEVEL_TOLERANCE_PCT)

        if tested_it and pushed_through and (flow is None or flow.buy_dominant):
            wall_still_there = _same_level(curr.wall_ask_price, prev.wall_ask_price)
            if wall_still_there:
                shrink_ratio = (prev.wall_ask_size - curr.wall_ask_size) / prev.wall_ask_size if prev.wall_ask_size > 0 else 0.0
            else:
                shrink_ratio = 1.0   # wall disappeared entirely

            if shrink_ratio >= WALL_SHRINK_THRESHOLD:
                return WallSignalResult(
                    is_setup=True,
                    event="ask_absorption",
                    wall_side="ask",
                    wall_price=prev.wall_ask_price,
                    wall_size_usd=prev.wall_ask_size,
                    distance_pct=round(distance_prev, 4),
                    bonus_score=_bonus(shrink_ratio),
                )

    # --- Bid wall (support) repulsion: price was testing it, then moved back
    # away, while the wall held its size ---
    if prev.wall_bid_price > 0:
        distance_prev = (prev.price - prev.wall_bid_price) / prev.wall_bid_price
        tested_it = 0 <= distance_prev <= WALL_SCAN_PCT
        bounced_away = curr.price > prev.price

        if tested_it and bounced_away and (flow is None or flow.sell_dominant):
            wall_held = _same_level(curr.wall_bid_price, prev.wall_bid_price) and (
                prev.wall_bid_size <= 0 or curr.wall_bid_size >= prev.wall_bid_size * (1 - WALL_SHRINK_THRESHOLD)
            )
            if wall_held:
                bounce_strength = ((curr.price - prev.price) / prev.price) / WALL_SCAN_PCT if WALL_SCAN_PCT > 0 else 0.0
                return WallSignalResult(
                    is_setup=True,
                    event="bid_repulsion",
                    wall_side="bid",
                    wall_price=prev.wall_bid_price,
                    wall_size_usd=curr.wall_bid_size,
                    distance_pct=round(distance_prev, 4),
                    bonus_score=_bonus(bounce_strength),
                )

    return _NO_SETUP


class WallTracker:
    """
    Holds the last scan cycle's order-book wall state per symbol so
    absorption/repulsion can be judged from how a wall behaved between
    consecutive cycles.

    If `state_path` is given, state is loaded from disk on construction and
    can be written back out with `.save()`. This matters because live_scan.py
    runs as a fresh process on each scheduled invocation (no in-memory state
    survives between cron ticks) — without disk persistence the tracker
    would never see a "previous cycle" and no wall signal could ever fire.
    """

    def __init__(self, history: int = 2, state_path: Union[str, Path, None] = None) -> None:
        self._history: Dict[str, Deque[_WallSnapshot]] = {}
        self._maxlen = history
        self.state_path = Path(state_path) if state_path else None
        if self.state_path:
            self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text())
            for symbol, snapshots in raw.items():
                buf: Deque[_WallSnapshot] = deque(maxlen=self._maxlen)
                for s in snapshots:
                    buf.append(_WallSnapshot(**s))
                self._history[symbol] = buf
        except Exception as exc:
            logger.warning("Could not load wall-tracker state from %s: %s", self.state_path, exc)

    def save(self) -> None:
        if not self.state_path:
            return
        raw = {symbol: [asdict(snap) for snap in buf] for symbol, buf in self._history.items()}
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(raw))

    def update(
        self,
        symbol: str,
        price: float,
        ob_signals: Optional[OrderBookSignals],
        flow: Optional[FlowSignals] = None,
    ) -> WallSignalResult:
        if ob_signals is None or price <= 0:
            return _NO_SETUP

        current = _WallSnapshot(
            price=price,
            wall_bid_price=ob_signals.wall_bid_price,
            wall_bid_size=ob_signals.wall_bid_size_usd,
            wall_ask_price=ob_signals.wall_ask_price,
            wall_ask_size=ob_signals.wall_ask_size_usd,
        )

        buf = self._history.setdefault(symbol, deque(maxlen=self._maxlen))
        if not buf:
            buf.append(current)
            return _NO_SETUP

        prev = buf[-1]
        result = _classify(prev, current, flow)
        buf.append(current)
        return result
