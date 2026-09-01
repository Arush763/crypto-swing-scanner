"""
Open-position tracking and lifecycle state.

The scanner has always been fire-and-forget: it announced an entry and never
looked again, so nothing in the system could report whether a target was hit,
a stop was taken out, or the trade is still running. That makes the alert
feed unfalsifiable — there is no record against which the strategy's claimed
edge can be checked.

This module holds the state that closes that loop. A `Position` is the live
counterpart of `backtesting.engine.Trade`: same economics, but marked to
market on every scan cycle and persisted to disk so a restart doesn't
silently abandon open risk.

Costs are tracked per leg rather than as a single round-trip figure, because
a position that scales out at TP1 pays three legs, not two, and a P&L number
that assumes two is wrong in the direction that flatters the strategy.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """One open (or recently closed) trade."""

    symbol: str
    side: str                       # "long" | "short"
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float

    opened_at: str = ""             # ISO8601 UTC
    closed_at: str = ""

    # Deadline for a time-based exit, ISO8601 UTC. The crowd-short signal is a
    # statement about forward return over a fixed 16-hour horizon, with no stop
    # and no target — holding past the horizon is not "letting a winner run",
    # it is trading a position the study never measured. Empty means the
    # position exits on price alone, as the wall signals do.
    expires_at: str = ""

    # Base units represented by ONE unit of `quantity`. 1.0 for spot and for
    # perps quoted in base terms; 0.01 for a nano BTC futures contract, where
    # `quantity` counts contracts rather than coins.
    #
    # This is load-bearing. Without it a position of 1 nano BTC contract books
    # its notional as one whole BTC — a hundredfold overstatement of exposure,
    # P&L and fees, in a direction that makes a losing strategy look like a
    # spectacular one. Every economic quantity below multiplies by it.
    contract_size: float = 1.0

    # Flat commission per contract per side, for venues that charge that way
    # instead of a percentage of notional. Zero means the percentage cost model
    # applies. See src/execution/contracts.py.
    fee_per_contract_usd: float = 0.0

    signal_type: str = ""
    is_paper: bool = True

    # Scale-out state. TP1 closes `tp1_scale_out_fraction` of the position and
    # moves the stop to breakeven; the remainder runs to TP2 or the stop.
    tp1_scale_out_fraction: float = 0.5
    tp1_hit: bool = False
    remaining_quantity: float = 0.0

    # Realised economics, accumulated across legs.
    realised_pnl_usd: float = 0.0
    fees_paid_usd: float = 0.0
    entry_cost_pct: float = 0.0

    status: str = "open"            # "open" | "closed"
    close_reason: str = ""
    exit_price: float = 0.0

    # Exchange linkage (live mode only)
    entry_order_id: str = ""
    stop_order_id: str = ""

    def __post_init__(self) -> None:
        if not self.opened_at:
            self.opened_at = datetime.now(timezone.utc).isoformat()
        if self.remaining_quantity == 0.0:
            self.remaining_quantity = self.quantity

    # -- derived ------------------------------------------------------

    @property
    def direction(self) -> int:
        """
        +1 for a long, -1 for a short.

        Every P&L expression below multiplies by this rather than branching, so
        there is no path where one of them gets the sign right and another
        doesn't — the class of bug that makes a losing short look like a
        winning one in the log while the account disagrees.
        """
        return -1 if self.side == "short" else 1

    @property
    def is_short(self) -> bool:
        return self.side == "short"

    @property
    def base_quantity(self) -> float:
        """Position size in base units, whatever `quantity` counts."""
        return self.quantity * self.contract_size

    @property
    def remaining_base_quantity(self) -> float:
        return self.remaining_quantity * self.contract_size

    @property
    def notional_usd(self) -> float:
        return self.entry_price * self.base_quantity

    @property
    def remaining_notional_usd(self) -> float:
        return self.entry_price * self.remaining_base_quantity

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    def unrealised_pnl_pct(self, mark_price: float) -> float:
        """Gross percentage move on the still-open portion, in the trade's favour."""
        if self.entry_price <= 0:
            return 0.0
        return (mark_price - self.entry_price) / self.entry_price * 100 * self.direction

    def unrealised_pnl_usd(self, mark_price: float) -> float:
        return ((mark_price - self.entry_price)
                * self.remaining_base_quantity * self.direction)

    def leg_fee_usd(self, price: float, quantity: float, fee_pct: float) -> float:
        """
        Commission for one leg of `quantity` units at `price`.

        A per-contract venue charges a flat amount that does not scale with
        price, so applying a percentage there would drift as the underlying
        moves. Preferring the flat charge when one is set keeps the two fee
        conventions from being silently mixed.
        """
        if self.fee_per_contract_usd > 0:
            return abs(quantity) * self.fee_per_contract_usd
        return abs(price * quantity * self.contract_size) * fee_pct / 100.0

    # -- time-based exit ----------------------------------------------

    def set_hold_hours(self, hours: float) -> None:
        """Stamp a deadline `hours` from the open time."""
        try:
            opened = datetime.fromisoformat(self.opened_at)
        except (ValueError, TypeError):
            opened = datetime.now(timezone.utc)
        self.expires_at = (opened + timedelta(hours=hours)).isoformat()

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        """Whether a time-based exit is due. False when no deadline is set."""
        if not self.expires_at:
            return False
        try:
            deadline = datetime.fromisoformat(self.expires_at)
        except (ValueError, TypeError):
            logger.warning("Unparseable expires_at on %s: %r", self.symbol, self.expires_at)
            return False
        return (now or datetime.now(timezone.utc)) >= deadline

    def holding_description(self) -> str:
        try:
            opened = datetime.fromisoformat(self.opened_at)
        except (ValueError, TypeError):
            return "unknown"
        end = datetime.now(timezone.utc)
        if self.closed_at:
            try:
                end = datetime.fromisoformat(self.closed_at)
            except (ValueError, TypeError):
                pass
        seconds = max(0, int((end - opened).total_seconds()))
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m"

    # -- transitions --------------------------------------------------

    def record_partial_exit(self, price: float, quantity: float, fee_usd: float) -> float:
        """
        Book a partial close. Returns the net USD P&L of this leg.

        The fee is deducted here rather than at final close so that a
        position's realised P&L is correct at every intermediate point, not
        only once it is fully flat.
        """
        gross = (price - self.entry_price) * quantity * self.contract_size * self.direction
        net = gross - fee_usd
        self.realised_pnl_usd += net
        self.fees_paid_usd += fee_usd
        self.remaining_quantity = max(0.0, self.remaining_quantity - quantity)
        return net

    def close(self, price: float, reason: str, fee_usd: float = 0.0) -> float:
        """Close whatever remains. Returns the net USD P&L of the final leg."""
        net = self.record_partial_exit(price, self.remaining_quantity, fee_usd)
        self.status = "closed"
        self.close_reason = reason
        self.exit_price = price
        self.closed_at = datetime.now(timezone.utc).isoformat()
        return net

    def total_pnl_pct(self) -> float:
        """Realised P&L as a percentage of the original notional, net of fees."""
        notional = self.notional_usd
        if notional <= 0:
            return 0.0
        return self.realised_pnl_usd / notional * 100


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class PositionStore:
    """
    Thread-safe, disk-backed collection of positions.

    Persistence matters more here than it does for the wall tracker: losing
    the wall tracker's state costs one cycle of signal quality, whereas
    losing this means the bot forgets it has money at risk and will happily
    open a duplicate position on the same symbol.

    Writes are atomic (temp file + replace) so a crash mid-write can't leave
    a truncated JSON file that would fail to load on restart — which, given
    what this file represents, would be the worst possible failure mode.
    """

    def __init__(self, path: Optional[str] = "data/state/positions.json") -> None:
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        self.open_positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.load()

    # -- persistence --------------------------------------------------

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Could not read position store at %s: %s", self.path, exc)
            return

        with self._lock:
            self.open_positions = {
                sym: Position(**data) for sym, data in raw.get("open", {}).items()
            }
            # Only the recent tail is kept in memory; the full history lives
            # in the trade log (see log_closed_trade).
            self.closed_positions = [Position(**d) for d in raw.get("closed", [])[-200:]]

        if self.open_positions:
            logger.info(
                "Restored %d open position(s) from disk: %s",
                len(self.open_positions), ", ".join(self.open_positions),
            )

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                "open": {sym: asdict(p) for sym, p in self.open_positions.items()},
                "closed": [asdict(p) for p in self.closed_positions[-200:]],
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
        tmp = self.path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            logger.error("Could not persist position store: %s", exc)

    # -- accessors ----------------------------------------------------

    def add(self, position: Position) -> None:
        with self._lock:
            self.open_positions[position.symbol] = position
        self.save()

    def get(self, symbol: str) -> Optional[Position]:
        with self._lock:
            return self.open_positions.get(symbol)

    def has_open(self, symbol: str) -> bool:
        with self._lock:
            return symbol in self.open_positions

    def all_open(self) -> List[Position]:
        with self._lock:
            return list(self.open_positions.values())

    @property
    def open_count(self) -> int:
        with self._lock:
            return len(self.open_positions)

    def retire(self, position: Position) -> None:
        """Move a closed position out of the open set."""
        with self._lock:
            self.open_positions.pop(position.symbol, None)
            self.closed_positions.append(position)
        self.save()

    # -- reporting ----------------------------------------------------

    def closed_since(self, since_iso: str) -> List[Position]:
        with self._lock:
            return [
                p for p in self.closed_positions
                if p.closed_at and p.closed_at >= since_iso
            ]

    def daily_stats(self, day: Optional[str] = None) -> dict:
        """Aggregate realised performance for one UTC day (default: today)."""
        day = day or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        todays = [p for p in self.closed_positions if p.closed_at.startswith(day)]
        wins = [p for p in todays if p.realised_pnl_usd > 0]
        return {
            "date": day,
            "trades": len(todays),
            "win_rate_pct": (len(wins) / len(todays) * 100) if todays else 0.0,
            "realised_pnl_usd": sum(p.realised_pnl_usd for p in todays),
            "fees_usd": sum(p.fees_paid_usd for p in todays),
            "open_positions": self.open_count,
        }


# ---------------------------------------------------------------------------
# Trade log
# ---------------------------------------------------------------------------

TRADE_LOG_PATH = Path("logs/trades.jsonl")


def log_closed_trade(position: Position, path: Path = TRADE_LOG_PATH) -> None:
    """
    Append a closed position to an immutable JSONL trade log.

    Separate from the position store, which only keeps a recent tail: this is
    the permanent record used to check realised performance against what the
    backtest predicted. Without it there is no way to tell whether the
    modelled costs in src/execution/costs.py match reality — and that
    comparison is the only real validation the cost model can get.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(position)) + "\n")
    except OSError as exc:
        logger.error("Could not append to trade log: %s", exc)
