"""
Hard risk limits — the circuit breakers that stand between a bug and the
account.

Every check here is a refusal, not a warning. An automated system trading on
a 60-second loop can open more positions in an hour than a human would in a
month, so the failure mode that matters is not "one bad trade" but "the same
bad trade, three hundred times, while nobody is watching". These limits bound
the blast radius of that scenario, including the case where the signal logic
itself is wrong.

State is persisted so the breakers survive a restart. That is the whole
point: a daily-loss limit that resets when the process crashes is not a limit,
it's a suggestion that disappears exactly when things are going worst.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


@dataclass
class RiskLimits:
    """Configured ceilings. See config.py for the rationale behind defaults."""
    max_concurrent_positions: int = 3
    max_position_usd: float = 500.0
    max_daily_loss_usd: float = 100.0
    max_daily_trades: int = 60
    risk_per_trade_pct: float = 0.01


@dataclass
class RiskState:
    """Mutable daily counters, persisted across restarts."""
    date: str = field(default_factory=_today_utc)
    realised_pnl_usd: float = 0.0
    trades_today: int = 0
    halted: bool = False
    halt_reason: str = ""
    kill_switch: bool = False

    def roll_if_new_day(self) -> bool:
        """Reset daily counters at the UTC boundary. Returns True if rolled."""
        today = _today_utc()
        if self.date == today:
            return False
        self.date = today
        self.realised_pnl_usd = 0.0
        self.trades_today = 0
        # A daily-loss halt expires with the day that caused it. The kill
        # switch does not — that one is a human decision and only a human
        # should clear it.
        if self.halted and not self.kill_switch:
            self.halted = False
            self.halt_reason = ""
        return True


class RiskManager:
    """
    Gatekeeper for new positions.

    Usage:
        ok, reason = risk.can_open(symbol, notional_usd, open_count)
        if not ok:
            ...
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        state_path: Optional[str] = "data/state/risk.json",
    ) -> None:
        self.limits = limits or RiskLimits()
        self.path = Path(state_path) if state_path else None
        self.state = RiskState()
        self.load()

    # -- persistence --------------------------------------------------

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.state = RiskState(**data)
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            logger.error("Could not read risk state (%s) — starting fresh", exc)
            self.state = RiskState()
        self.state.roll_if_new_day()

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(asdict(self.state), indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            logger.error("Could not persist risk state: %s", exc)

    # -- gates --------------------------------------------------------

    def can_open(
        self,
        symbol: str,
        notional_usd: float,
        open_count: int,
        already_open_symbol: bool = False,
    ) -> Tuple[bool, str]:
        """
        Returns (allowed, reason). `reason` is empty when allowed.

        Checks are ordered cheapest-and-most-absolute first so that a halted
        system short-circuits before doing any sizing arithmetic.
        """
        self.state.roll_if_new_day()

        if self.state.kill_switch:
            return False, "kill switch engaged"

        if self.state.halted:
            return False, f"trading halted: {self.state.halt_reason}"

        if already_open_symbol:
            # One position per symbol. Without this, a signal that keeps
            # firing on consecutive cycles silently pyramids into a position
            # several times the intended size — the most common way an
            # automated entry loop blows past its own risk budget.
            return False, f"already holding {symbol}"

        if open_count >= self.limits.max_concurrent_positions:
            return False, (
                f"at max concurrent positions "
                f"({open_count}/{self.limits.max_concurrent_positions})"
            )

        if self.state.trades_today >= self.limits.max_daily_trades:
            return False, (
                f"daily trade cap reached "
                f"({self.state.trades_today}/{self.limits.max_daily_trades})"
            )

        if notional_usd > self.limits.max_position_usd:
            return False, (
                f"position ${notional_usd:,.2f} exceeds cap "
                f"${self.limits.max_position_usd:,.2f}"
            )

        if notional_usd <= 0:
            return False, "computed position size is zero"

        return True, ""

    # -- bookkeeping --------------------------------------------------

    def record_open(self) -> None:
        self.state.roll_if_new_day()
        self.state.trades_today += 1
        self.save()

    def record_close(self, pnl_usd: float) -> Optional[str]:
        """
        Book a realised result. Returns a halt reason if this trip the daily
        loss breaker, else None.
        """
        self.state.roll_if_new_day()
        self.state.realised_pnl_usd += pnl_usd

        if (
            not self.state.halted
            and self.state.realised_pnl_usd <= -abs(self.limits.max_daily_loss_usd)
        ):
            self.state.halted = True
            self.state.halt_reason = (
                f"daily loss ${self.state.realised_pnl_usd:,.2f} "
                f"hit limit ${self.limits.max_daily_loss_usd:,.2f}"
            )
            self.save()
            logger.error("RISK HALT: %s", self.state.halt_reason)
            return self.state.halt_reason

        self.save()
        return None

    # -- manual control -----------------------------------------------

    def engage_kill_switch(self, reason: str = "manual") -> None:
        self.state.kill_switch = True
        self.state.halted = True
        self.state.halt_reason = f"kill switch ({reason})"
        self.save()
        logger.error("KILL SWITCH ENGAGED: %s", reason)

    def release_kill_switch(self) -> None:
        self.state.kill_switch = False
        self.state.halted = False
        self.state.halt_reason = ""
        self.save()
        logger.warning("Kill switch released — trading re-enabled")

    # -- sizing -------------------------------------------------------

    def position_size(
        self,
        equity_usd: float,
        entry_price: float,
        stop_price: float,
        side: str = "long",
    ) -> Tuple[float, float]:
        """
        Risk-based sizing: stake the amount that loses exactly
        `risk_per_trade_pct` of equity if the stop is hit.

        Returns (quantity, notional_usd), both clamped to max_position_usd.

        Note this sizes off the *stop distance*, not off a fixed notional —
        a tight stop earns a larger position and a wide one a smaller,
        which is what makes risk-per-trade constant across setups whose
        volatility differs by an order of magnitude.

        A short's stop sits *above* its entry, so the distance is taken as an
        absolute value and the side is checked separately. A stop on the wrong
        side of the entry is rejected rather than silently absolute-valued into
        a plausible-looking size: it means the caller has the direction
        confused, and sizing that position at all is the wrong response.
        """
        if entry_price <= 0 or stop_price <= 0:
            return 0.0, 0.0

        side = side.lower()
        if side == "long" and stop_price >= entry_price:
            return 0.0, 0.0
        if side == "short" and stop_price <= entry_price:
            return 0.0, 0.0

        risk_usd = equity_usd * self.limits.risk_per_trade_pct
        risk_per_unit = abs(entry_price - stop_price)
        if risk_per_unit <= 0:
            return 0.0, 0.0

        quantity = risk_usd / risk_per_unit
        notional = quantity * entry_price

        if notional > self.limits.max_position_usd:
            notional = self.limits.max_position_usd
            quantity = notional / entry_price

        return quantity, notional

    def notional_size(
        self,
        equity_usd: float,
        entry_price: float,
        fraction_of_equity: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Fixed-notional sizing for a strategy that has no stop.

        The crowd-short signal deliberately carries neither stop nor target —
        it is a statement about return over a fixed horizon, and the parameter
        search that would have added exit levels was exhausted and found
        nothing. Stop-distance sizing is undefined for such a position, so size
        is capped by notional instead.

        Because there is no stop, `max_position_usd` is not merely a ceiling
        here — it IS the risk limit, and the worst case is bounded only by how
        far price can run inside the hold window. On a perp, add exchange-side
        liquidation distance to that reasoning before raising it.
        """
        if entry_price <= 0:
            return 0.0, 0.0

        fraction = fraction_of_equity or self.limits.risk_per_trade_pct
        notional = min(equity_usd * fraction, self.limits.max_position_usd)
        if notional <= 0:
            return 0.0, 0.0
        return notional / entry_price, notional
