"""
Contract mechanics for Coinbase Derivatives Exchange (CDE) futures.

Everything in this repo before now assumed a spot-like instrument: you can buy
any fractional quantity, you pay a percentage fee, and the market is always
open. A CFTC-regulated futures contract breaks all three assumptions, and each
break is the kind that produces a wrong number rather than an exception.

  1. Size is an INTEGER number of contracts. `base_increment` is 1 and
     `base_min_size` is 1, so the smallest BTC position is one nano contract
     (0.01 BTC, ~$770). A risk model that computes "$500 of BTC" and rounds
     down gets zero contracts and silently never trades; one that rounds up
     takes 54% more risk than it was authorised to.

  2. Fees are charged PER CONTRACT, not as a percentage of notional. This
     inverts the usual intuition: a flat fee is proportionally *cheaper* on a
     large contract and brutally expensive on a small one. At $1.00/side, BTC's
     $770 contract costs 0.26% round trip while AVAX's $72 contract costs 2.8%
     — against a measured edge of +0.655%, the first is fine and the second is
     fatal. Percentage-of-notional thinking gets this exactly backwards.

  3. The market has SESSIONS. FCM products close daily (21:00 UTC at the time
     of writing) and for maintenance. An order placed into a closed session is
     rejected, and — worse for a fixed-hold strategy — a position whose exit
     deadline lands inside a closed session cannot be flattened on time.

The per-contract fee is the number that decides viability, and it is the one
number here that is NOT known from the public API. `DEFAULT_PER_CONTRACT_FEE_USD`
is a deliberately pessimistic placeholder; `scripts/coinbase_preflight.py` reads
the real figure from an authenticated account and `register_contract_fee` pins
it. Trading on the placeholder is trading on a guess about the exact quantity
that killed the previous strategy.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Pessimistic placeholder, in USD per contract per side. Chosen high on
# purpose: an overestimate refuses trades that would have been fine, which
# costs opportunity; an underestimate takes trades that lose money, which costs
# money. Override from the account's real schedule before trading.
DEFAULT_PER_CONTRACT_FEE_USD = 1.00

_MEASURED_CONTRACT_FEES: Dict[str, float] = {}


def register_contract_fee(venue: str, fee_usd_per_contract: float) -> None:
    """Pin the account's real per-contract commission, as read from the API."""
    _MEASURED_CONTRACT_FEES[venue] = float(fee_usd_per_contract)
    logger.info("Per-contract fee for %s set to $%.4f/side",
                venue, fee_usd_per_contract)


def contract_fee_usd(venue: str) -> float:
    return _MEASURED_CONTRACT_FEES.get(venue, DEFAULT_PER_CONTRACT_FEE_USD)


def fee_is_measured(venue: str) -> bool:
    """Whether the fee in use came from the account rather than the placeholder."""
    return venue in _MEASURED_CONTRACT_FEES


# ---------------------------------------------------------------------------
# Contract description
# ---------------------------------------------------------------------------

@dataclass
class ContractSpec:
    """The tradeable facts about one CDE contract, read from ccxt's market."""

    symbol: str                 # unified ccxt symbol, e.g. BTC/USD:USD-301220
    contract_id: str            # venue id, e.g. BIP-20DEC30-CDE
    base: str
    contract_size: float        # base units per contract, e.g. 0.01 BTC
    min_contracts: int
    max_contracts: int
    display_name: str = ""
    expiry_iso: str = ""

    def notional_usd(self, price: float) -> float:
        """Dollar value of ONE contract at `price`."""
        return price * self.contract_size

    def fee_pct_per_side(self, price: float, venue: str) -> float:
        """
        Per-contract commission expressed as a percent of notional.

        This is the conversion that makes a flat fee comparable to the
        percentage-based cost model the rest of the repo speaks. Note it is a
        function of PRICE: the same contract gets proportionally cheaper as the
        underlying rises, which is a real effect and not a modelling artefact.
        """
        notional = self.notional_usd(price)
        if notional <= 0:
            return float("inf")
        return contract_fee_usd(venue) / notional * 100.0

    def round_trip_fee_pct(self, price: float, venue: str) -> float:
        return 2.0 * self.fee_pct_per_side(price, venue)


def spec_from_market(symbol: str, market: dict) -> ContractSpec:
    """Build a ContractSpec from a ccxt market entry."""
    info = market.get("info", {}) or {}
    details = info.get("future_product_details", {}) or {}
    limits = (market.get("limits", {}) or {}).get("amount", {}) or {}
    return ContractSpec(
        symbol=symbol,
        contract_id=str(market.get("id", "")),
        base=str(market.get("base", "")),
        contract_size=float(market.get("contractSize") or details.get("contract_size") or 0),
        min_contracts=int(float(limits.get("min") or 1)),
        max_contracts=int(float(limits.get("max") or 10_000)),
        display_name=str(info.get("display_name", "")),
        expiry_iso=str(market.get("expiryDatetime") or ""),
    )


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

@dataclass
class ContractSizing:
    contracts: int
    notional_usd: float
    base_quantity: float
    fee_pct_round_trip: float
    reason: str = ""

    @property
    def tradeable(self) -> bool:
        return self.contracts > 0


def size_in_contracts(
    spec: ContractSpec,
    price: float,
    max_notional_usd: float,
    venue: str,
    allow_overshoot: bool = False,
) -> ContractSizing:
    """
    Convert a dollar budget into a whole number of contracts.

    Rounds DOWN, and returns zero rather than one contract when even the
    minimum exceeds the budget. That refusal is the point: `max_notional_usd`
    is a risk limit set by a human, and the alternative — quietly taking a
    position 54% larger than authorised because the contract happened not to
    divide evenly — is how an automated system ends up outside the risk budget
    it was audited against. `allow_overshoot` exists so the caller can make
    that trade-off explicitly, never by accident.
    """
    if price <= 0 or spec.contract_size <= 0:
        return ContractSizing(0, 0.0, 0.0, float("inf"), "no price or contract size")

    per_contract = spec.notional_usd(price)
    contracts = int(math.floor(max_notional_usd / per_contract))

    if contracts < spec.min_contracts:
        if not allow_overshoot:
            return ContractSizing(
                0, 0.0, 0.0, spec.round_trip_fee_pct(price, venue),
                reason=(
                    f"one contract is ${per_contract:,.2f}, over the "
                    f"${max_notional_usd:,.2f} cap — raise MAX_POSITION_USD above "
                    f"${per_contract:,.2f} to trade {spec.base}, or skip it"
                ),
            )
        contracts = spec.min_contracts

    contracts = min(contracts, spec.max_contracts)
    notional = contracts * per_contract
    return ContractSizing(
        contracts=contracts,
        notional_usd=notional,
        base_quantity=contracts * spec.contract_size,
        fee_pct_round_trip=spec.round_trip_fee_pct(price, venue),
        reason="",
    )


# ---------------------------------------------------------------------------
# Trading sessions
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    is_open: bool
    close_time_iso: str = ""
    open_time_iso: str = ""
    state: str = ""
    reason: str = ""

    def closes_within(self, hours: float, now: Optional[datetime] = None) -> bool:
        """
        Whether the session ends inside `hours` from now.

        A fixed-hold strategy needs this before entering, not after: a 16-hour
        hold opened four hours before a session close cannot be exited on
        schedule, and an exit that waits for the next open happens at whatever
        price the gap left behind.
        """
        if not self.close_time_iso:
            return False
        try:
            close = datetime.fromisoformat(self.close_time_iso.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return False
        now = now or datetime.now(timezone.utc)
        return (close - now).total_seconds() < hours * 3600


def session_from_market(market: dict) -> SessionState:
    """Read FCM session state out of a ccxt market entry."""
    details = (market.get("info", {}) or {}).get("fcm_trading_session_details") or {}
    if not details:
        # A market with no session block is a continuously-traded instrument.
        return SessionState(is_open=True, state="continuous")
    return SessionState(
        is_open=bool(details.get("is_session_open")),
        close_time_iso=str(details.get("close_time") or ""),
        open_time_iso=str(details.get("open_time") or ""),
        state=str(details.get("session_state") or ""),
        reason=str(details.get("closed_reason") or ""),
    )


def session_allows_entry(
    session: SessionState,
    hold_hours: float,
    now: Optional[datetime] = None,
) -> Tuple[bool, str]:
    """
    Whether a position can be opened right now. Returns (allowed, reason).

    Only the *current* session state blocks an entry. It is tempting to also
    refuse when the hold would outlast the session, but FCM sessions run a full
    24 hours with a short daily break rather than closing overnight: refusing
    every entry within 16 hours of the close would reject roughly two thirds of
    all signals, on a strategy whose edge is already concentrated in rare
    extremes. Crossing the break is normal — the position persists, only
    trading pauses.

    What genuinely matters is whether the EXIT lands inside the break, which
    delays the flatten rather than preventing it. That is a warning, surfaced
    by `exit_falls_in_break`, not a refusal.
    """
    if not session.is_open:
        return False, (
            f"trading session is closed ({session.state or 'unknown'}"
            + (f", {session.reason}" if session.reason and "UNDEFINED" not in session.reason else "")
            + ")"
        )
    return True, ""


def exit_falls_in_break(
    session: SessionState,
    hold_hours: float,
    now: Optional[datetime] = None,
    break_hours: float = 1.0,
) -> Tuple[bool, str]:
    """
    Whether this hold's deadline lands inside the daily session break.

    A delayed exit is not a disaster, but it is a deviation from the horizon
    the strategy was measured over, and an unlogged deviation is how a backtest
    quietly stops describing the account. `break_hours` is how long the venue
    is down; CDE's daily maintenance is about an hour.
    """
    if not session.close_time_iso:
        return False, ""
    try:
        close = datetime.fromisoformat(session.close_time_iso.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return False, ""

    now = now or datetime.now(timezone.utc)
    deadline = now.timestamp() + hold_hours * 3600
    gap_start, gap_end = close.timestamp(), close.timestamp() + break_hours * 3600

    if gap_start <= deadline < gap_end:
        return True, (
            f"the {hold_hours:g}h exit lands in the session break after "
            f"{session.close_time_iso}; the flatten will be delayed to the reopen"
        )
    return False, ""
