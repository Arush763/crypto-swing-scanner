"""
Venue capability registry.

The executor used to assume one venue (OKX), one product type (spot), and one
direction (long). Routing to Coinbase breaks all three assumptions at once,
and they break in ways that silently produce the wrong trade rather than an
error — so they are made explicit here instead of being discovered live.

The distinction that matters most
---------------------------------
"Coinbase" is not one venue. It is three, with different products, different
credentials, and — decisively for this project — different answers to "can I
be short?":

  coinbaseadvanced      US-available spot. Long only. Selling means selling
                        coins you already hold; there is no borrow, so a
                        short is not expressible at all.
  coinbaseexchange      The institutional spot venue (ex-Coinbase Pro). Same
                        long-only constraint for a retail account.
  coinbaseinternational Perpetual futures. Shorts are native. Not available
                        to US persons.

The one validated strategy in this repo (src/modules/crowd_signal.py) is
SHORT-ONLY. So the choice of Coinbase product is not a configuration detail,
it is the difference between the strategy being expressible and not. A venue
that cannot short does not get to silently take the trade as a long — that
would be a different strategy, and the mirror trade was measured at
-0.240%/trade.

Credentials
-----------
Coinbase has two generations of key and ccxt takes them in the same fields:

  CDP / Cloud keys (current)  apiKey = "organizations/{org}/apiKeys/{id}"
                              secret = the EC private key PEM, newlines and all
  Legacy HMAC keys            apiKey/secret/passphrase triple

Passing a PEM through an environment variable is where this usually goes
wrong: a `.env` file or CI secret often collapses the newlines into the two
literal characters backslash-n, and ccxt then fails to parse the key with an
error that does not mention newlines. `load_credentials` repairs that case
rather than leaving it to be debugged at 3am against a live account.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VenueSpec:
    """Everything the executor needs to know about one venue."""

    name: str                       # key used in config and FEE_SCHEDULES
    ccxt_id: str                    # exchange class in ccxt
    product: str                    # "spot" | "perp"
    can_short: bool
    quote: str                      # settlement/quote currency for symbols
    description: str

    # Env vars searched for credentials, in order of preference. The generic
    # EXCHANGE_* names are kept last so an existing OKX setup does not get
    # picked up as Coinbase credentials by accident.
    key_vars: Tuple[str, ...] = ("EXCHANGE_API_KEY",)
    secret_vars: Tuple[str, ...] = ("EXCHANGE_API_SECRET",)
    passphrase_vars: Tuple[str, ...] = ("EXCHANGE_API_PASSWORD",)

    # Whether ccxt's set_sandbox_mode does something real here. Coinbase
    # Advanced has no public sandbox, which matters: for that venue the only
    # rehearsal available is this repo's paper mode.
    has_sandbox: bool = False

    # Symbol suffix appended to BASE/QUOTE for this product. ccxt unified
    # perp symbols look like "BTC/USDC:USDC"; spot symbols have no suffix.
    settle: Optional[str] = None

    us_available: bool = True
    notes: str = ""

    # Whether symbols must be resolved against live markets rather than built
    # from a template. Dated futures roll, so their symbol carries an expiry
    # that no template can know — see `resolve_contract`.
    dynamic_symbols: bool = False

    def symbol_for(self, base: str) -> str:
        """
        Map a bare base ('BTC') to this venue's unified ccxt symbol.

        Only valid for venues whose symbols are templated (spot and perps). A
        dated-futures venue raises instead of returning a symbol that looks
        plausible and does not exist.
        """
        base = base.split("/")[0].upper().strip()
        if self.dynamic_symbols:
            raise ValueError(
                f"{self.name} lists dated contracts whose symbols carry an expiry; "
                f"use resolve_contract(client, {base!r}) against live markets."
            )
        pair = f"{base}/{self.quote}"
        return f"{pair}:{self.settle}" if self.settle else pair


VENUES: Dict[str, VenueSpec] = {
    # ---------------- Coinbase ----------------
    "coinbaseadvanced": VenueSpec(
        name="coinbaseadvanced",
        ccxt_id="coinbaseadvanced",
        product="spot",
        can_short=False,
        quote="USD",
        description="Coinbase Advanced Trade - US spot",
        key_vars=("COINBASE_API_KEY", "EXCHANGE_API_KEY"),
        secret_vars=("COINBASE_API_SECRET", "EXCHANGE_API_SECRET"),
        passphrase_vars=("COINBASE_API_PASSPHRASE", "EXCHANGE_API_PASSWORD"),
        has_sandbox=False,
        us_available=True,
        notes=(
            "The SPOT leg only: long only, no borrow. The same ccxt client also "
            "carries the CFTC-regulated CDE futures, which US persons CAN short - "
            "those are the separate 'coinbasederivatives' entry below, not this one."
        ),
    ),
    "coinbasederivatives": VenueSpec(
        name="coinbasederivatives",
        ccxt_id="coinbaseadvanced",          # same client, different markets
        product="future",
        can_short=True,
        quote="USD",
        settle="USD",
        description="Coinbase Derivatives Exchange (CDE) - CFTC-regulated US futures",
        key_vars=("COINBASE_API_KEY", "EXCHANGE_API_KEY"),
        secret_vars=("COINBASE_API_SECRET", "EXCHANGE_API_SECRET"),
        passphrase_vars=("COINBASE_API_PASSPHRASE", "EXCHANGE_API_PASSWORD"),
        has_sandbox=False,
        us_available=True,
        dynamic_symbols=True,
        notes=(
            "The one Coinbase product a US person can short. Contracts are dated, "
            "so the symbol cannot be built from a template - resolve it against "
            "live markets (see resolve_contract). The furthest-dated series "
            "(Dec 2030) is the perpetual-style product and is the right one for a "
            "16h hold; a near-dated contract would expire underneath the strategy. "
            "TWO THINGS TO VERIFY BEFORE TRADING: fees are charged PER CONTRACT, "
            "not as a percentage of notional, so small contracts are "
            "proportionally expensive - and the account must be futures-enabled."
        ),
    ),
    "coinbaseexchange": VenueSpec(
        name="coinbaseexchange",
        ccxt_id="coinbaseexchange",
        product="spot",
        can_short=False,
        quote="USD",
        description="Coinbase Exchange (ex-Coinbase Pro) - institutional spot",
        key_vars=("COINBASE_API_KEY", "EXCHANGE_API_KEY"),
        secret_vars=("COINBASE_API_SECRET", "EXCHANGE_API_SECRET"),
        passphrase_vars=("COINBASE_API_PASSPHRASE", "EXCHANGE_API_PASSWORD"),
        has_sandbox=True,
        us_available=True,
        notes="Long only for a retail account.",
    ),
    "coinbaseinternational": VenueSpec(
        name="coinbaseinternational",
        ccxt_id="coinbaseinternational",
        product="perp",
        can_short=True,
        quote="USDC",
        settle="USDC",
        description="Coinbase International Exchange - perpetual futures",
        key_vars=("COINBASE_INTX_API_KEY", "COINBASE_API_KEY", "EXCHANGE_API_KEY"),
        secret_vars=("COINBASE_INTX_API_SECRET", "COINBASE_API_SECRET", "EXCHANGE_API_SECRET"),
        passphrase_vars=("COINBASE_INTX_API_PASSPHRASE", "COINBASE_API_PASSPHRASE",
                         "EXCHANGE_API_PASSWORD"),
        has_sandbox=True,
        us_available=False,
        notes=(
            "The only Coinbase product on which the crowd-short strategy is "
            "expressible. Not open to US persons. Funding is charged periodically "
            "and is NOT in this repo's cost model - see docs/COINBASE.md."
        ),
    ),

    # ---------------- incumbent ----------------
    "okx": VenueSpec(
        name="okx",
        ccxt_id="okx",
        product="spot",
        can_short=False,
        quote="USDT",
        description="OKX spot - the repo's existing default",
        key_vars=("OKX_API_KEY", "EXCHANGE_API_KEY"),
        secret_vars=("OKX_API_SECRET", "EXCHANGE_API_SECRET"),
        passphrase_vars=("OKX_API_PASSPHRASE", "EXCHANGE_API_PASSWORD"),
        has_sandbox=True,
        us_available=False,
    ),
}


def venue_spec(name: str) -> VenueSpec:
    """
    Look up a venue, raising rather than defaulting.

    Deliberately not falling back to a default: silently routing an order to a
    venue the caller did not name is the worst failure this module could have.
    """
    try:
        return VENUES[name]
    except KeyError:
        known = ", ".join(sorted(VENUES))
        raise KeyError(f"Unknown venue {name!r}. Known venues: {known}") from None


def is_coinbase(name: str) -> bool:
    return name.startswith("coinbase")


# ---------------------------------------------------------------------------
# Dated-contract resolution
# ---------------------------------------------------------------------------

# Coinbase Derivatives Exchange contract ids end in this suffix. Filtering on
# it separates the CFTC-regulated US product from the International perps that
# the same ccxt client also lists but a US account cannot trade — a distinction
# that is invisible in the unified symbol and decisive in practice.
CDE_SUFFIX = "-CDE"


def resolve_contract(
    client,
    base: str,
    venue: str = "coinbasederivatives",
    min_days_to_expiry: int = 90,
    now_ms: Optional[int] = None,
) -> Optional[str]:
    """
    Pick the contract to trade for `base`, or None if there isn't a usable one.

    Chooses the FURTHEST-dated active series. Coinbase lists both near-dated
    quarterlies and a long-dated (Dec 2030) series that functions as a
    perpetual; the long-dated one is what a fixed-hold strategy wants, because
    a near-dated contract can expire mid-position and settle the trade on the
    calendar's schedule rather than the strategy's.

    `min_days_to_expiry` is a floor rather than a preference: a contract close
    to expiry drifts from spot and thins out, so it misprices the very signal
    being traded even when the hold would technically fit inside it.
    """
    base = base.split("/")[0].upper().strip()
    spec = venue_spec(venue)
    horizon_ms = (now_ms or 0) + min_days_to_expiry * 86_400_000

    candidates = []
    for symbol, market in (client.markets or {}).items():
        if market.get("base") != base or not market.get("active"):
            continue
        if not str(market.get("id", "")).endswith(CDE_SUFFIX):
            continue
        expiry = market.get("expiry") or 0
        if now_ms is not None and expiry and expiry < horizon_ms:
            continue
        candidates.append((expiry, symbol))

    if not candidates:
        logger.info("No usable %s contract for %s", spec.name, base)
        return None
    return max(candidates)[1]


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------

def _first_env(names: Tuple[str, ...]) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def repair_pem(secret: str) -> str:
    """
    Restore a PEM private key mangled by environment-variable transport.

    Shells, .env parsers and CI secret stores routinely turn the newlines in a
    PEM into the two literal characters backslash-n. The key then looks correct
    to the eye and fails to parse, with an error that says nothing about
    newlines. Cheap to repair, expensive to diagnose.
    """
    if "BEGIN" not in secret:
        return secret
    if "\\n" in secret and "\n" not in secret:
        return secret.replace("\\n", "\n")
    return secret


@dataclass
class Credentials:
    api_key: str = ""
    secret: str = ""
    passphrase: str = ""

    @property
    def present(self) -> bool:
        return bool(self.api_key and self.secret)

    @property
    def is_cdp_key(self) -> bool:
        """CDP keys name an organization; legacy HMAC keys are opaque strings."""
        return self.api_key.startswith("organizations/") or "BEGIN" in self.secret

    def missing(self, spec: VenueSpec) -> List[str]:
        """Which env vars would need setting, for an error message worth reading."""
        gaps: List[str] = []
        if not self.api_key:
            gaps.append(spec.key_vars[0])
        if not self.secret:
            gaps.append(spec.secret_vars[0])
        return gaps


def load_credentials(spec: VenueSpec) -> Credentials:
    """Read credentials for `spec` from the environment."""
    return Credentials(
        api_key=_first_env(spec.key_vars),
        secret=repair_pem(_first_env(spec.secret_vars)),
        passphrase=_first_env(spec.passphrase_vars),
    )


def ccxt_options(spec: VenueSpec, creds: Credentials) -> dict:
    """
    Build the ccxt constructor options for this venue.

    A CDP key carries no passphrase; passing an empty one makes some ccxt
    versions attempt legacy HMAC signing and fail confusingly, so the field is
    omitted rather than blanked.
    """
    options: dict = {
        "apiKey": creds.api_key,
        "secret": creds.secret,
        "enableRateLimit": True,
    }
    if creds.passphrase and not creds.is_cdp_key:
        options["password"] = creds.passphrase
    if spec.product == "perp":
        options["options"] = {"defaultType": "swap"}
    return options


# ---------------------------------------------------------------------------
# Direction checking
# ---------------------------------------------------------------------------

class DirectionUnsupported(Exception):
    """Raised when a venue cannot express the side a signal asks for."""


def check_direction(spec: VenueSpec, side: str) -> None:
    """
    Verify `side` is tradeable on `spec`, raising with the reason if not.

    Called before sizing, not after - a refusal that arrives once an order is
    half-built is a refusal that eventually gets worked around.
    """
    side = side.lower()
    if side == "long":
        return
    if side != "short":
        raise DirectionUnsupported(f"Unknown side {side!r}")
    if not spec.can_short:
        raise DirectionUnsupported(
            f"{spec.name} is {spec.product} and cannot hold a short position "
            f"({spec.description}). The crowd-short signal is short-only and its "
            f"mirror trade was measured at -0.240%/trade, so taking this as a "
            f"long is not a fallback - it is a different, losing strategy."
        )
