"""
Trading cost model — fees, spread, and market impact.

Why this module exists
----------------------
`TapeBacktestConfig.commission_pct` has existed since the first backtest and
was never referenced by the engine: every P&L figure this repo has ever
produced was gross of fees, spread, and slippage. That is survivable when the
average trade is held for days and the edge is measured in whole percent. It
is fatal at high frequency, where the same fixed round-trip toll is paid far
more often against a much smaller per-trade move.

Concretely, from the 2,086 labelled setups in labeled_setups.pkl, the 4h tape
signal's gross expectancy is +0.242%/trade. Charge a realistic 0.30%
round-trip and that becomes -0.058%/trade — the strategy flips from
profitable to loss-making purely on costs. Any decision about trading
frequency that isn't made net of this model is uninformed.

Cost decomposition
------------------
Total round-trip cost has three parts, only one of which is a constant:

  1. Fees        — exchange commission, per side. Maker and taker differ
                   substantially, and that difference is the single largest
                   lever available at high frequency.
  2. Spread      — a taker order crosses the book and pays half the bid/ask
                   spread. A maker order does not; it earns it. This is why
                   post-only entries matter far more than they look.
  3. Impact      — orders larger than the touch consume successive levels.
                   Modelled with the standard square-root law against
                   participation rate; see `impact_bps` for why not linear.

All functions here work in *percent of notional* (0.1 == 0.1%) to match the
convention used by Trade.pnl_pct throughout the backtester, with basis points
used only where an input is conventionally quoted that way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

BPS_PER_PCT = 100.0


# ---------------------------------------------------------------------------
# Venue fee schedules
# ---------------------------------------------------------------------------
# Spot fees at the lowest (no-volume, no-token-holding) VIP tier, which is the
# correct assumption for a new account — quoting a tier you have not reached
# is how a backtest flatters itself. Negative maker fees are rebates.
#
# Sourced from each venue's public spot fee schedule. If you hold exchange
# tokens or clear volume tiers your real fees will be lower; override via
# CostModel(fee_override_bps=...) rather than editing these, so the defaults
# stay conservative.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeeSchedule:
    """Maker/taker commission for one venue, in basis points per side."""
    venue: str
    maker_bps: float
    taker_bps: float

    @property
    def maker_pct(self) -> float:
        return self.maker_bps / BPS_PER_PCT

    @property
    def taker_pct(self) -> float:
        return self.taker_bps / BPS_PER_PCT


FEE_SCHEDULES: Dict[str, FeeSchedule] = {
    "okx":      FeeSchedule("okx",      maker_bps=8.0,  taker_bps=10.0),
    "binance":  FeeSchedule("binance",  maker_bps=10.0, taker_bps=10.0),
    "bybit":    FeeSchedule("bybit",    maker_bps=10.0, taker_bps=10.0),
    "kucoin":   FeeSchedule("kucoin",   maker_bps=10.0, taker_bps=10.0),
    "coinbase": FeeSchedule("coinbase", maker_bps=40.0, taker_bps=60.0),
    "kraken":   FeeSchedule("kraken",   maker_bps=25.0, taker_bps=40.0),
    "gateio":   FeeSchedule("gateio",   maker_bps=20.0, taker_bps=20.0),
}

DEFAULT_VENUE = "okx"


# ---------------------------------------------------------------------------
# Typical half-spreads by liquidity tier
# ---------------------------------------------------------------------------
# A taker order pays half the quoted spread on entry and half on exit. On
# BTC/ETH this is genuinely negligible (sub-basis-point); on a thin alt it can
# exceed the fee itself, which is the entire reason restricting the universe
# to majors changes the viability arithmetic.
#
# These are fallbacks used by the backtester, which has no archived book to
# measure against. Live code should pass the real measured spread from
# OrderBookSignals.spread_pct instead — see CostModel.round_trip_pct.
# ---------------------------------------------------------------------------

HALF_SPREAD_BPS_BY_TIER: Dict[str, float] = {
    "mega":  0.5,    # BTC, ETH — order of a tenth of a basis point in practice
    "large": 1.5,    # SOL, XRP, BNB, ADA, DOGE …
    "mid":   6.0,    # top-100 alts
    "small": 25.0,   # everything else; wide, and the quote moves as you lift it
}


# ---------------------------------------------------------------------------
# Market impact
# ---------------------------------------------------------------------------

def impact_bps(
    order_size_usd: float,
    reference_volume_usd: float,
    impact_coefficient: float = 10.0,
) -> float:
    """
    Estimate temporary market impact in basis points via the square-root law:

        impact_bps = coefficient * sqrt(order_size / reference_volume)

    `reference_volume_usd` is the dollar volume traded over the bar (or other
    window) the order executes into.

    Square-root rather than linear: the linear "Kyle's lambda" form used in
    `OrderBookSnapshot.estimate_slippage` walks a *live* book and is right for
    that purpose, since it reads real resting size. Applied to historical bars
    it badly overestimates, because it ignores that the book refills as you
    trade — resting liquidity is a flow, not the fixed stock a snapshot makes
    it look like. The square-root law is the standard empirical form for
    execution over time and is what the frequency question needs.

    The default coefficient of 10 is a mid-range value from the published
    equity/crypto impact literature: a trade equal to 1% of bar volume costs
    about 1bp. It is a modelling assumption, not a measurement — if you can
    measure your own fills, calibrate it and pass your own value.
    """
    if reference_volume_usd <= 0 or order_size_usd <= 0:
        return 0.0
    participation = order_size_usd / reference_volume_usd
    return impact_coefficient * math.sqrt(participation)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

@dataclass
class CostModel:
    """
    Full round-trip trading cost for one position.

    Usage in a backtest, where only the tier is known:
        cost = CostModel(venue="okx", liquidity_tier="mega")
        pct  = cost.round_trip_pct()

    Usage live, where the real book has been measured:
        pct = cost.round_trip_pct(
            entry_half_spread_pct=ob.spread_pct / 2,
            order_size_usd=1_000,
            reference_volume_usd=bar_dollar_volume,
        )
    """

    venue: str = DEFAULT_VENUE
    liquidity_tier: str = "mega"

    # Order types per leg. Post-only entries are the highest-leverage cost
    # reduction available: a maker leg pays the (lower) maker fee *and* skips
    # the half-spread, roughly halving round-trip cost on a major. The cost of
    # that saving is fill uncertainty, which this model does not price —
    # see `PAPER_FILL_NOTE` in src/execution/executor.py.
    entry_is_maker: bool = False
    exit_is_maker: bool = False

    # Override the venue schedule (e.g. you hold exchange tokens or have
    # cleared a volume tier). Applied to both sides.
    fee_override_bps: Optional[float] = None

    impact_coefficient: float = 10.0

    def _fee_pct(self, is_maker: bool) -> float:
        if self.fee_override_bps is not None:
            return self.fee_override_bps / BPS_PER_PCT
        schedule = FEE_SCHEDULES.get(self.venue, FEE_SCHEDULES[DEFAULT_VENUE])
        return schedule.maker_pct if is_maker else schedule.taker_pct

    def _half_spread_pct(self, override: Optional[float]) -> float:
        if override is not None:
            return override
        tier = HALF_SPREAD_BPS_BY_TIER.get(self.liquidity_tier, HALF_SPREAD_BPS_BY_TIER["mid"])
        return tier / BPS_PER_PCT

    def leg_cost_pct(
        self,
        is_maker: bool,
        half_spread_pct: Optional[float] = None,
        order_size_usd: float = 0.0,
        reference_volume_usd: float = 0.0,
    ) -> float:
        """
        Cost of a single leg (one entry or one exit) as a percent of notional.

        A maker leg pays no spread — it is quoted at the touch and waits to be
        crossed. It also incurs no impact in this model, since it supplies
        liquidity rather than consuming it.
        """
        fee = self._fee_pct(is_maker)
        if is_maker:
            return fee

        spread = self._half_spread_pct(half_spread_pct)
        impact = impact_bps(order_size_usd, reference_volume_usd, self.impact_coefficient) / BPS_PER_PCT
        return fee + spread + impact

    def round_trip_pct(
        self,
        entry_half_spread_pct: Optional[float] = None,
        exit_half_spread_pct: Optional[float] = None,
        order_size_usd: float = 0.0,
        reference_volume_usd: float = 0.0,
    ) -> float:
        """
        Total cost of entering and exiting one position, in percent of
        notional. Subtract directly from a trade's gross percentage return.
        """
        entry = self.leg_cost_pct(
            self.entry_is_maker, entry_half_spread_pct,
            order_size_usd, reference_volume_usd,
        )
        exit_ = self.leg_cost_pct(
            self.exit_is_maker,
            exit_half_spread_pct if exit_half_spread_pct is not None else entry_half_spread_pct,
            order_size_usd, reference_volume_usd,
        )
        return entry + exit_

    def breakeven_move_pct(self, **kwargs) -> float:
        """
        The gross price move a trade must capture just to break even.

        This is the number to compare a candidate timeframe's average bar
        range against before trading it: if a typical favourable move on that
        timeframe is not several multiples of this, the strategy cannot pay
        for itself no matter how good the signal is.
        """
        return self.round_trip_pct(**kwargs)


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------

# Deliberately small and explicit. These are the only assets where the
# spread/impact terms are small enough for a sub-1% average trade to survive
# costs. Membership is by sustained real liquidity, not market cap ranking.
MEGA_CAP_BASES = {"BTC", "ETH"}

LARGE_CAP_BASES = {
    "SOL", "XRP", "BNB", "ADA", "DOGE", "AVAX", "LINK", "DOT",
    "MATIC", "LTC", "BCH", "TRX", "UNI", "ATOM", "XLM", "NEAR",
    "APT", "ARB", "OP", "FIL", "ETC", "HBAR", "ICP", "SUI",
}


def base_of(symbol: str) -> str:
    """'BTC/USDT' -> 'BTC'. Tolerates already-bare bases."""
    return symbol.split("/")[0].upper().strip()


def liquidity_tier(symbol: str) -> str:
    """Classify a symbol into a spread tier for the cost model."""
    base = base_of(symbol)
    if base in MEGA_CAP_BASES:
        return "mega"
    if base in LARGE_CAP_BASES:
        return "large"
    return "mid"


def cost_model_for(
    symbol: str,
    venue: str = DEFAULT_VENUE,
    entry_is_maker: bool = False,
    exit_is_maker: bool = False,
) -> CostModel:
    """Build a CostModel with the spread tier inferred from the symbol."""
    return CostModel(
        venue=venue,
        liquidity_tier=liquidity_tier(symbol),
        entry_is_maker=entry_is_maker,
        exit_is_maker=exit_is_maker,
    )
