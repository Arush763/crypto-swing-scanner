"""
Pre-trade viability gate — refuse trades the venue's costs already beat.

Why this is a hard gate and not a warning
-----------------------------------------
This repo's whole measured history says the same thing: the signals are not
wrong, they are too small. The exhaustive sweep in
`scripts/run_full_combo_sweep.py` searched 1,728 parameter combinations on a
full year of data and found **0 of 1,008** usable combinations net-positive
out of sample, while 13% were gross-positive. The gap between those two
numbers is the fee. Nothing in the parameter space escaped it.

So the failure mode this module exists to prevent is not "a bad signal fires".
It is "a genuinely positive signal fires at a venue whose round trip costs
more than the signal is worth, and the loop dutifully pays that toll several
hundred times". That is precisely what the 96 paper trades in `logs/trades.jsonl`
recorded: 50% win rate and -0.854%/trade.

Routing to Coinbase makes this urgent rather than theoretical. Coinbase's
retail spot tier is roughly an order of magnitude above OKX's, so a strategy
that is marginal on OKX is decisively negative on Coinbase spot — and the
arithmetic that shows it is three lines long, which is exactly why it should
be run automatically before every entry rather than remembered.

What counts as "expected edge"
------------------------------
Only a number that was actually measured out of sample, with its provenance
attached. A signal with no measured edge gets `None`, and `None` does not pass
— an unmeasured strategy cannot be shown to clear its costs, and assuming it
does is how every previously falsified idea in this repo got funded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from src.execution.costs import CostModel, cost_model_for

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Measured gross edges
# ---------------------------------------------------------------------------
# Gross (pre-cost) mean return per trade, from the out-of-sample half of the
# study named in each entry. Gross, not net, because the cost side is what
# this module computes — quoting a net figure measured at a different venue's
# fees and then charging fees again would double-count.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MeasuredEdge:
    signal_type: str
    gross_pct_per_trade: float
    t_stat: float
    oos_trades: int
    source: str

    @property
    def is_significant(self) -> bool:
        return abs(self.t_stat) >= 2.0


MEASURED_EDGES: Dict[str, MeasuredEdge] = {
    "crowd_short": MeasuredEdge(
        signal_type="crowd_short",
        gross_pct_per_trade=0.655,
        t_stat=2.24,
        oos_trades=246,
        source=(
            "scripts/study_crowd_short.py — 12 majors, 365d, 1h bars, 16h hold. "
            "Net +0.429%/trade at the fees it was measured against; beats an "
            "always-short benchmark by +0.509% (t=2.58). Passes 6/6 robustness "
            "checks including the top-3 tail trim."
        ),
    ),
    # Kept so the guard has something to say about the legacy signals rather
    # than falling through to 'unknown'. This is the best out-of-sample gross
    # figure any of 1,008 usable combinations produced, i.e. a ceiling, not an
    # expectation — and it is below every fee schedule in costs.py.
    "ob_wall": MeasuredEdge(
        signal_type="ob_wall",
        gross_pct_per_trade=0.148,
        t_stat=0.0,
        oos_trades=310,
        source=(
            "scripts/run_full_combo_sweep.py — best gross of 1,008 combos on a "
            "full year. 0 of 1,008 were net-positive out of sample. This is a "
            "ceiling from a best-of-N search, so the true expectation is lower."
        ),
    ),
}

# Aliases for the signal_type strings the scanner actually emits.
_EDGE_ALIASES = {
    "ob_wall_ask_absorption": "ob_wall",
    "ob_wall_bid_repulsion": "ob_wall",
    "ask_absorption": "ob_wall",
    "bid_repulsion": "ob_wall",
}


def measured_edge(signal_type: str) -> Optional[MeasuredEdge]:
    """The out-of-sample gross edge for a signal type, or None if unmeasured."""
    key = _EDGE_ALIASES.get(signal_type, signal_type)
    return MEASURED_EDGES.get(key)


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

@dataclass
class EdgeVerdict:
    """The full arithmetic behind an allow/refuse, so a refusal is auditable."""

    allowed: bool
    reason: str
    symbol: str = ""
    venue: str = ""
    gross_edge_pct: float = 0.0
    round_trip_cost_pct: float = 0.0
    funding_pct: float = 0.0
    net_edge_pct: float = 0.0
    required_margin_pct: float = 0.0

    def explain(self) -> str:
        return (
            f"{self.symbol} @ {self.venue}: gross {self.gross_edge_pct:+.3f}% "
            f"- costs {self.round_trip_cost_pct:.3f}% "
            f"- funding {self.funding_pct:.3f}% "
            f"= net {self.net_edge_pct:+.3f}% "
            f"(needs > {self.required_margin_pct:.3f}%)"
        )


class EdgeGuard:
    """
    Decides whether a signal is worth its execution costs at a given venue.

    `safety_margin_pct` is the cushion a trade must clear *beyond* breakeven.
    Zero would mean trading at a modelled net of exactly zero, which in
    practice is a loss: the cost model omits maker adverse selection (see
    PAPER_FILL_NOTE in executor.py) and every omission points the same way.
    """

    def __init__(
        self,
        venue: str,
        entry_is_maker: bool = True,
        exit_is_maker: bool = False,
        safety_margin_pct: float = 0.05,
        require_significant: bool = True,
    ) -> None:
        self.venue = venue
        self.entry_is_maker = entry_is_maker
        self.exit_is_maker = exit_is_maker
        self.safety_margin_pct = safety_margin_pct
        self.require_significant = require_significant

    # ------------------------------------------------------------------

    def cost_model(self, symbol: str) -> CostModel:
        return cost_model_for(
            symbol,
            venue=self.venue,
            entry_is_maker=self.entry_is_maker,
            exit_is_maker=self.exit_is_maker,
        )

    def round_trip_cost_pct(
        self,
        symbol: str,
        half_spread_pct: Optional[float] = None,
        order_size_usd: float = 0.0,
        reference_volume_usd: float = 0.0,
        fee_pct_round_trip: Optional[float] = None,
    ) -> float:
        """
        Total cost of a round trip, in percent of notional.

        `fee_pct_round_trip` replaces the venue's percentage fee schedule
        entirely, for venues that charge per contract rather than per dollar —
        see src/execution/contracts.py. Spread and impact are still added on
        top, because those are properties of the book and do not care how the
        commission is quoted.
        """
        model = self.cost_model(symbol)
        cost = model.round_trip_pct(
            entry_half_spread_pct=half_spread_pct,
            order_size_usd=order_size_usd,
            reference_volume_usd=reference_volume_usd,
        )
        if fee_pct_round_trip is None:
            return cost

        # Strip the schedule's fees back out and substitute the flat charge.
        scheduled_fees = (model._fee_pct(self.entry_is_maker)
                          + model._fee_pct(self.exit_is_maker))
        return cost - scheduled_fees + fee_pct_round_trip

    # ------------------------------------------------------------------

    def evaluate(
        self,
        symbol: str,
        signal_type: str,
        hold_hours: float = 0.0,
        funding_rate_pct_per_hour: float = 0.0,
        half_spread_pct: Optional[float] = None,
        order_size_usd: float = 0.0,
        reference_volume_usd: float = 0.0,
        fee_pct_round_trip: Optional[float] = None,
    ) -> EdgeVerdict:
        """
        Judge one prospective trade.

        `funding_rate_pct_per_hour` follows the exchange's own sign convention:
        positive means longs pay shorts. This function charges only the
        unfavourable case and never books a credit — a short that expects to be
        *paid* funding should not have that windfall counted toward clearing
        its fees, because the rate can flip inside a 16-hour hold and the
        strategy was never validated on funding income.
        """
        edge = measured_edge(signal_type)
        if edge is None:
            return EdgeVerdict(
                allowed=False,
                symbol=symbol,
                venue=self.venue,
                reason=(
                    f"no out-of-sample edge has been measured for signal type "
                    f"{signal_type!r}, so it cannot be shown to clear costs"
                ),
            )

        if self.require_significant and not edge.is_significant:
            return EdgeVerdict(
                allowed=False,
                symbol=symbol,
                venue=self.venue,
                gross_edge_pct=edge.gross_pct_per_trade,
                reason=(
                    f"{signal_type} has t={edge.t_stat:.2f} on {edge.oos_trades} "
                    f"out-of-sample trades — not distinguishable from zero"
                ),
            )

        cost = self.round_trip_cost_pct(
            symbol, half_spread_pct, order_size_usd, reference_volume_usd,
            fee_pct_round_trip=fee_pct_round_trip,
        )
        funding = max(0.0, -funding_rate_pct_per_hour) * max(0.0, hold_hours)
        net = edge.gross_pct_per_trade - cost - funding

        allowed = net > self.safety_margin_pct
        verdict = EdgeVerdict(
            allowed=allowed,
            symbol=symbol,
            venue=self.venue,
            gross_edge_pct=edge.gross_pct_per_trade,
            round_trip_cost_pct=cost,
            funding_pct=funding,
            net_edge_pct=net,
            required_margin_pct=self.safety_margin_pct,
            reason="",
        )
        verdict.reason = (
            "clears costs" if allowed
            else f"costs exceed the measured edge — {verdict.explain()}"
        )
        return verdict

    # ------------------------------------------------------------------

    def breakeven_report(self, symbol: str, signal_type: str) -> str:
        """One-line summary for logs and preflight output."""
        edge = measured_edge(signal_type)
        cost = self.round_trip_cost_pct(symbol)
        if edge is None:
            return f"{symbol:16s} cost {cost:.3f}%/round-trip, edge unmeasured"
        net = edge.gross_pct_per_trade - cost
        flag = "OK " if net > self.safety_margin_pct else "NO "
        return (
            f"{flag}{symbol:16s} gross {edge.gross_pct_per_trade:+.3f}%  "
            f"cost {cost:.3f}%  net {net:+.3f}%"
        )
