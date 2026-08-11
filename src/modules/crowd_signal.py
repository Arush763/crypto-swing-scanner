"""
Crowd-short signal — short symbols whose retail long/short ratio is extreme.

The one validated edge in this project. Measured on Binance futures
positioning metrics, 12 majors, 365 days, 1h bars, 16-hour fixed hold:

    out-of-sample   +0.429%/trade   t=2.24   246 trades   +105.7% total
    vs shorting everything over the same window (-0.080%/trade): +0.509%, t=2.58
    all six robustness checks pass, including the tail trim that killed
    every previous candidate

Mechanism: crowded leveraged longs are the fuel for a liquidation cascade,
and a cascade is directional by construction. This is why the signal has
directional information when every spot-price-derived feature tested in this
project had none.

Two properties that are easy to get wrong
-----------------------------------------
1. It is ONE-SIDED. The mirror trade (long when the crowd is least long)
   loses -0.240%/trade, and running both as a spread nets to +0.009%.
   Retail is structurally long-biased in crypto, so crowded shorts do not
   produce a symmetric squeeze. Short only.

2. It ranks WITHIN each symbol. Absolute ratio levels differ by ~2x across
   symbols and across venues, so a global threshold would trade some symbols
   constantly and others never. The percentile is the signal; the level is not.

Data source caveat — read before enabling live
-----------------------------------------------
The signal was validated on Binance data. Binance's live API is geo-blocked
from this project's environment (451, same block documented in config.py).
OKX is reachable, but its ratio only tracks Binance's closely for BTC, ETH and
SOL (rank correlation 0.98/0.92/0.89); for others it is weak (DOGE 0.24) or
inverted (AVAX -0.41, LINK -0.23). And BTC/ETH/SOL alone do NOT carry the
edge — 58 out-of-sample trades, t=1.14, and the tail trim collapses it to
+0.008%.

So `require_validated_source=True` (the default) refuses to emit live signals
from a source that was not validated, rather than silently trading a
different signal than the one that was tested. See
scripts/compare_venue_ratios.py for the measurement behind this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Percentile of a symbol's own ratio history above which it is a short
# candidate. 0.90 is what was validated; loosening it trades more often but
# was not tested and the edge is concentrated in the extreme.
DEFAULT_SHORT_PERCENTILE = 0.90

# Observations needed before a percentile is meaningful. The backtest ranked
# against ~6,000 hourly points per symbol; live starts from whatever the API
# returns (OKX gives 720) and grows. Below this the gate stays shut — for a
# signal that only fires on extremes, a bad percentile estimate means firing
# at non-extremes, which is worse than not trading.
MIN_HISTORY = 300

# Hold period in hours. Fixed by the validation: the signal is a statement
# about forward return over this horizon, not a trend to be ridden. There is
# deliberately no stop or target — adding them reintroduces the parameter
# search that was exhausted in scripts/run_full_combo_sweep.py.
HOLD_HOURS = 16

# Venues whose ratio was shown to track the validated Binance series closely
# enough to carry the signal (rank correlation >= 0.85 per symbol).
VALIDATED_SOURCES = {"binance"}
PARTIAL_SOURCES = {"okx": {"BTC/USDT", "ETH/USDT", "SOL/USDT"}}


@dataclass
class CrowdSignal:
    symbol: str
    timestamp: datetime
    direction: int              # always -1; the long side has no edge
    ratio: float
    percentile: float
    observations: int
    hold_hours: int
    source: str
    reason: str

    @property
    def is_short(self) -> bool:
        return self.direction < 0


@dataclass
class CrowdVerdict:
    """Why a symbol did or did not produce a signal — kept for logging."""
    symbol: str
    fired: bool
    reason: str
    percentile: float = float("nan")
    observations: int = 0


class CrowdShortSignal:
    """
    Evaluates positioning readings and emits short signals.

    Stateless with respect to history: the caller supplies each symbol's
    ratio history (from src/data/positioning.py, which persists it), so this
    class stays testable without network or disk.
    """

    def __init__(
        self,
        short_percentile: float = DEFAULT_SHORT_PERCENTILE,
        min_history: int = MIN_HISTORY,
        hold_hours: int = HOLD_HOURS,
        source: str = "okx",
        require_validated_source: bool = True,
    ) -> None:
        self.short_percentile = short_percentile
        self.min_history = min_history
        self.hold_hours = hold_hours
        self.source = source
        self.require_validated_source = require_validated_source

    # ------------------------------------------------------------------

    def source_is_trusted(self, symbol: str) -> bool:
        """
        Whether this (source, symbol) pair was shown to track the series the
        signal was validated on.
        """
        if not self.require_validated_source:
            return True
        if self.source in VALIDATED_SOURCES:
            return True
        allowed = PARTIAL_SOURCES.get(self.source)
        return bool(allowed and symbol in allowed)

    def evaluate(
        self,
        symbol: str,
        current_ratio: float,
        history: pd.Series,
    ) -> CrowdVerdict:
        """
        Judge one symbol. `history` must NOT contain `current_ratio` — a
        reading compared against a window containing itself is pulled toward
        that window's middle, which for an extremes-only signal means missing
        the very readings it exists to catch.
        """
        if not np.isfinite(current_ratio) or current_ratio <= 0:
            return CrowdVerdict(symbol, False, "ratio unavailable")

        if not self.source_is_trusted(symbol):
            return CrowdVerdict(
                symbol, False,
                f"{self.source} ratio for {symbol} was not validated against the "
                f"backtested Binance series — refusing to trade an untested signal",
            )

        clean = history.dropna() if history is not None else pd.Series(dtype=float)
        n = len(clean)
        if n < self.min_history:
            return CrowdVerdict(
                symbol, False,
                f"only {n}/{self.min_history} observations — percentile not yet reliable",
                observations=n,
            )

        values = clean.to_numpy(dtype=float)
        percentile = float((values <= current_ratio).mean())
        cutoff = float(np.quantile(values, self.short_percentile))

        if current_ratio < cutoff:
            return CrowdVerdict(
                symbol, False,
                f"crowd not extreme (p{percentile*100:.0f})",
                percentile=percentile, observations=n,
            )

        return CrowdVerdict(
            symbol, True,
            f"crowd heavily long (ratio {current_ratio:.2f}, "
            f"p{percentile*100:.0f} of {n} observations)",
            percentile=percentile, observations=n,
        )

    def generate(
        self,
        snapshots: Dict[str, "object"],
    ) -> tuple:
        """
        Evaluate every snapshot. Returns (signals, verdicts).

        `snapshots` maps symbol -> PositioningSnapshot (see
        src/data/positioning.py); anything exposing `long_short_ratio` and
        `history` works, which keeps this decoupled from the fetcher.
        """
        signals: List[CrowdSignal] = []
        verdicts: List[CrowdVerdict] = []

        for symbol, snap in snapshots.items():
            verdict = self.evaluate(
                symbol,
                getattr(snap, "long_short_ratio", float("nan")),
                getattr(snap, "history", None),
            )
            verdicts.append(verdict)
            if not verdict.fired:
                logger.debug("No crowd signal for %s — %s", symbol, verdict.reason)
                continue

            signals.append(CrowdSignal(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                direction=-1,
                ratio=float(snap.long_short_ratio),
                percentile=verdict.percentile,
                observations=verdict.observations,
                hold_hours=self.hold_hours,
                source=self.source,
                reason=verdict.reason,
            ))
            logger.info("CROWD SHORT %s — %s", symbol, verdict.reason)

        return signals, verdicts
