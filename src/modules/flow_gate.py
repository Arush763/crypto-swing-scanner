"""
Live flow-concentration gate.

Implements, for the live scanner, the filter validated offline in
scripts/study_combo.py and scripts/study_gate_as_filter.py: suppress entries
taken while flow is concentrated into a few very large prints, because that
condition precedes SMALLER subsequent moves.

Backtested effect of using this as a filter on the existing tape signal
(15m, 12 majors, 90d, train/test split by time):

    gate open   133 trades   +0.224% gross/trade   39.1% win rate
    gate shut  2591 trades   -0.052% gross/trade   27.9% win rate

replicating at 1h with a +0.386% gross swing and a 27.8% -> 42.9% win-rate
lift. Out-of-sample the gated subset was net-positive at 15m, though on only
46 trades — the effect is consistent in direction across timeframes but the
absolute magnitude is not yet established on a large sample.

Why a per-symbol percentile rather than a fixed cutoff
------------------------------------------------------
Whale share differs by nearly 5x across majors — measured over 90 days, the
30th percentile is 0.374 for BTC but 0.082 for LINK. A single absolute
threshold would suppress every BTC signal and no LINK signal, which is not
the filter that was validated. So each symbol is judged against its own
recent distribution, which is also what the offline study did (quantiles were
computed per symbol).
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Window of past observations each symbol is judged against. Long enough to
# describe a distribution, short enough to track regime change.
HISTORY_WINDOW = 300

# Observations required before the gate will express an opinion. Below this
# the percentile estimate is too noisy to act on, and the gate stays OPEN —
# failing permissive rather than silently blocking every signal during
# warm-up, which would look identical to "the strategy stopped working".
MIN_OBSERVATIONS = 40

# Suppress when whale share sits above this percentile of the symbol's own
# history. 0.30 in the study was the "low whale share" bucket, i.e. trade
# only the bottom 30%; that is the aggressive setting. 0.50 is the moderate
# one and is the default here — see PERCENTILE note in config.
DEFAULT_MAX_PERCENTILE = 0.50

# Minimum prints in a window before its whale share means anything.
#
# The quantity was validated on 15m bars carrying thousands of prints. Live,
# the window is one scan interval, and a short interval on a quiet symbol can
# return a handful of trades — at which point "the top 1% of prints" is just
# the single largest trade and the share is an artifact of the sample size,
# not a property of the market. Observed directly: a 6-second window on
# BTC/USDT returned 5 trades and a whale share of 67.8%; a 1-trade window
# gives 100.0%.
#
# Below this count the reading is discarded rather than acted on OR recorded
# — letting it into the history would poison the very distribution later
# readings are judged against.
MIN_TRADES_FOR_WHALE_SHARE = 50


@dataclass
class GateVerdict:
    is_open: bool
    whale_share: float
    percentile: float
    reason: str
    warming_up: bool = False
    unreliable: bool = False    # too few prints for the reading to mean anything


class FlowConcentrationGate:
    """
    Tracks each symbol's own whale-share distribution and judges the current
    reading against it.

    Persisted to disk because the live scanner is a fresh process on every
    scheduled run; without persistence every cycle would be a cold start and
    the gate would never leave warm-up.
    """

    def __init__(
        self,
        state_path: Optional[str] = "data/state/flow_gate.json",
        max_percentile: float = DEFAULT_MAX_PERCENTILE,
        window: int = HISTORY_WINDOW,
        min_observations: int = MIN_OBSERVATIONS,
    ) -> None:
        self.path = Path(state_path) if state_path else None
        self.max_percentile = max_percentile
        self.window = window
        self.min_observations = min_observations
        self.history: Dict[str, Deque[float]] = {}
        self.load()

    # -- persistence --------------------------------------------------

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read flow-gate state (%s) — starting fresh", exc)
            return
        self.history = {
            sym: deque(vals[-self.window:], maxlen=self.window)
            for sym, vals in raw.items()
        }

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {sym: list(vals) for sym, vals in self.history.items()}
        tmp = self.path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            logger.error("Could not persist flow-gate state: %s", exc)

    # -- gate ---------------------------------------------------------

    def observe(self, symbol: str, whale_share: float) -> None:
        """Record a reading without judging it."""
        if not np.isfinite(whale_share):
            return
        self.history.setdefault(symbol, deque(maxlen=self.window)).append(float(whale_share))

    def check(
        self,
        symbol: str,
        whale_share: float,
        trade_count: Optional[int] = None,
        observe: bool = True,
    ) -> GateVerdict:
        """
        Judge the current whale share against this symbol's own history.

        The reading is judged BEFORE being added to the history, so a symbol
        is never compared against a distribution that already contains the
        value being tested — a subtle self-reference that would pull every
        observation toward the middle of its own window.

        `trade_count` is the number of prints the share was computed from.
        Pass it whenever it is known: a share derived from a handful of
        trades is meaningless (see MIN_TRADES_FOR_WHALE_SHARE) and is
        neither acted on nor recorded.
        """
        if trade_count is not None and trade_count < MIN_TRADES_FOR_WHALE_SHARE:
            return GateVerdict(
                is_open=True, whale_share=whale_share, percentile=float("nan"),
                reason=(
                    f"only {trade_count} prints in window "
                    f"(need {MIN_TRADES_FOR_WHALE_SHARE}) — reading discarded"
                ),
                unreliable=True,
            )

        history = self.history.get(symbol)
        n = len(history) if history else 0

        if n < self.min_observations:
            if observe:
                self.observe(symbol, whale_share)
            return GateVerdict(
                is_open=True, whale_share=whale_share, percentile=float("nan"),
                reason=f"warming up ({n}/{self.min_observations} observations)",
                warming_up=True,
            )

        arr = np.fromiter(history, dtype=float)
        percentile = float((arr <= whale_share).mean())
        cutoff = float(np.quantile(arr, self.max_percentile))

        if observe:
            self.observe(symbol, whale_share)

        if whale_share <= cutoff:
            return GateVerdict(
                is_open=True, whale_share=whale_share, percentile=percentile,
                reason=f"flow is broad (p{percentile*100:.0f})",
            )

        return GateVerdict(
            is_open=False, whale_share=whale_share, percentile=percentile,
            reason=(
                f"flow concentrated in large prints "
                f"(whale share {whale_share:.1%}, p{percentile*100:.0f} of this "
                f"symbol's history) — precedes smaller moves"
            ),
        )
