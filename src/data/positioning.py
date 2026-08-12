"""
Live leveraged-positioning data.

The crowd-short signal (src/modules/crowd_signal.py) was validated on Binance's
historical futures metrics archive. That archive is published daily with a
lag, so it cannot drive live trading — and Binance's live REST API returns 451
from this environment, the same geo-block documented for spot in config.py.

OKX publishes equivalent statistics through its free, unauthenticated "rubik"
endpoints, is already this project's execution venue, and is reachable from
here. So history comes from Binance and live readings come from OKX.

That substitution needs stating plainly, because it is the weakest joint in
the live system: OKX's long/short ratio measures a different population of
traders than Binance's, and the two are not guaranteed to move together. Two
things mitigate it — the signal ranks each symbol against its OWN history, so
absolute level differences between venues cancel; and
`scripts/compare_venue_ratios.py` measures the actual correlation over the
overlapping window rather than assuming it. Read that number before trusting
live signals.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OKX_BASE = "https://www.okx.com/api/v5/rubik/stat"

# OKX keys these by base currency, not instrument id.
LONG_SHORT_URL = f"{OKX_BASE}/contracts/long-short-account-ratio"
OPEN_INTEREST_URL = f"{OKX_BASE}/contracts/open-interest-volume"
TAKER_VOLUME_URL = f"{OKX_BASE}/taker-volume"

# OKX's rubik endpoints rate-limit harder than the documented 20-per-2s
# suggests: at 0.25s spacing, 4 of 12 symbols came back with code 50011
# ("too many requests"). A missed symbol is a missed trade, so this is
# deliberately conservative and paired with a retry.
MIN_REQUEST_INTERVAL = 1.1
RATE_LIMIT_CODE = "50011"
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF = 2.0


@dataclass
class PositioningSnapshot:
    """Current positioning for one symbol, plus the history it is ranked against."""
    symbol: str
    long_short_ratio: float
    history: pd.Series           # indexed by timestamp, oldest first
    fetched_at: float

    @property
    def observations(self) -> int:
        return len(self.history)

    def percentile_of_current(self) -> float:
        """Where the current reading sits within its own history, 0..1."""
        if self.history.empty:
            return float("nan")
        return float((self.history <= self.long_short_ratio).mean())


class PositioningFetcher:
    """
    Fetches long/short account ratio and open interest from OKX.

    History is persisted so the rolling window survives process restarts and
    can grow beyond the 720 points OKX returns in one call — the signal's
    percentile estimate is only as good as the distribution behind it.
    """

    def __init__(
        self,
        state_path: Optional[str] = "data/state/positioning_history.json",
        max_history: int = 2000,
        timeout: int = 20,
    ) -> None:
        self.path = Path(state_path) if state_path else None
        self.max_history = max_history
        self.timeout = timeout
        self._last_call = 0.0
        self.history: Dict[str, pd.Series] = {}

        self.session = requests.Session()
        self.load()

    # -- persistence --------------------------------------------------

    def load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read positioning history (%s) — starting fresh", exc)
            return
        for symbol, records in raw.items():
            if not records:
                continue
            idx = pd.to_datetime([r[0] for r in records], utc=True)
            self.history[symbol] = pd.Series(
                [float(r[1]) for r in records], index=idx,
            ).sort_index()

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            symbol: [[ts.isoformat(), float(v)] for ts, v in series.tail(self.max_history).items()]
            for symbol, series in self.history.items()
        }
        tmp = self.path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            logger.error("Could not persist positioning history: %s", exc)

    # -- fetching -----------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < MIN_REQUEST_INTERVAL:
            time.sleep(MIN_REQUEST_INTERVAL - elapsed)
        self._last_call = time.time()

    def _get(self, url: str, params: dict) -> Optional[list]:
        """
        GET with backoff on rate limiting. A rate-limited symbol is a symbol
        the strategy cannot evaluate this cycle, i.e. a missed trade — worth
        retrying rather than dropping.
        """
        for attempt in range(RATE_LIMIT_RETRIES):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                payload = resp.json()
            except Exception as exc:
                logger.warning("Positioning fetch failed (%s): %s", url, exc)
                return None

            code = payload.get("code")
            if code == "0":
                return payload.get("data") or None

            if code == RATE_LIMIT_CODE and attempt < RATE_LIMIT_RETRIES - 1:
                delay = RATE_LIMIT_BACKOFF * (attempt + 1)
                logger.debug("Rate limited on %s — retrying in %.1fs", params.get("ccy"), delay)
                time.sleep(delay)
                continue

            logger.warning("OKX returned code=%s for %s (%s)", code, url, params.get("ccy"))
            return None
        return None

    @staticmethod
    def _base_of(symbol: str) -> str:
        return symbol.split("/")[0].upper()

    def fetch(self, symbol: str, period: str = "1H") -> Optional[PositioningSnapshot]:
        """
        Current long/short account ratio plus history for `symbol`.

        OKX returns newest-first; this reverses to oldest-first and merges with
        any persisted history so the ranking window can exceed one API call.
        """
        rows = self._get(LONG_SHORT_URL, {"ccy": self._base_of(symbol), "period": period})
        if not rows:
            return None

        try:
            idx = pd.to_datetime([int(r[0]) for r in rows], unit="ms", utc=True)
            fresh = pd.Series([float(r[1]) for r in rows], index=idx).sort_index()
        except (ValueError, IndexError, TypeError) as exc:
            logger.warning("Malformed positioning payload for %s: %s", symbol, exc)
            return None

        if fresh.empty:
            return None

        prior = self.history.get(symbol)
        merged = fresh if prior is None else pd.concat([prior, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        self.history[symbol] = merged.tail(self.max_history)

        # The reading being judged is excluded from the distribution it is
        # judged against — same discipline as the flow gate.
        current = float(merged.iloc[-1])
        return PositioningSnapshot(
            symbol=symbol,
            long_short_ratio=current,
            history=merged.iloc[:-1],
            fetched_at=time.time(),
        )

    def fetch_many(self, symbols: List[str], period: str = "1H") -> Dict[str, PositioningSnapshot]:
        out: Dict[str, PositioningSnapshot] = {}
        for symbol in symbols:
            snap = self.fetch(symbol, period)
            if snap is not None:
                out[symbol] = snap
        self.save()
        return out

    def open_interest(self, symbol: str, period: str = "1H") -> Optional[pd.Series]:
        """Open-interest history — not used by the signal, kept for diagnostics."""
        rows = self._get(OPEN_INTEREST_URL, {"ccy": self._base_of(symbol), "period": period})
        if not rows:
            return None
        idx = pd.to_datetime([int(r[0]) for r in rows], unit="ms", utc=True)
        return pd.Series([float(r[1]) for r in rows], index=idx).sort_index()
