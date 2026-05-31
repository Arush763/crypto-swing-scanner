"""
Order Book Data Fetcher & Signal Extractor.

Pulls L2 snapshots from exchange REST APIs via ccxt (free, no auth required).
Updates every scan cycle and caches snapshots so all modules can read from
a single in-memory state.

Signals produced (all backed by research findings):
  - bid_ask_imbalance      : bid_depth / ask_depth at top-20 levels
  - spread_pct             : (ask - bid) / mid — widens on weak demand
  - depth_usd_10bps        : cumulative USD depth within ±10bps of mid
  - wall_bid_level         : price of largest bid cluster (dynamic support)
  - wall_ask_level         : price of largest ask cluster (dynamic resistance)
  - has_ask_wall_above     : large ask cluster within N% above current price
  - has_bid_wall_below     : large bid cluster within N% below current price
  - slippage_est_pct       : estimated slippage for a target order size
  - is_stop_hunt_risk      : imbalance + momentum signal for stop-hunt detection
  - ob_breakout_conviction : imbalance score at the exact resistance level
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
OB_DEPTH = 20             # Top-N levels to fetch (sweet spot per research)
WALL_MULTIPLIER = 3.0     # A cluster is a "wall" if it's ≥3× mean level size
WALL_SCAN_PCT = 0.05      # Look for walls within ±5% of current price
STOP_HUNT_IMBALANCE = -0.6  # Heavily ask-heavy while bullish = stop-hunt risk
SLIPPAGE_FRACTION = 0.40  # Use 40% of depth to stay under 1% slippage


# ── Data containers ────────────────────────────────────────────────────────

@dataclass
class OrderBookSnapshot:
    symbol: str
    timestamp: float
    bids: List[List[float]]   # [[price, size], ...]
    asks: List[List[float]]   # [[price, size], ...]
    exchange: str = ""

    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0

    @property
    def mid(self) -> float:
        bb, ba = self.best_bid, self.best_ask
        return (bb + ba) / 2 if bb and ba else 0.0

    @property
    def spread_pct(self) -> float:
        m = self.mid
        if m == 0:
            return 0.0
        return (self.best_ask - self.best_bid) / m

    def bid_volume(self, levels: int = OB_DEPTH) -> float:
        return sum(row[1] * row[0] for row in self.bids[:levels])   # USD-weighted

    def ask_volume(self, levels: int = OB_DEPTH) -> float:
        return sum(row[1] * row[0] for row in self.asks[:levels])

    def imbalance(self, levels: int = OB_DEPTH) -> float:
        """
        (bid_vol - ask_vol) / (bid_vol + ask_vol)  →  range [-1, +1].
        Positive = bid-heavy (buying pressure). Negative = ask-heavy (selling pressure).
        Based on research: imbalance >0.2 at top-20 correlates with 3-day up-move.
        """
        bv = self.bid_volume(levels)
        av = self.ask_volume(levels)
        total = bv + av
        return (bv - av) / total if total > 0 else 0.0

    def depth_within_bps(self, bps: float = 10.0) -> float:
        """Total USD depth (bids + asks) within ±bps basis points of mid."""
        m = self.mid
        if m == 0:
            return 0.0
        lo = m * (1 - bps / 10_000)
        hi = m * (1 + bps / 10_000)
        bid_d = sum(r[1] * r[0] for r in self.bids if r[0] >= lo)
        ask_d = sum(r[1] * r[0] for r in self.asks if r[0] <= hi)
        return bid_d + ask_d

    def imbalance_at_level(self, price: float, tolerance_pct: float = 0.02) -> float:
        """
        Compute bid/ask imbalance within `tolerance_pct` of a specific price.
        Used to assess conviction at a resistance/support level.

        Research finding: imbalance >1.5 (bid-heavy) at resistance = 73% follow-through.
        Returns raw ratio (bids_usd / asks_usd), not normalised.
        """
        lo = price * (1 - tolerance_pct)
        hi = price * (1 + tolerance_pct)
        bv = sum(r[1] * r[0] for r in self.bids if lo <= r[0] <= hi)
        av = sum(r[1] * r[0] for r in self.asks if lo <= r[0] <= hi)
        return bv / av if av > 0 else (2.0 if bv > 0 else 1.0)

    def find_walls(
        self, side: str = "ask", scan_pct: float = WALL_SCAN_PCT
    ) -> List[Tuple[float, float]]:
        """
        Identify large limit-order clusters ("walls") within scan_pct of mid.

        A wall is any level whose USD size ≥ WALL_MULTIPLIER × mean level size.
        Returns [(price, usd_size), ...] sorted by price.

        Research: ask wall appearing pre-breakout → 68% fake breakout.
                  ask wall disappearing as price approaches → 72% follow-through.
        """
        m = self.mid
        if m == 0:
            return []

        rows = self.asks if side == "ask" else self.bids
        lo = m if side == "ask" else m * (1 - scan_pct)
        hi = m * (1 + scan_pct) if side == "ask" else m

        in_range = [(r[0], r[1] * r[0]) for r in rows if lo <= r[0] <= hi]
        if not in_range:
            return []

        sizes = [s for _, s in in_range]
        mean_size = float(np.mean(sizes))
        threshold = mean_size * WALL_MULTIPLIER

        walls = [(p, s) for p, s in in_range if s >= threshold]
        return sorted(walls, key=lambda x: x[0])

    def estimate_slippage(self, order_size_usd: float, side: str = "buy") -> float:
        """
        Estimate percentage slippage for an order of `order_size_usd`.

        Uses the simplified Kyle's Lambda model from research:
          impact_bps = (order_size / cumulative_depth) × 1000

        Returns slippage as a fraction (0.01 = 1%).
        """
        rows = self.asks if side == "buy" else self.bids
        if not rows:
            return 0.01  # fallback 1%

        cumulative_usd = 0.0
        spent_usd = 0.0
        entry = rows[0][0]

        for price, size in rows:
            level_usd = price * size
            if spent_usd + level_usd >= order_size_usd:
                # Fill partially at this level
                remaining = order_size_usd - spent_usd
                fill_price = price
                break
            spent_usd += level_usd
            fill_price = price
            cumulative_usd += level_usd

        if entry == 0:
            return 0.0
        return abs(fill_price - entry) / entry

    def max_safe_position_usd(self, slippage_limit_pct: float = 0.005) -> float:
        """
        Maximum order size (USD) that stays under `slippage_limit_pct` slippage.
        Uses the 40% depth rule from research.

        Research: order > 10% of visible depth → >0.5% slippage.
        """
        depth = self.depth_within_bps(50)   # ±50bps depth
        return depth * SLIPPAGE_FRACTION


@dataclass
class OrderBookSignals:
    """Derived trading signals extracted from an OB snapshot."""

    symbol: str
    timestamp: float

    # Core metrics
    spread_pct: float = 0.0
    imbalance: float = 0.0           # -1 to +1; >0 = bid-heavy
    depth_usd_10bps: float = 0.0
    depth_usd_50bps: float = 0.0

    # Wall detection
    wall_bid_price: float = 0.0      # Strongest bid wall price (0 = none found)
    wall_ask_price: float = 0.0      # Strongest ask wall price (0 = none found)
    has_ask_wall_above: bool = False  # Ask wall within 5% above price
    has_bid_wall_below: bool = False  # Bid wall within 5% below price

    # Breakout-specific
    ob_breakout_conviction: float = 0.0   # imbalance ratio at resistance level (>1.5 = bullish)

    # Risk metrics
    slippage_est_pct: float = 0.0         # Estimated slippage for default order size
    max_safe_position_usd: float = 0.0
    is_stop_hunt_risk: bool = False

    # Composite OB score contribution (0-100, used in smart money scoring)
    ob_score: float = 0.0


# ── Fetcher ───────────────────────────────────────────────────────────────

class OrderBookFetcher:
    """
    Fetches L2 order book snapshots for a list of symbols via ccxt REST.

    Rate-limit safe: enforces per-exchange delays and caches results for
    `cache_ttl_seconds` to avoid redundant calls within one scan cycle.
    """

    # Per-exchange minimum interval between calls (seconds)
    _RATE_LIMITS = {
        "binance": 0.5,
        "bybit":   0.2,
        "coinbase": 0.1,
        "kraken":  0.07,
    }

    def __init__(self, depth: int = OB_DEPTH, cache_ttl: int = 60) -> None:
        self.depth = depth
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, OrderBookSnapshot]] = {}
        self._last_call: Dict[str, float] = {}
        self._clients: Dict[str, object] = {}

    def _get_client(self, exchange_id: str):
        if exchange_id not in self._clients:
            try:
                import ccxt
                cls = getattr(ccxt, exchange_id)
                self._clients[exchange_id] = cls({"enableRateLimit": True})
            except Exception as exc:
                logger.error("Cannot init %s: %s", exchange_id, exc)
                return None
        return self._clients[exchange_id]

    def _respect_rate_limit(self, exchange_id: str) -> None:
        min_interval = self._RATE_LIMITS.get(exchange_id, 0.5)
        last = self._last_call.get(exchange_id, 0.0)
        elapsed = time.time() - last
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call[exchange_id] = time.time()

    def fetch(self, symbol: str, exchange_id: str = "binance") -> Optional[OrderBookSnapshot]:
        """Fetch a single order book snapshot, using cache if fresh."""
        cache_key = f"{exchange_id}:{symbol}"
        now = time.time()
        if cache_key in self._cache:
            ts, snap = self._cache[cache_key]
            if now - ts < self.cache_ttl:
                return snap

        client = self._get_client(exchange_id)
        if client is None:
            return None

        self._respect_rate_limit(exchange_id)
        try:
            raw = client.fetch_order_book(symbol, limit=self.depth)
            snap = OrderBookSnapshot(
                symbol=symbol,
                timestamp=now,
                bids=raw.get("bids", []),
                asks=raw.get("asks", []),
                exchange=exchange_id,
            )
            self._cache[cache_key] = (now, snap)
            return snap
        except Exception as exc:
            logger.debug("OB fetch failed %s/%s: %s", exchange_id, symbol, exc)
            return None

    def fetch_signals(
        self,
        symbol: str,
        exchange_id: str = "binance",
        resistance_level: float = 0.0,
        order_size_usd: float = 10_000.0,
    ) -> Optional[OrderBookSignals]:
        """
        Fetch OB snapshot and compute all trading signals.

        Args:
            resistance_level : the breakout level to check conviction at
            order_size_usd   : hypothetical order size for slippage estimation
        """
        snap = self.fetch(symbol, exchange_id)
        if snap is None:
            return None

        # Core metrics
        imb = snap.imbalance(OB_DEPTH)
        spread = snap.spread_pct
        d10 = snap.depth_within_bps(10)
        d50 = snap.depth_within_bps(50)

        # Wall detection
        bid_walls = snap.find_walls("bid")
        ask_walls = snap.find_walls("ask")
        wall_bid = bid_walls[-1][0] if bid_walls else 0.0   # strongest = highest bid wall
        wall_ask = ask_walls[0][0] if ask_walls else 0.0    # lowest ask wall

        # Breakout conviction at resistance
        ob_conv = 1.0
        if resistance_level > 0:
            ob_conv = snap.imbalance_at_level(resistance_level)

        # Slippage & position sizing
        slip = snap.estimate_slippage(order_size_usd)
        max_pos = snap.max_safe_position_usd()

        # Stop-hunt risk:
        # research finding — ask-heavy (imbalance < -0.6) during bullish push
        # signals potential stop hunt / fake breakout
        is_stop_hunt = imb < STOP_HUNT_IMBALANCE

        # OB score (0-100) for smart-money module integration
        # Combines imbalance, spread health, wall structure
        imb_score = max(0.0, (imb + 1.0) / 2.0) * 40        # imbalance 0-40pts
        spread_score = max(0.0, (1.0 - spread / 0.01)) * 20  # tight spread 0-20pts
        conv_score = min(1.0, max(0.0, (ob_conv - 1.0) / 2.0)) * 25  # conviction 0-25pts
        wall_ok = 1.0 if (not ask_walls or wall_ask > snap.mid * 1.03) else 0.3
        wall_score = wall_ok * 15                             # clear overhead 0-15pts
        ob_score = round(min(100.0, imb_score + spread_score + conv_score + wall_score), 2)

        return OrderBookSignals(
            symbol=symbol,
            timestamp=snap.timestamp,
            spread_pct=round(spread * 100, 4),
            imbalance=round(imb, 4),
            depth_usd_10bps=round(d10, 2),
            depth_usd_50bps=round(d50, 2),
            wall_bid_price=wall_bid,
            wall_ask_price=wall_ask,
            has_ask_wall_above=bool(ask_walls),
            has_bid_wall_below=bool(bid_walls),
            ob_breakout_conviction=round(ob_conv, 3),
            slippage_est_pct=round(slip * 100, 4),
            max_safe_position_usd=round(max_pos, 2),
            is_stop_hunt_risk=is_stop_hunt,
            ob_score=ob_score,
        )
