"""
Main Scanner Orchestrator.

Pipeline per scan cycle:
  1. Fetch universe OHLCV from exchanges.
  2. Fetch order book snapshots (top-20 L2, free REST API).
  3. Score every asset (trend 30%, momentum 30%, liquidity 20%, smart money 20%).
  4. Track order-book walls per symbol across cycles and classify
     absorption (continuation) vs. repulsion (bounce) — the sole signal source.
  5. Apply the wall-signal bonus to the composite score.
  6. Use OB data to:
       - Augment smart money score with live imbalance
       - Validate/suppress signals (stop-hunt risk)
       - Provide dynamic stop placement and position sizing
  7. Generate signals for qualifying assets.
  8. Build ranked leaderboards.

Note: order-book wall tracking needs at least one prior scan cycle's
snapshot per symbol, so enable_orderbook must be True for signals to fire
at all, and the first cycle after a cold start won't produce any.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from src.config.config import (
    EXCHANGES,
    MIN_DAILY_VOLUME_USD,
    OHLCV_LIMIT,
    SCAN_TIMEFRAME,
    SIGNAL_SCORE_THRESHOLD,
)
from src.data.fetcher import MarketDataFetcher
from src.data.orderbook import OrderBookFetcher, OrderBookSignals
from src.scoring.composite import score_asset, ScoreResult
from src.modules.wall_signal import WallTracker, WallSignalResult
from src.indicators.volatility import atr_latest
from src.signals.generator import generate_signal, Signal, format_signal_table
from src.ranking.ranker import rank_results, leaderboard_summary

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    timestamp: datetime
    scores: List[ScoreResult]
    signals: List[Signal]
    ranked_df: pd.DataFrame
    leaderboards: Dict[str, pd.DataFrame]
    signal_table: pd.DataFrame
    duration_seconds: float
    assets_scanned: int
    ob_enabled: bool = False


class Scanner:
    """
    Orchestrates the full scanning pipeline.

    Usage:
        scanner = Scanner(enable_orderbook=True)
        result  = scanner.run()
    """

    def __init__(
        self,
        exchange_ids: List[str] = EXCHANGES,
        min_volume: float = MIN_DAILY_VOLUME_USD,
        score_threshold: float = SIGNAL_SCORE_THRESHOLD,
        enable_orderbook: bool = True,
        ob_order_size_usd: float = 10_000.0,
    ) -> None:
        self.fetcher        = MarketDataFetcher(exchange_ids=exchange_ids)
        self.min_volume     = min_volume
        self.score_threshold = score_threshold
        self.enable_orderbook = enable_orderbook
        self.ob_order_size  = ob_order_size_usd
        self.ob_fetcher     = OrderBookFetcher() if enable_orderbook else None
        self.wall_tracker   = WallTracker()

    def run(self) -> ScanResult:
        start = time.time()
        logger.info("=== Scan cycle started (OB=%s) ===", self.enable_orderbook)

        raw_universe = self.fetcher.fetch_universe(min_volume=self.min_volume)
        btc_ohlcv    = self.fetcher.fetch_btc()
        btc_close    = btc_ohlcv["close"] if not btc_ohlcv.empty else None

        logger.info("Scoring %d assets…", len(raw_universe))

        scores: List[ScoreResult] = []
        signals: List[Signal]     = []

        for symbol, ohlcv in raw_universe.items():
            try:
                # Determine primary exchange for OB fetch
                primary_exchange = "binance"
                price = float(ohlcv["close"].iloc[-1])

                ob_sigs  = None
                wall_sig = None
                if self.enable_orderbook and self.ob_fetcher:
                    ob_sigs = self.ob_fetcher.fetch_signals(
                        symbol, primary_exchange,
                        order_size_usd=self.ob_order_size,
                    )
                    wall_sig = self.wall_tracker.update(symbol, price, ob_sigs)

                result = self._score_one(symbol, ohlcv, btc_close, ob_sigs, wall_sig)
                scores.append(result)

                sig = generate_signal(
                    result, ohlcv,
                    wall_signal=wall_sig,
                    ob_signals=ob_sigs,
                    score_threshold=self.score_threshold,
                )
                if sig is not None:
                    signals.append(sig)

            except Exception as exc:
                logger.warning("Failed to score %s: %s", symbol, exc)

        ranked_df    = rank_results(scores)
        leaderboards = leaderboard_summary(ranked_df)
        signal_table = format_signal_table(signals)

        duration = round(time.time() - start, 2)
        logger.info("=== Scan complete: %d assets, %d signals, %.1fs ===",
                    len(scores), len(signals), duration)

        return ScanResult(
            timestamp=datetime.now(timezone.utc),
            scores=scores,
            signals=signals,
            ranked_df=ranked_df,
            leaderboards=leaderboards,
            signal_table=signal_table,
            duration_seconds=duration,
            assets_scanned=len(scores),
            ob_enabled=self.enable_orderbook,
        )

    def _score_one(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        btc_close: Optional[pd.Series],
        ob_sigs: Optional[OrderBookSignals],
        wall_sig: Optional[WallSignalResult],
    ) -> ScoreResult:
        high   = ohlcv["high"]
        low    = ohlcv["low"]
        close  = ohlcv["close"]
        volume = ohlcv["volume"]

        is_setup    = bool(wall_sig and wall_sig.is_setup)
        wall_bonus  = wall_sig.bonus_score if is_setup else 0.0
        wall_event  = wall_sig.event if wall_sig else "none"

        current_atr = atr_latest(high, low, close)

        return score_asset(
            symbol=symbol,
            ohlcv=ohlcv,
            btc_close=btc_close,
            wall_bonus=wall_bonus,
            is_wall_signal=is_setup,
            wall_event=wall_event,
            atr=current_atr,
            ob_signals=ob_sigs,
        )
