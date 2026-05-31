"""
Main Scanner Orchestrator.

Pipeline per scan cycle:
  1. Fetch universe OHLCV from exchanges.
  2. Fetch order book snapshots (top-20 L2, free REST API).
  3. Score every asset (trend 30%, momentum 30%, liquidity 20%, smart money 20%).
  4. Run detection modules (breakout, retest, squeeze).
  5. Apply detection bonuses to composite score.
  6. Use OB data to:
       - Augment smart money score with live imbalance
       - Validate/suppress signals (wall detection, stop-hunt risk)
       - Provide dynamic stop placement and position sizing
  7. Generate signals for qualifying assets.
  8. Build ranked leaderboards.
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
from src.modules.breakout import detect_breakout
from src.modules.retest import detect_retest
from src.modules.squeeze import detect_squeeze
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

                ob_sigs = None
                if self.enable_orderbook and self.ob_fetcher:
                    breakout_res = detect_breakout(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
                    resistance   = breakout_res.resistance_level if breakout_res.is_breakout else 0.0
                    ob_sigs = self.ob_fetcher.fetch_signals(
                        symbol, primary_exchange,
                        resistance_level=resistance,
                        order_size_usd=self.ob_order_size,
                    )

                result = self._score_one(symbol, ohlcv, btc_close, ob_sigs)
                scores.append(result)

                # Re-detect with the same logic for signal generation
                breakout = detect_breakout(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
                retest   = detect_retest(ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"], breakout)
                squeeze  = detect_squeeze(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])

                sig = generate_signal(
                    result, ohlcv,
                    breakout=breakout, retest=retest, squeeze=squeeze,
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
    ) -> ScoreResult:
        high   = ohlcv["high"]
        low    = ohlcv["low"]
        close  = ohlcv["close"]
        volume = ohlcv["volume"]

        breakout = detect_breakout(high, low, close, volume)
        retest   = detect_retest(ohlcv["open"], high, low, close, breakout)
        squeeze  = detect_squeeze(high, low, close, volume)

        bo_bonus = breakout.bonus_score if breakout.is_breakout else 0.0
        rt_bonus = retest.bonus_score   if retest.is_retest     else 0.0
        sq_bonus = squeeze.bonus_score  if (squeeze.squeeze_breakout or squeeze.in_squeeze) else 0.0

        current_atr = atr_latest(high, low, close)

        return score_asset(
            symbol=symbol,
            ohlcv=ohlcv,
            btc_close=btc_close,
            breakout_bonus=bo_bonus,
            retest_bonus=rt_bonus,
            squeeze_bonus=sq_bonus,
            is_breakout=breakout.is_breakout,
            is_retest=retest.is_retest,
            is_squeeze=squeeze.squeeze_breakout or squeeze.in_squeeze,
            atr=current_atr,
            ob_signals=ob_sigs,
        )
