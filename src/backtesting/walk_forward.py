"""
Expanding-window walk-forward backtest with weekly cumulative ML retraining.

Unlike run_backtest (src.backtesting.engine), which trains the ML signal
filter once up front and replays a single fixed historical window, this
module simulates trading week by week through history: each week's trades
are decided using only the model trained on *prior* weeks' outcomes, then
that week's own setups (labelled independently via simulate_exit) are added
to the cumulative training set and the model is refit before the next week
starts. The model only ever sees the past relative to whatever week it's
currently deciding — there is no lookahead in what the model knows, even
though training-label generation for a given week's setups does look at
that setup's own future bars to determine its outcome (the same approach
src.modules.signal_filter / scripts/train_signal_filter.py already use;
unavoidable for supervised learning — the alternative, waiting for every
real trade to close before learning anything, would leave most setups
unlabelled for months).

The traded universe expands over time: a symbol only starts contributing
trades and training rows once it has enough of its own bar history (no
survivorship bias from including coins that didn't exist yet).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config.config import TapeBacktestConfig
from src.indicators.trend import ema
from src.indicators.volatility import atr as compute_atr
from src.modules.tape_signal import detect_tape_signals
from src.modules.signal_filter import FEATURE_NAMES, tape_features, select_tape_features
from src.backtesting.engine import (
    Trade, _PositionState, _step_bar_range, _daily_trend_bull,
    _build_equity_curve, simulate_exit,
)
from src.backtesting.metrics import compute_all_metrics

MIN_TRAINING_ROWS_TO_FIT = 20

# Generic features (is_ask, distance_pct, ...) carry no symbol identity, so a
# structurally bad symbol can look statistically fine to the model if its
# setups happen to land on feature values the model has learned to favor
# (this is exactly what happened with ZEC/USDT in the 5-year run — see
# data/walk_forward_5y_report.txt). This trailing per-symbol win rate gives
# the shared model a way to learn "this name specifically has been bad to me
# lately" without one-hot identity (which would overfit symbols with only a
# handful of trades). Computed strictly from that symbol's own past labelled
# setups -- no lookahead.
TRAILING_WIN_RATE_WINDOW = 15
TRAILING_WIN_RATE_NEUTRAL = 0.5

# symbol_trailing_win_rate only sees win/loss as a coin flip -- a symbol that
# loses 70% of the time by -1% each and wins 30% by +0.5% looks identical to
# one that loses 70% of the time by -8% each, even though the second is the
# one actually capable of producing a -50%+ cumulative drawdown on its own.
# This magnitude-aware sibling feature (trailing *average pnl*, not just
# win/loss direction) gives the model a way to learn that distinction and
# respond to *how bad* a symbol's setups have been, not just how often they
# lose. Same no-lookahead, same-symbol-only, same window as the win-rate
# feature above.
TRAILING_PNL_NEUTRAL = 0.0
WALK_FORWARD_FEATURE_NAMES = FEATURE_NAMES + ["symbol_trailing_win_rate", "symbol_trailing_avg_pnl_pct"]

# Several of the worst-losing symbols (ZEC, TAO, JTO, WLD, BICO, AVAX, DOT)
# sit in the bottom half of the universe by dollar trading volume -- thin
# liquidity is more directly tied to pump-and-dump risk than market cap
# (Binance kline data has no market cap field anyway, and we already have
# volume). Rolling, not a single full-history figure, so a symbol that was
# thin early on but became liquid later isn't punished forever, and vice
# versa -- and shifted by one bar so the gate at bar i only ever uses volume
# known strictly before i, same no-lookahead convention as _daily_trend_bull.
ROLLING_VOLUME_WINDOW_BARS = 180  # ~30 days of 4h bars

# Research finding: AVAX/USDT's worst losses came from setups the filter
# passed on marginal confidence -- its raw setup pool has a normal ~36% win
# rate, but the 25 setups actually traded skewed to 20%, and per-symbol
# checking showed the trailing-performance features carry no real
# *symbol-specific* signal (correlation indistinguishable from noise at
# n=171, p=0.88) even though they're genuinely predictive pooled across the
# whole universe (p<0.0001). A hierarchical/per-symbol model would be
# fitting that noise. Instead of trying to detect *which* marginal trades
# are bad (the data says that's not reliably knowable per-symbol), shrink
# the bet size on all of them: scale risk by how confident the model
# actually is, linearly from MIN_CONFIDENCE_SCALE at the pass/fail threshold
# up to full size at proba=1.0. A symbol whose passing setups cluster near
# the threshold (like AVAX's did) now loses less per trade even though it
# still isn't blocked outright.
MIN_CONFIDENCE_SCALE = 0.3


def _confidence_scale(proba: np.ndarray, threshold: float, min_scale: float = MIN_CONFIDENCE_SCALE) -> np.ndarray:
    span = max(1.0 - threshold, 1e-9)
    raw = min_scale + (1 - min_scale) * (proba - threshold) / span
    return np.clip(raw, min_scale, 1.0)


def _trailing_win_rate(history: deque) -> float:
    if not history:
        return TRAILING_WIN_RATE_NEUTRAL
    return sum(history) / len(history)


def _trailing_avg_pnl(history: deque) -> float:
    if not history:
        return TRAILING_PNL_NEUTRAL
    return sum(history) / len(history)


@dataclass
class WeeklyLogEntry:
    week_start: pd.Timestamp
    active_symbols: int
    new_trades: int
    new_training_rows: int
    cumulative_training_rows: int
    model_fitted: bool


@dataclass
class WalkForwardResult:
    trades: List[Trade]
    equity_curve: pd.Series
    metrics: dict
    weekly_log: List[WeeklyLogEntry] = field(default_factory=list)
    training_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    final_model: object = None


def _score_batch(pipeline, feat_df: pd.DataFrame) -> np.ndarray:
    if pipeline is None or feat_df.empty:
        return np.ones(len(feat_df))
    return pipeline.predict_proba(feat_df[WALK_FORWARD_FEATURE_NAMES])[:, 1]


def _fit_pipeline(training_df: pd.DataFrame):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    y = training_df["win"]
    if len(training_df) < MIN_TRAINING_ROWS_TO_FIT or y.nunique() < 2:
        return None
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    pipeline.fit(training_df[WALK_FORWARD_FEATURE_NAMES], y)
    return pipeline


def _btc_in_bull_series(universe: Dict[str, pd.DataFrame], cfg: TapeBacktestConfig) -> Optional[pd.Series]:
    btc_bars = universe.get("BTC/USDT")
    if btc_bars is None:
        btc_bars = universe.get("BTC/USD")
    if btc_bars is None:
        return None
    btc_close = btc_bars["close"]
    in_bull = (btc_close > ema(btc_close, cfg.ema_mid)) & (btc_close > ema(btc_close, cfg.ema_long))
    return in_bull


def _precompute_symbol(
    symbol: str,
    bars: pd.DataFrame,
    cfg: TapeBacktestConfig,
    btc_in_bull: Optional[pd.Series],
    min_dollar_volume: Optional[float] = None,
) -> Optional[dict]:
    min_len = max(cfg.lookback, 20) + 15
    if len(bars) < min_len:
        return None

    atr_ser = compute_atr(bars["high"], bars["low"], bars["close"])
    raw_signals = detect_tape_signals(
        bars,
        lookback=cfg.lookback,
        proximity_pct=cfg.proximity_pct,
        volume_spike_mult=cfg.volume_spike_mult,
        bonus_pts=cfg.bonus_pts,
        ml_filter=None,
        two_phase_absorption=cfg.two_phase_absorption,
        two_phase_window=cfg.two_phase_window,
        two_phase_narrow_mult=cfg.two_phase_narrow_mult,
        cvd_filter=cfg.cvd_filter,
        cvd_window=cfg.cvd_window,
        stacked_bars=cfg.stacked_bars,
        enable_ask_absorption=cfg.enable_ask_absorption,
        enable_bid_repulsion=cfg.enable_bid_repulsion,
        enable_liquidity_sweep=cfg.enable_liquidity_sweep,
        sweep_window=cfg.sweep_window,
        sweep_volume_mult=cfg.sweep_volume_mult,
        enable_climax_exhaustion=cfg.enable_climax_exhaustion,
        climax_window=cfg.climax_window,
        climax_volume_mult=cfg.climax_volume_mult,
        climax_wide_mult=cfg.climax_wide_mult,
        enable_delta_divergence=cfg.enable_delta_divergence,
        enable_vwap_fade=cfg.enable_vwap_fade,
        vwap_window=cfg.vwap_window,
        vwap_stretch_pct=cfg.vwap_stretch_pct,
        vwap_volume_mult=cfg.vwap_volume_mult,
        enable_momentum_breakout=cfg.enable_momentum_breakout,
        breakout_margin_pct=cfg.breakout_margin_pct,
        breakout_volume_mult=cfg.breakout_volume_mult,
        breakout_dominance_ratio=cfg.breakout_dominance_ratio,
    )

    raw_feat = tape_features(bars, cfg.lookback)
    is_ask_mask = raw_signals["event"] == "ask_absorption"
    feat = select_tape_features(raw_feat, is_ask_mask)

    daily_trend_bull = _daily_trend_bull(bars, cfg.daily_trend_ema_period) if cfg.enable_daily_trend_filter else None
    btc_aligned = None
    if cfg.btc_regime_filter and btc_in_bull is not None:
        btc_aligned = btc_in_bull.reindex(bars.index, method="ffill").fillna(False).values.astype(bool)

    volume_ok = None
    if min_dollar_volume is not None:
        rolling_vol = bars["volume"].rolling(ROLLING_VOLUME_WINDOW_BARS, min_periods=ROLLING_VOLUME_WINDOW_BARS).mean().shift(1)
        volume_ok = (rolling_vol >= min_dollar_volume).fillna(False).values

    effective_signals = raw_signals.copy()
    effective_signals["is_setup"] = False
    effective_signals["risk_pct"] = cfg.risk_per_trade_pct

    return {
        "bars": bars,
        "atr_ser": atr_ser,
        "raw_signals": raw_signals,
        "effective_signals": effective_signals,
        "feat": feat,
        "daily_trend_bull": daily_trend_bull,
        "btc_in_bull": btc_aligned,
        "volume_ok": volume_ok,
        "state": _PositionState(),
        "cursor": min_len,
        "trailing_wins": deque(maxlen=TRAILING_WIN_RATE_WINDOW),
        "trailing_pnls": deque(maxlen=TRAILING_WIN_RATE_WINDOW),
        "cumulative_pnl_pct": 0.0,
        "disabled": False,
    }


def run_walk_forward(
    universe: Dict[str, pd.DataFrame],
    cfg: Optional[TapeBacktestConfig] = None,
    start: Optional[date] = None,
    end: Optional[date] = None,
    week_days: int = 7,
    ml_threshold: float = 0.5,
    blacklist_win_rate_threshold: Optional[float] = None,
    blacklist_min_samples: int = TRAILING_WIN_RATE_WINDOW,
    min_dollar_volume: Optional[float] = None,
    confidence_sizing: bool = False,
    max_symbol_loss_pct: Optional[float] = None,
) -> WalkForwardResult:
    """
    Run the expanding-window, weekly-retrained walk-forward backtest.

    `universe` maps symbol -> that symbol's full available bar history
    (different symbols may start at different dates — see module docstring).
    `start`/`end` default to the full overlap of all symbols' histories.

    `blacklist_win_rate_threshold`: a hard per-symbol cutoff layered on top
    of the soft `symbol_trailing_win_rate` model feature. That feature only
    *discounts* a struggling symbol's setups proportionally to one learned
    coefficient shared across every symbol — it can meaningfully reduce but
    not zero out a structurally bad name (confirmed empirically: ZEC/USDT's
    pass rate only dropped from 21.5% to 20.1% against the model alone, vs.
    13.8% for the rest of the universe). Setting this threshold blocks ANY
    setup for a symbol once its own trailing win rate (over its last
    `blacklist_min_samples` labelled setups, no lookahead) drops below it,
    regardless of what the model itself thinks of that particular setup.
    `None` disables this (default — matches prior behaviour exactly).

    `min_dollar_volume`: a hard liquidity floor (see ROLLING_VOLUME_WINDOW_BARS).
    Blocks any setup for a symbol while its trailing rolling dollar volume
    sits below this. `None` disables this (default).

    `confidence_sizing`: scale each trade's risk fraction by the model's own
    predicted win probability (see _confidence_scale) instead of using a
    fixed risk_per_trade for everything that passes the threshold. `False`
    disables this (default — matches prior behaviour exactly). Empirically
    this alone doesn't help a specific bad symbol (see max_symbol_loss_pct)
    because the model's passing probabilities cluster too close to the
    threshold to differentiate trades -- it mostly just de-levers everything
    uniformly.

    `max_symbol_loss_pct`: a hard, deterministic circuit breaker -- once a
    symbol's own realized cumulative pnl_pct (summed across its closed
    trades, same measure the loss report's per-symbol breakdown uses) drops
    to or below -max_symbol_loss_pct, no new entries are taken on that
    symbol for the rest of the run (any already-open position still exits
    normally). Unlike every soft/ML-driven lever above, this is the only
    mechanism that can actually *guarantee* no single symbol's realized loss
    exceeds the stated cap -- a probabilistic classifier can only shift
    odds, never enforce a hard ceiling. `None` disables this (default).
    """
    cfg = cfg or TapeBacktestConfig()
    btc_in_bull = _btc_in_bull_series(universe, cfg)

    precomp: Dict[str, dict] = {}
    for symbol, bars in universe.items():
        p = _precompute_symbol(symbol, bars, cfg, btc_in_bull, min_dollar_volume)
        if p is not None:
            precomp[symbol] = p

    if not precomp:
        raise ValueError("No symbol has enough history to run the walk-forward.")

    momentum_atr_mult = (
        cfg.momentum_atr_trailing_stop_mult
        if cfg.momentum_atr_trailing_stop_mult is not None
        else cfg.atr_trailing_stop_mult
    )

    all_start = min(p["bars"].index.min() for p in precomp.values()).date()
    all_end = max(p["bars"].index.max() for p in precomp.values()).date()
    start = start or all_start
    end = end or all_end

    current_pipeline = None
    training_rows: List[dict] = []
    all_trades: List[Trade] = []
    weekly_log: List[WeeklyLogEntry] = []

    week_start = start
    while week_start <= end:
        week_end_date = min(week_start + timedelta(days=week_days), end + timedelta(days=1))
        week_end_ts = pd.Timestamp(week_end_date, tz="UTC")

        active_symbols = 0
        new_trades_count = 0
        new_rows: List[dict] = []

        for symbol, p in precomp.items():
            bars = p["bars"]
            n = len(bars)
            hi = int(bars.index.searchsorted(week_end_ts))
            lo = p["cursor"]
            if hi <= lo or lo >= n - 1:
                continue

            active_symbols += 1
            raw_is_setup = p["raw_signals"]["is_setup"]
            week_raw_setup = raw_is_setup.iloc[lo:hi]

            if week_raw_setup.any():
                setup_positions = np.where(week_raw_setup.values)[0] + lo
                feat_rows = p["feat"].iloc[setup_positions].copy()

                # simulate_exit doesn't depend on the model at all, so labels
                # (and the deque update they feed) can be resolved in this
                # first, strictly-sequential pass -- each position's trailing
                # rate must reflect labels from strictly earlier positions of
                # this *same* symbol, including any already labelled earlier
                # this same week, not just the rate from before the week
                # started.
                trailing_rates = []
                trailing_pnls_feat = []
                trailing_samples = []
                labels = []
                for pos in setup_positions:
                    trailing_rates.append(_trailing_win_rate(p["trailing_wins"]))
                    trailing_pnls_feat.append(_trailing_avg_pnl(p["trailing_pnls"]))
                    trailing_samples.append(len(p["trailing_wins"]))
                    pnl = simulate_exit(bars, p["atr_ser"], int(pos), atr_mult=cfg.atr_trailing_stop_mult)
                    if pnl is None:
                        labels.append(None)
                        continue
                    win = int(pnl > 0)
                    p["trailing_wins"].append(win)
                    p["trailing_pnls"].append(pnl)
                    labels.append((win, pnl))
                feat_rows["symbol_trailing_win_rate"] = trailing_rates
                feat_rows["symbol_trailing_avg_pnl_pct"] = trailing_pnls_feat

                proba = _score_batch(current_pipeline, feat_rows)
                passed = proba >= ml_threshold

                if blacklist_win_rate_threshold is not None:
                    blacklisted = np.array([
                        samples >= blacklist_min_samples and rate < blacklist_win_rate_threshold
                        for samples, rate in zip(trailing_samples, trailing_rates)
                    ])
                    passed = passed & ~blacklisted

                if p["volume_ok"] is not None:
                    passed = passed & p["volume_ok"][setup_positions]

                if p["disabled"]:
                    passed = np.zeros(len(setup_positions), dtype=bool)

                p["effective_signals"].iloc[setup_positions, p["effective_signals"].columns.get_loc("is_setup")] = passed

                if confidence_sizing:
                    scale = _confidence_scale(proba, ml_threshold)
                    risk_col = p["effective_signals"].columns.get_loc("risk_pct")
                    p["effective_signals"].iloc[setup_positions, risk_col] = cfg.risk_per_trade_pct * scale

                for label_idx, pos in enumerate(setup_positions):
                    label = labels[label_idx]
                    if label is None:
                        continue
                    win, pnl = label
                    row = feat_rows.iloc[label_idx].to_dict()
                    row["win"] = win
                    row["symbol"] = symbol
                    # bars.index[pos] is the *signal* bar; the actual fill
                    # happens the bar after (mirroring _step_bar_range's own
                    # entry_bar = pos + 1) -- match that here so this column
                    # is directly joinable against Trade.entry_time instead
                    # of silently sitting one bar off from it.
                    row["entry_time"] = bars.index[int(pos) + 1]
                    row["pnl_pct"] = pnl
                    new_rows.append(row)

            new_trades = _step_bar_range(
                symbol, bars, p["atr_ser"], p["effective_signals"], cfg, momentum_atr_mult,
                p["state"], lo, hi, p["btc_in_bull"], p["daily_trend_bull"],
            )
            all_trades.extend(new_trades)
            new_trades_count += len(new_trades)
            p["cursor"] = hi

            if max_symbol_loss_pct is not None and not p["disabled"]:
                p["cumulative_pnl_pct"] += sum(t.pnl_pct for t in new_trades if not t.is_open)
                if p["cumulative_pnl_pct"] <= -max_symbol_loss_pct:
                    p["disabled"] = True

        training_rows.extend(new_rows)
        model_fitted = False
        if new_rows:
            training_df_so_far = pd.DataFrame(training_rows)
            refit = _fit_pipeline(training_df_so_far)
            if refit is not None:
                current_pipeline = refit
                model_fitted = True

        weekly_log.append(WeeklyLogEntry(
            week_start=pd.Timestamp(week_start, tz="UTC"),
            active_symbols=active_symbols,
            new_trades=new_trades_count,
            new_training_rows=len(new_rows),
            cumulative_training_rows=len(training_rows),
            model_fitted=model_fitted,
        ))

        week_start += timedelta(days=week_days)

    # close out any still-open positions at the end of their own history
    for symbol, p in precomp.items():
        state = p["state"]
        bars = p["bars"]
        if state.in_trade:
            price = float(bars["close"].iloc[-1])
            all_trades.append(Trade(
                symbol=symbol,
                entry_bar=state.entry_bar,
                entry_price=state.entry_price,
                exit_bar=len(bars) - 1,
                exit_price=price,
                exit_reason="end_of_data",
                pnl_pct=(price - state.entry_price) / state.entry_price * 100,
                holding_bars=len(bars) - 1 - state.entry_bar,
                is_open=True,
                signal_event=state.entry_event,
                entry_time=bars.index[state.entry_bar],
                risk_pct=state.entry_risk_pct,
            ))

    # Trades were appended in whatever order symbols happened to be
    # processed within each week, not true chronological order -- sort the
    # whole list (not just the completed subset fed to the equity curve) so
    # any consumer of `result.trades` (e.g. the loss-diagnostics report)
    # doesn't fall into the exact scrambled-order trap fixed in engine.py.
    all_trades.sort(key=lambda t: t.entry_time)
    completed = [t for t in all_trades if not t.is_open]
    equity = _build_equity_curve(completed, cfg.initial_capital, cfg.risk_per_trade_pct)
    pnls = [t.pnl_pct for t in completed]
    metrics = compute_all_metrics(equity, pnls) if pnls else {
        "total_return_pct": 0.0, "win_rate_pct": 0.0, "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0, "profit_factor": 0.0, "expectancy_pct": 0.0,
        "sortino_ratio": 0.0, "calmar_ratio": 0.0, "num_trades": 0,
    }

    return WalkForwardResult(
        trades=all_trades,
        equity_curve=equity,
        metrics=metrics,
        weekly_log=weekly_log,
        training_df=pd.DataFrame(training_rows),
        final_model=current_pipeline,
    )
