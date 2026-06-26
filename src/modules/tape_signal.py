"""
Tape-Based Wall Signal — historical proxy for the live order-book wall signal.

src.modules.wall_signal classifies absorption/repulsion from resting L2
depth, which is only ever observable live (no free historical archive of
full order-book depth exists). For backtesting, this module infers the same
concept from free historical trade-tick data instead: a burst of aggressor
(taker) volume clustering at a recent swing level is treated as the
"wall" being tested, and judged by what price does next:

  - ask_absorption : price was testing a swing-high level, a burst of
                      sell-side aggressor volume hit it (supply being sold
                      into), and price still pushed through with buyers
                      remaining dominant -> the supply got absorbed.
  - bid_repulsion   : price was testing a swing-low level, a burst of
                      sell-side aggressor volume hit it, and price bounced
                      away without breaking the level -> selling got
                      absorbed by resting demand and rejected.

This is a proxy, not the live signal itself — useful only for backtesting
the strategy's edge, since live trading uses the L2-based wall_signal.py.

Variant flags (all default to the original single-pass behaviour, so
existing callers/tests are unaffected):

  - two_phase_absorption : require a distinct stall bar at resistance
                            (high volume, narrow range) followed within
                            `two_phase_window` bars by a separate
                            confirmation bar that closes through the level
                            with buy-side delta dominant. Targets the
                            exhaustion/absorption confusion in the original
                            single-bar rule, where a bar that simply spiked
                            volume and closed up looks identical whether
                            supply was genuinely absorbed or the move was
                            already exhausted.
  - cvd_filter            : gate ask_absorption on a rising cumulative
                            volume delta (net buying pressure building into
                            the breakout) and gate bid_repulsion on delta
                            having been falling into the test and then
                            turning up on the bounce bar.
  - stacked_bars          : instead of a single bar of elevated one-sided
                            volume, require this many *consecutive* prior
                            bars to each show one-sided (sell-dominant)
                            volume. 1 reproduces the original single-bar
                            check.
  - enable_ask_absorption / enable_bid_repulsion : drop one event type
                            entirely (e.g. to isolate bid_repulsion, which
                            historically carries the strategy's edge).

Additional independent (long-only) order-flow event types, each gated by
its own enable flag and defaulting to off:

  - liquidity_sweep    : price wicks below a known support level (sweeping
                          resting stop-losses) on a volume spike, then
                          closes back above it within `sweep_window` bars —
                          fades a failed breakdown rather than confirming a
                          continuation, which tends to win more often than
                          absorption/repulsion at the cost of being rarer.
  - climax_exhaustion   : an extreme-volume, wide-range, weak-closing bar
                          that prints a new swing low (a capitulation/selling
                          climax), followed within `climax_window` bars by a
                          bar that reclaims the climax bar's high — fades
                          exhaustion instead of confirming continuation.
  - delta_divergence    : price prints a new swing low (breaks support) but
                          cumulative volume delta does not make a
                          correspondingly deep new low over the same
                          lookback window (less aggressive selling beneath
                          the new price low), and the bar turns up — a
                          leading reversal signal.
  - vwap_fade           : price stretches `vwap_stretch_pct` or more below
                          its rolling volume-weighted average price on
                          elevated volume, then turns up — a mean-reversion
                          fade back toward the volume-weighted reference.
                          Works best range-bound, weaker in strong trends.
  - momentum_breakout   : structurally different from every signal above —
                          those all bet a level *holds* (repulsion, climax,
                          vwap_fade) or that supply gets *absorbed* into a
                          hold (absorption); this bets a breakout *keeps
                          running*. Close clears resistance by at least
                          `breakout_margin_pct` (not just within proximity
                          of it) on a volume spike with buyers strongly
                          dominant (`breakout_dominance_ratio`x). Intended to
                          be paired with a much wider trailing stop than the
                          mean-reversion family, to let the trend run instead
                          of cutting it short.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.config.config import (
    TAPE_LEVEL_LOOKBACK,
    TAPE_PROXIMITY_PCT,
    TAPE_VOLUME_SPIKE_MULT,
    TAPE_SIGNAL_BONUS,
    TAPE_SWEEP_WINDOW,
    TAPE_SWEEP_VOLUME_MULT,
    TAPE_CLIMAX_WINDOW,
    TAPE_CLIMAX_VOLUME_MULT,
    TAPE_CLIMAX_WIDE_MULT,
    TAPE_VWAP_WINDOW,
    TAPE_VWAP_STRETCH_PCT,
    TAPE_VWAP_VOLUME_MULT,
    TAPE_BREAKOUT_MARGIN_PCT,
    TAPE_BREAKOUT_VOLUME_MULT,
    TAPE_BREAKOUT_DOMINANCE_RATIO,
    ML_FILTER_THRESHOLD,
)
from src.modules.signal_filter import SignalFilter, tape_features, select_tape_features


@dataclass
class TapeSignalResult:
    is_setup: bool
    event: str          # "ask_absorption" | "bid_repulsion" | "none"
    level_price: float
    bonus_score: float


def _stacked_one_sided_spike(
    elevated_and_dominant: pd.Series, stacked_bars: int
) -> pd.Series:
    """
    True on bar i if each of the `stacked_bars` bars immediately before i
    (i.e. i-stacked_bars .. i-1) independently satisfied
    `elevated_and_dominant` — replaces a single elevated-volume bar with a
    run of consecutive one-sided bars (stronger evidence the level is
    genuinely being leaned on, not just hit by one outlier print).
    """
    counts = elevated_and_dominant.rolling(stacked_bars).sum()
    return (counts.shift(1) >= stacked_bars).fillna(False)


def detect_tape_signals(
    bars: pd.DataFrame,
    lookback: int = TAPE_LEVEL_LOOKBACK,
    proximity_pct: float = TAPE_PROXIMITY_PCT,
    volume_spike_mult: float = TAPE_VOLUME_SPIKE_MULT,
    bonus_pts: float = TAPE_SIGNAL_BONUS,
    ml_filter: Optional[SignalFilter] = None,
    ml_threshold: float = ML_FILTER_THRESHOLD,
    two_phase_absorption: bool = False,
    two_phase_window: int = 5,
    two_phase_narrow_mult: float = 0.7,
    cvd_filter: bool = False,
    cvd_window: int = 5,
    stacked_bars: int = 1,
    enable_ask_absorption: bool = True,
    enable_bid_repulsion: bool = True,
    enable_liquidity_sweep: bool = False,
    sweep_window: int = TAPE_SWEEP_WINDOW,
    sweep_volume_mult: float = TAPE_SWEEP_VOLUME_MULT,
    enable_climax_exhaustion: bool = False,
    climax_window: int = TAPE_CLIMAX_WINDOW,
    climax_volume_mult: float = TAPE_CLIMAX_VOLUME_MULT,
    climax_wide_mult: float = TAPE_CLIMAX_WIDE_MULT,
    enable_delta_divergence: bool = False,
    enable_vwap_fade: bool = False,
    vwap_window: int = TAPE_VWAP_WINDOW,
    vwap_stretch_pct: float = TAPE_VWAP_STRETCH_PCT,
    vwap_volume_mult: float = TAPE_VWAP_VOLUME_MULT,
    enable_momentum_breakout: bool = False,
    breakout_margin_pct: float = TAPE_BREAKOUT_MARGIN_PCT,
    breakout_volume_mult: float = TAPE_BREAKOUT_VOLUME_MULT,
    breakout_dominance_ratio: float = TAPE_BREAKOUT_DOMINANCE_RATIO,
) -> pd.DataFrame:
    """
    Vectorised over the whole bar series (fast enough for backtesting many
    assets/days). `bars` must have columns: open, high, low, close, volume,
    buy_volume, sell_volume (see src.data.trade_tape.resample_to_bars).

    Returns a DataFrame aligned to `bars.index` with columns:
    is_setup, event, level_price, bonus_score.

    If `ml_filter` is given and trained, every raw setup (either event type)
    is additionally scored and only kept if its predicted win probability
    clears `ml_threshold` — see src.modules.signal_filter for why.

    See module docstring for the variant flags.
    """
    close = bars["close"]
    high  = bars["high"]
    low   = bars["low"]
    volume     = bars["volume"]
    buy_volume = bars["buy_volume"]
    sell_volume = bars["sell_volume"]

    # Level known *before* the current bar — excludes it to avoid lookahead
    resistance = high.rolling(lookback).max().shift(1)
    support    = low.rolling(lookback).min().shift(1)
    avg_volume = volume.rolling(lookback).mean().shift(1)

    prev_close = close.shift(1)
    prev_sell  = sell_volume.shift(1)

    # --- One-sided volume dominance test: single elevated bar (baseline),
    # or a run of `stacked_bars` consecutive sell-dominant bars ---
    if stacked_bars <= 1:
        sell_spike = prev_sell >= (avg_volume * volume_spike_mult / 2.0)
    else:
        elevated_and_dominant = (
            (sell_volume >= avg_volume * volume_spike_mult / 2.0) & (sell_volume > buy_volume)
        )
        sell_spike = _stacked_one_sided_spike(elevated_and_dominant, stacked_bars)

    # --- Ask absorption: tested resistance from below, heavy sell-side
    # tape hit it, price still closed through with buyers still dominant ---
    dist_to_res = (resistance - prev_close) / resistance
    tested_res  = (dist_to_res >= 0) & (dist_to_res <= proximity_pct)
    pushed_through = close >= resistance * 0.999
    buy_dominant = buy_volume > sell_volume

    if two_phase_absorption:
        bar_range_pct = (high - low) / close
        avg_range = bar_range_pct.rolling(lookback).mean().shift(1)
        narrow_range = bar_range_pct <= (avg_range * two_phase_narrow_mult)

        dist_high_to_res = (resistance - high).abs() / resistance
        tested_res_self = (dist_high_to_res <= proximity_pct) & resistance.notna()
        stall_volume_spike = volume >= (avg_volume * volume_spike_mult)

        stall_bar = tested_res_self & narrow_range & stall_volume_spike
        # A stall in any of the `two_phase_window` bars strictly before the
        # current one (current bar is the candidate confirmation bar).
        stall_recent = (
            stall_bar.rolling(two_phase_window).sum().shift(1) > 0
        ).fillna(False)

        ask_absorption = stall_recent & pushed_through & buy_dominant & resistance.notna()
    else:
        ask_absorption = tested_res & sell_spike & pushed_through & buy_dominant & resistance.notna()

    # --- Bid repulsion: tested support from above, heavy sell-side tape
    # hit it, price bounced away without breaking the level ---
    dist_to_sup  = (prev_close - support) / support
    tested_sup   = (dist_to_sup >= 0) & (dist_to_sup <= proximity_pct)
    bounced      = close > prev_close
    held_support = close > support
    bid_repulsion = tested_sup & sell_spike & bounced & held_support & support.notna() & ~ask_absorption

    if cvd_filter:
        cvd = (buy_volume - sell_volume).cumsum()
        rising_cvd = cvd > cvd.shift(cvd_window)
        ask_absorption = ask_absorption & rising_cvd.fillna(False)

        was_falling_into_test = cvd.shift(1) < cvd.shift(cvd_window + 1)
        now_reversing_up = cvd > cvd.shift(1)
        bid_repulsion = bid_repulsion & (was_falling_into_test & now_reversing_up).fillna(False)

    if not enable_ask_absorption:
        ask_absorption = ask_absorption & False
    if not enable_bid_repulsion:
        bid_repulsion = bid_repulsion & False
    bid_repulsion = bid_repulsion & ~ask_absorption

    sell_volume_safe = sell_volume.where(sell_volume > 0, 1.0)
    abs_strength = ((buy_volume - sell_volume) / sell_volume_safe).clip(lower=0, upper=1).fillna(0)
    rep_strength = (((close - prev_close) / prev_close) / proximity_pct).clip(lower=0, upper=1).fillna(0)

    bonus_absorption = bonus_pts * (0.6 + 0.4 * abs_strength)
    bonus_repulsion   = bonus_pts * (0.6 + 0.4 * rep_strength)

    claimed = ask_absorption | bid_repulsion

    # --- Liquidity sweep: wick below support on a volume spike (resting
    # stops swept), then close back above it within `sweep_window` bars ---
    liquidity_sweep = pd.Series(False, index=bars.index)
    sweep_strength = pd.Series(0.0, index=bars.index)
    if enable_liquidity_sweep:
        sweep_bar = (low < support) & (volume >= avg_volume * sweep_volume_mult) & support.notna()
        swept_recently = sweep_bar.rolling(sweep_window).max().astype(bool)
        liquidity_sweep = swept_recently & (close > support) & support.notna() & ~claimed
        sweep_strength = (((close - support) / support) / proximity_pct).clip(lower=0, upper=1).fillna(0)
        claimed = claimed | liquidity_sweep

    # --- Climax exhaustion: extreme-volume wide-range red bar prints a new
    # swing low (capitulation), then a later bar reclaims its high ---
    climax_exhaustion = pd.Series(False, index=bars.index)
    climax_strength = pd.Series(0.0, index=bars.index)
    if enable_climax_exhaustion:
        bar_range_pct = (high - low) / close
        avg_range = bar_range_pct.rolling(lookback).mean().shift(1)
        wide_range = bar_range_pct >= (avg_range * climax_wide_mult)
        extreme_volume = volume >= (avg_volume * climax_volume_mult)
        new_low = low <= support
        red_bar = close < bars["open"]
        climax_bar = wide_range & extreme_volume & new_low & red_bar & support.notna()

        climax_recent_high = (
            high.where(climax_bar).rolling(climax_window, min_periods=1).max().shift(1)
        )
        climax_exhaustion = climax_recent_high.notna() & (close > climax_recent_high) & ~claimed
        climax_strength = (((close - climax_recent_high) / climax_recent_high) / proximity_pct).clip(
            lower=0, upper=1
        ).fillna(0)
        claimed = claimed | climax_exhaustion

    # --- Delta divergence: a new swing low in price that cumulative delta
    # doesn't confirm (delta stays above its own recent low), then the bar
    # turns up ---
    delta_divergence = pd.Series(False, index=bars.index)
    divergence_strength = pd.Series(0.0, index=bars.index)
    if enable_delta_divergence:
        cvd = (buy_volume - sell_volume).cumsum()
        cvd_min_lookback = cvd.rolling(lookback).min().shift(1)
        is_new_low = (low <= support) & support.notna()
        bullish_divergence = is_new_low & (cvd > cvd_min_lookback)
        turning_up = close > prev_close
        delta_divergence = bullish_divergence & turning_up & ~claimed
        cvd_range = (cvd - cvd_min_lookback).abs().replace(0, 1.0)
        divergence_strength = ((cvd - cvd_min_lookback) / cvd_range).clip(lower=0, upper=1).fillna(0)
        claimed = claimed | delta_divergence

    # --- VWAP fade: price stretched below its rolling volume-weighted
    # average on elevated volume, then turns up ---
    vwap_fade = pd.Series(False, index=bars.index)
    vwap_fade_strength = pd.Series(0.0, index=bars.index)
    if enable_vwap_fade:
        vwap = (close * volume).rolling(vwap_window).sum() / volume.rolling(vwap_window).sum()
        stretch_pct = (vwap - close) / vwap
        stretched_below = stretch_pct >= vwap_stretch_pct
        elevated_volume = volume >= (avg_volume * vwap_volume_mult)
        turning_up = close > prev_close
        vwap_fade = stretched_below & elevated_volume & turning_up & vwap.notna() & ~claimed
        vwap_fade_strength = (stretch_pct / (vwap_stretch_pct * 2)).clip(lower=0, upper=1).fillna(0)
        claimed = claimed | vwap_fade

    # --- Momentum breakout: close clears resistance by a real margin (not
    # just within proximity of it) on a volume spike with buyers strongly
    # dominant — bets the breakout keeps running, not that a level holds ---
    momentum_breakout = pd.Series(False, index=bars.index)
    breakout_strength = pd.Series(0.0, index=bars.index)
    if enable_momentum_breakout:
        sell_volume_safe_bo = sell_volume.where(sell_volume > 0, 1e-9)
        cleared_resistance = close >= resistance * (1 + breakout_margin_pct)
        breakout_volume_spike = volume >= (avg_volume * breakout_volume_mult)
        strongly_dominant = buy_volume >= sell_volume_safe_bo * breakout_dominance_ratio
        momentum_breakout = (
            cleared_resistance & breakout_volume_spike & strongly_dominant & resistance.notna() & ~claimed
        )
        breakout_strength = (((close - resistance) / resistance) / (breakout_margin_pct * 2)).clip(
            lower=0, upper=1
        ).fillna(0)
        claimed = claimed | momentum_breakout

    is_setup = claimed

    if ml_filter is not None and ml_filter.is_trained and is_setup.any():
        raw_feat = tape_features(bars, lookback)
        feat = select_tape_features(raw_feat, ask_absorption)
        win_proba = pd.Series(0.0, index=bars.index)
        win_proba[is_setup] = ml_filter.score_batch(feat[is_setup])
        is_setup = is_setup & (win_proba >= ml_threshold)
        ask_absorption = ask_absorption & is_setup
        bid_repulsion = bid_repulsion & is_setup
        liquidity_sweep = liquidity_sweep & is_setup
        climax_exhaustion = climax_exhaustion & is_setup
        delta_divergence = delta_divergence & is_setup
        vwap_fade = vwap_fade & is_setup
        momentum_breakout = momentum_breakout & is_setup

    event = pd.Series("none", index=bars.index)
    event[ask_absorption] = "ask_absorption"
    event[bid_repulsion] = "bid_repulsion"
    event[liquidity_sweep] = "liquidity_sweep"
    event[climax_exhaustion] = "climax_exhaustion"
    event[delta_divergence] = "delta_divergence"
    event[vwap_fade] = "vwap_fade"
    event[momentum_breakout] = "momentum_breakout"

    level_price = pd.Series(0.0, index=bars.index)
    level_price[ask_absorption] = resistance[ask_absorption]
    level_price[bid_repulsion] = support[bid_repulsion]
    level_price[liquidity_sweep] = support[liquidity_sweep]
    level_price[climax_exhaustion] = support[climax_exhaustion]
    level_price[delta_divergence] = support[delta_divergence]
    if enable_vwap_fade:
        level_price[vwap_fade] = vwap[vwap_fade]
    level_price[momentum_breakout] = resistance[momentum_breakout]

    bonus_score = pd.Series(0.0, index=bars.index)
    bonus_score[ask_absorption] = bonus_absorption[ask_absorption]
    bonus_score[bid_repulsion] = bonus_repulsion[bid_repulsion]
    bonus_score[liquidity_sweep] = (bonus_pts * (0.6 + 0.4 * sweep_strength))[liquidity_sweep]
    bonus_score[climax_exhaustion] = (bonus_pts * (0.6 + 0.4 * climax_strength))[climax_exhaustion]
    bonus_score[delta_divergence] = (bonus_pts * (0.6 + 0.4 * divergence_strength))[delta_divergence]
    bonus_score[vwap_fade] = (bonus_pts * (0.6 + 0.4 * vwap_fade_strength))[vwap_fade]
    bonus_score[momentum_breakout] = (bonus_pts * (0.6 + 0.4 * breakout_strength))[momentum_breakout]

    return pd.DataFrame({
        "is_setup": is_setup.fillna(False),
        "event": event,
        "level_price": level_price.round(8),
        "bonus_score": bonus_score.round(2),
    })
