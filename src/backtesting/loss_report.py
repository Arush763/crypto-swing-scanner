"""
End-of-run "response system" — turns a WalkForwardResult into a plain-English
report of where the strategy lost money and why, instead of just a metrics
table. Three angles, cross-referenced against each other:

  1. Realized trades, broken down by signal event and by symbol — where did
     the actual losses come from.
  2. The cumulative training set (every raw setup seen, win/lose labelled
     independently of portfolio state) — which feature values separate
     winners from losers.
  3. The final model's own learned coefficients — what it concluded,
     in its own terms.

A cluster only gets called out as a real "losing point" if more than one of
these angles agrees with it (e.g. a feature with a wide win/loss mean gap
AND a sizeable model coefficient pointing the same direction) — single-angle
findings are reported as observations, not as the headline explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from src.backtesting.engine import Trade
from src.backtesting.walk_forward import WalkForwardResult, WALK_FORWARD_FEATURE_NAMES as FEATURE_NAMES

MIN_TRADES_FOR_BREAKDOWN = 3


@dataclass
class EventBreakdown:
    event: str
    num_trades: int
    win_rate_pct: float
    total_pnl_pct: float
    avg_pnl_pct: float


@dataclass
class SymbolBreakdown:
    symbol: str
    num_trades: int
    win_rate_pct: float
    total_pnl_pct: float


def _event_breakdown(trades: List[Trade]) -> List[EventBreakdown]:
    if not trades:
        return []
    df = pd.DataFrame([{"event": t.signal_event, "pnl_pct": t.pnl_pct} for t in trades])
    out = []
    for event, grp in df.groupby("event"):
        wins = (grp["pnl_pct"] > 0).sum()
        out.append(EventBreakdown(
            event=event,
            num_trades=len(grp),
            win_rate_pct=100.0 * wins / len(grp),
            total_pnl_pct=float(grp["pnl_pct"].sum()),
            avg_pnl_pct=float(grp["pnl_pct"].mean()),
        ))
    return sorted(out, key=lambda e: e.total_pnl_pct)


def _symbol_breakdown(trades: List[Trade]) -> List[SymbolBreakdown]:
    if not trades:
        return []
    df = pd.DataFrame([{"symbol": t.symbol, "pnl_pct": t.pnl_pct} for t in trades])
    out = []
    for symbol, grp in df.groupby("symbol"):
        wins = (grp["pnl_pct"] > 0).sum()
        out.append(SymbolBreakdown(
            symbol=symbol,
            num_trades=len(grp),
            win_rate_pct=100.0 * wins / len(grp),
            total_pnl_pct=float(grp["pnl_pct"].sum()),
        ))
    return sorted(out, key=lambda s: s.total_pnl_pct)


def _feature_separation(training_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every feature, the mean among losers vs. winners and the gap
    between them (in units of the feature's own overall standard
    deviation, so features on very different scales are comparable) —
    a quick read on which raw-setup characteristics most distinguish
    eventual winners from losers, independent of the model.
    """
    if training_df.empty or "win" not in training_df.columns:
        return pd.DataFrame()

    rows = []
    for feat in FEATURE_NAMES:
        if feat not in training_df.columns:
            continue
        overall_std = training_df[feat].std()
        if overall_std == 0 or pd.isna(overall_std):
            overall_std = 1.0
        win_mean = training_df.loc[training_df["win"] == 1, feat].mean()
        lose_mean = training_df.loc[training_df["win"] == 0, feat].mean()
        rows.append({
            "feature": feat,
            "win_mean": win_mean,
            "lose_mean": lose_mean,
            "gap_in_std": (win_mean - lose_mean) / overall_std,
        })
    return pd.DataFrame(rows).sort_values("gap_in_std", key=abs, ascending=False)


def _model_coefficients(model) -> pd.DataFrame:
    """
    LogisticRegression coefficients are in standardized-feature space (the
    pipeline's StandardScaler runs first) — already comparable across
    features without further normalization. Positive = pushes win
    probability up.
    """
    if model is None:
        return pd.DataFrame()
    clf = model.named_steps.get("clf")
    if clf is None or not hasattr(clf, "coef_"):
        return pd.DataFrame()
    coefs = clf.coef_[0]
    return pd.DataFrame({"feature": FEATURE_NAMES, "coefficient": coefs}).sort_values(
        "coefficient", key=abs, ascending=False
    )


def _narrate_event(eb: EventBreakdown) -> str:
    direction = "a net loser" if eb.total_pnl_pct < 0 else "marginal"
    return (
        f"  - {eb.event}: {eb.num_trades} trades, {eb.win_rate_pct:.1f}% win rate, "
        f"{eb.total_pnl_pct:+.1f}% total pnl ({eb.avg_pnl_pct:+.2f}% avg/trade) — {direction}."
    )


def _narrate_feature(row: pd.Series, coef_row: pd.Series | None) -> str:
    feat = row["feature"]
    gap = row["gap_in_std"]
    direction = "higher" if gap > 0 else "lower"
    text = (
        f"  - {feat}: winners average {direction} values than losers "
        f"({row['win_mean']:.4f} vs {row['lose_mean']:.4f}, a {abs(gap):.2f} std-dev gap)."
    )
    if coef_row is not None:
        agrees = (gap > 0) == (coef_row["coefficient"] > 0)
        if agrees:
            text += f" The model agrees — it weighted {feat} with coefficient {coef_row['coefficient']:+.3f} in the same direction."
        else:
            text += f" The model disagrees here (coefficient {coef_row['coefficient']:+.3f}) — treat this one with caution, it may not hold up out of sample."
    return text


def generate_loss_report(result: WalkForwardResult) -> str:
    completed = [t for t in result.trades if not t.is_open]
    losers = [t for t in completed if t.pnl_pct < 0]
    winners = [t for t in completed if t.pnl_pct >= 0]

    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("WALK-FORWARD LOSS-DIAGNOSTICS REPORT")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Total trades: {len(completed)}  |  Wins: {len(winners)}  |  Losses: {len(losers)}")
    lines.append(f"Total return: {result.metrics.get('total_return_pct', 0):.2f}%  |  "
                  f"Max drawdown: {result.metrics.get('max_drawdown_pct', 0):.2f}%  |  "
                  f"Sharpe: {result.metrics.get('sharpe_ratio', 0):.2f}")
    lines.append(f"Weeks simulated: {len(result.weekly_log)}  |  "
                  f"Cumulative training rows by end of run: {len(result.training_df)}")
    lines.append("")

    lines.append("-" * 78)
    lines.append("BY SIGNAL EVENT (worst first)")
    lines.append("-" * 78)
    event_rows = _event_breakdown(completed)
    if not event_rows:
        lines.append("  (no completed trades)")
    for eb in event_rows:
        lines.append(_narrate_event(eb))
    lines.append("")

    lines.append("-" * 78)
    lines.append("WORST SYMBOLS BY TOTAL PNL")
    lines.append("-" * 78)
    symbol_rows = [s for s in _symbol_breakdown(completed) if s.num_trades >= MIN_TRADES_FOR_BREAKDOWN]
    for sb in symbol_rows[:10]:
        lines.append(f"  - {sb.symbol}: {sb.num_trades} trades, {sb.win_rate_pct:.1f}% win rate, "
                      f"{sb.total_pnl_pct:+.1f}% total pnl")
    if not symbol_rows:
        lines.append(f"  (no symbol reached {MIN_TRADES_FOR_BREAKDOWN}+ trades)")
    lines.append("")

    lines.append("-" * 78)
    lines.append("FEATURE SEPARATION: WINNERS vs. LOSERS (all raw setups seen, not just acted-on trades)")
    lines.append("-" * 78)
    feat_df = _feature_separation(result.training_df)
    coef_df = _model_coefficients(result.final_model)
    coef_lookup = {row["feature"]: row for _, row in coef_df.iterrows()} if not coef_df.empty else {}
    if feat_df.empty:
        lines.append("  (no labelled training data)")
    for _, row in feat_df.iterrows():
        lines.append(_narrate_feature(row, coef_lookup.get(row["feature"])))
    lines.append("")

    lines.append("-" * 78)
    lines.append("FINAL MODEL'S LEARNED COEFFICIENTS (standardized-feature space)")
    lines.append("-" * 78)
    if coef_df.empty:
        lines.append("  (model never reached enough labelled data to fit)")
    else:
        for _, row in coef_df.iterrows():
            push = "raises" if row["coefficient"] > 0 else "lowers"
            lines.append(f"  - {row['feature']}: {row['coefficient']:+.3f} ({push} predicted win probability)")
    lines.append("")

    lines.append("-" * 78)
    lines.append("SUMMARY: MAIN LOSING POINTS")
    lines.append("-" * 78)
    summary_lines = _build_summary(event_rows, feat_df, coef_lookup)
    lines.extend(summary_lines if summary_lines else ["  (insufficient data to draw conclusions)"])
    lines.append("")

    return "\n".join(lines)


def _build_summary(event_rows: List[EventBreakdown], feat_df: pd.DataFrame, coef_lookup: dict) -> List[str]:
    out = []
    losing_events = [e for e in event_rows if e.total_pnl_pct < 0 and e.num_trades >= MIN_TRADES_FOR_BREAKDOWN]
    for e in losing_events:
        out.append(
            f"  - '{e.event}' setups lost money overall ({e.total_pnl_pct:+.1f}% across {e.num_trades} "
            f"trades, {e.win_rate_pct:.1f}% win rate) — the model should have learned to demand a higher "
            f"bar for this event type; check whether it's still being approved too often."
        )

    if not feat_df.empty:
        agreeing = []
        for _, row in feat_df.iterrows():
            coef_row = coef_lookup.get(row["feature"])
            if coef_row is None or abs(row["gap_in_std"]) < 0.15:
                continue
            agrees = (row["gap_in_std"] > 0) == (coef_row["coefficient"] > 0)
            if agrees:
                agreeing.append((row, coef_row))
        for row, coef_row in agreeing[:3]:
            # gap = win_mean - lose_mean: positive means *winners* skew higher,
            # which means it's the *lower* end of this feature that's
            # associated with losses (and vice versa) -- inverted from the
            # gap's own sign.
            loss_direction = "lower" if row["gap_in_std"] > 0 else "higher"
            out.append(
                f"  - {row['feature']} is a confirmed loss driver: both the raw win/loss split and the "
                f"model's own coefficient agree that {loss_direction} {row['feature']} predicts losses "
                f"(gap {row['gap_in_std']:+.2f} std-dev, model coefficient {coef_row['coefficient']:+.3f})."
            )

    return out
