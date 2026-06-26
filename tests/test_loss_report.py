"""Unit tests for the walk-forward loss-diagnostics report."""

import pandas as pd

from src.backtesting.loss_report import _build_summary, _feature_separation


def test_feature_separation_gap_sign_matches_winners_minus_losers():
    # winners have LOWER distance_pct than losers -> gap should be negative
    training_df = pd.DataFrame({
        "is_ask": [0, 0, 0, 0],
        "distance_pct": [0.01, 0.01, 0.05, 0.05],
        "dominance_margin": [1.0, 1.0, 1.0, 1.0],
        "move_strength_pct": [1.0, 1.0, 1.0, 1.0],
        "volume_ratio": [1.0, 1.0, 1.0, 1.0],
        "win": [1, 1, 0, 0],
    })
    result = _feature_separation(training_df)
    row = result[result["feature"] == "distance_pct"].iloc[0]
    assert row["gap_in_std"] < 0
    assert row["win_mean"] < row["lose_mean"]


def test_summary_loss_direction_is_inverted_from_gap_sign():
    # gap > 0 (winners skew higher) must be reported as "lower X predicts
    # losses", not "higher X predicts losses" -- regression test for a sign
    # bug where the summary said the opposite of what the win/loss split
    # actually showed.
    feat_df = pd.DataFrame([{"feature": "dominance_margin", "win_mean": 2.0, "lose_mean": 1.0, "gap_in_std": 0.5}])
    coef_lookup = {"dominance_margin": pd.Series({"feature": "dominance_margin", "coefficient": 0.3})}
    summary = _build_summary([], feat_df, coef_lookup)
    assert any("lower dominance_margin predicts losses" in line for line in summary)
    assert not any("higher dominance_margin predicts losses" in line for line in summary)


def test_summary_skips_features_below_significance_threshold():
    feat_df = pd.DataFrame([{"feature": "volume_ratio", "win_mean": 1.0, "lose_mean": 1.01, "gap_in_std": 0.05}])
    coef_lookup = {"volume_ratio": pd.Series({"feature": "volume_ratio", "coefficient": 0.3})}
    summary = _build_summary([], feat_df, coef_lookup)
    assert summary == []
