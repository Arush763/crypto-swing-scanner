"""
Train the ML signal filter (src.modules.signal_filter) on real per-signal
trade outcomes across the live universe's historical tape data.

For every bar where the existing tape_signal rule would fire (either
ask_absorption or bid_repulsion), this script computes the shared
FEATURE_NAMES and labels it by simulating the same 2-ATR trailing-stop exit
the backtest engine uses (src.backtesting.engine.simulate_exit), independent
of the one-trade-at-a-time state machine so every setup gets its own label.

Usage:
    python scripts/train_signal_filter.py --days 365
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd


def build_dataset(days: int, max_workers: int) -> pd.DataFrame:
    from src.data.fetcher import MarketDataFetcher
    from src.data.trade_tape import TradeTapeFetcher, dedupe_by_binance_symbol
    from src.config.config import TAPE_LEVEL_LOOKBACK, TAPE_PROXIMITY_PCT, TAPE_VOLUME_SPIKE_MULT
    from src.indicators.volatility import atr as compute_atr
    from src.modules.tape_signal import detect_tape_signals
    from src.modules.signal_filter import tape_features, select_tape_features
    from src.backtesting.engine import simulate_exit

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=days)

    logger.info("Resolving live universe symbol list…")
    raw_symbols = MarketDataFetcher().list_universe_symbols()
    symbols = dedupe_by_binance_symbol(raw_symbols)
    logger.info("%d universe symbols -> %d unique Binance tickers", len(raw_symbols), len(symbols))

    fetcher = TradeTapeFetcher()
    logger.info("Fetching %d days of tick data for %d symbols (concurrent)…", days, len(symbols))
    bars_by_symbol = fetcher.fetch_many_bars(symbols, start, end, timeframe="4h", max_workers=max_workers)

    rows = []
    for symbol, bars in bars_by_symbol.items():
        if len(bars) < TAPE_LEVEL_LOOKBACK + 10:
            continue

        signals = detect_tape_signals(bars)
        is_setup = signals["is_setup"]
        if not is_setup.any():
            continue

        is_ask_mask = signals["event"] == "ask_absorption"
        raw_feat = tape_features(bars, TAPE_LEVEL_LOOKBACK)
        feat = select_tape_features(raw_feat, is_ask_mask)

        high, low, close = bars["high"], bars["low"], bars["close"]
        atr_ser = compute_atr(high, low, close)

        setup_idxs = np.where(is_setup.values)[0]
        for i in setup_idxs:
            pnl = simulate_exit(bars, atr_ser, i)
            if pnl is None:
                continue
            row = feat.iloc[i].to_dict()
            row["symbol"] = symbol
            row["timestamp"] = bars.index[i]
            row["pnl_pct"] = pnl
            row["win"] = int(pnl > 0)
            rows.append(row)

    df = pd.DataFrame(rows).dropna()
    logger.info("Built %d labeled setups", len(df))
    return df


def train(df: pd.DataFrame, model_path: Path) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, classification_report
    from src.modules.signal_filter import FEATURE_NAMES
    import joblib

    df = df.sort_values("timestamp").reset_index(drop=True)
    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    logger.info(
        "Train/test split: %d train (%s -> %s) / %d test (%s -> %s)",
        len(train_df), train_df["timestamp"].min(), train_df["timestamp"].max(),
        len(test_df), test_df["timestamp"].min(), test_df["timestamp"].max(),
    )

    X_train, y_train = train_df[FEATURE_NAMES], train_df["win"]
    X_test, y_test = test_df[FEATURE_NAMES], test_df["win"]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    pipeline.fit(X_train, y_train)

    proba_test = pipeline.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)
    logger.info("Held-out test AUC: %.3f", roc_auc_score(y_test, proba_test))
    print(classification_report(y_test, pred_test, target_names=["loss", "win"]))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info("Saved model to %s", model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ML signal filter")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--model-path", default="data/models/signal_filter.joblib")
    args = parser.parse_args()

    df = build_dataset(args.days, args.workers)
    if df.empty:
        print("No labeled setups built — aborting.")
        return
    train(df, Path(args.model_path))


if __name__ == "__main__":
    main()
