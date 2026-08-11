"""
Does OKX's long/short ratio track the Binance series the signal was fitted on?

The crowd-short signal was validated entirely on Binance's historical futures
metrics. Live trading has to read OKX, because Binance's live API is
geo-blocked here and its archive is a day stale. That substitution is an
assumption, and an unverified assumption sitting under a live strategy is
exactly the kind of thing that looks fine until it costs money.

Two properties matter, and they are different:

  LEVEL correlation   whether the two venues report similar absolute ratios.
                      Largely irrelevant — the signal ranks each symbol
                      against its own history, so a constant offset cancels.
  RANK correlation    whether a reading that is extreme on Binance is also
                      extreme on OKX. This is the one the signal depends on
                      entirely. If it is weak, the live signal fires at
                      different times than the backtested one and the
                      historical result does not transfer.

Also reported: how often the two venues agree on the top-decile call, which is
the actual trading decision.

Usage:
    python scripts/compare_venue_ratios.py
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.ERROR, format="%(levelname)s  %(message)s")
logger = logging.getLogger("venue")
logger.setLevel(logging.INFO)

import numpy as np
import pandas as pd

from src.config.config import MAJOR_BASES
from src.data.positioning import PositioningFetcher

BINANCE_METRICS = ROOT / "data" / "cache" / "study" / "futures_metrics_365d_12s.pkl"


def main() -> None:
    if not BINANCE_METRICS.exists():
        print(f"Binance metrics cache missing: {BINANCE_METRICS}")
        return
    with BINANCE_METRICS.open("rb") as fh:
        binance = pickle.load(fh)

    fetcher = PositioningFetcher(state_path=None)
    symbols = [f"{b}/USDT" for b in MAJOR_BASES[:12]]

    print()
    print("=" * 92)
    print("  OKX vs BINANCE long/short account ratio — overlapping window")
    print("  Rank correlation is what matters: the signal ranks within symbol, so a")
    print("  constant level offset between venues is harmless but a rank mismatch is fatal.")
    print("=" * 92)
    print(f"{'symbol':<12}{'overlap':>9}{'okx mean':>11}{'bnb mean':>11}"
          f"{'level r':>10}{'RANK r':>9}{'top-decile agree':>19}")
    print("-" * 92)

    rank_rs, agrees = [], []
    for symbol in symbols:
        snap = fetcher.fetch(symbol, period="1H")
        if snap is None:
            print(f"{symbol:<12}{'no okx data':>9}")
            continue

        okx = pd.concat([snap.history, pd.Series([snap.long_short_ratio],
                                                 index=[snap.history.index[-1] + pd.Timedelta(hours=1)])]) \
            if not snap.history.empty else pd.Series(dtype=float)

        bnb_df = binance.get(symbol)
        if bnb_df is None or "count_long_short_ratio" not in bnb_df.columns:
            print(f"{symbol:<12}{'no binance data':>9}")
            continue
        bnb = bnb_df["count_long_short_ratio"].dropna()
        bnb = bnb.resample("1h").last().dropna()

        joined = pd.concat([okx.rename("okx"), bnb.rename("bnb")], axis=1).dropna()
        if len(joined) < 50:
            print(f"{symbol:<12}{len(joined):>9}   (too little overlap)")
            continue

        level_r = float(joined["okx"].corr(joined["bnb"]))
        rank_r = float(joined["okx"].corr(joined["bnb"], method="spearman"))

        okx_top = joined["okx"] >= joined["okx"].quantile(0.90)
        bnb_top = joined["bnb"] >= joined["bnb"].quantile(0.90)
        agree = float((okx_top == bnb_top).mean())

        rank_rs.append(rank_r)
        agrees.append(agree)
        print(f"{symbol:<12}{len(joined):>9}{joined['okx'].mean():>11.3f}"
              f"{joined['bnb'].mean():>11.3f}{level_r:>10.3f}{rank_r:>9.3f}{agree:>18.1%}")

    print("=" * 92)
    print()
    if rank_rs:
        mean_rank = float(np.mean(rank_rs))
        mean_agree = float(np.mean(agrees))
        print(f"  Mean rank correlation:      {mean_rank:.3f}")
        print(f"  Mean top-decile agreement:  {mean_agree:.1%}")
        print()
        if mean_rank >= 0.7:
            print("  -> OKX tracks Binance closely enough to carry the signal live.")
        elif mean_rank >= 0.4:
            print("  -> Partial agreement. The live signal will fire at somewhat different")
            print("     times than the backtest; expect degraded but related performance.")
        else:
            print("  -> WEAK. OKX measures a materially different population. The backtested")
            print("     result does NOT transfer to an OKX-driven live signal — either source")
            print("     Binance data another way, or re-validate the signal on OKX history.")
    print()


if __name__ == "__main__":
    main()
