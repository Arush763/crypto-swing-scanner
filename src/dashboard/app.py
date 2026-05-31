"""
Streamlit Dashboard for the Crypto Swing Trading Scanner.

Sections:
  1. Live Signals         — assets that fired an alert this scan cycle
  2. Top Overall          — full leaderboard sorted by final score
  3. Breakouts            — assets with confirmed breakout
  4. Retest Opportunities — assets retesting a prior breakout level
  5. Squeeze Candidates   — assets in or breaking out of compression
  6. Top Momentum         — highest momentum assets
  7. Volume Leaders       — highest volume-expansion assets
  8. Backtester           — run historical performance on cached data

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config.config import (
    DASHBOARD_REFRESH_SECONDS,
    DASHBOARD_TOP_N,
    SIGNAL_SCORE_THRESHOLD,
    BacktestConfig,
)
from src.scanner import Scanner, ScanResult
from src.backtesting.engine import run_backtest, optimise_parameters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Crypto Swing Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("⚙️ Scanner Settings")
    min_volume = st.number_input(
        "Min Daily Volume (USD)", min_value=1_000_000, max_value=500_000_000,
        value=5_000_000, step=1_000_000, format="%d",
    )
    score_threshold = st.slider(
        "Signal Score Threshold", min_value=50, max_value=100,
        value=int(SIGNAL_SCORE_THRESHOLD), step=1,
    )
    auto_refresh = st.checkbox("Auto-refresh every 5 minutes", value=False)
    run_scan = st.button("🔍 Run Scan Now", type="primary", use_container_width=True)

    st.divider()
    st.markdown("### Filters")
    show_breakouts_only = st.checkbox("Breakouts only", value=False)
    show_retests_only = st.checkbox("Retests only", value=False)
    show_squeezes_only = st.checkbox("Squeezes only", value=False)
    min_rr = st.slider("Min Risk-Reward", 0.0, 5.0, 1.5, 0.1)

    st.divider()
    st.caption("Crypto Swing Scanner v1.0")

# ---------------------------------------------------------------------------
# Session state — cache scan results
# ---------------------------------------------------------------------------

if "scan_result" not in st.session_state:
    st.session_state.scan_result = None
if "last_scan_time" not in st.session_state:
    st.session_state.last_scan_time = 0.0

# ---------------------------------------------------------------------------
# Helper: run and cache scan
# ---------------------------------------------------------------------------

def do_scan(min_vol: float, threshold: float) -> ScanResult:
    with st.spinner("Fetching data and scanning universe…"):
        scanner = Scanner(min_volume=min_vol, score_threshold=threshold)
        result = scanner.run()
    st.session_state.scan_result = result
    st.session_state.last_scan_time = time.time()
    return result


# Auto-refresh logic
if auto_refresh:
    elapsed = time.time() - st.session_state.last_scan_time
    if elapsed > DASHBOARD_REFRESH_SECONDS:
        do_scan(min_volume, score_threshold)

if run_scan:
    do_scan(min_volume, score_threshold)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

result: ScanResult | None = st.session_state.scan_result

st.title("📈 Crypto Swing Trading Scanner")

if result is None:
    st.info("Click **Run Scan Now** in the sidebar to fetch live data and generate signals.")
    st.stop()

# Header stats
col1, col2, col3, col4 = st.columns(4)
col1.metric("Assets Scanned", result.assets_scanned)
col2.metric("Signals Generated", len(result.signals))
col3.metric("Scan Duration", f"{result.duration_seconds}s")
col4.metric("Last Scan", result.timestamp.strftime("%H:%M:%S UTC"))

st.divider()

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tabs = st.tabs([
    "🚨 Signals",
    "🏆 Leaderboard",
    "💥 Breakouts",
    "🔄 Retests",
    "🌀 Squeezes",
    "🚀 Momentum",
    "📊 Volume",
    "🧪 Backtester",
])

# ── 1. Signals ──────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("🚨 Live Signals")
    sig_df = result.signal_table.copy()

    if sig_df.empty:
        st.warning("No signals generated this cycle. Lower the score threshold to see more.")
    else:
        # Apply sidebar filters
        if show_breakouts_only:
            sig_df = sig_df[sig_df["Type"] == "breakout"]
        if show_retests_only:
            sig_df = sig_df[sig_df["Type"] == "retest"]
        if show_squeezes_only:
            sig_df = sig_df[sig_df["Type"] == "squeeze_breakout"]
        sig_df = sig_df[sig_df["R:R"] >= min_rr]

        if sig_df.empty:
            st.info("No signals match current filters.")
        else:
            # Colour-code by strength
            def highlight_strength(row):
                colour = "#1f4e23" if row["Strength"] == "strong" else "#2e3a1f"
                return [f"background-color: {colour}"] * len(row)

            st.dataframe(
                sig_df.style.apply(highlight_strength, axis=1),
                use_container_width=True,
                height=400,
            )

            # Detailed card for selected signal
            selected_sym = st.selectbox("View signal detail", sig_df["Symbol"].tolist())
            row = sig_df[sig_df["Symbol"] == selected_sym].iloc[0]
            with st.expander(f"📋 {selected_sym} — Full Signal Detail", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Final Score", row["Score"])
                c2.metric("Signal Type", row["Type"].replace("_", " ").title())
                c3.metric("R:R Ratio", row["R:R"])

                c4, c5, c6 = st.columns(3)
                c4.metric("Entry Zone", f"{row['Entry Low']:.6g} – {row['Entry High']:.6g}")
                c5.metric("Stop Loss", f"{row['Stop Loss']:.6g}")
                c6.metric("Resistance", f"{row['Resistance']:.6g}")

                st.progress(int(row["Score"]), text=f"Score: {row['Score']}/100")

                score_data = pd.DataFrame({
                    "Category": ["Trend", "Momentum", "Liquidity", "Smart Money"],
                    "Score": [row["Trend"], row["Momentum"], row["Liquidity"], row["SmartMoney"]],
                })
                fig = px.bar(score_data, x="Category", y="Score", range_y=[0, 100],
                             color="Score", color_continuous_scale="RdYlGn",
                             title="Score Breakdown")
                st.plotly_chart(fig, use_container_width=True)


# ── 2. Leaderboard ──────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("🏆 Full Leaderboard")
    ranked = result.ranked_df

    sort_col = st.selectbox("Sort by", ["final_score", "momentum_score", "trend_score",
                                         "liquidity_score", "smart_money_score", "volume_ratio"])
    display_df = ranked.sort_values(sort_col, ascending=False).head(DASHBOARD_TOP_N)

    display_cols = [
        "symbol", "final_score", "trend_score", "momentum_score",
        "liquidity_score", "smart_money_score", "latest_price",
        "is_breakout", "is_retest", "is_squeeze",
    ]
    st.dataframe(display_df[display_cols], use_container_width=True, height=600)

    # Score distribution chart
    fig = px.histogram(ranked, x="final_score", nbins=20, title="Score Distribution",
                       labels={"final_score": "Final Score"})
    st.plotly_chart(fig, use_container_width=True)


# ── 3. Breakouts ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("💥 Confirmed Breakouts")
    bo_df = result.leaderboards.get("breakouts", pd.DataFrame())
    if bo_df.empty:
        st.info("No breakouts detected in this scan.")
    else:
        st.dataframe(bo_df.head(DASHBOARD_TOP_N), use_container_width=True)


# ── 4. Retests ───────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("🔄 Retest Opportunities")
    rt_df = result.leaderboards.get("retests", pd.DataFrame())
    if rt_df.empty:
        st.info("No retest setups detected in this scan.")
    else:
        st.dataframe(rt_df.head(DASHBOARD_TOP_N), use_container_width=True)


# ── 5. Squeezes ──────────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("🌀 Volatility Squeeze Candidates")
    sq_df = result.leaderboards.get("squeezes", pd.DataFrame())
    if sq_df.empty:
        st.info("No squeeze setups detected in this scan.")
    else:
        st.dataframe(sq_df.head(DASHBOARD_TOP_N), use_container_width=True)


# ── 6. Momentum ──────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("🚀 Top Momentum Assets")
    mom_df = result.leaderboards.get("top_momentum", pd.DataFrame())
    if not mom_df.empty:
        fig = px.bar(
            mom_df.head(15),
            x="symbol", y="momentum_score",
            color="momentum_score", color_continuous_scale="Viridis",
            title="Top 15 Momentum Assets",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(mom_df.head(DASHBOARD_TOP_N), use_container_width=True)


# ── 7. Volume Leaders ────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("📊 Highest Volume Expansion")
    vol_df = result.leaderboards.get("top_volume_growth", pd.DataFrame())
    if not vol_df.empty:
        fig = px.bar(
            vol_df.head(15),
            x="symbol", y="volume_ratio",
            color="volume_ratio", color_continuous_scale="Oranges",
            title="Top 15 Volume Expansion (ratio vs 30d avg)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(vol_df.head(DASHBOARD_TOP_N), use_container_width=True)


# ── 8. Backtester ─────────────────────────────────────────────────────────────
with tabs[7]:
    st.subheader("🧪 Strategy Backtester")
    st.markdown(
        "Run the scanner's signal logic against cached historical data "
        "to evaluate performance metrics."
    )

    with st.form("backtest_form"):
        bc1, bc2, bc3 = st.columns(3)
        bt_ema_short = bc1.number_input("EMA Short", 5, 50, 20)
        bt_ema_mid = bc2.number_input("EMA Mid", 20, 100, 50)
        bt_vol_mult = bc3.number_input("Volume Multiplier", 1.0, 5.0, 2.0, 0.1)
        bt_capital = st.number_input("Initial Capital (USD)", 1000, 1_000_000, 10_000, 1000)
        bt_risk = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5)
        run_bt = st.form_submit_button("▶ Run Backtest")

    if run_bt:
        # Build universe from scan result's raw data (re-use cached parquet)
        cache_dir = Path("data/cache")
        parquet_files = list(cache_dir.glob("*.parquet")) if cache_dir.exists() else []

        if not parquet_files:
            st.warning("No cached data found. Run a scan first to populate the data cache.")
        else:
            with st.spinner("Running backtest…"):
                bt_universe = {}
                for f in parquet_files[:50]:  # cap at 50 for speed
                    try:
                        df = pd.read_parquet(f)
                        symbol = f.stem.split("_", 1)[1].replace("_", "/")
                        bt_universe[symbol] = df
                    except Exception:
                        pass

                cfg = BacktestConfig(
                    ema_short=bt_ema_short,
                    ema_mid=bt_ema_mid,
                    volume_multiplier=bt_vol_mult,
                    initial_capital=float(bt_capital),
                    risk_per_trade_pct=bt_risk / 100,
                )
                bt_result = run_backtest(bt_universe, cfg)

            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Return", f"{bt_result.total_return_pct:.1f}%")
            m2.metric("Win Rate", f"{bt_result.win_rate:.1f}%")
            m3.metric("Sharpe Ratio", bt_result.sharpe_ratio)
            m4.metric("Max Drawdown", f"{bt_result.max_drawdown_pct:.1f}%")

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Profit Factor", bt_result.profit_factor)
            m6.metric("Avg Return/Trade", f"{bt_result.avg_return_pct:.2f}%")
            m7.metric("Avg Hold (bars)", bt_result.avg_holding_bars)
            m8.metric("Total Trades", bt_result.num_trades)

            # Equity curve
            eq = bt_result.equity_curve
            fig_eq = px.line(
                x=list(range(len(eq))), y=eq.tolist(),
                title="Equity Curve", labels={"x": "Trade #", "y": "Portfolio Value (USD)"},
            )
            fig_eq.add_hline(y=float(bt_capital), line_dash="dash", line_color="gray")
            st.plotly_chart(fig_eq, use_container_width=True)

            # Trade log
            if bt_result.trades:
                trade_rows = [
                    {
                        "Entry Bar": t.entry_bar,
                        "Entry Price": t.entry_price,
                        "Exit Bar": t.exit_bar,
                        "Exit Price": t.exit_price,
                        "P&L %": round(t.pnl_pct, 2),
                        "Hold Bars": t.holding_bars,
                        "Exit Reason": t.exit_reason,
                    }
                    for t in bt_result.trades
                ]
                st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, height=300)
