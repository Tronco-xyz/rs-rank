# ==============================================================================
# Streamlit Web App: IBD-Style RS Rank Scanner + ATR Stop Distance
# - Curated universes via CSVs in /universes
# - Session-only edits + CSV download
# ==============================================================================

import pandas as pd
import streamlit as st

from src.scanner import run_scan
from src.universes import load_all_universes

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="RS Rank Scanner",
    layout="wide",
)

st.title("IBD-Style RS Rank Scanner + ATR Stop")

# -----------------------------
# Load universes
# -----------------------------
@st.cache_data(show_spinner=False)
def _load_universes():
    return load_all_universes()

UNIVERSES = _load_universes()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Universe")

    universe_label = st.selectbox(
        "Choose universe",
        options=list(UNIVERSES.keys()),
        index=0,
    )

    universe = UNIVERSES[universe_label].copy()
    st.caption(f"{len(universe)} tickers loaded")

    st.divider()
    st.header("Scan settings")

    benchmark = st.text_input("Benchmark", value="SPY")
    top_n = st.number_input("Top N (0 = all)", min_value=0, max_value=2000, value=100, step=10)
    lookback_days = st.number_input("Lookback days", min_value=200, max_value=2000, value=600, step=50)

    st.subheader("IBD RS windows (trading days)")
    p12 = st.number_input("12M", value=252)
    p9  = st.number_input("9M", value=189)
    p6  = st.number_input("6M", value=126)
    p3  = st.number_input("3M", value=63)

    st.subheader("IBD weights")
    w12 = st.number_input("W12", value=0.20)
    w9  = st.number_input("W9", value=0.20)
    w6  = st.number_input("W6", value=0.20)
    w3  = st.number_input("W3", value=0.40)

    st.subheader("ATR")
    atr_window = st.number_input("ATR window", value=14)
    atr_mult = st.number_input("ATR multiplier", value=2.5)

    st.divider()
    run_btn = st.button("Run scan", type="primary", use_container_width=True)

# -----------------------------
# Run
# -----------------------------
if run_btn:
    with st.spinner("Downloading data and computing RS Rank..."):
        df, dropped = run_scan(
            universe=universe,
            benchmark=benchmark,
            lookback_days=int(lookback_days),
            top_n=None if top_n == 0 else int(top_n),
            p12=int(p12), p9=int(p9), p6=int(p6), p3=int(p3),
            w12=float(w12), w9=float(w9), w6=float(w6), w3=float(w3),
            atr_window=int(atr_window),
            atr_multiplier=float(atr_mult),
        )

    st.subheader("Results")
    st.caption(f"{universe_label} • Rows: {len(df)}")

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download results CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="rs_rank_results.csv",
        mime="text/csv",
    )

    st.subheader("Dropped tickers")
    if dropped.empty:
        st.success("None")
    else:
        st.dataframe(dropped, use_container_width=True, hide_index=True)
        st.download_button(
            "Download dropped CSV",
            data=dropped.to_csv(index=False).encode("utf-8"),
            file_name="dropped_tickers.csv",
            mime="text/csv",
        )
