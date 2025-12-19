# ==============================================================================
# Streamlit Web App: IBD-Style RS Rank Scanner + ATR Stop Distance
# - Universe presets stored as CSV in /universes (manual updates via GitHub)
# - Supports session-only edits + dropped tickers table
# - Works: local / Streamlit Cloud
# ==============================================================================

import pandas as pd
import streamlit as st

from src.scanner import run_scan
from src.universes import (
    PRESET_FILES,
    load_universe_csv,
    parse_ticker_text,
    normalize_tickers,
    quick_validate_tickers,
)

st.set_page_config(page_title="RS Rank Scanner", layout="wide")
st.title("IBD-Style RS Rank Scanner + ATR Stop")

# -----------------------------
# Sidebar: Universe + Settings
# -----------------------------
with st.sidebar:
    st.header("Universe")

    preset = st.selectbox(
        "Choose universe",
        options=["Mega Caps", "S&P 500", "ETFs", "Custom (paste/upload)"],
        index=0,
    )

    st.divider()
    st.header("Scan Settings")

    benchmark = st.text_input("Benchmark", value="SPY")

    top_n = st.number_input("Top N (0 = All)", min_value=0, max_value=2000, value=0, step=10)
    lookback_days = st.number_input("Lookback (calendar days)", min_value=200, max_value=2000, value=600, step=50)

    st.subheader("IBD RS windows (trading days)")
    p12 = st.number_input("P12", min_value=126, max_value=400, value=252, step=1)
    p9  = st.number_input("P9",  min_value=90,  max_value=300, value=189, step=1)
    p6  = st.number_input("P6",  min_value=63,  max_value=260, value=126, step=1)
    p3  = st.number_input("P3",  min_value=21,  max_value=200, value=63, step=1)

    st.subheader("IBD weights (must sum ~ 1)")
    w12 = st.number_input("W12", min_value=0.0, max_value=1.0, value=0.20, step=0.05, format="%.2f")
    w9  = st.number_input("W9",  min_value=0.0, max_value=1.0, value=0.20, step=0.05, format="%.2f")
    w6  = st.number_input("W6",  min_value=0.0, max_value=1.0, value=0.20, step=0.05, format="%.2f")
    w3  = st.number_input("W3",  min_value=0.0, max_value=1.0, value=0.40, step=0.05, format="%.2f")

    st.subheader("ATR")
    atr_window = st.number_input("ATR Window", min_value=5, max_value=50, value=14, step=1)
    atr_mult = st.number_input("ATR Multiplier", min_value=0.5, max_value=10.0, value=2.5, step=0.1, format="%.1f")

    st.divider()
    st.header("Performance horizons (trading days)")
    d1w = st.number_input("1W", min_value=1, max_value=20, value=5, step=1)
    d1m = st.number_input("1M", min_value=5, max_value=60, value=21, step=1)
    d1q = st.number_input("1Q", min_value=21, max_value=126, value=63, step=1)

    st.divider()
    validate_toggle = st.checkbox("Quick-validate tickers (fast)", value=True)
    run_btn = st.button("Run scan", type="primary", use_container_width=True)

# -----------------------------
# Load/Build Universe
# -----------------------------
universe = []
universe_source_label = ""

if preset in PRESET_FILES:
    universe_source_label = f"Preset: {preset}"
    universe = load_universe_csv(PRESET_FILES[preset])

    st.caption(f"Loaded {len(universe)} tickers — {universe_source_label}")

    with st.expander("Update this universe (session-only)"):
        st.write("Paste tickers (comma or newline). This updates only your current session.")
        txt = st.text_area("Tickers", value="\n".join(universe), height=160)

        colA, colB, colC = st.columns([1, 1, 1])
        with colA:
            if st.button("Update universe (this session)"):
                universe = normalize_tickers(parse_ticker_text(txt))
                st.success(f"Universe updated for this session: {len(universe)} tickers")
        with colB:
            csv_bytes = ("Ticker\n" + "\n".join(universe) + "\n").encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name=f"{preset.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with colC:
            st.write("To persist changes: replace the CSV in your GitHub repo and push.")

else:
    universe_source_label = "Custom"
    st.caption("Custom universe: paste tickers or upload a CSV with column 'Ticker'.")

    col1, col2 = st.columns([2, 1])
    with col1:
        txt = st.text_area("Paste tickers (comma or newline)", value="", height=180)
    with col2:
        uploaded = st.file_uploader("Upload CSV (Ticker column)", type=["csv"])

    if uploaded is not None:
        dfu = pd.read_csv(uploaded)
        if "Ticker" not in dfu.columns:
            st.error("CSV must contain a 'Ticker' column.")
        else:
            universe = dfu["Ticker"].astype(str).tolist()
    else:
        universe = parse_ticker_text(txt)

    universe = normalize_tickers(universe)
    st.caption(f"Custom universe tickers: {len(universe)}")

# -----------------------------
# Run Scan
# -----------------------------
if run_btn:
    if len(universe) < 2:
        st.error("Universe is empty or too small. Provide at least 2 tickers.")
        st.stop()

    invalid = []
    if validate_toggle:
        with st.spinner("Quick-validating tickers..."):
            invalid = quick_validate_tickers(universe, max_batch=200)

        if invalid:
            st.warning(f"Invalid/unavailable tickers (removed): {invalid}")
            universe = [t for t in universe if t not in invalid]

    params = dict(
        benchmark=benchmark,
        lookback_days=int(lookback_days),
        top_n=None if int(top_n) == 0 else int(top_n),
        # IBD windows/weights
        p12=int(p12), p9=int(p9), p6=int(p6), p3=int(p3),
        w12=float(w12), w9=float(w9), w6=float(w6), w3=float(w3),
        # Perf horizons
        d1w=int(d1w), d1m=int(d1m), d1q=int(d1q),
        # ATR
        atr_window=int(atr_window),
        atr_multiplier=float(atr_mult),
    )

    with st.spinner("Downloading prices and computing RS/ATR..."):
        df_out, dropped = run_scan(universe=universe, **params)

    st.subheader("Results")
    st.caption(f"{universe_source_label} • Universe used: {len(universe)} • Rows: {len(df_out)}")

    st.dataframe(
        df_out,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Perf_1W": st.column_config.NumberColumn(format="%.2f"),
            "Perf_1M": st.column_config.NumberColumn(format="%.2f"),
            "Perf_1Q": st.column_config.NumberColumn(format="%.2f"),
            "ATR_14": st.column_config.NumberColumn(format="%.2f"),
            "Stop_ATR": st.column_config.NumberColumn(format="%.2f"),
            "Dist_to_ATR_%": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.download_button(
        "Download results CSV",
        data=df_out.to_csv(index=False).encode("utf-8"),
        file_name="rs_rank_results.csv",
        mime="text/csv",
    )

    st.subheader("Dropped tickers")
    if dropped.empty:
        st.success("None")
    else:
        st.dataframe(dropped, use_container_width=True, hide_index=True)
        st.download_button(
            "Download dropped tickers CSV",
            data=dropped.to_csv(index=False).encode("utf-8"),
            file_name="dropped_tickers.csv",
            mime="text/csv",
        )
