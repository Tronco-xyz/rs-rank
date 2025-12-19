from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import time
import io

import numpy as np
import pandas as pd
import yfinance as yf
import requests


# =============================================================================
# Helpers
# =============================================================================

def last_valid(s: pd.Series):
    s = s.dropna()
    return s.iloc[-1] if len(s) else np.nan


def pct_return(close: pd.DataFrame, periods: int) -> pd.Series:
    r = close.pct_change(periods)
    return r.iloc[-1]


def atr_wilder(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1)
    tr = tr.groupby(level=1, axis=1).max()

    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


# =============================================================================
# RS
# =============================================================================

def ibd_rs_raw(
    close: pd.DataFrame,
    benchmark: str,
    p12: int, p9: int, p6: int, p3: int,
    w12: float, w9: float, w6: float, w3: float,
) -> pd.Series:

    r12 = close.pct_change(p12)
    r9  = close.pct_change(p9)
    r6  = close.pct_change(p6)
    r3  = close.pct_change(p3)

    m12 = r12[benchmark].iloc[-1]
    m9  = r9[benchmark].iloc[-1]
    m6  = r6[benchmark].iloc[-1]
    m3  = r3[benchmark].iloc[-1]

    rs = (
        w3  * (r3.iloc[-1]  - m3) +
        w6  * (r6.iloc[-1]  - m6) +
        w9  * (r9.iloc[-1]  - m9) +
        w12 * (r12.iloc[-1] - m12)
    )

    rs = rs.drop(labels=[benchmark], errors="ignore").dropna()
    rs.name = "RS_Raw"
    return rs


def to_rs_rank_1_99(rs_raw: pd.Series) -> pd.Series:
    rs_rank = (rs_raw.rank(pct=True) * 99).round().clip(1, 99).astype(int)
    rs_rank.name = "RS_Rank"
    return rs_rank


# =============================================================================
# DATA DOWNLOAD – STOOQ (fallback-first for Streamlit Cloud)
# =============================================================================

def _stooq_symbol(ticker: str) -> str:
    return ticker.lower()


def _fetch_stooq_ohlc(ticker: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={_stooq_symbol(ticker)}&i=d"
    r = requests.get(url, timeout=20)
    if r.status_code != 200 or "Date" not in r.text:
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(r.text))
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    df.columns = [c.capitalize() for c in df.columns]

    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            return pd.DataFrame()

    if "Volume" not in df.columns:
        df["Volume"] = np.nan

    return df[["Open", "High", "Low", "Close", "Volume"]]


def _download_ohlc_stooq(
    tickers: List[str],
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:

    frames = []
    for t in tickers:
        df = _fetch_stooq_ohlc(t)
        if not df.empty:
            df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]
            df.columns = pd.MultiIndex.from_product([[t], df.columns], names=["Ticker", "Field"])
            frames.append(df)
        time.sleep(0.2)

    if not frames:
        raise RuntimeError("STOOQ returned no usable data for any ticker.")

    return pd.concat(frames, axis=1).sort_index(axis=1)


# =============================================================================
# MAIN SCAN
# =============================================================================

def run_scan(
    universe: List[str],
    benchmark: str = "SPY",
    lookback_days: int = 600,
    top_n: Optional[int] = None,
    p12: int = 252, p9: int = 189, p6: int = 126, p3: int = 63,
    w12: float = 0.20, w9: float = 0.20, w6: float = 0.20, w3: float = 0.40,
    d1w: int = 5, d1m: int = 21, d1q: int = 63,
    atr_window: int = 14,
    atr_multiplier: float = 2.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    benchmark = benchmark.upper()
    tickers = sorted(set([t.upper() for t in universe] + [benchmark]))

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=int(lookback_days))

    # 🔴 Yahoo es inestable en Streamlit Cloud → usamos STOOQ directamente
    data = _download_ohlc_stooq(tickers, start_dt, end_dt)

    close = data.xs("Close", level=1, axis=1)
    high  = data.xs("High",  level=1, axis=1)
    low   = data.xs("Low",   level=1, axis=1)

    rs_raw  = ibd_rs_raw(close, benchmark, p12, p9, p6, p3, w12, w9, w6, w3)
    rs_rank = to_rs_rank_1_99(rs_raw)

    perf_1w = pct_return(close, d1w).rename("Perf_1W") * 100
    perf_1m = pct_return(close, d1m).rename("Perf_1M") * 100
    perf_1q = pct_return(close, d1q).rename("Perf_1Q") * 100

    atr = atr_wilder(high, low, close, window=atr_window)
    atr_last = atr.apply(last_valid).rename("ATR_14")
    price_last = close.apply(last_valid).rename("Price")

    df = pd.DataFrame({
        "RS_Rank": rs_rank,
        "Price": price_last,
        "Perf_1W": perf_1w,
        "Perf_1M": perf_1m,
        "Perf_1Q": perf_1q,
        "ATR_14": atr_last,
    }).dropna()

    df["Stop_ATR"] = df["Price"] - df["ATR_14"] * atr_multiplier
    df["Dist_to_ATR_%"] = (df["ATR_14"] * atr_multiplier / df["Price"]) * 100

    df = df.sort_values("RS_Rank", ascending=False)
    df.insert(0, "Pos", range(1, len(df) + 1))

    if top_n:
        df = df.head(top_n)

    df = df.reset_index().rename(columns={"index": "Ticker"})

    dropped = pd.DataFrame(columns=["Ticker", "Reason"])
    return df, dropped
