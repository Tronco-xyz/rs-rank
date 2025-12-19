# FILE: src/scanner.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf


def last_valid(s: pd.Series):
    s = s.dropna()
    return s.iloc[-1] if len(s) else np.nan


def pct_return(close: pd.DataFrame, periods: int) -> pd.Series:
    # Return last row of pct_change(periods)
    r = close.pct_change(periods)
    return r.iloc[-1]


def atr_wilder(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Wilder ATR using EWM.
    Fixes 'All-NaN slice encountered' by using pandas concat + max across axis=1.
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    # robust max across the three TR candidates
    tr = pd.concat([tr1, tr2, tr3], axis=1, keys=["tr1", "tr2", "tr3"])
    # reshape back: index x columns, max over the "tr*" level
    tr = tr.groupby(level=1, axis=1).max()

    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def ibd_rs_raw(
    close: pd.DataFrame,
    benchmark: str,
    p12: int, p9: int, p6: int, p3: int,
    w12: float, w9: float, w6: float, w3: float,
) -> pd.Series:
    if benchmark not in close.columns:
        raise ValueError(f"Benchmark '{benchmark}' not found in Close data.")

    r12 = close.pct_change(p12)
    r9 = close.pct_change(p9)
    r6 = close.pct_change(p6)
    r3 = close.pct_change(p3)

    m12 = r12[benchmark].iloc[-1]
    m9 = r9[benchmark].iloc[-1]
    m6 = r6[benchmark].iloc[-1]
    m3 = r3[benchmark].iloc[-1]

    rs = (
        w3 * (r3.iloc[-1] - m3) +
        w6 * (r6.iloc[-1] - m6) +
        w9 * (r9.iloc[-1] - m9) +
        w12 * (r12.iloc[-1] - m12)
    )

    rs = rs.drop(labels=[benchmark], errors="ignore").dropna()
    rs.name = "RS_Raw"
    return rs


def to_rs_rank_1_99(rs_raw: pd.Series) -> pd.Series:
    rs_rank = (rs_raw.rank(pct=True) * 99).round().clip(1, 99).astype(int)
    rs_rank.name = "RS_Rank"
    return rs_rank


def _download_ohlc(
    tickers: List[str],
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    data = yf.download(
        tickers=tickers,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        auto_adjust=True,
        group_by="column",
        threads=True,
        progress=False,
    )

    if data is None or data.empty or not isinstance(data.columns, pd.MultiIndex):
        raise RuntimeError("Download failed or unexpected yfinance format (expected OHLC MultiIndex).")

    # Normalize columns to (Ticker, Field)
    return data.swaplevel(axis=1).sort_index(axis=1)


def _min_required_bars(p12: int, d1q: int, atr_window: int) -> int:
    # Need enough bars for the longest lookback + ATR warm-up
    return max(p12, d1q, atr_window) + 5


def run_scan(
    universe: List[str],
    benchmark: str = "SPY",
    lookback_days: int = 600,
    top_n: Optional[int] = None,
    # IBD windows/weights
    p12: int = 252, p9: int = 189, p6: int = 126, p3: int = 63,
    w12: float = 0.20, w9: float = 0.20, w6: float = 0.20, w3: float = 0.40,
    # Perf horizons
    d1w: int = 5, d1m: int = 21, d1q: int = 63,
    # ATR
    atr_window: int = 14,
    atr_multiplier: float = 2.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not universe or len(universe) < 2:
        raise ValueError("Universe is empty or too small.")

    benchmark = benchmark.strip().upper()
    tickers = sorted(set([t.strip().upper() for t in universe] + [benchmark]))

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=int(lookback_days))

    data = _download_ohlc(tickers, start_dt, end_dt)

    close = data.xs("Close", level=1, axis=1)
    high = data.xs("High", level=1, axis=1)
    low = data.xs("Low", level=1, axis=1)

    # Drop tickers with insufficient OHLC history (prevents yfinance tz issues / NaN ATR)
    min_bars = _min_required_bars(p12, d1q, atr_window)

    def valid_col(col: str) -> bool:
        # Must have enough non-NaN in Close/High/Low
        c_ok = close[col].dropna().shape[0] >= min_bars
        h_ok = high[col].dropna().shape[0] >= min_bars
        l_ok = low[col].dropna().shape[0] >= min_bars
        return bool(c_ok and h_ok and l_ok)

    keep = []
    drop = []
    for t in tickers:
        if t == benchmark:
            keep.append(t)
            continue
        if t in close.columns and t in high.columns and t in low.columns and valid_col(t):
            keep.append(t)
        else:
            drop.append(t)

    # Ensure benchmark still ok
    if benchmark not in keep or not valid_col(benchmark):
        raise RuntimeError(f"Benchmark '{benchmark}' has insufficient history or missing data.")

    if drop:
        dropped_df = pd.DataFrame({"Ticker": drop, "Reason": "insufficient OHLC history"})
    else:
        dropped_df = pd.DataFrame(columns=["Ticker", "Reason"])

    # Filter data to keep
    close = close[keep]
    high = high[keep]
    low = low[keep]

    # RS
    rs_raw = ibd_rs_raw(close, benchmark, p12, p9, p6, p3, w12, w9, w6, w3)
    rs_rank = to_rs_rank_1_99(rs_raw)

    # Perf (absolute)
    perf_1w = pct_return(close, d1w).rename("Perf_1W") * 100
    perf_1m = pct_return(close, d1m).rename("Perf_1M") * 100
    perf_1q = pct_return(close, d1q).rename("Perf_1Q") * 100

    # ATR
    atr = atr_wilder(high, low, close, window=atr_window)
    atr_last = atr.apply(last_valid).rename(f"ATR_{atr_window}")

    price_last = close.apply(last_valid).rename("Price")

    idx = rs_raw.index.intersection(price_last.index)

    df = pd.DataFrame({
        "RS_Rank": rs_rank.reindex(idx),
        "RS_Raw": rs_raw.reindex(idx),
        "Price": price_last.reindex(idx),
        "Perf_1W": perf_1w.reindex(idx),
        "Perf_1M": perf_1m.reindex(idx),
        "Perf_1Q": perf_1q.reindex(idx),
        f"ATR_{atr_window}": atr_last.reindex(idx),
    }).dropna(subset=["RS_Rank", "Price", f"ATR_{atr_window}"])

    # ATR stop & distance
    atr_col = f"ATR_{atr_window}"
    df["Stop_ATR"] = df["Price"] - (df[atr_col] * atr_multiplier)
    df["Dist_to_ATR_%"] = (df[atr_col] * atr_multiplier / df["Price"]) * 100

    # Sort
    df = df.sort_values(["RS_Rank", "RS_Raw"], ascending=[False, False])

    # Pos
    df.insert(0, "Pos", range(1, len(df) + 1))

    # Limit
    if top_n is not None:
        df = df.head(int(top_n)).copy()

    # Presentation
    df = df.rename_axis("Ticker").reset_index()

    df["RS_Raw"] = df["RS_Raw"].round(6)
    df["Price"] = df["Price"].round(2)
    df[atr_col] = df[atr_col].round(2)
    df["Stop_ATR"] = df["Stop_ATR"].round(2)
    df["Dist_to_ATR_%"] = df["Dist_to_ATR_%"].round(2)
    df["Perf_1W"] = df["Perf_1W"].round(2)
    df["Perf_1M"] = df["Perf_1M"].round(2)
    df["Perf_1Q"] = df["Perf_1Q"].round(2)

    # Final columns (requested)
    df_out = df[[
        "Pos", "Ticker", "RS_Rank",
        "Perf_1W", "Perf_1M", "Perf_1Q",
        atr_col, "Stop_ATR", "Dist_to_ATR_%"
    ]].rename(columns={atr_col: "ATR_14" if atr_window == 14 else atr_col})

    return df_out, dropped_df

