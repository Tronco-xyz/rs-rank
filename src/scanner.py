from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# -----------------------------
# Helpers
# -----------------------------

def last_valid(s: pd.Series):
    s = s.dropna()
    return s.iloc[-1] if len(s) else np.nan


def pct_return(close: pd.DataFrame, periods: int) -> pd.Series:
    r = close.pct_change(periods)
    return r.iloc[-1]


def atr_wilder(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    """
    Wilder ATR using EWM.
    Robust against all-NaN slices.
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    # MultiIndex safe: concat keeps (date, ticker) aligned; groupby level=1 does max per ticker
    tr = pd.concat([tr1, tr2, tr3], axis=1)
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


# -----------------------------
# Core
# -----------------------------

def _download_ohlc(
    tickers: List[str],
    start_dt: datetime,
    end_dt: datetime,
    chunk_size: int = 25,      # más bajo para Streamlit Cloud
    max_retries: int = 4,      # reintentos por chunk
) -> pd.DataFrame:
    """
    Robust yfinance downloader for Streamlit Cloud:
    - Downloads tickers in chunks
    - Retries with backoff when Yahoo returns empty
    - Fallback per-ticker history() if download() keeps failing
    - Normalizes output to MultiIndex columns (Ticker, Field)
    """
    import time

    fields = {"Open", "High", "Low", "Close", "Volume"}

    def _normalize(data: pd.DataFrame, tickers_in_chunk: List[str]) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame()

        if not isinstance(data.columns, pd.MultiIndex):
            # single ticker collapsed format
            t = tickers_in_chunk[0] if len(tickers_in_chunk) == 1 else "SINGLE"
            data.columns = pd.MultiIndex.from_tuples(
                [(t, str(c)) for c in data.columns],
                names=["Ticker", "Field"],
            )
            return data.sort_index(axis=1)

        # MultiIndex could be (Field, Ticker) or (Ticker, Field)
        lvl0 = set(map(str, data.columns.get_level_values(0)))
        lvl1 = set(map(str, data.columns.get_level_values(1)))
        lvl0_is_fields = len(lvl0.intersection(fields)) >= 3
        lvl1_is_fields = len(lvl1.intersection(fields)) >= 3

        if lvl0_is_fields and not lvl1_is_fields:
            data = data.swaplevel(0, 1, axis=1)

        data.columns = pd.MultiIndex.from_tuples(
            [(str(t), str(f)) for t, f in data.columns],
            names=["Ticker", "Field"],
        )
        return data.sort_index(axis=1)

    def _download_chunk(chunk: List[str]) -> pd.DataFrame:
        # retry loop
        for attempt in range(max_retries):
            try:
                df = yf.download(
                    tickers=chunk,
                    start=start_dt.strftime("%Y-%m-%d"),
                    end=end_dt.strftime("%Y-%m-%d"),
                    auto_adjust=True,
                    group_by="column",
                    threads=False,     # importante en Streamlit Cloud
                    progress=False,
                )
                df = _normalize(df, chunk)
                if not df.empty:
                    return df
            except Exception:
                pass

            # backoff
            time.sleep(1.0 + attempt * 1.5)

        return pd.DataFrame()

    def _fallback_per_ticker(t: str) -> pd.DataFrame:
        # último recurso: history() por ticker
        try:
            h = yf.Ticker(t).history(
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                auto_adjust=True,
            )
            if h is None or h.empty:
                return pd.DataFrame()

            keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in h.columns]
            h = h[keep_cols].copy()
            h.columns = pd.MultiIndex.from_tuples([(t, c) for c in h.columns], names=["Ticker", "Field"])
            return h.sort_index(axis=1)
        except Exception:
            return pd.DataFrame()

    # clean tickers
    tickers_unique = list(dict.fromkeys([t.strip().upper() for t in tickers if t and str(t).strip()]))

    parts: List[pd.DataFrame] = []

    # 1) chunk download (con retries)
    for i in range(0, len(tickers_unique), chunk_size):
        chunk = tickers_unique[i:i + chunk_size]
        d = _download_chunk(chunk)
        if not d.empty:
            parts.append(d)

    # 2) si no hay nada, fallback por ticker (más lento, pero salva Cloud)
    if not parts:
        fb_parts: List[pd.DataFrame] = []
        for t in tickers_unique:
            d = _fallback_per_ticker(t)
            if not d.empty:
                fb_parts.append(d)
            time.sleep(0.2)  # pequeño throttle

        if not fb_parts:
            raise RuntimeError(
                "yfinance returned empty data for ALL tickers. "
                "En Streamlit Cloud suele ser rate-limit/bloqueo temporal de Yahoo. "
                "Reintenta más tarde o reduce el universo."
            )

        data = pd.concat(fb_parts, axis=1).sort_index(axis=1)
        return data

    # merge chunk results
    data = pd.concat(parts, axis=1).sort_index(axis=1)
    if not isinstance(data.columns, pd.MultiIndex):
        raise RuntimeError("Download normalization failed (expected MultiIndex).")
    return data



def _min_required_bars(p12: int, d1q: int, atr_window: int) -> int:
    return max(p12, d1q, atr_window) + 5


def run_scan(
    universe: List[str],
    benchmark: str = "SPY",
    lookback_days: int = 600,
    top_n: Optional[int] = None,
    # IBD windows / weights
    p12: int = 252, p9: int = 189, p6: int = 126, p3: int = 63,
    w12: float = 0.20, w9: float = 0.20, w6: float = 0.20, w3: float = 0.40,
    # Performance horizons
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

    # If some chunks returned partial columns, ensure fields exist before xs()
    needed_fields = {"Close", "High", "Low"}
    got_fields = set(map(str, data.columns.get_level_values(1)))
    missing = needed_fields.difference(got_fields)
    if missing:
        raise RuntimeError(f"Downloaded data missing required fields: {sorted(missing)}")

    close = data.xs("Close", level=1, axis=1)
    high  = data.xs("High",  level=1, axis=1)
    low   = data.xs("Low",   level=1, axis=1)

    min_bars = _min_required_bars(p12, d1q, atr_window)

    def valid_col(t: str) -> bool:
        return (
            t in close.columns and t in high.columns and t in low.columns and
            close[t].dropna().shape[0] >= min_bars and
            high[t].dropna().shape[0]  >= min_bars and
            low[t].dropna().shape[0]   >= min_bars
        )

    keep, drop = [], []
    for t in tickers:
        if t == benchmark:
            keep.append(t)
        elif valid_col(t):
            keep.append(t)
        else:
            drop.append(t)

    if benchmark not in keep or not valid_col(benchmark):
        raise RuntimeError(f"Benchmark '{benchmark}' has insufficient data.")

    dropped_df = (
        pd.DataFrame({"Ticker": drop, "Reason": "insufficient OHLC history / missing in download"})
        if drop else
        pd.DataFrame(columns=["Ticker", "Reason"])
    )

    close = close[keep]
    high  = high[keep]
    low   = low[keep]

    rs_raw  = ibd_rs_raw(close, benchmark, p12, p9, p6, p3, w12, w9, w6, w3)
    rs_rank = to_rs_rank_1_99(rs_raw)

    perf_1w = pct_return(close, d1w).rename("Perf_1W") * 100
    perf_1m = pct_return(close, d1m).rename("Perf_1M") * 100
    perf_1q = pct_return(close, d1q).rename("Perf_1Q") * 100

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

    atr_col = f"ATR_{atr_window}"
    df["Stop_ATR"] = df["Price"] - (df[atr_col] * atr_multiplier)
    df["Dist_to_ATR_%"] = (df[atr_col] * atr_multiplier / df["Price"]) * 100

    df = df.sort_values(["RS_Rank", "RS_Raw"], ascending=[False, False])
    df.insert(0, "Pos", range(1, len(df) + 1))

    if top_n is not None:
        df = df.head(int(top_n)).copy()

    df = df.rename_axis("Ticker").reset_index()

    for c in ["RS_Raw", "Price", atr_col, "Stop_ATR", "Dist_to_ATR_%", "Perf_1W", "Perf_1M", "Perf_1Q"]:
        df[c] = df[c].round(2 if c != "RS_Raw" else 6)

    df_out = df[[
        "Pos", "Ticker", "RS_Rank",
        "Perf_1W", "Perf_1M", "Perf_1Q",
        atr_col, "Stop_ATR", "Dist_to_ATR_%"
    ]].rename(columns={atr_col: "ATR_14" if atr_window == 14 else atr_col})

    return df_out, dropped_df
