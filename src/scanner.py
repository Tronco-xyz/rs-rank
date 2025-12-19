from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Tuple, Optional
import time
import io

import numpy as np
import pandas as pd
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

    if benchmark not in close.columns:
        raise RuntimeError(
            f"Benchmark '{benchmark}' not found in downloaded data. "
            f"Available columns example: {list(close.columns)[:10]}"
        )

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
# STOOQ download (robust symbol resolution)
# =============================================================================

def _stooq_candidates(ticker: str) -> List[str]:
    """
    Stooq often uses:
      - US equities/ETFs: <ticker>.us   (e.g. spy.us, aapl.us)
      - Some symbols may prefer '.' instead of '-' (e.g. brk.b.us)
    We'll try several candidates in order.
    """
    t = (ticker or "").strip().lower()
    if not t:
        return []

    t_dot = t.replace("-", ".")
    cands = []

    # Most common for US
    cands.append(f"{t}.us")
    if t_dot != t:
        cands.append(f"{t_dot}.us")

    # Fallbacks (sometimes without .us)
    cands.append(t)
    if t_dot != t:
        cands.append(t_dot)

    # de-dupe preserving order
    out = []
    seen = set()
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _fetch_stooq_ohlc_for_symbol(symbol: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    r = requests.get(url, timeout=20)
    if r.status_code != 200 or "Date" not in r.text:
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(r.text))
    if df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

    # normalize cols
    df.columns = [str(c).capitalize() for c in df.columns]

    needed = {"Open", "High", "Low", "Close"}
    if not needed.issubset(set(df.columns)):
        return pd.DataFrame()

    if "Volume" not in df.columns:
        df["Volume"] = np.nan

    return df[["Open", "High", "Low", "Close", "Volume"]]


def _fetch_stooq_ohlc(ticker: str) -> pd.DataFrame:
    for sym in _stooq_candidates(ticker):
        df = _fetch_stooq_ohlc_for_symbol(sym)
        if not df.empty:
            return df
    return pd.DataFrame()


def _download_ohlc_stooq(
    tickers: List[str],
    start_dt: datetime,
    end_dt: datetime,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Returns:
      - OHLC dataframe with MultiIndex columns (Ticker, Field)
      - list of tickers that failed on Stooq
    """
    frames = []
    failed = []

    for t in tickers:
        df = _fetch_stooq_ohlc(t)
        if df.empty:
            failed.append(t)
        else:
            df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]
            df.columns = pd.MultiIndex.from_product([[t], df.columns], names=["Ticker", "Field"])
            frames.append(df)

        time.sleep(0.15)

    if not frames:
        raise RuntimeError("STOOQ returned no usable data for any ticker.")

    data = pd.concat(frames, axis=1).sort_index(axis=1)
    return data, failed


# =============================================================================
# MAIN
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

    if not universe or len(universe) < 2:
        raise ValueError("Universe is empty or too small.")

    benchmark = (benchmark or "").strip().upper()
    tickers = sorted(set([t.strip().upper() for t in universe if str(t).strip()] + [benchmark]))

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=int(lookback_days))

    data, failed = _download_ohlc_stooq(tickers, start_dt, end_dt)

    # Required fields
    close = data.xs("Close", level=1, axis=1)
    high  = data.xs("High",  level=1, axis=1)
    low   = data.xs("Low",   level=1, axis=1)

    # If benchmark failed or missing, stop with a clear message
    if benchmark not in close.columns:
        raise RuntimeError(
            f"Benchmark '{benchmark}' missing from data (likely unsupported by Stooq symbol lookup). "
            f"Try benchmark='SPY' (default) or 'QQQ'. Failed tickers: {failed[:20]}"
        )

    # RS
    rs_raw  = ibd_rs_raw(close, benchmark, p12, p9, p6, p3, w12, w9, w6, w3)
    rs_rank = to_rs_rank_1_99(rs_raw)

    # Perf
    perf_1w = pct_return(close, d1w).rename("Perf_1W") * 100
    perf_1m = pct_return(close, d1m).rename("Perf_1M") * 100
    perf_1q = pct_return(close, d1q).rename("Perf_1Q") * 100

    # ATR
    atr = atr_wilder(high, low, close, window=atr_window)
    atr_last = atr.apply(last_valid).rename("ATR_14")
    price_last = close.apply(last_valid).rename("Price")

    idx = rs_raw.index.intersection(price_last.index)

    df = pd.DataFrame({
        "RS_Rank": rs_rank.reindex(idx),
        "RS_Raw": rs_raw.reindex(idx),
        "Price": price_last.reindex(idx),
        "Perf_1W": perf_1w.reindex(idx),
        "Perf_1M": perf_1m.reindex(idx),
        "Perf_1Q": perf_1q.reindex(idx),
        "ATR_14": atr_last.reindex(idx),
    }).dropna()

    df["Stop_ATR"] = df["Price"] - df["ATR_14"] * atr_multiplier
    df["Dist_to_ATR_%"] = (df["ATR_14"] * atr_multiplier / df["Price"]) * 100

    df = df.sort_values(["RS_Rank", "RS_Raw"], ascending=[False, False])
    df.insert(0, "Pos", range(1, len(df) + 1))

    if top_n is not None:
        df = df.head(int(top_n)).copy()

    df = df.rename_axis("Ticker").reset_index()

    # rounding
    df["RS_Raw"] = df["RS_Raw"].round(6)
    for c in ["Price", "ATR_14", "Stop_ATR", "Dist_to_ATR_%", "Perf_1W", "Perf_1M", "Perf_1Q"]:
        df[c] = df[c].round(2)

    df_out = df[[
        "Pos", "Ticker", "RS_Rank",
        "Perf_1W", "Perf_1M", "Perf_1Q",
        "ATR_14", "Stop_ATR", "Dist_to_ATR_%"
    ]]

    # Dropped/failed report
    dropped = pd.DataFrame(
        {"Ticker": failed, "Reason": "no data from Stooq (unsupported symbol / temporary issue)"}
    ) if failed else pd.DataFrame(columns=["Ticker", "Reason"])

    # Also include tickers that were in universe but didn't survive RS (rare)
    # (kept minimal on purpose)

    return df_out, dropped
