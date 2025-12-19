# FILE: src/universes.py
from __future__ import annotations

import pandas as pd
import yfinance as yf

PRESET_FILES = {
    "Mega Caps": "universes/megacaps.csv",
    "S&P 500": "universes/sp500.csv",
    "ETFs": "universes/etfs.csv",
}


def normalize_tickers(tickers: list[str]) -> list[str]:
    out = []
    for t in tickers:
        if t is None:
            continue
        t = str(t).strip().upper()
        if not t:
            continue
        # Yahoo format: BRK.B -> BRK-B
        t = t.replace(".", "-")
        out.append(t)
    # dedupe preserving order
    seen = set()
    res = []
    for t in out:
        if t not in seen:
            seen.add(t)
            res.append(t)
    return res


def parse_ticker_text(text: str) -> list[str]:
    if not text:
        return []
    # accept commas, spaces, newlines
    raw = text.replace(",", "\n").replace(" ", "\n").splitlines()
    return [r for r in raw if r and r.strip()]


def load_universe_csv(path: str) -> list[str]:
    df = pd.read_csv(path)
    if "Ticker" not in df.columns:
        raise ValueError(f"Universe CSV '{path}' must contain column 'Ticker'.")
    return normalize_tickers(df["Ticker"].astype(str).tolist())


def quick_validate_tickers(tickers: list[str], max_batch: int = 200) -> list[str]:
    """
    Fast-ish validation: downloads 5d Close for batches and flags those with no data.
    """
    tickers = normalize_tickers(tickers)
    invalid = []

    for i in range(0, len(tickers), max_batch):
        batch = tickers[i : i + max_batch]
        try:
            data = yf.download(
                batch,
                period="5d",
                auto_adjust=True,
                group_by="column",
                threads=True,
                progress=False,
            )
            if data is None or data.empty:
                invalid.extend(batch)
                continue

            # Handle both single and multi index
            if isinstance(data.columns, pd.MultiIndex):
                data = data.swaplevel(axis=1).sort_index(axis=1)
                close = data.xs("Close", level=1, axis=1)
                for t in batch:
                    if t not in close.columns or close[t].dropna().empty:
                        invalid.append(t)
            else:
                # single ticker case; if it's empty, mark all
                pass

        except Exception:
            # If yfinance blows up, mark the whole batch invalid to be safe
            invalid.extend(batch)

    # dedupe
    return sorted(set(invalid))
