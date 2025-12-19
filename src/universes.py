# FILE: src/universes.py
# Universe presets + helpers for Streamlit app
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


# -----------------------------
# Paths / Presets
# -----------------------------

def _project_root() -> Path:
    # src/universes.py -> parents[1] == repo root
    return Path(__file__).resolve().parents[1]


# These keys MUST match what app.py uses in the sidebar selectbox
PRESET_FILES: Dict[str, Path] = {
    "Mega Caps": _project_root() / "universes" / "megacaps.csv",
    "S&P 500": _project_root() / "universes" / "sp500.csv",
    "ETFs": _project_root() / "universes" / "etfs.csv",
}


# -----------------------------
# Ticker parsing / normalization
# -----------------------------

def _clean_ticker(t: str) -> str:
    """
    Normalize tickers for yfinance:
    - strip spaces
    - uppercase
    - convert '.' to '-' (e.g., BRK.B -> BRK-B)
    """
    if t is None:
        return ""
    return str(t).strip().upper().replace(".", "-")


def parse_ticker_text(text: str) -> List[str]:
    """
    Accept comma/space/newline separated tickers.
    """
    if not text:
        return []
    # Replace common separators with newline
    s = text.replace(",", "\n").replace(";", "\n").replace("\t", "\n")
    parts = [p.strip() for p in s.splitlines()]
    parts = [p for p in parts if p]
    return parts


def normalize_tickers(tickers: List[str]) -> List[str]:
    """
    Clean + de-duplicate while preserving order.
    """
    seen = set()
    out: List[str] = []
    for t in tickers or []:
        ct = _clean_ticker(t)
        if not ct:
            continue
        if ct not in seen:
            seen.add(ct)
            out.append(ct)
    return out


# -----------------------------
# CSV loaders (repo universes)
# -----------------------------

def load_universe_csv(csv_path: Path) -> List[str]:
    """
    Load from CSV. Accepts either:
      - column 'Ticker' (recommended)
      - column 'ticker' (legacy)
    Returns a cleaned, de-duplicated list preserving file order.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Universe file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    col = None
    for c in ["Ticker", "ticker"]:
        if c in df.columns:
            col = c
            break

    if col is None:
        raise ValueError(f"Universe CSV must contain a 'Ticker' column. File: {csv_path.name}")

    tickers = df[col].astype(str).tolist()
    tickers = normalize_tickers(tickers)

    if not tickers:
        raise ValueError(f"No valid tickers found in: {csv_path.name}")

    return tickers


# -----------------------------
# Quick validation (optional)
# -----------------------------

def quick_validate_tickers(tickers: List[str], max_batch: int = 200) -> List[str]:
    """
    Fast-ish availability check against yfinance:
    - downloads 1d history for batches
    - any ticker with all-NaN Close is considered invalid/unavailable

    Returns list of invalid tickers.
    """
    tickers = normalize_tickers(tickers)
    if not tickers:
        return []

    invalid: List[str] = []

    for i in range(0, len(tickers), max_batch):
        batch = tickers[i : i + max_batch]
        try:
            data = yf.download(
                tickers=batch,
                period="7d",
                interval="1d",
                auto_adjust=True,
                group_by="column",
                threads=True,
                progress=False,
            )
        except Exception:
            # If yfinance fails hard, skip validation (treat as valid)
            continue

        # Expected MultiIndex columns. If not, skip validation for this batch.
        if data is None or data.empty or not isinstance(data.columns, pd.MultiIndex):
            continue

        data = data.swaplevel(axis=1).sort_index(axis=1)

        # Close dataframe: columns = tickers
        try:
            close = data.xs("Close", level=1, axis=1)
        except Exception:
            continue

        for t in batch:
            if t not in close.columns:
                invalid.append(t)
                continue
            series = close[t]
            if series.dropna().empty:
                invalid.append(t)

    # de-dupe preserve order
    seen = set()
    out = []
    for t in invalid:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
