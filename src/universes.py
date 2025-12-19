# src/universes.py
# Helpers for loading and managing ticker universes (Streamlit-compatible)

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf


# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
def _project_root() -> Path:
    # This file lives in <root>/src/universes.py
    return Path(__file__).resolve().parents[1]


UNIVERSES_DIR = _project_root() / "universes"


# -------------------------------------------------------------------
# Preset universes exposed in the UI
# Keys MUST match app.py selectbox labels
# -------------------------------------------------------------------
PRESET_FILES: Dict[str, Path] = {
    "Mega Caps": UNIVERSES_DIR / "megacaps.csv",
    "S&P 500": UNIVERSES_DIR / "sp500.csv",
    "ETFs": UNIVERSES_DIR / "etfs.csv",
}


# -------------------------------------------------------------------
# Ticker utilities
# -------------------------------------------------------------------
def _clean_ticker(t: str) -> str:
    """
    Normalize tickers for yfinance:
    - strip spaces
    - uppercase
    - convert '.' to '-' (BRK.B -> BRK-B)
    """
    if t is None:
        return ""
    return str(t).strip().upper().replace(".", "-")


def normalize_tickers(tickers: List[str]) -> List[str]:
    """
    Clean + de-duplicate tickers preserving order.
    """
    out: List[str] = []
    seen = set()
    for t in tickers:
        ct = _clean_ticker(t)
        if ct and ct not in seen:
            seen.add(ct)
            out.append(ct)
    return out


def parse_ticker_text(text: str) -> List[str]:
    """
    Parse pasted text (comma, space or newline separated).
    """
    if not text:
        return []
    raw = (
        text.replace(",", "\n")
        .replace(";", "\n")
        .splitlines()
    )
    return normalize_tickers(raw)


# -------------------------------------------------------------------
# CSV loaders
# -------------------------------------------------------------------
def load_universe_csv(csv_path: Path) -> List[str]:
    """
    Load a universe from CSV.
    Accepts column 'Ticker' or 'ticker'.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Universe file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    col = None
    if "Ticker" in df.columns:
        col = "Ticker"
    elif "ticker" in df.columns:
        col = "ticker"
    else:
        raise ValueError(
            f"{csv_path.name} must contain a 'Ticker' column"
        )

    tickers = df[col].astype(str).tolist()
    tickers = normalize_tickers(tickers)

    if not tickers:
        raise ValueError(f"No valid tickers found in {csv_path.name}")

    return tickers


# -------------------------------------------------------------------
# Fast availability check (optional, used by app.py)
# -------------------------------------------------------------------
def quick_validate_tickers(
    tickers: List[str],
    max_batch: int = 200,
) -> List[str]:
    """
    Very fast availability check using yfinance metadata.
    Returns tickers that appear invalid/unavailable.
    """
    invalid: List[str] = []

    for i in range(0, len(tickers), max_batch):
        batch = tickers[i : i + max_batch]
        try:
            data = yf.download(
                tickers=batch,
                period="5d",
                group_by="column",
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception:
            invalid.extend(batch)
            continue

        if data is None or data.empty:
            invalid.extend(batch)
            continue

        # If single ticker, yfinance returns single-index columns
        if not isinstance(data.columns, pd.MultiIndex):
            continue

        available = set(data.columns.get_level_values(0))
        for t in batch:
            if t not in available:
                invalid.append(t)

    return sorted(set(invalid))
