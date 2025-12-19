# src/universes.py
# Loads curated ticker universes from CSV files under /universes

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class UniverseSpec:
    """Defines one selectable universe in the app."""
    key: str          # internal id (stable)
    label: str        # UI label
    csv_path: Path    # path to CSV file with a 'ticker' column


def _project_root() -> Path:
    """
    Resolve project root assuming this file lives in: <root>/src/universes.py
    """
    return Path(__file__).resolve().parents[1]


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


def load_universe_from_csv(csv_path: Path) -> List[str]:
    """
    Load tickers from a CSV with at least a 'ticker' column.
    Returns a de-duplicated, cleaned list preserving original order.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Universe file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "ticker" not in df.columns:
        raise ValueError(
            f"Universe CSV must contain a 'ticker' column. Missing in: {csv_path.name}"
        )

    # Clean + drop empty
    tickers_raw = df["ticker"].astype(str).map(_clean_ticker)
    tickers_raw = tickers_raw[tickers_raw != ""]

    # De-duplicate preserving order
    seen = set()
    tickers: List[str] = []
    for t in tickers_raw.tolist():
        if t not in seen:
            seen.add(t)
            tickers.append(t)

    if len(tickers) == 0:
        raise ValueError(f"No valid tickers found in: {csv_path.name}")

    return tickers


def get_default_universe_specs(universes_dir: Optional[Path] = None) -> List[UniverseSpec]:
    """
    Define the curated universes exposed in the UI.
    CSVs live at: <root>/universes/*.csv
    """
    root = _project_root()
    udir = universes_dir or (root / "universes")

    return [
        UniverseSpec(key="megacaps", label="MegaCaps", csv_path=udir / "megacaps.csv"),
        UniverseSpec(key="sp500", label="S&P 500", csv_path=udir / "sp500.csv"),
        UniverseSpec(key="etfs", label="ETFs", csv_path=udir / "etfs.csv"),
    ]


def load_all_universes(
    specs: Optional[List[UniverseSpec]] = None,
    universes_dir: Optional[Path] = None
) -> Dict[str, List[str]]:
    """
    Loads all universes and returns a dict keyed by spec.label for UI use.

    Example return:
    {
      "MegaCaps": [...],
      "S&P 500": [...],
      "ETFs": [...]
    }
    """
    specs = specs or get_default_universe_specs(universes_dir=universes_dir)

    out: Dict[str, List[str]] = {}
    for spec in specs:
        out[spec.label] = load_universe_from_csv(spec.csv_path)
    return out


def load_universe_by_key(
    key: str,
    specs: Optional[List[UniverseSpec]] = None,
    universes_dir: Optional[Path] = None
) -> List[str]:
    """
    Load a single universe by its stable key (e.g., 'megacaps', 'sp500', 'etfs').
    """
    specs = specs or get_default_universe_specs(universes_dir=universes_dir)
    key_norm = (key or "").strip().lower()

    for spec in specs:
        if spec.key.lower() == key_norm:
            return load_universe_from_csv(spec.csv_path)

    valid = ", ".join([s.key for s in specs])
    raise ValueError(f"Unknown universe key '{key}'. Valid: {valid}")
