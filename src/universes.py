from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# -----------------------------
# Models
# -----------------------------

@dataclass(frozen=True)
class UniverseSpec:
    """
    Defines one selectable universe in the app.
    """
    key: str          # stable internal id (e.g. "megacaps")
    label: str        # UI label (e.g. "Mega Caps")
    csv_path: Path    # CSV path with a 'Ticker' column


# -----------------------------
# Helpers
# -----------------------------

def _project_root() -> Path:
    """
    Resolve project root assuming this file lives in:
    <root>/src/universes.py
    """
    return Path(__file__).resolve().parents[1]


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


# -----------------------------
# Loaders
# -----------------------------

def load_universe_csv(csv_path: Path) -> List[str]:
    """
    Load tickers from a CSV with a 'Ticker' column.
    De-duplicates while preserving order.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Universe file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if "Ticker" not in df.columns:
        raise ValueError(
            f"Universe CSV must contain a 'Ticker' column: {csv_path.name}"
        )

    raw = df["Ticker"].astype(str).map(_clean_ticker)
    raw = raw[raw != ""]

    seen = set()
    out: List[str] = []
    for t in raw.tolist():
        if t not in seen:
            seen.add(t)
            out.append(t)

    if not out:
        raise ValueError(f"No valid tickers found in {csv_path.name}")

    return out


def get_default_universe_specs(
    universes_dir: Optional[Path] = None
) -> List[UniverseSpec]:
    """
    Curated universes exposed in the UI.
    CSVs live in: <root>/universes/*.csv
    """
    root = _project_root()
    udir = universes_dir or (root / "universes")

    return [
        UniverseSpec("megacaps", "Mega Caps", udir / "megacaps.csv"),
        UniverseSpec("sp500", "S&P 500", udir / "sp500.csv"),
        UniverseSpec("etfs", "ETFs", udir / "etfs.csv"),
    ]


def load_all_universes(
    specs: Optional[List[UniverseSpec]] = None,
    universes_dir: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """
    Load all universes and return:
    { "Mega Caps": [...], "S&P 500": [...], "ETFs": [...] }
    """
    specs = specs or get_default_universe_specs(universes_dir)

    out: Dict[str, List[str]] = {}
    for spec in specs:
        out[spec.label] = load_universe_csv(spec.csv_path)
    return out


def load_universe_by_key(
    key: str,
    specs: Optional[List[UniverseSpec]] = None,
    universes_dir: Optional[Path] = None,
) -> List[str]:
    """
    Load a single universe by its stable key
    (megacaps / sp500 / etfs)
    """
    specs = specs or get_default_universe_specs(universes_dir)
    key = (key or "").strip().lower()

    for spec in specs:
        if spec.key == key:
            return load_universe_csv(spec.csv_path)

    valid = ", ".join(s.key for s in specs)
    raise ValueError(f"Unknown universe key '{key}'. Valid: {valid}")
