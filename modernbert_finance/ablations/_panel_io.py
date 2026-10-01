"""Panel I/O for ablations without importing torch-backed dataset.py."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd


def load_panel(path: Path) -> dict[str, pd.DataFrame]:
    with Path(path).open("rb") as handle:
        panel = pickle.load(handle)
    if not isinstance(panel, dict):
        raise ValueError(f"{path} does not contain a symbol -> DataFrame panel")
    return {str(symbol): frame for symbol, frame in panel.items()}
