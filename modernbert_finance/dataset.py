"""Numeric Dataset adapter for the Kronos ModernBERT decision model."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from modernbert_finance.build_dataset import FEATURES, prepare_frame
from modernbert_finance.build_targets import WINDOW


LOOKBACK = 120
UP_COLUMNS = ("up_003", "up_005", "up_008", "up_012")
DOWN_COLUMNS = ("down_003", "down_005", "down_008", "down_012")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_panel(path: Path) -> dict[str, pd.DataFrame]:
    with path.open("rb") as handle:
        panel = pickle.load(handle)
    if not isinstance(panel, dict):
        raise ValueError(f"{path} does not contain a symbol -> DataFrame panel")
    return {str(symbol): frame for symbol, frame in panel.items()}


def load_sector_vocabulary(path: Path) -> tuple[dict[str, int], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = payload.get("sector_labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"{path} does not contain sector_labels")
    mapping = {str(label): index for index, label in enumerate(labels)}
    unknown_id = int(payload.get("unknown_sector_id", len(labels)))
    return mapping, unknown_id


class ModernBERTWindowDataset(Dataset):
    """Read numeric decision samples without materializing a new input dataset.

    The target Parquet file is aligned by deterministic symbol/date ordering.
    The manifest's source-panel hash prevents pairing it with a different panel;
    filtered date bounds keep the label row offsets dense without a large
    Python dictionary keyed by every sample identity.
    """

    def __init__(
        self,
        panel_path: Path,
        targets_path: Path,
        sector_vocabulary_path: Path,
        signal_start: str | None = None,
        signal_end: str | None = None,
        clip: float = 5.0,
        verify_source: bool = True,
    ) -> None:
        self.panel_path = Path(panel_path)
        self.targets_path = Path(targets_path)
        self.clip = float(clip)
        if self.clip <= 0:
            raise ValueError("clip must be positive")

        target_manifest = self._load_target_manifest(verify_source)
        split_name = (
            "validation"
            if "validation" in self.targets_path.name or "val" in self.targets_path.name
            else "train"
        )
        split_stats = target_manifest.get("splits", {}).get(split_name, {})
        if signal_start is None:
            signal_start = split_stats.get("signal_start")
        if signal_end is None:
            signal_end = split_stats.get("signal_end")
        sector_ids, unknown_sector_id = load_sector_vocabulary(
            Path(sector_vocabulary_path)
        )
        panel = load_panel(self.panel_path)
        target_table = pq.read_table(
            self.targets_path,
            columns=["mfe10", "mae10", *UP_COLUMNS, *DOWN_COLUMNS],
        )
        target_count = target_table.num_rows
        target_arrays = {
            name: np.asarray(target_table[name]).copy()
            for name in ["mfe10", "mae10", *UP_COLUMNS, *DOWN_COLUMNS]
        }

        start_date = pd.Timestamp(signal_start).date() if signal_start else None
        end_date = pd.Timestamp(signal_end).date() if signal_end else None
        symbol_names: list[str] = []
        symbol_ids: list[int] = []
        starts: list[int] = []
        target_rows: list[int] = []
        values_by_symbol: dict[str, np.ndarray] = {}
        sector_by_symbol: dict[str, np.ndarray] = {}
        size_by_symbol: dict[str, np.ndarray] = {}
        dates_by_symbol: dict[str, np.ndarray] = {}
        sidecar_row = 0

        for symbol in sorted(panel):
            frame = prepare_frame(panel[symbol], symbol)
            if len(frame) < WINDOW:
                continue
            frame = frame.sort_index()
            values_by_symbol[symbol] = frame.loc[:, FEATURES].to_numpy(
                dtype=np.float32
            )
            labels = frame.get("sector_id")
            if labels is not None:
                sector = pd.to_numeric(labels, errors="coerce").fillna(
                    unknown_sector_id
                ).to_numpy(dtype=np.int64)
            else:
                raw_labels = frame.get(
                    "sector", pd.Series("unknown", index=frame.index)
                ).astype(str)
                sector = raw_labels.map(sector_ids).fillna(
                    unknown_sector_id
                ).to_numpy(dtype=np.int64)
            sector_by_symbol[symbol] = sector

            raw_size = pd.to_numeric(
                frame.get(
                    "size_percentile",
                    pd.Series(0.5, index=frame.index),
                ),
                errors="coerce",
            ).clip(0.0, 1.0)
            size_by_symbol[symbol] = raw_size.fillna(0.5).to_numpy(
                dtype=np.float32
            )
            dates_by_symbol[symbol] = frame.index.to_numpy(dtype="datetime64[D]")

            symbol_id = len(symbol_names)
            symbol_names.append(symbol)
            max_start = len(frame) - WINDOW + 1
            for start in range(max_start):
                asof_date = pd.Timestamp(
                    dates_by_symbol[symbol][start + LOOKBACK - 1]
                ).date()
                if start_date and asof_date < start_date:
                    continue
                if end_date and asof_date > end_date:
                    continue
                symbol_ids.append(symbol_id)
                starts.append(start)
                target_rows.append(sidecar_row)
                sidecar_row += 1

        if sidecar_row != target_count:
            raise ValueError(
                "Target row count does not match filtered panel window enumeration: "
                f"{target_count} != {sidecar_row}; check sidecar split/date range "
                "and source panel"
            )

        self.symbol_names = tuple(symbol_names)
        self.symbol_ids = np.asarray(symbol_ids, dtype=np.int32)
        self.starts = np.asarray(starts, dtype=np.int32)
        self.target_rows = np.asarray(target_rows, dtype=np.int64)
        self.values_by_symbol = values_by_symbol
        self.sector_by_symbol = sector_by_symbol
        self.size_by_symbol = size_by_symbol
        self.dates_by_symbol = dates_by_symbol
        self.target_arrays = target_arrays

    def _load_target_manifest(self, verify_source: bool) -> dict[str, Any]:
        manifest_path = self.targets_path.with_name("targets_manifest.json")
        if not manifest_path.is_file():
            manifest_path = self.targets_path.with_name(
                "decision_targets_manifest.json"
            )
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        window = manifest.get("window", {})
        if (
            int(window.get("lookback", -1)) != LOOKBACK
            or int(window.get("source_window", -1)) != WINDOW
        ):
            raise ValueError("Target manifest window contract does not match dataset")
        if verify_source:
            source = manifest.get("source", {})
            dataset_source = manifest.get("source_dataset", {})
            is_validation = (
                "validation" in self.targets_path.name
                or "val" in self.targets_path.name
            )
            expected = (
                dataset_source.get("validation_panel_sha256")
                if is_validation
                else dataset_source.get("train_panel_sha256")
            )
            if not expected:
                expected = source.get(
                    "validation_panel_sha256"
                    if is_validation
                    else "train_panel_sha256"
                )
            if not expected:
                raise ValueError("Target manifest is missing the source panel SHA256")
            if sha256_file(self.panel_path) != expected:
                raise ValueError("Panel SHA256 does not match target manifest")
        return manifest

    def __len__(self) -> int:
        return int(self.symbol_ids.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        position = int(index)
        symbol_id = int(self.symbol_ids[position])
        symbol = self.symbol_names[symbol_id]
        start = int(self.starts[position])
        target_row = int(self.target_rows[position])
        history = self.values_by_symbol[symbol][start:start + LOOKBACK].copy()
        mean = history.mean(axis=0)
        std = history.std(axis=0)
        history = (history - mean) / (std + 1e-5)
        history = np.clip(history, -self.clip, self.clip).astype(np.float32)
        asof_position = start + LOOKBACK - 1
        targets = self.target_arrays
        return {
            "history": torch.from_numpy(history),
            "sector_id": torch.tensor(
                self.sector_by_symbol[symbol][asof_position],
                dtype=torch.long,
            ),
            "size_percentile": torch.tensor(
                self.size_by_symbol[symbol][asof_position],
                dtype=torch.float32,
            ),
            "up_target": torch.tensor(
                [targets[name][target_row] for name in UP_COLUMNS],
                dtype=torch.float32,
            ),
            "down_target": torch.tensor(
                [targets[name][target_row] for name in DOWN_COLUMNS],
                dtype=torch.float32,
            ),
            "mfe10": torch.tensor(targets["mfe10"][target_row], dtype=torch.float32),
            "mae10": torch.tensor(targets["mae10"][target_row], dtype=torch.float32),
            "symbol": symbol,
            "start_index": start,
            "asof_date": str(
                pd.Timestamp(self.dates_by_symbol[symbol][asof_position]).date()
            ),
        }
