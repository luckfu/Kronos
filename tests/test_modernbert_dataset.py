import json
import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from modernbert_finance.build_targets import build_split
from modernbert_finance.dataset import ModernBERTWindowDataset


def test_numeric_dataset_returns_only_model_inputs_and_labels(tmp_path):
    rows = 131
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    close = np.linspace(100.0, 110.0, rows)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(rows, 1000.0),
            "amount": np.full(rows, 100000.0),
            "sector": ["J66货币金融服务"] * rows,
            "size_percentile": np.full(rows, 0.75),
        },
        index=dates,
    )
    panel_path = tmp_path / "train_data.pkl"
    with panel_path.open("wb") as handle:
        pickle.dump({"sh.600000": frame}, handle)

    vocab_path = tmp_path / "sector_vocabulary.json"
    vocab_path.write_text(
        json.dumps(
            {
                "sector_labels": ["J66货币金融服务"],
                "unknown_sector_id": 1,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    targets_dir = tmp_path / "targets"
    build_split(
        {"sh.600000": frame},
        targets_dir / "train_targets.parquet",
        None,
        None,
        0,
        100,
    )
    manifest = {
        "window": {"lookback": 120, "source_window": 131},
        "source": {"train_panel_sha256": "skip"},
    }
    (targets_dir / "targets_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    dataset = ModernBERTWindowDataset(
        panel_path,
        targets_dir / "train_targets.parquet",
        vocab_path,
        verify_source=False,
    )
    sample = dataset[0]

    assert len(dataset) == 1
    assert tuple(sample["history"].shape) == (120, 6)
    assert sample["history"].dtype == torch.float32
    assert int(sample["sector_id"]) == 0
    assert float(sample["size_percentile"]) == 0.75
    assert tuple(sample["up_target"].shape) == (4,)
    assert tuple(sample["down_target"].shape) == (4,)
    assert "summary" not in sample


def test_filtered_sidecar_uses_dense_rows_and_manifest_dates(tmp_path):
    rows = 133
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    close = np.linspace(100.0, 110.0, rows)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(rows, 1000.0),
            "amount": np.full(rows, 100000.0),
            "sector": ["test"] * rows,
            "size_percentile": np.full(rows, 0.5),
        },
        index=dates,
    )
    panel_path = tmp_path / "train_data.pkl"
    with panel_path.open("wb") as handle:
        pickle.dump({"sh.600000": frame}, handle)
    vocab_path = tmp_path / "sector_vocabulary.json"
    vocab_path.write_text(
        json.dumps({"sector_labels": ["test"], "unknown_sector_id": 1}),
        encoding="utf-8",
    )
    targets_dir = tmp_path / "targets"
    stats = build_split(
        {"sh.600000": frame},
        targets_dir / "train_targets.parquet",
        dates[120].date().isoformat(),
        dates[120].date().isoformat(),
        0,
        100,
    )
    (targets_dir / "decision_targets_manifest.json").write_text(
        json.dumps(
            {
                "window": {"lookback": 120, "source_window": 131},
                "splits": {"train": stats},
            }
        ),
        encoding="utf-8",
    )

    dataset = ModernBERTWindowDataset(
        panel_path,
        targets_dir / "train_targets.parquet",
        vocab_path,
        verify_source=False,
    )

    assert len(dataset) == 1
    assert dataset[0]["start_index"] == 1
    assert dataset[0]["asof_date"] == dates[120].date().isoformat()
