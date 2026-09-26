import json
import pickle

import numpy as np
import pandas as pd
import pytest

from modernbert_finance.prepare_temporal_holdout_labels import (
    find_temporal_holdout_root,
    prepare_labels,
)


def write_panel(root, name, rows=131):
    processed = root / "processed_datasets"
    processed.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    close = np.linspace(100.0, 110.0, rows)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1000.0,
            "amount": 100000.0,
            "sector": "test",
            "size_percentile": 0.5,
        },
        index=dates,
    )
    with (processed / name).open("wb") as handle:
        pickle.dump({"sh.600000": frame}, handle)


def test_temporal_holdout_labels_use_source_manifest_boundaries(tmp_path):
    root = tmp_path / "holdout"
    root.mkdir()
    write_panel(root, "train_data.pkl")
    write_panel(root, "val_data.pkl")
    (root / "data_manifest.json").write_text(
        json.dumps(
            {
                "split": {
                    "unit": "symbol_and_time",
                    "train_cutoff": "2020-05-01",
                },
                "window_contract": {
                    "lookback": 120,
                    "predict": 10,
                    "validation_signal_start": "2020-04-29",
                    "validation_signal_end": "2020-05-01",
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "targets"
    manifest = prepare_labels(tmp_path, output)

    assert manifest["join_key"] == ["symbol", "start_index", "asof_date"]
    assert manifest["splits"]["train"]["records"] == 1
    assert manifest["splits"]["validation"]["records"] == 1
    assert manifest["splits"]["train"]["path"] == "train_targets.parquet"
    assert manifest["splits"]["validation"]["path"] == "validation_targets.parquet"
    assert (output / "train_targets.parquet").is_file()
    assert (output / "validation_targets.parquet").is_file()
    assert (output / "decision_targets_manifest.json").is_file()


def test_discovery_rejects_multiple_panel_datasets(tmp_path):
    for name in ("one", "two"):
        root = tmp_path / name
        root.mkdir()
        write_panel(root, "train_data.pkl")
        write_panel(root, "val_data.pkl")
        (root / "data_manifest.json").write_text(
            json.dumps(
                {
                    "split": {"unit": "symbol_and_time"},
                    "window_contract": {"lookback": 120, "predict": 10},
                }
            ),
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="exactly one"):
        find_temporal_holdout_root(tmp_path)
