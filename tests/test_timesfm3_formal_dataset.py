import importlib.util
import json
import pickle

import numpy as np
import pandas as pd
import pytest


def module():
    spec = importlib.util.spec_from_file_location(
        "formal", "finetune/timesfm3_prepare_formal_dataset.py",
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def frame():
    return pd.DataFrame(
        np.ones((140, 7), dtype=np.float32),
        columns=module().FEATURES, index=pd.bdate_range("2020-01-01", periods=140),
    )


def test_matches_kronos_131_row_eligibility():
    value = module()
    data = frame()
    _, dates, indices = value.window_indices(data)
    assert len(indices) == 10
    _, _, filtered = value.window_indices(data, str(dates[120]), str(dates[122]))
    assert filtered.tolist() == [1, 2, 3]


def test_uses_formal_panels_and_exports_all_validation(tmp_path):
    value = module()
    root = tmp_path / "source"
    (root / "processed_datasets").mkdir(parents=True)
    data = frame()
    for split, symbol in (("train", "train-stock"), ("val", "holdout-stock")):
        with (root / f"processed_datasets/{split}_data.pkl").open("wb") as handle:
            pickle.dump({symbol: data}, handle)
    pd.DataFrame({
        "symbol": ["train-stock", "holdout-stock"], "split": ["train", "validation"],
    }).to_csv(root / "symbol_split.csv", index=False)
    (root / "data_manifest.json").write_text(json.dumps({
        "split": {"unit": "symbol_and_time", "train_cutoff": "2019-12-31"},
        "window_contract": {
            "lookback": 120, "predict": 10,
            "validation_signal_start": "2020-01-01",
            "validation_signal_end": "2021-01-01",
        },
    }))
    assert value.discover(tmp_path) == root
    result = value.prepare(root, tmp_path / "output")
    assert result["splits"]["val"]["samples"] == 10
    shards = sorted((tmp_path / "output/val").glob("shard_*.npz"))
    assert sum(np.load(path)["contexts"].shape[0] for path in shards) == 10
    assert len((tmp_path / "output/val/identities.jsonl").read_text().splitlines()) == 10


def test_rejects_bad_dates_and_nonfinite():
    value = module()
    data = frame()
    with pytest.raises(ValueError, match="sorted"):
        value.window_indices(data.iloc[::-1])
    data.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        value.window_indices(data)
