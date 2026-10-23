"""Builder for the temporal Rank IC panel. No full price panel required."""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
FINETUNE = ROOT / "finetune"
for path in (ROOT, FINETUNE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_temporal_rank_ic_panel as builder
from temporal_rank_ic_selection import (
    TemporalRankICSelectionError,
    load_precomputed_rows,
    manifest_sha256,
    validate_manifest_calendar,
)


def _frame(index, scale=1.0, sector="bank"):
    close = np.linspace(10.0, 20.0, len(index)) * scale
    return pd.DataFrame({
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": np.full(len(index), 100.0),
        "amount": np.full(len(index), 1000.0),
        "sector": np.full(len(index), sector),
        "size_percentile": np.full(len(index), 0.4),
    }, index=pd.DatetimeIndex(index))


def _manifest(path, dates, train_bound="2024-12-31"):
    body = {
        "protocol_id": "temporal_rank_ic_v1",
        "split_kind": "temporal",
        "horizon": "D10",
        "n_paths": 1,
        "path_aggregation": "median",
        "temperature": 0.65,
        "top_p": 0.8,
        "top_k": 0,
        "path_seed": 20260906,
        "lookback": 120,
        "predict": 10,
        "train_time_upper_bound": train_bound,
        "signal_dates": list(dates),
        "sha256": manifest_sha256(dates),
        "panel_sha256": "",
        "panel_filename": "temporal_rank_ic_panel.pkl",
    }
    path.write_text(json.dumps(body), encoding="utf-8")
    return body


def _dataset(root, cutoff="2024-12-31"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "processed_datasets").mkdir()
    document = {
        "split": {"unit": "symbol_and_time", "train_cutoff": cutoff},
    }
    manifest = root / "data_manifest.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    metadata = pd.DataFrame({
        "symbol": ["000001", "000002"],
        "date": ["2020-01-01", "2020-01-01"],
        "sector": ["bank", "steel"],
        "size_percentile": [0.2, 0.8],
    })
    metadata.to_csv(root / "asset_metadata.csv", index=False)
    return manifest


def _panels():
    early = pd.date_range("2025-01-01", "2026-03-31", freq="D")
    late = pd.date_range("2026-04-01", "2026-07-31", freq="D")
    full = early.append(late)
    train = {
        "000001": _frame(early, scale=1.0, sector="bank"),
    }
    val = {
        "000001": _frame(late, scale=1.0, sector="bank"),
        "000002": _frame(full, scale=2.0, sector="steel"),
    }
    return train, val


def test_assemble_bakes_ids_and_is_deterministic(tmp_path):
    train, val = _panels()
    dates = ["2026-04-01", "2026-04-02"]
    data = tmp_path / "data"
    _dataset(data)
    metadata = pd.read_csv(data / "asset_metadata.csv")
    first, first_rows = builder.assemble_panel(
        train, val, dates, metadata=metadata, min_names=2,
    )
    second, second_rows = builder.assemble_panel(
        train, val, dates, metadata=metadata, min_names=2,
    )
    assert len(first_rows) == len(second_rows)
    assert [row["symbol"] for row in first_rows] == [row["symbol"] for row in second_rows]
    left = tmp_path / "a" / "temporal_rank_ic_panel.pkl"
    right = tmp_path / "b" / "temporal_rank_ic_panel.pkl"
    assert builder.write_panel(first, left) == builder.write_panel(second, right)
    with left.open("rb") as handle:
        loaded = pickle.load(handle)
    assert list(loaded) == ["000001", "000002"]
    assert int(loaded["000001"]["sector_id"].iloc[0]) == 0
    assert int(loaded["000002"]["sector_id"].iloc[0]) == 1
    assert "sector" not in loaded["000001"].columns
    assert loaded["000001"].index.min() <= pd.Timestamp("2026-03-31")
    assert loaded["000001"].index.max() >= pd.Timestamp("2026-04-01")
    with pytest.raises(TemporalRankICSelectionError, match="requires"):
        builder.assemble_panel(train, val, dates, metadata=metadata, min_names=3)


def test_build_pins_sha_without_touching_the_calendar(tmp_path):
    data = tmp_path / "dataset"
    manifest_path = _dataset(data)
    selection = tmp_path / "selection_window.json"
    dates = ["2026-04-01", "2026-04-02"]
    original = _manifest(selection, dates)
    train, val = _panels()
    with (data / "processed_datasets" / "train_data.pkl").open("wb") as handle:
        pickle.dump(train, handle, protocol=4)
    with (data / "processed_datasets" / "val_data.pkl").open("wb") as handle:
        pickle.dump(val, handle, protocol=4)
    output = tmp_path / "out" / "temporal_rank_ic_panel.pkl"
    summary = builder.build_from_dataset(
        data,
        selection,
        output,
        expected_sha=builder.sha256_file(manifest_path),
        expected_cutoff="2024-12-31",
        min_names=2,
    )
    pinned = json.loads(selection.read_text(encoding="utf-8"))
    assert pinned["signal_dates"] == original["signal_dates"]
    assert pinned["sha256"] == original["sha256"]
    assert pinned["panel_sha256"] == summary["panel_sha256"]
    assert summary["panel_sha256"] == builder.sha256_file(output)
    rows_path = output.with_name("temporal_rank_ic_rows.pkl")
    assert summary["rows_sha256"] == builder.sha256_file(rows_path)
    assert pinned["rows_sha256"] == summary["rows_sha256"]
    assert pinned["rows_filename"] == "temporal_rank_ic_rows.pkl"
    assert summary["n_dates"] == 2
    assert pinned["signal_dates"] == original["signal_dates"]
    assert pinned["sha256"] == original["sha256"]
    validate_manifest_calendar(pinned)
    again = builder.build_from_dataset(
        data,
        selection,
        output,
        expected_sha=builder.sha256_file(manifest_path),
        expected_cutoff="2024-12-31",
        min_names=2,
    )
    assert again["panel_sha256"] == summary["panel_sha256"]
    assert again["rows_sha256"] == summary["rows_sha256"]
    assert again["calendar_sha256"] == original["sha256"]
    rewritten = json.loads(selection.read_text(encoding="utf-8"))
    assert rewritten["panel_sha256"] == summary["panel_sha256"]
    assert rewritten["sha256"] == original["sha256"]
    assert rewritten["signal_dates"] == original["signal_dates"]
    loaded_rows = load_precomputed_rows(
        rows_path,
        {
            "signal_dates": pinned["signal_dates"],
            "panel_sha256": pinned["panel_sha256"],
            "sha256": pinned["sha256"],
            "panel_path": str(output),
        },
        expected_sha=pinned["rows_sha256"],
        min_names=2,
    )
    assert len(loaded_rows) == summary["n_rows"]
    assert {row["asof_date"] for row in loaded_rows} == set(dates)


def test_build_refuses_a_non_session_and_a_wrong_contract(tmp_path):
    selection = tmp_path / "selection_window.json"
    _manifest(selection, ["2026-04-06"])
    with pytest.raises(TemporalRankICSelectionError, match="non-sessions"):
        builder.build_from_dataset(tmp_path / "missing", selection, tmp_path / "temporal_rank_ic_panel.pkl")

    data = tmp_path / "other"
    _dataset(data, cutoff="2020-01-01")
    _manifest(selection, ["2026-04-01", "2026-04-02"])
    with pytest.raises(TemporalRankICSelectionError, match="train_cutoff"):
        builder.build_from_dataset(
            data,
            selection,
            tmp_path / "temporal_rank_ic_panel.pkl",
            expected_sha=builder.sha256_file(data / "data_manifest.json"),
            expected_cutoff="2024-12-31",
        )
    with pytest.raises(TemporalRankICSelectionError, match="clean data_manifest"):
        builder.build_from_dataset(
            data,
            selection,
            tmp_path / "temporal_rank_ic_panel.pkl",
        )


def test_output_cannot_be_val_data_or_a_holdout_path(tmp_path):
    with pytest.raises(TemporalRankICSelectionError, match="val_data.pkl"):
        builder.assert_output_path(tmp_path / "val_data.pkl")
    with pytest.raises(TemporalRankICSelectionError, match="symbol-holdout"):
        builder.assert_output_path(
            tmp_path / "a-share-120d-temporal-symbol-holdout" / "temporal_rank_ic_panel.pkl",
        )


def test_pin_keeps_panel_sha_until_the_bytes_change(tmp_path):
    selection = tmp_path / "selection_window.json"
    dates = ["2026-04-01", "2026-04-02"]
    original = _manifest(selection, dates)
    panel_sha = "a" * 64
    first = builder.pin_selection_digests(
        selection, panel_sha256=panel_sha, rows_sha256="b" * 64,
    )
    assert first["panel_sha256"] == panel_sha
    builder.pin_selection_digests(
        selection, panel_sha256=panel_sha, rows_sha256="c" * 64,
    )
    stored = json.loads(selection.read_text(encoding="utf-8"))
    assert stored["panel_sha256"] == panel_sha
    assert stored["rows_sha256"] == "c" * 64
    assert stored["sha256"] == original["sha256"]
    assert stored["signal_dates"] == original["signal_dates"]
    changed = builder.pin_selection_digests(
        selection, panel_sha256="d" * 64, rows_sha256="c" * 64,
    )
    assert changed["panel_sha256"] == "d" * 64
    final = json.loads(selection.read_text(encoding="utf-8"))
    assert final["sha256"] == original["sha256"]
    assert final["signal_dates"] == dates


def test_clean_sha_constant_matches_launch():
    launch = (ROOT / "finetune/small_clean_bigbatch_relay/launch.py").read_text(
        encoding="utf-8",
    )
    assert builder.CLEAN_DATA_MANIFEST_SHA256 in launch
    assert builder.CLEAN_TRAIN_CUTOFF == "2024-12-31"
