import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "finetune/kaggle_kronos_small_v21.py"


def load_runner():
    namespace = {
        "__name__": "kaggle_kronos_small_v21_test",
        "__file__": str(RUNNER),
    }
    exec(compile(RUNNER.read_text(), str(RUNNER), "exec"), namespace)
    return namespace


def test_small_runner_keeps_24_7m_architecture_and_no_physical_layer_expansion():
    source = RUNNER.read_text()
    assert '"n_layers": 8' in source
    assert '"d_model": 512' in source
    assert '"physical_layer_expansion": False' in source
    assert '"KRONOS_CONTEXT_LAYER": "6"' in source


def test_stage_contracts_are_independent_from_existing_beta_scripts():
    ns = load_runner()
    assert ns["STAGES"]["bootstrap"]["auxiliary"] == "0"
    assert ns["STAGES"]["bootstrap"]["predictor_lr"] == "1e-6"
    assert ns["STAGES"]["bootstrap"]["condition_lr"] == "1e-5"
    assert ns["STAGES"]["main"]["best_metric"] == "forecast"
    assert ns["STAGES"]["main"]["predictor_lr"] == "1e-5"
    assert ns["STAGES"]["main"]["condition_lr"] == "1e-5"
    assert ns["STAGES"]["v21"]["auxiliary"] == "1"
    assert ns["STAGES"]["v21"]["best_metric"] == "beta_v21_score"


def test_find_data_root_requires_a_unique_train_val_pair(tmp_path, monkeypatch):
    ns = load_runner()
    input_root = tmp_path / "input"
    data_root = input_root / "bundle" / "processed_datasets"
    data_root.mkdir(parents=True)
    (data_root / "train_data.pkl").touch()
    (data_root / "val_data.pkl").touch()
    assert ns["find_data_root"](input_root) == data_root.parent

    second = input_root / "second" / "processed_datasets"
    second.mkdir(parents=True)
    (second / "train_data.pkl").touch()
    (second / "val_data.pkl").touch()
    with pytest.raises(SystemExit, match="exactly one data root"):
        ns["find_data_root"](input_root)


def test_metadata_is_required_unless_unknown_only_mode_is_explicit(tmp_path, monkeypatch):
    ns = load_runner()
    with pytest.raises(SystemExit, match="requires asset_metadata"):
        ns["metadata_path"](tmp_path)
    monkeypatch.setenv("KRONOS_SMALL_V21_ALLOW_MISSING_METADATA", "1")
    assert ns["metadata_path"](tmp_path) == tmp_path / "asset_metadata.csv"


def test_manifest_records_source_hashes_and_stage(tmp_path):
    ns = load_runner()
    data_root = tmp_path / "data"
    processed = data_root / "processed_datasets"
    processed.mkdir(parents=True)
    (processed / "train_data.pkl").write_bytes(b"train")
    (processed / "val_data.pkl").write_bytes(b"val")
    predictor = tmp_path / "predictor"
    tokenizer = tmp_path / "tokenizer"
    predictor.mkdir()
    tokenizer.mkdir()
    for root in (predictor, tokenizer):
        (root / "model.safetensors").write_bytes(b"weights")
    out = tmp_path / "out"
    ns["write_manifest"](out, "bootstrap", data_root, predictor, tokenizer)
    manifest = json.loads((out / "small_v21_manifest.json").read_text())
    assert manifest["stage"] == "bootstrap"
    assert manifest["architecture"]["n_layers"] == 8
    assert manifest["data"]["train_sha256"]


def test_cross_stage_start_disables_same_stage_continuation(tmp_path, monkeypatch):
    ns = load_runner()
    monkeypatch.setenv("KRONOS_SMALL_V21_DISABLE_AUTO_CONTINUATION", "1")
    assert ns["copy_continuation_if_requested"](tmp_path / "output") is False
    assert not (tmp_path / "output").exists()


def test_main_stage_uses_full_natural_validation(tmp_path, monkeypatch):
    ns = load_runner()
    data_root = tmp_path / "data"
    processed = data_root / "processed_datasets"
    processed.mkdir(parents=True)
    (processed / "train_data.pkl").write_bytes(b"train")
    (processed / "val_data.pkl").write_bytes(b"val")
    (data_root / "asset_metadata.csv").write_text("symbol\n")
    (data_root / "data_manifest.json").write_text(json.dumps({
        "window_contract": {
            "validation_signal_start": "2025-07-01",
            "validation_signal_end": "2026-07-02",
        }
    }))
    monkeypatch.setenv("KRONOS_BATCH_SIZE", "64")
    env = ns["build_environment"](
        "main", data_root, tmp_path / "predictor", tmp_path / "tokenizer",
        tmp_path / "runtime", "small_0.1_main",
    )
    assert env["KRONOS_BATCH_SIZE"] == "64"
    assert env["KRONOS_PREDICTOR_LEARNING_RATE"] == "1e-5"
    assert env["KRONOS_CONDITION_LEARNING_RATE"] == "1e-5"
    assert env["KRONOS_VALIDATION_SAMPLES"] == "0"
    assert env["KRONOS_VALIDATION_LARGE_SAMPLES"] == "0"
    assert env["KRONOS_VAL_SIGNAL_START"] == "2025-07-01"
    assert env["KRONOS_VAL_SIGNAL_END"] == "2026-07-02"
    assert "KRONOS_FIXED_VALIDATION_MANIFEST_PATH" not in env
