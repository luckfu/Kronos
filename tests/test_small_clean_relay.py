import hashlib
import json
import runpy
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_launcher():
    return runpy.run_path(str(ROOT / "finetune/small_clean_relay/launch.py"))


def test_relay_preserves_submitted_identity_after_git_commit(tmp_path, monkeypatch):
    ns = runpy.run_path(str(ROOT / "finetune/small_clean_relay/push_chunk.py"))
    main = ns["main"]
    actual_here = main.__globals__["HERE"]
    previous = tmp_path / "build" / ns["slug"](1) / "build_manifest.json"
    previous.parent.mkdir(parents=True)
    previous.write_text(json.dumps({
        "source_sha256": hashlib.sha256(b"original source").hexdigest(),
        "source_commit": "submitted-commit",
    }))
    monkeypatch.setitem(main.__globals__, "HERE", tmp_path)
    monkeypatch.setitem(main.__globals__, "source_bundle", lambda: b"original source")
    monkeypatch.setitem(main.__globals__, "local_credential", lambda: "")
    monkeypatch.setattr(ns["subprocess"], "check_output", lambda *a, **k: "new-head\n")
    original_run_path = runpy.run_path
    monkeypatch.setattr(runpy, "run_path",
                        lambda path: original_run_path(str(actual_here / "launch.py")))
    monkeypatch.setattr(sys, "argv", ["push_chunk.py", "2", "--dry-run"])
    main()
    built = json.loads((tmp_path / "build" / ns["slug"](2) / "build_manifest.json").read_text())
    assert built["source_commit"] == "submitted-commit"
    monkeypatch.setitem(main.__globals__, "source_bundle", lambda: b"changed source")
    with pytest.raises(RuntimeError, match="Source changed across chunks"):
        main()


def test_flat_kaggle_upload_restores_manifest_layout(tmp_path):
    ns = load_launcher()
    source = tmp_path / "input" / "dataset"
    source.mkdir(parents=True)
    data = b"clean_train"
    (source / "train_data.pkl").write_bytes(data)
    manifest = {
        "files": [{
            "name": "processed_datasets/train_data.pkl",
            "sha256": hashlib.sha256(data).hexdigest(),
        }],
        "stats": {}, "split": {}, "window_contract": {},
    }
    manifest_path = source / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    prepare = ns["prepare_data"]
    prepare.__globals__["DATA_SHA"] = ns["sha256"](manifest_path)
    target = prepare(tmp_path / "input", tmp_path / "runtime")
    assert (target / "processed_datasets/train_data.pkl").read_bytes() == data
    (source / "train_data.pkl").write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        prepare(tmp_path / "input", tmp_path / "other")


def test_output_gate_rejects_different_segment_boundary(tmp_path):
    ns = load_launcher()
    for name in ns["REQUIRED"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (tmp_path / "experiment_manifest.json").write_text('{"recipe": {}}')
    (tmp_path / "progress.json").write_text(
        '{"status": "stopped", "current_segment": 4}'
    )
    (tmp_path / "summary.json").write_text(
        '{"final_result": {"status": "stopped", "completed_segments": 4}}'
    )

    class FakeTorch:
        @staticmethod
        def load(*args, **kwargs):
            return {"next_epoch": 3, "resume_step": 0}

    with pytest.raises(RuntimeError, match="durable segment boundary"):
        ns["validate_output"](tmp_path, FakeTorch, {})

    with pytest.raises(RuntimeError, match="recipe changed"):
        ns["validate_output"](tmp_path, FakeTorch, {"changed": True})

    class CompleteState:
        @staticmethod
        def load(*args, **kwargs):
            return {
                "next_epoch": 4, "resume_step": 0,
                "scheduler_type": "warmup_constant",
                "effective_epochs": ns["TOTAL_SEGMENTS"],
                "optimizer": {"state": {"parameter": {"step": 12}}},
                "amp_scaler": {}, "rng_state": {"python": "state"},
                "scheduler": {"last_epoch": 12}, "batch_idx_global": 12,
            }

    assert ns["validate_output"](tmp_path, CompleteState, {}) == 4
