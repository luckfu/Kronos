"""Recover CANCEL_ACKNOWLEDGED Output into a durable relay parent."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
RELAY = ROOT / "finetune/small_clean_bigbatch_relay"
FINETUNE = ROOT / "finetune"
for path in (ROOT, FINETUNE, RELAY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from recover_incomplete_chunk import (
    inspect_incomplete_boundary,
    write_durable_sidecar,
)


class _FakeTorch:
    @staticmethod
    def load(path, map_location=None, weights_only=False):
        del map_location, weights_only
        return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_incomplete(tmp_path, *, next_epoch=29, resume_step=0):
    root = tmp_path / "exp"
    ckpt = root / "checkpoints"
    best = ckpt / "best_model"
    best.mkdir(parents=True)
    (root / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "small_v21_manifest.json").write_text("{}\n", encoding="utf-8")
    (best / "model.safetensors").write_bytes(b"weights")
    (best / "config.json").write_text("{}\n", encoding="utf-8")
    (best / "best_metric.json").write_text(
        json.dumps({"daily_rank_ic_mean": 0.089, "segment": 10}) + "\n",
        encoding="utf-8",
    )
    progress = {
        "status": "running",
        "phase": "validation",
        "current_segment": 30,
        "total_segments": 520,
        "best_val_loss": 0.089,
    }
    (root / "progress.json").write_text(json.dumps(progress) + "\n", encoding="utf-8")
    state = {
        "next_epoch": next_epoch,
        "resume_step": resume_step,
        "batch_idx_global": 4553,
        "effective_epochs": 520,
        "scheduler_type": "warmup_constant",
        "optimizer": {"state": {"0": {}}},
        "amp_scaler": {},
        "rng_state": {"python": 1},
        "scheduler": {"last_epoch": 4553},
    }
    # Store JSON so FakeTorch.load can read it without real torch.
    (ckpt / "last_state.pt").write_text(json.dumps(state) + "\n", encoding="utf-8")
    return root, progress


def test_inspect_accepts_durable_last_state(tmp_path):
    root, _progress = _write_incomplete(tmp_path)
    boundary, state, progress = inspect_incomplete_boundary(root, _FakeTorch)
    assert boundary == 29
    assert state["resume_step"] == 0
    assert progress["status"] == "running"


def test_inspect_rejects_mid_segment_resume(tmp_path):
    root, _progress = _write_incomplete(tmp_path, resume_step=12)
    with pytest.raises(RuntimeError, match="resume_step"):
        inspect_incomplete_boundary(root, _FakeTorch)


def test_write_durable_sidecar_matches_validate_contract(tmp_path):
    root, progress = _write_incomplete(tmp_path)
    recipe = {
        "experiment": "small_0.1_clean_bigbatch2_from_e03b_20261023",
        "parent_best_sha256": "abc",
        "world_size": 2,
        "total_segments": 520,
    }
    write_durable_sidecar(
        root,
        29,
        progress,
        recipe,
        chunk=1,
        parent_kernel="wynstonliu/kronos-small-clean-bigbatch2-e01e-from-e03b",
        reason="unit-test recovery",
    )
    progress2 = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert progress2["status"] == "stopped"
    assert progress2["completed_segments"] == 29
    assert progress2["current_segment"] == 29
    assert summary["status"] == "stopped"
    assert summary["completed_segments"] == 29
    assert summary["final_result"]["completed_segments"] == 29
    assert summary["final_result"]["status"] == "stopped"
    assert manifest["recipe"]["parent_best_sha256"] == "abc"
    assert manifest["recovered_boundary_next_epoch"] == 29
    # Best Rank IC is untouched.
    best = json.loads((root / "checkpoints/best_model/best_metric.json").read_text())
    assert best["daily_rank_ic_mean"] == 0.089
    assert best["segment"] == 10
