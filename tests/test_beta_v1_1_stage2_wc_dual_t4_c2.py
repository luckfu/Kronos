import importlib.util
import json
from pathlib import Path

import pytest


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "finetune/kaggle_beta_v1_1_stage2_wc_dual_t4_c2"
    / "kaggle_beta_v1_1_stage2_wc_dual_t4_c2.py"
)
spec = importlib.util.spec_from_file_location("beta_v1_1_c2", RUNNER)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_parent(tmp_path):
    parent = tmp_path / "c1" / module.OUTPUT_NAME
    for name in (
        "run.log", "metrics.jsonl", "checkpoints/last_state.pt",
        "checkpoints/last_model/model.safetensors",
        "checkpoints/best_model/model.safetensors",
    ):
        target = parent / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    (parent / "progress.json").write_text(json.dumps({
        "current_segment": 10, "unique_windows_covered": 200000,
        "total_segments": 534, "status": "stopped",
    }))
    (parent / "summary.json").write_text(json.dumps({
        "world_size": 2, "final_result": {
            "completed_segments": 10, "resume_segment": 11,
            "stop_reason": "segment_limit", "train_selection": {"coverage_seed": 20260924},
        },
    }))
    (parent / "experiment_manifest.json").write_text(json.dumps({
        "parent": {"checkpoint": "Best@818"},
        "training": {"gpu": "2x Tesla T4"},
        "data": {"data_manifest_sha256":
                 "32cfcbf606dcee81c9416f9ad399ee7303b6790cf9025bf5888638f552d66598"},
    }))
    return parent


def test_only_completed_c1_can_resume(tmp_path):
    parent = make_parent(tmp_path)
    assert module.validate_parent(tmp_path) == parent
    summary = parent / "summary.json"
    data = json.loads(summary.read_text())
    data["final_result"]["resume_segment"] = 10
    summary.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="resume_segment"):
        module.validate_parent(tmp_path)


def test_missing_checkpoint_refuses_fresh_start(tmp_path):
    with pytest.raises(RuntimeError, match="exactly one C1 checkpoint"):
        module.validate_parent(tmp_path)


def test_timing_average_uses_current_chunk_not_global_segment():
    class Run:
        def __init__(self):
            self.rows = []

        def log(self, payload, **kwargs):
            self.rows.append(payload)

    run = Run()
    logger = module.TimingRun(run)
    logger.log({"timing/segment_seconds": 400, "segment": 11})
    logger.log({"timing/segment_seconds": 420, "segment": 12})
    assert [row["timing/avg_segment_seconds"] for row in run.rows] == [400, 410]
