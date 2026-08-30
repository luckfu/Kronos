import ast
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
PREPARE = ROOT / "finetune/prepare_v1_beta_evaluation.py"
EVALUATOR = ROOT / "finetune/evaluate_v1_beta_checkpoints.py"
KAGGLE_RUNNER = ROOT / "finetune/kaggle_v1_beta_checkpoint_evaluation.py"
MANIFEST = (
    ROOT / "data/a_share_v1_beta_eval_20260826/package/evaluation_manifest.json"
)


def load_functions(path, names, namespace=None):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    result = dict(namespace or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), result)
    return result


def test_future_evaluation_is_strictly_after_parent_latest_target():
    manifest = json.loads(MANIFEST.read_text())
    isolation = manifest["temporal_isolation"]

    assert isolation["parent_latest_training_target"] == "2026-07-31"
    assert isolation["future_signal_start"] == "2026-08-03"
    assert isolation["future_signal_start"] > isolation["parent_latest_training_target"]
    assert isolation["strictly_after_parent_latest_training_target"]
    assert manifest["sample_sets"]["future_all"]["date_count"] == 6
    assert manifest["sample_sets"]["future_all"]["samples"] == 30930


def test_future_balanced_set_is_exactly_equal_by_direction():
    manifest = json.loads(MANIFEST.read_text())
    balanced = manifest["sample_sets"]["future_balanced"]

    assert balanced["samples"] == 3000
    assert balanced["direction_counts"] == {
        "long": 1000,
        "neutral": 1000,
        "short": 1000,
    }


def test_direction_thresholds_match_validation_contract():
    direction = load_functions(
        PREPARE,
        {"direction_name"},
        {"DIRECTION_NAMES": {-1: "short", 0: "neutral", 1: "long"}},
    )["direction_name"]

    assert direction(-0.010001) == "short"
    assert direction(-0.01) == "neutral"
    assert direction(0.01) == "neutral"
    assert direction(0.010001) == "long"


def test_three_class_prediction_thresholds_are_identical():
    classify = load_functions(EVALUATOR, {"three_class"}, {"np": np})["three_class"]

    assert classify([-0.02, -0.01, 0.0, 0.01, 0.02]).tolist() == [
        "short",
        "neutral",
        "neutral",
        "neutral",
        "long",
    ]


def test_evaluator_is_read_only_with_respect_to_model_training():
    source = EVALUATOR.read_text()

    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "optimizer.step" not in source
    assert '"training_performed": False' in source
    assert "EXPECTED_BEST_SEGMENT = 467" in source
    assert "EXPECTED_LAST_SEGMENT = 1058" in source


def test_kaggle_runner_guards_p100_cuda_architecture():
    source = KAGGLE_RUNNER.read_text()

    assert '"sm_60"' not in source  # The generated Python check uses single quotes.
    assert "'sm_60'" in source
    assert '"torch==2.5.1"' in source
    assert '"https://download.pytorch.org/whl/cu121"' in source
