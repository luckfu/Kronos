import ast
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
TRAINER = (
    ROOT
    / "finetune/kaggle_v1_beta_minimal/Kronos/finetune/train_predictor.py"
)
DATASET = ROOT / "finetune/kaggle_v1_beta_minimal/Kronos/finetune/dataset.py"
RUNNER = (
    ROOT
    / "finetune/kaggle_v1_beta_minimal/Kronos/finetune/"
    "kaggle_uniform_balanced_runner.py"
)
MANIFEST_ROOT = (
    ROOT / "data/a_share_full_market_v1_beta/balanced_validation_v1"
)


def load_functions(path, names, namespace=None):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    result = dict(namespace or {})
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), result)
    return result


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fixed_balanced_validation_artifacts_are_signed_and_ordered():
    manifest_path = MANIFEST_ROOT / "balanced_validation_manifest.json"
    samples_path = MANIFEST_ROOT / "balanced_validation_samples.jsonl"
    manifest = json.loads(manifest_path.read_text())
    records = [json.loads(line) for line in samples_path.read_text().splitlines()]

    assert sha256(manifest_path) == (
        "14afe39e260a90e438051512167f50ee29c4d8561a27fe09896ec261c79a4ba6"
    )
    assert sha256(samples_path) == manifest["selection"]["samples_file_sha256"]
    assert len(records) == 12_000
    assert all(record["quick"] for record in records[:3_000])
    assert not any(record["quick"] for record in records[3_000:])
    assert Counter(record["direction"] for record in records) == {
        "short": 4_000,
        "neutral": 4_000,
        "long": 4_000,
    }
    assert Counter(record["direction"] for record in records[:3_000]) == {
        "short": 1_000,
        "neutral": 1_000,
        "long": 1_000,
    }
    isolation = manifest["training_isolation"]
    assert isolation["training_candidate_overlap_samples"] == 11_999
    assert isolation["not_in_training_candidate_samples"] == 1
    assert isolation["not_in_training_candidates"] == [{
        "symbol": "sz.300332",
        "start_index": 116,
        "asof_date": "2026-06-22",
        "target_date": "2026-07-17",
    }]


def test_fixed_validation_excludes_the_reserved_july_holdout():
    manifest = json.loads(
        (MANIFEST_ROOT / "balanced_validation_manifest.json").read_text()
    )
    records = [
        json.loads(line)
        for line in (
            MANIFEST_ROOT / "balanced_validation_samples.jsonl"
        ).read_text().splitlines()
    ]

    assert max(record["asof_date"] for record in records) == "2026-06-30"
    isolation = manifest["final_test_isolation"]
    assert isolation["reserved_tuning_holdout_start"] == "2026-07-01"
    assert isolation["reserved_tuning_holdout_excluded_from_quick_and_large"]
    assert isolation["deployment_final_test_status"] == "pending_future_data"


def test_large_validation_milestones_are_first_tens_and_final():
    helper = load_functions(
        TRAINER, {"should_run_large_validation"}
    )["should_run_large_validation"]

    selected = [segment for segment in range(1, 121) if helper(segment, 529, 10)]
    assert selected == [1] + list(range(10, 121, 10))
    assert helper(529, 529, 10)
    assert not helper(528, 529, 10)
    with pytest.raises(ValueError, match="must be positive"):
        helper(1, 529, 0)


def test_uniform_scheduler_rejects_any_family_lr_difference():
    validate = load_functions(
        TRAINER, {"validate_uniform_learning_rate_config"}
    )["validate_uniform_learning_rate_config"]
    config = {
        "scheduler_type": "uniform_cosine",
        "predictor_learning_rate": 1e-6,
        "condition_learning_rate": 1e-6,
        "predictor_warmup_start_learning_rate": 5e-7,
        "condition_warmup_start_learning_rate": 5e-7,
        "predictor_min_learning_rate": 2e-7,
        "condition_min_learning_rate": 2e-7,
    }

    validate(config)
    config["condition_learning_rate"] = 2e-6
    with pytest.raises(ValueError, match="identical adaptation and condition peak LR"):
        validate(config)


def test_uniform_lr_trace_hits_start_peak_and_minimum_identically():
    schedule = load_functions(
        TRAINER, {"warmup_cosine_multiplier"}, {"math": math}
    )["warmup_cosine_multiplier"]
    total_steps = 330_000
    warmup_steps = 3_300
    for step, expected in ((0, 5e-7), (warmup_steps, 1e-6), (total_steps, 2e-7)):
        adaptation = 1e-6 * schedule(
            step, total_steps, warmup_steps, 5e-7, 1e-6, 2e-7
        )
        condition = 1e-6 * schedule(
            step, total_steps, warmup_steps, 5e-7, 1e-6, 2e-7
        )
        assert math.isclose(adaptation, expected)
        assert math.isclose(condition, expected)
        assert adaptation == condition


def test_resume_guard_signs_fixed_validation_and_exclusion_policy():
    source = TRAINER.read_text()

    assert "'fixed_validation_manifest_sha256'" in source
    assert "'validation_quick_samples'" in source
    assert "'validation_large_samples'" in source
    assert "'validation_large_interval_segments'" in source
    assert "'exclude_fixed_validation_from_training'" in source


def test_dataset_hard_fails_if_training_exclusion_count_is_incomplete():
    source = DATASET.read_text()

    assert "eligible &= ~excluded" in source
    assert "excluded_fixed_validation_samples != expected_training_exclusions" in source
    assert "Fixed validation training exclusion count mismatch" in source
    assert "Fixed validation panel SHA does not match its manifest" in source


def test_runner_bootstraps_from_parent_last_without_optimizer_state():
    source = RUNNER.read_text()

    assert "EXPECTED_PARENT_SEGMENT = 1058" in source
    assert "EXPECTED_PARENT_BEST_SEGMENT = 467" in source
    assert '"KRONOS_PREDICTOR_PATH": str(source_root / "checkpoints/last_model")' in source
    assert '"KRONOS_BOOTSTRAP_COMPLETED_SEGMENTS": "0"' in source
    assert '"KRONOS_RESUME_TRAINING": "1" if resume_training else "0"' in source
    assert 'config.get("KRONOS_SCHEDULER") != "two_speed"' in source
    assert 'config.get("KRONOS_SCHEDULER") != "uniform_cosine"' in source


def test_runner_declares_uniform_balanced_stage_contract():
    source = RUNNER.read_text()

    required_config = {
        '"KRONOS_PREDICTOR_LEARNING_RATE": "1e-6"',
        '"KRONOS_CONDITION_LEARNING_RATE": "1e-6"',
        '"KRONOS_PREDICTOR_WARMUP_START_LR": "5e-7"',
        '"KRONOS_CONDITION_WARMUP_START_LR": "5e-7"',
        '"KRONOS_PREDICTOR_MIN_LR": "2e-7"',
        '"KRONOS_CONDITION_MIN_LR": "2e-7"',
        '"KRONOS_SCHEDULER": "uniform_cosine"',
        '"KRONOS_COVERAGE_PASSES": "1"',
        '"KRONOS_MAX_SEGMENTS_PER_RUN": "150"',
        '"KRONOS_EXCLUDE_FIXED_VALIDATION_FROM_TRAINING": "1"',
    }
    for line in required_config:
        assert line in source
    assert 'output_validation = output_root / "validation"' in source
    assert 'VALIDATION_MANIFEST_FILENAME = "balanced_validation_manifest.json"' in source
    assert 'VALIDATION_SAMPLES_FILENAME = "balanced_validation_samples.jsonl"' in source
    assert '"parent_best_reference.json"' in source


def test_new_swanlab_parsers_include_missing_and_large_metrics():
    functions = load_functions(
        RUNNER,
        {"parse_validation_summary_line", "parse_large_validation_line"},
        {
            "VALIDATION_SUMMARY_RE": re_compile(
                r"Validation Train Average/Best: ([0-9.eE+-]+) / ([0-9.eE+-]+)"
            ),
            "LARGE_VALIDATION_LOG_RE": re_compile(
                r"Large Validation Objective/Forecast/History/Full: "
                r"([0-9.eE+-]+) / ([0-9.eE+-]+) / "
                r"([0-9.eE+-]+) / ([0-9.eE+-]+)"
            ),
        },
    )

    assert functions["parse_validation_summary_line"](
        "Validation Train Average/Best: 2.401234 / 2.429001"
    ) == {"train_average": 2.401234, "best_loss": 2.429001}
    assert functions["parse_large_validation_line"](
        "Large Validation Objective/Forecast/History/Full: "
        "2.430000 / 2.380000 / 2.500000 / 2.490000"
    ) == {
        "objective_loss": 2.43,
        "forecast_loss": 2.38,
        "history_loss": 2.5,
        "full_loss": 2.49,
    }


def re_compile(pattern):
    import re

    return re.compile(pattern)
