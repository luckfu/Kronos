import runpy
import io
import json
import os
import re
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "finetune/kaggle_v1_beta_minimal/Kronos/finetune/kaggle_v3_bootstrap_runner.py"
BUILDER = ROOT / "finetune/build_kaggle_v3_bootstrap_kernel.py"


def test_v3_bootstrap_is_isolated_and_uses_two_thousand_windows():
    source = RUNNER.read_text()

    assert 'STAGE_ID = "beta_v3_base_dynamic_size_path_bootstrap_v1"' in source
    assert '"KRONOS_VALIDATION_QUICK_SAMPLES": "2000"' in source
    assert '"KRONOS_VALIDATION_LARGE_SAMPLES": "2000"' in source
    assert '"KRONOS_USE_SIZE_PATH": "1"' in source
    assert '"KRONOS_USE_SIZE_PERCENTILE": "0"' in source
    assert '"KRONOS_PREDICTOR_LEARNING_RATE": "1e-6"' in source
    assert '"KRONOS_CONDITION_LEARNING_RATE": "1e-5"' in source
    assert '"KRONOS_MAX_SEGMENTS_PER_RUN": "10"' in source
    assert 'NeoQuasar/Kronos-base' in source
    assert 'NeoQuasar/Kronos-Tokenizer-base' in source
    assert 'SWANLAB_RUN_ID = "kronos-beta-v3-base-dynamic-size-bootstrap"' in source
    assert '"torch==2.5.1"' in source
    assert '"monitor/{key}"' in source
    assert '"progress/coverage_fraction"' in source


def test_v3_dashboard_parses_training_validation_and_condition_metrics():
    namespace = runpy.run_path(str(RUNNER))
    train = namespace["parse_train_line"](
        "[Rank 0, Segment 3/473, Step 100/625] "
        "Adaptation LR 9.0e-7, Condition LR 9.0e-6, "
        "Loss: 2.40, Forecast: 2.38, History: 1.00"
    )
    validation = namespace["parse_validation_line"](
        "Validation Forecast/History/Full: 2.38 / 1.00 / 2.12"
    )
    monitor = namespace["parse_condition_monitor_line"](
        'Condition Monitor JSON: {"condition_trunk_norm_ratio": 0.1, '
        '"size_path_mlp_output_weight_norm": 0.2}'
    )

    assert train == {
        "segment": 3,
        "total_segments": 473,
        "step": 100,
        "total_steps": 625,
        "adaptation_learning_rate": 9e-7,
        "condition_learning_rate": 9e-6,
        "objective_loss": 2.4,
        "forecast_loss": 2.38,
        "history_loss": 1.0,
    }
    assert validation == {
        "objective_loss": 2.4,
        "forecast_loss": 2.38,
        "history_loss": 1.0,
        "full_loss": 2.12,
    }
    assert monitor["condition_trunk_norm_ratio"] == 0.1
    assert monitor["size_path_mlp_output_weight_norm"] == 0.2


def test_v3_dashboard_streams_required_metric_groups():
    namespace = runpy.run_path(str(RUNNER))

    class FakeRun:
        def __init__(self):
            self.records = []

        def log(self, payload, step):
            self.records.append((payload, step))

    lines = [
        "[Rank 0, Segment 1/473, Step 100/625] Adaptation LR 1e-6, "
        "Condition LR 1e-5, Loss: 2.4, Forecast: 2.38, History: 1.0",
        'Condition Monitor JSON: {"condition_trunk_norm_ratio": 0.1, '
        '"sector_embedding_weight_norm": 0.2, '
        '"size_path_mlp_output_weight_norm": 0.3, '
        '"condition_grad_total_l2": 0.4, "adaptation_grad_total_l2": 0.5}',
        "Validation Forecast/History/Full: 2.36 / 1.0 / 2.1",
    ]
    child_code = "print(" + "); print(".join(repr(line) for line in lines) + ")"
    run = FakeRun()
    namespace["stream_training"](
        [sys.executable, "-c", child_code], os.environ.copy(), run
    )
    payload = {key: value for record, _step in run.records for key, value in record.items()}

    assert payload["train/objective_loss"] == 2.4
    assert payload["validation/objective_loss"] == 2.38
    assert payload["learning_rate/trunk"] == 1e-6
    assert payload["learning_rate/condition"] == 1e-5
    assert payload["monitor/condition_trunk_norm_ratio"] == 0.1
    assert payload["monitor/sector_embedding_weight_norm"] == 0.2
    assert payload["monitor/size_path_mlp_output_weight_norm"] == 0.3
    assert payload["monitor/condition_grad_total_l2"] == 0.4
    assert payload["monitor/adaptation_grad_total_l2"] == 0.5
    assert 0 < payload["progress/coverage_fraction"] < 1


def test_v3_kernel_uses_dedicated_data_and_no_v2_parent_kernel():
    namespace = runpy.run_path(str(BUILDER))
    metadata = namespace["METADATA"]

    assert metadata["id"] == "luckfu/kronos-beta-v3-base-dynamic-size-bootstrap"
    assert metadata["kernel_sources"] == []
    assert metadata["dataset_sources"] == [
        "luckfu/kronos-a-share-full-market-v1-beta-120d"
    ]


def test_embedded_bootstrap_validation_contains_exactly_two_thousand_records():
    archive = runpy.run_path(str(BUILDER))["build_archive"]()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        manifest = json.load(bundle.extractfile(
            "Kronos/finetune/v3_data_contract/natural_validation_manifest.json"
        ))
        records = bundle.extractfile(
            "Kronos/finetune/v3_data_contract/natural_validation_samples.jsonl"
        ).read().splitlines()

    assert len(records) == 2000
    assert manifest["selection"]["quick_samples"] == 2000
    assert manifest["selection"]["large_samples"] == 2000
    assert manifest["training_isolation"]["not_in_training_candidate_samples"] == 2000
    assert manifest["bootstrap_subset"]["source_full_samples"] == 123982


def test_embedded_v3_source_contains_no_literal_swanlab_key():
    archive = runpy.run_path(str(BUILDER))["build_archive"]()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        python_sources = "\n".join(
            bundle.extractfile(member).read().decode("utf-8")
            for member in bundle.getmembers()
            if member.isfile() and member.name.endswith(".py")
        )

    assert "UserSecretsClient().get_secret(\"SWANLAB_API_KEY\")" in python_sources
    assert re.search(r"api_key\s*=\s*['\"][A-Za-z0-9]{16,}['\"]", python_sources) is None
