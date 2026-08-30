import ast
import base64
import io
import json
import re
import tarfile
from pathlib import Path


RUNNER = Path(__file__).parents[1] / "finetune" / "kaggle_v1_beta.py"
BEST109_RUNNER = (
    Path(__file__).parents[1]
    / "finetune"
    / "kaggle_v1_beta_best109_friend.py"
)


def test_kaggle_runners_load_swanlab_credentials_externally():
    for runner in (RUNNER, BEST109_RUNNER):
        tree = ast.parse(runner.read_text())
        assignment = next(
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "SWANLAB_API_KEY"
                for target in node.targets
            )
        )

        assert isinstance(assignment.value, ast.Call)
        assert runner.read_text().count("UserSecretsClient") == 2


def load_parser_functions(runner=RUNNER):
    tree = ast.parse(runner.read_text())
    wanted = {
        "TRAIN_LOG_RE",
        "VALIDATION_LOG_RE",
        "parse_swanlab_train_line",
        "parse_swanlab_validation_line",
        "parse_condition_monitor_line",
        "parse_condition_ablation_line",
        "CONDITION_MONITOR_PREFIX",
        "CONDITION_ABLATION_RE",
        "normalize_resume_position",
        "SWANLAB_STEPS_PER_SEGMENT",
        "swanlab_global_step",
    }
    nodes = [
        node for node in tree.body
        if getattr(node, "name", None) in wanted
        or any(getattr(target, "id", None) in wanted for target in getattr(node, "targets", []))
    ]
    namespace = {"re": re, "json": json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(runner), "exec"), namespace)
    return namespace


def test_best109_swanlab_axis_is_independent_of_physical_batch_steps():
    global_step = load_parser_functions(BEST109_RUNNER)["swanlab_global_step"]

    assert global_step(109, 1250, 1250) == 136250
    assert global_step(109, 625, 625) == 136250
    assert global_step(110, 1, 625) > 136250
    assert global_step(110, 625, 625) == 137500
    assert global_step(110, 1250, 1250) == 137500


def test_best109_swanlab_axis_preserves_segment_boundary_continuity():
    global_step = load_parser_functions(BEST109_RUNNER)["swanlab_global_step"]

    assert global_step(114, 625, 625) == 142500
    assert global_step(115, 0, 625) == 142500
    assert global_step(115, 100, 625) == 142700


def test_training_log_parser_captures_segment_and_step_totals():
    parser = load_parser_functions()["parse_swanlab_train_line"]
    record = parser(
        "[Rank 0, Segment 27/1058, Step 1150/1250] "
        "Adaptation LR 1.0000000000e-05, Condition LR 1.0000000000e-04, "
        "Loss: 2.3836, Forecast: 2.3319, History: 2.5830"
    )

    assert record == {
        "segment": 27,
        "total_segments": 1058,
        "step": 1150,
        "total_steps": 1250,
        "learning_rate": 1e-5,
        "adaptation_learning_rate": 1e-5,
        "condition_learning_rate": 1e-4,
        "loss": 2.3836,
        "forecast_loss": 2.3319,
        "history_loss": 2.583,
    }


def test_validation_log_parser_uses_latest_training_segment():
    parser = load_parser_functions()["parse_swanlab_validation_line"]
    record = parser(
        "Validation Forecast/History/Full: 2.4241 / 2.5945 / 2.5804",
        segment=27,
        total_steps=1250,
    )

    assert record == {
        "segment": 27,
        "forecast_loss": 2.4241,
        "history_loss": 2.5945,
        "full_loss": 2.5804,
        "step": 1250,
    }


def test_training_log_parser_preserves_high_precision_scientific_lr():
    parser = load_parser_functions()["parse_swanlab_train_line"]
    record = parser(
        "[Rank 0, Segment 57/1058, Step 100/1250] "
        "Adaptation LR 1.1234567890e-06, "
        "Condition LR 1.1234567890e-05, "
        "Loss: 2.3836, Forecast: 2.3319, History: 2.5830"
    )

    assert record["adaptation_learning_rate"] == 1.123456789e-6
    assert record["condition_learning_rate"] == 1.123456789e-5


def test_condition_monitor_parser_accepts_only_numeric_json():
    parser = load_parser_functions()["parse_condition_monitor_line"]
    record = parser(
        'Condition Monitor JSON: {"condition_output_norm": 0.25, '
        '"condition_grad_rms": 1.2e-5}'
    )

    assert record == {
        "condition_output_norm": 0.25,
        "condition_grad_rms": 1.2e-5,
    }


def test_condition_ablation_parser_preserves_signed_deltas():
    parser = load_parser_functions()["parse_condition_ablation_line"]
    record = parser(
        "Validation Condition Full/None/Shuffled Forecast: "
        "2.390100 / 2.395200 / 2.397300; "
        "Delta Full-None/Full-Shuffled: -0.005100 / -0.007200"
    )

    assert record == {
        "full_forecast_loss": 2.3901,
        "none_forecast_loss": 2.3952,
        "shuffled_forecast_loss": 2.3973,
        "full_minus_none_forecast_loss": -0.0051,
        "full_minus_shuffled_forecast_loss": -0.0072,
    }


def test_embedded_bundle_logs_learning_rate_with_high_precision():
    tree = ast.parse(RUNNER.read_text())
    archive_text = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id == "EMBEDDED_KRONOS_ARCHIVE_B64"
        and isinstance(node.value, ast.Constant)
    )
    archive_bytes = base64.b64decode(archive_text)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as bundle:
        trainer = bundle.extractfile("Kronos/finetune/train_predictor.py").read().decode()

    assert 'f"Adaptation LR {adaptation_lr:.10e}, "' in trainer
    assert 'f"Condition LR {condition_lr:.10e}, Loss:' in trainer


def test_swanlab_payloads_do_not_include_string_phase_metrics():
    source = RUNNER.read_text()

    assert 'payload["phase"]' not in source
    assert '"phase": "train"' not in source
    assert '"phase": "validation"' not in source


def test_resume_position_normalizes_a_completed_segment_boundary():
    normalize = load_parser_functions()["normalize_resume_position"]

    assert normalize(45, 1250, 1250) == (46, 0)
    assert normalize(59, 700, 1250) == (59, 700)


def test_runner_audits_only_same_segment_uncheckpointed_progress():
    source = RUNNER.read_text()

    assert "recoverable_same_segment_rollback" in source
    assert '"type": "discard_uncheckpointed_progress"' in source
    assert "discarded_metric_rows" in source
    assert "an unsafe rollback" in source
    assert "last_state.pt resume_step must be zero" in source
    assert '"resume_rebuild": "last_model"' in source
    required_history = source[source.index("required_history = ["):source.index("]", source.index("required_history = ["))]
    assert '"summary.json"' not in required_history
    assert '"checkpoints/last_model/model.safetensors"' not in required_history


def test_embedded_bundle_uses_segment_level_resume_only():
    tree = ast.parse(RUNNER.read_text())
    archive_text = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id == "EMBEDDED_KRONOS_ARCHIVE_B64"
        and isinstance(node.value, ast.Constant)
    )
    archive_bytes = base64.b64decode(archive_text)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as bundle:
        trainer = bundle.extractfile("Kronos/finetune/train_predictor.py").read().decode()
        config = bundle.extractfile("Kronos/finetune/config.py").read().decode()

    assert "Periodic resume checkpoint saved at segment" not in trainer
    assert "resume_checkpoint_interval_steps" not in config
    assert "KRONOS_RESUME_CHECKPOINT_INTERVAL_STEPS" not in RUNNER.read_text()
    assert "persist_resume_checkpoint(next_segment)" in trainer
    assert "persist_resume_checkpoint(start_epoch)" in trainer
    assert "resume_step=0" in trainer
    assert "if resume_step != 0" in trainer
    assert "persist_resume_checkpoint(epoch_idx" not in trainer
    assert trainer.index("save_pretrained_with_retry(", trainer.index("if improved:")) < trainer.index(
        "persist_resume_checkpoint(next_segment)"
    )
    assert "best_model', 'best_metric.json'" in trainer


def test_embedded_bundle_uses_the_optimized_training_plan():
    tree = ast.parse(RUNNER.read_text())
    archive_text = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id == "EMBEDDED_KRONOS_ARCHIVE_B64"
        and isinstance(node.value, ast.Constant)
    )
    archive_bytes = base64.b64decode(archive_text)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as bundle:
        trainer = bundle.extractfile("Kronos/finetune/train_predictor.py").read().decode()
        config = bundle.extractfile("Kronos/finetune/config.py").read().decode()

    assert "torch.optim.lr_scheduler.LambdaLR" in trainer
    assert "warmup_cosine_multiplier" in trainer
    assert "two_speed_multiplier" in trainer
    assert "condition_fast_decay_steps" in trainer
    assert "parameter_family_statistics" in trainer
    assert "Validation Condition Full/None/Shuffled Forecast" in trainer
    assert "parameter_uses_weight_decay" in trainer
    assert "scheduler_warmup_steps=warmup_steps" in trainer
    assert '"warmup_cosine"' in config
    assert 'KRONOS_SCHEDULER_WARMUP_RATIO", "0.02"' in config


def test_twospeed_runner_bootstraps_from_parent_best_without_model_download():
    source = RUNNER.read_text()

    assert 'expected_parent_segment = 11' in source
    assert 'expected_parent_objective = 2.439993896484375' in source
    assert 'source_scheduler == "two_speed"' in source
    assert 'base_model_dir = bootstrap_best_dir' in source
    assert '"KRONOS_RESUME_TRAINING": "1" if' in source
    assert '"KRONOS_SCHEDULER": "two_speed"' in source
    assert '"KRONOS_PREDICTOR_LEARNING_RATE": "3e-6"' in source
    assert '"KRONOS_CONDITION_LEARNING_RATE": "3e-5"' in source
    assert '"KRONOS_MAX_SEGMENTS_PER_RUN": "120"' in source
    assert '"KRONOS_BATCH_SIZE": "32"' in source
    assert '"KRONOS_USE_AMP": "1"' in source
    assert 'modelscope", "download"' not in source


def test_embedded_bundle_persists_amp_for_exact_continuation():
    tree = ast.parse(RUNNER.read_text())
    archive_text = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id == "EMBEDDED_KRONOS_ARCHIVE_B64"
        and isinstance(node.value, ast.Constant)
    )
    archive_bytes = base64.b64decode(archive_text)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as bundle:
        trainer = bundle.extractfile("Kronos/finetune/train_predictor.py").read().decode()
        config = bundle.extractfile("Kronos/finetune/config.py").read().decode()

    assert 'self.use_amp = os.getenv(' in config
    assert "'batch_size', 'use_amp', 'n_train_iter'" in trainer
    assert "scaler = torch.amp.GradScaler('cuda', enabled=scale_gradients)" in trainer
    assert 'tokenizer encoding remains float32' in trainer
    assert 'scaler.scale(loss).backward()' in trainer
    assert 'scaler.unscale_(optimizer)' in trainer
    assert 'scaler.step(optimizer)' in trainer
    assert 'amp_scaler=scaler.state_dict()' in trainer
    assert "scaler.load_state_dict(resume_state['amp_scaler'])" in trainer
