import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).parents[1] / "finetune" / "relay_a800_metrics_to_swanlab.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("a800_swanlab_relay", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeRun:
    def __init__(self):
        self.calls = []

    def log(self, payload, step):
        self.calls.append((payload, step))


class FakeSwanlab:
    def __init__(self):
        self.login_calls = []

    def login(self, **kwargs):
        self.login_calls.append(kwargs)


def test_login_uses_local_profile_without_an_environment_key():
    relay = load_module()
    swanlab = FakeSwanlab()

    relay.login_swanlab(swanlab, "")

    assert swanlab.login_calls == [{}]


def test_login_uses_explicit_environment_key_when_available():
    relay = load_module()
    swanlab = FakeSwanlab()

    relay.login_swanlab(swanlab, "test-key")

    assert swanlab.login_calls == [{"api_key": "test-key"}]


def test_metric_payload_flattens_period_metrics():
    relay = load_module()
    record = {
        "type": "validation",
        "segment": 5,
        "loss": 2.46,
        "period_metrics": {"2025H2": {"objective_loss": 2.44, "samples": 3122}},
    }

    payload = relay.metric_payload(record)

    assert payload["validation/quick/loss"] == 2.46
    assert payload["validation/quick/period_metrics/2025H2/objective_loss"] == 2.44
    assert payload["validation/quick/period_metrics/2025H2/samples"] == 3122
    assert payload["progress/segment"] == 5


def test_metric_payload_exposes_progress_and_named_consistency_horizons():
    relay = load_module()
    record = {
        "type": "validation_large",
        "segment": 2,
        "total_segments": 946,
        "beta_v21_score": 0.9286,
        "return_path_consistency": {
            "mae": [0.7, 0.8, 0.6, 0.9],
            "sign_agreement": [0.51, 0.52, 0.53, 0.54],
            "samples": 2048,
        },
    }

    payload = relay.metric_payload(record)

    assert payload["validation/full/beta_v21_score"] == 0.9286
    assert payload["validation/full/return_path_consistency/mae/D5"] == 0.6
    assert (
        payload["validation/full/return_path_consistency/sign_agreement/D10"]
        == 0.54
    )
    assert payload["progress/completed_segments"] == 2
    assert payload["progress/total_segments"] == 946
    assert payload["progress/completion_percent"] == 100 * 2 / 946


def test_train_progress_includes_current_segment_fraction():
    relay = load_module()
    record = {
        "type": "train",
        "segment": 3,
        "total_segments": 10,
        "step": 156,
        "total_steps": 312,
    }

    payload = relay.metric_payload(record)

    assert payload["progress/completed_segments"] == 2.5
    assert payload["progress/completion_percent"] == 25.0


def test_global_step_is_monotonic_across_segment_boundary():
    relay = load_module()

    assert relay.global_step({"type": "train", "segment": 1, "step": 625, "total_steps": 625}) == 625
    assert relay.global_step({"type": "validation", "segment": 1, "total_steps": 625}) == 625
    assert relay.global_step({"type": "train", "segment": 2, "step": 100, "total_steps": 625}) == 725


def test_explicit_steps_per_segment_fixes_validation_without_total_steps():
    relay = load_module()

    train = {"type": "train", "segment": 2, "step": 287, "total_steps": 313}
    validation = {"type": "validation_large", "segment": 2, "step": 0}
    next_train = {"type": "train", "segment": 3, "step": 74, "total_steps": 313}

    assert relay.global_step(train, 313) == 600
    assert relay.global_step(validation, 313) == 626
    assert relay.global_step(next_train, 313) == 700


def test_complete_line_checkpoint_does_not_consume_partial_json(tmp_path):
    relay = load_module()
    state_path = tmp_path / "state.json"
    state = {"offset": 0, "baseline_logged": False, "records_logged": 0}
    first = json.dumps({
        "type": "train", "segment": 2, "step": 100,
        "total_steps": 625, "loss": 2.3,
    }).encode() + b"\n"
    partial = b'{"type":"train"'
    run = FakeRun()

    relay.log_complete_lines(run, state_path, state, 0, first + partial)

    assert len(run.calls) == 1
    assert state["offset"] == len(first)
    assert state["records_logged"] == 1
    assert json.loads(state_path.read_text())["offset"] == len(first)
