import ast
import copy
import math
from pathlib import Path

ROOT = Path(__file__).parents[1]
TRAINER = ROOT / "finetune/train_predictor.py"
CONFIG = ROOT / "finetune/config.py"
RUNNER = (
    ROOT
    / "finetune/kaggle_small_0_1_stage2_cosine_refinement_c1_kernel"
    / "kaggle_small_0_1_stage2_cosine_refinement_c1.py"
)


def load_helpers():
    tree = ast.parse(TRAINER.read_text())
    wanted = {
        "optimizer_to",
        "restore_optimizer_for_scheduler_transition",
        "warmup_cosine_multiplier",
    }
    nodes = [node for node in tree.body if getattr(node, "name", None) in wanted]
    class FakeTorch:
        @staticmethod
        def is_tensor(_value):
            return False

    namespace = {"math": math, "torch": FakeTorch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAINER), "exec"), namespace)
    return namespace


def test_zero_warmup_cosine_starts_at_peak_and_ends_at_minimum():
    schedule = load_helpers()["warmup_cosine_multiplier"]
    total_steps = 83_571
    assert math.isclose(1e-5 * schedule(0, total_steps, 0, 1e-5, 1e-5, 1e-6), 1e-5)
    assert math.isclose(1e-5 * schedule(total_steps, total_steps, 0, 1e-5, 1e-5, 1e-6), 1e-6)


def test_scheduler_transition_preserves_adamw_moments_but_uses_new_lr_plan():
    class FakeOptimizer:
        def __init__(self, groups):
            self.param_groups = groups
            self.state = {}

        def load_state_dict(self, state):
            self.state = copy.deepcopy(state["state"])
            self.param_groups = copy.deepcopy(state["param_groups"])

    source_state = {
        "state": {0: {"step": 100, "exp_avg": [0.125], "exp_avg_sq": [0.25]}},
        "param_groups": [{
            "params": [0], "name": "adaptation_decay", "family": "adaptation",
            "lr": 9e-6, "peak_lr": 1e-5, "warmup_start_lr": 1e-5,
            "min_lr": 1e-6, "weight_decay": 0.1,
        }],
    }
    target = FakeOptimizer([{
        "params": [0], "name": "adaptation_decay", "family": "adaptation",
        "lr": 1e-5, "initial_lr": 1e-5, "peak_lr": 1e-5,
        "warmup_start_lr": 1e-5, "min_lr": 1e-6, "weight_decay": 0.1,
    }])
    target_plan = [
        {key: value for key, value in target.param_groups[0].items() if key != "params"}
    ]
    load_helpers()["restore_optimizer_for_scheduler_transition"](
        target, {"optimizer": source_state}, target_plan, "cpu"
    )

    assert target.state[0]["exp_avg"] == [0.125]
    assert target.param_groups[0]["lr"] == 1e-5
    assert target.param_groups[0]["min_lr"] == 1e-6


def test_refinement_kernel_declares_new_stage_contract():
    source = RUNNER.read_text()
    required = (
        'EXPECTED_PARENT_SEGMENT = 534',
        'REFINEMENT_SEGMENTS = 267',
        'MAX_RUNTIME_SECONDS = 39600',
        'COVERAGE_SEED = "20260912"',
        '"KRONOS_SCHEDULER": "uniform_cosine"',
        '"KRONOS_SCHEDULER_WARMUP_RATIO": "0"',
        '"KRONOS_PREDICTOR_MIN_LR": "1e-6"',
        '"KRONOS_CONDITION_MIN_LR": "1e-6"',
        '"KRONOS_SCHEDULER_TRANSITION_STATE": str(source_state)',
        '"SWANLAB_RUN_ID": "small_0.1_stage2_cosine_refinement"',
    )
    for declaration in required:
        assert declaration in source
    assert "KRONOS_SMALL_V21_CONTINUATION_ROOT" not in source


def test_config_accepts_literal_zero_warmup():
    source = CONFIG.read_text()
    assert "0 <= self.scheduler_warmup_ratio < 1" in source
    assert "KRONOS_SCHEDULER_TRANSITION_STATE" in source
