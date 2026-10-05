"""Phase 1 trunk/head AdamW split and dual-T4 kernel recipe."""
import ast
import math
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).parents[1]
TRAINER = ROOT / "finetune/train_predictor.py"
RUNNER = ROOT / "finetune/kaggle_beta_v21_c1_dual_t4.py"


def load_helpers():
    tree = ast.parse(TRAINER.read_text())
    wanted = {
        "is_condition_parameter",
        "is_adaptation_head_parameter",
        "parameter_optimizer_family",
        "parameter_uses_weight_decay",
        "build_optimizer_groups",
        "learning_rates_by_family",
        "best_selection_value",
    }
    nodes = []
    for node in tree.body:
        if getattr(node, "name", None) in wanted:
            nodes.append(node)
        elif isinstance(node, ast.Assign):
            targets = [getattr(target, "id", None) for target in node.targets]
            if "ADAPTATION_HEAD_PREFIXES" in targets:
                nodes.append(node)
    namespace = {"math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAINER), "exec"), namespace)
    return namespace


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(4, 4)
        self.transformer = nn.ModuleList([nn.Linear(4, 4)])
        self.norm = nn.LayerNorm(4)
        self.dep_layer = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)
        self.return_head = nn.Linear(4, 4)
        self.barrier_head = nn.Linear(4, 3)
        self.sector_emb = nn.Embedding(3, 4)
        self.size_mlp = nn.Linear(2, 4)


def _config(split):
    return {
        "split_trunk_head_learning_rate": split,
        "predictor_learning_rate": 1e-6,
        "condition_learning_rate": 1e-5,
        "predictor_warmup_start_learning_rate": 1e-6,
        "condition_warmup_start_learning_rate": 1e-5,
        "predictor_min_learning_rate": 1e-6,
        "condition_min_learning_rate": 1e-5,
        "adam_weight_decay": 0.1,
    }


def _families(split):
    helpers = load_helpers()
    model = _Tiny()
    groups = helpers["build_optimizer_groups"](model, _config(split))
    by_id = {}
    for group in groups:
        for parameter in group["params"]:
            by_id[id(parameter)] = (
                group["family"], group["peak_lr"], group["warmup_start_lr"]
            )
    mapping = {}
    for name, parameter in model.named_parameters():
        mapping[name] = by_id[id(parameter)]
    return mapping


def test_default_keeps_heads_on_predictor_lr():
    mapping = _families(False)
    family, peak, start = mapping["transformer.0.weight"]
    assert family == "adaptation" and peak == 1e-6 and start == 1e-6
    for name in (
        "norm.weight", "dep_layer.weight", "head.weight",
        "return_head.weight", "barrier_head.weight",
    ):
        family, peak, _start = mapping[name]
        assert family == "adaptation" and peak == 1e-6, name
    family, peak, start = mapping["sector_emb.weight"]
    assert family == "condition" and peak == 1e-5 and start == 1e-5
    family, peak, _start = mapping["size_mlp.weight"]
    assert family == "condition" and peak == 1e-5


def test_phase1_split_puts_heads_and_condition_at_1e5_and_trunk_at_1e6():
    mapping = _families(True)
    trunk = ("embedding.weight", "transformer.0.weight", "transformer.0.bias")
    heads = (
        "norm.weight", "norm.bias", "dep_layer.weight", "dep_layer.bias",
        "head.weight", "head.bias", "return_head.weight", "return_head.bias",
        "barrier_head.weight", "barrier_head.bias",
        "sector_emb.weight", "size_mlp.weight", "size_mlp.bias",
    )
    for name in trunk:
        family, peak, start = mapping[name]
        assert family == "adaptation", name
        assert peak == 1e-6 and start == 1e-6, name
    for name in heads:
        family, peak, start = mapping[name]
        assert family == "condition", name
        assert peak == 1e-5 and start == 1e-5, name


def test_forecast_selection_uses_weighted_forecast():
    helper = load_helpers()["best_selection_value"]
    metrics = {
        "objective_loss": 1.0,
        "full_sequence_loss": 1.0,
        "forecast_loss": 2.40,
        "weighted_forecast_loss": 2.31,
        "history_loss": 1.0,
    }
    assert helper("forecast", metrics) == 2.31


def test_ranking_selection_is_ranking_loss_not_score():
    helper = load_helpers()["best_selection_value"]
    metrics = {
        "objective_loss": 1.0,
        "full_sequence_loss": 1.0,
        "forecast_loss": 2.40,
        "weighted_forecast_loss": 2.31,
        "history_loss": 1.0,
        "ranking_loss": 0.48,
        "beta_v21_score": 0.91,
    }
    assert helper("ranking", metrics) == 0.48
    assert helper("forecast", metrics) == 2.31


def test_phase1_kernel_recipe():
    """v20 recipe: unfreeze all, trunk 1e-7 constant, heads 1e-6->1e-5, pairwise best."""
    source = RUNNER.read_text()
    head = source.split("EMBEDDED_KRONOS_ARCHIVE_B64", 1)[0]
    assert 'OUTPUT_NAME = "beta_v2_1_c1_dual_t4_rank_unfreeze"' in head
    assert 'PARENT_OUTPUT_NAME = "beta_v2_1_c1_dual_t4_rank_frozen"' in head
    assert 'PARENT_KERNEL = "luckfu/kronos-beta-v2-1-c1-dual-t4"' in head
    assert 'SWANLAB_RUN_ID = "beta_v2_1_c1_dual_t4_rank_unfreeze"' in head
    assert 'KERNEL_VERSION = "v20"' in head
    assert "MAX_SEGMENTS_PER_RUN = 30" in source
    assert "MAX_RUNTIME_SECONDS = 43200" in source
    assert 'COVERAGE_SEED = "20261002"' in source
    assert '"KRONOS_USE_BETA_V21_AUXILIARY": "1"' in source
    assert '"KRONOS_SAME_DAY_RANKING_BATCHES": "1"' in source
    assert '"KRONOS_BETA_V21_RANKING_WEIGHT": RANKING_WEIGHT' in source
    assert 'RANKING_WEIGHT = "0.5"' in head
    assert 'PREDICTOR_LR = "1e-7"' in head
    assert 'PREDICTOR_WARMUP_START_LR = "1e-7"' in head
    assert 'CONDITION_LR = "1e-5"' in head
    assert 'CONDITION_WARMUP_START_LR = "1e-6"' in head
    assert 'WARMUP_RATIO = "0.05"' in head
    assert '"KRONOS_PREDICTOR_MIN_LR": PREDICTOR_LR' in source
    assert '"KRONOS_CONDITION_MIN_LR": CONDITION_LR' in source
    assert '"KRONOS_SPLIT_TRUNK_HEAD_LR": "1"' in source
    assert '"KRONOS_SCHEDULER": "warmup_constant"' in source
    assert '"KRONOS_TRAINABLE_TRANSFORMER_LAYERS": "-1"' in source
    assert '"KRONOS_TRAIN_BETA_V21_HEADS_ONLY": "0"' in source
    assert '"KRONOS_BEST_SELECTION_METRIC": "pairwise_accuracy"' in source
    assert '"KRONOS_KEEP_EXISTING_BEST": "1"' in source
    assert 'FORECAST_MONITOR_BASE = "2.31236787"' in head
    assert 'FORECAST_MONITOR_MARGIN = "0.015"' in head
    assert 'FORECAST_DRIFT_ALERT_BASE = "2.31236782"' in head
    assert 'FORECAST_DRIFT_ALERT_MARGIN = "0.005"' in head
    assert "same_day_batch_no_segment_date_sort" in head
    assert "SEGMENT_OFFSET = 0" in head
    assert "EXPECTED_PARENT_SEGMENT = 19" in head
    assert "EXPECTED_PARENT_PAIRWISE_ACCURACY = 0.65735263" in head
    assert "1fe7ae4bf6cfc2c068be4d6755e2c528550456325bbb81cf900ed064078d3d58" in head
    assert "find_seg19_pairwise_best" in source
    assert 'PARENT_DATASET = "luckfu/kronos-beta-v21-c1-rank-frozen-seg19-best"' in head
    assert '"KRONOS_BATCH_SIZE": "32"' in source
    assert '"KRONOS_AMP_DTYPE": "float16"' in source
    assert 'assert recipe["kernel_version"] == "v20"' in source
    assert 'assert recipe["heads_only"] == "0"' in source
    assert 'assert recipe["trainable_layers"] == "-1"' in source
    assert 'assert recipe["predictor_lr"] == "1e-7"' in source
    assert 'assert recipe["condition_lr"] == "1e-5"' in source


def _drift_helper():
    tree = ast.parse(TRAINER.read_text())
    nodes = [
        node for node in tree.body
        if getattr(node, "name", None) == "forecast_drift_alert_lines"
    ]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAINER), "exec"), namespace)
    return namespace["forecast_drift_alert_lines"]


def test_forecast_drift_alert_quiet_within_margin():
    helper = _drift_helper()
    lines = helper(2.3160, 2.31236782, 0.005, 1, 5, calibration_baseline=2.31236782)
    assert len(lines) == 1
    assert lines[0].startswith("Forecast drift monitor (not a stop): segment=1 ")
    assert "delta=+0.00363218" in lines[0]
    assert not any("WARNING" in line for line in lines)


def test_forecast_drift_alert_loud_and_early():
    helper = _drift_helper()
    lines = helper(2.3200, 2.31236782, 0.005, 2, 5)
    assert any("WARNING: EARLY FORECAST DRIFT ALERT segment=2" in line for line in lines)
    late = helper(2.3200, 2.31236782, 0.005, 9, 5)
    assert any(line.startswith("WARNING: FORECAST DRIFT ALERT segment=9") for line in late)
    assert helper(2.32, None, 0.005, 1, 5) == []


def test_drift_monitor_line_matches_runner_regex():
    import re
    helper = _drift_helper()
    line = helper(2.3200, 2.31236782, 0.005, 3, 5)[0]
    pattern = re.compile(
        r"Forecast drift monitor \(not a stop\): segment=(\d+) "
        r"weighted_forecast_loss=([0-9.eE+-]+) alert_base=([0-9.eE+-]+) "
        r"delta=([0-9.eE+-]+)"
    )
    match = pattern.search(line)
    assert match and float(match.group(4)) > 0.005


def test_heads_only_missing_adaptation_family_lr_print_safe():
    """Frozen-trunk heads-only: adaptation group empty; LR print must not KeyError."""
    helpers = load_helpers()
    model = _Tiny()
    # Mimic configure_trainable_parameters heads-only: only return/barrier heads.
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(('return_head.', 'barrier_head.'))
    config = _config(True)
    groups = helpers['build_optimizer_groups'](model, config)
    assert groups, 'expected non-empty condition groups'
    assert all(group['family'] == 'condition' for group in groups)
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    family_lrs = helpers['learning_rates_by_family'](optimizer)
    assert 'condition' in family_lrs and family_lrs['condition'] > 0
    assert family_lrs.get('adaptation', 0.0) == 0.0
    # Same f-string contract as train_model keep_existing_best / step logs.
    rendered = (
        f'adaptation_lr={family_lrs["adaptation"]:.10e}, '
        f'condition_lr={family_lrs["condition"]:.10e}.'
    )
    assert 'adaptation_lr=0.0000000000e+00' in rendered
    assert 'condition_lr=' in rendered
    step_line = (
        f"Adaptation LR {family_lrs['adaptation']:.10e}, "
        f"Condition LR {family_lrs['condition']:.10e}, Loss: 0.0000"
    )
    assert step_line.startswith('Adaptation LR 0.0000000000e+00, Condition LR ')

