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
    """Ranking diagnostic recipe: same-day batches, not the forecast-best board."""
    source = RUNNER.read_text()
    head = source.split("EMBEDDED_KRONOS_ARCHIVE_B64", 1)[0]
    assert 'OUTPUT_NAME = "beta_v2_1_c1_dual_t4_wc"' in source
    assert 'SWANLAB_RUN_ID = "beta_v2_1_c1_dual_t4_rank"' in head
    assert "MAX_SEGMENTS_PER_RUN = 30" in source
    assert "MAX_RUNTIME_SECONDS = 23400" in source
    assert '"KRONOS_USE_BETA_V21_AUXILIARY": "1"' in source
    assert '"KRONOS_SAME_DAY_RANKING_BATCHES": "1"' in source
    assert '"KRONOS_PREDICTOR_LEARNING_RATE": PREDICTOR_LR' in source
    assert 'PREDICTOR_LR = "1e-5"' in head
    assert '"KRONOS_CONDITION_LEARNING_RATE": PREDICTOR_LR' in source
    assert '"KRONOS_PREDICTOR_WARMUP_START_LR": WARMUP_START_LR' in source
    assert '"KRONOS_CONDITION_WARMUP_START_LR": WARMUP_START_LR' in source
    assert 'WARMUP_START_LR = "1e-6"' in head
    assert '"KRONOS_PREDICTOR_MIN_LR": PREDICTOR_LR' in source
    assert '"KRONOS_CONDITION_MIN_LR": PREDICTOR_LR' in source
    assert '"KRONOS_SPLIT_TRUNK_HEAD_LR": "0"' in source
    assert '"KRONOS_SCHEDULER": "warmup_constant"' in source
    assert 'WARMUP_RATIO = "0.05"' in head
    assert '"KRONOS_BEST_SELECTION_METRIC": "ranking"' in source
    assert '"KRONOS_HISTORY_LOSS_WEIGHT": "0.02"' in source
    assert "same_day_batch_no_segment_date_sort" in head
    assert "SEGMENT_OFFSET = 0" in head
    assert "LAST_VERIFIED_FINISHED_CHART_SEGMENT" not in head
    assert "EXPECTED_BEST_CHART_SEGMENT = 155" in head
    assert '"KRONOS_COLLECT_VALIDATION_AUXILIARY": "0"' in source
    assert '"KRONOS_BATCH_SIZE": "32"' in source
    assert '"KRONOS_AMP_DTYPE": "float16"' in source
    assert "beta_v2_1_c1_dual_t4_v6" not in head
    assert "beta_v2_1_c1_dual_t4_p1b" not in head
    assert 'KRONOS_SCHEDULER": "warmup_cosine"' not in source
    assert 'assert recipe["warmup_ratio"] == "0.05"' in source
    assert 'assert recipe["segment_offset"] == "0"' in source
    assert 'assert recipe["max_runtime_seconds"] == "23400"' in source
    assert 'assert recipe["best_metric"] == "ranking"' in source
