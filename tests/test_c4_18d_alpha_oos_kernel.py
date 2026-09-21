import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parents[1]
KERNEL_DIR = ROOT / "finetune/kaggle_c4_18d_alpha_oos"
KERNEL = KERNEL_DIR / "c4_18d_alpha_oos.py"
META = KERNEL_DIR / "kernel-metadata.json"


def load_module():
    spec = importlib.util.spec_from_file_location("c4_18d_alpha_oos", KERNEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kernel_is_c4_only_locked_decode_and_does_not_train():
    source = KERNEL.read_text()
    for token in (
        '"training_performed": False',
        '"decode_search_performed": False',
        "EXPECTED_EVAL_NAME = \"kronos_beta_v2_time_oos_through_20260903\"",
        "RESIDUAL_IC_BAR = 0.15",
        "NEWEY_WEST_LAG = 10",
        "small_0.1_stage2_c2_best_wc_1e5_c4/checkpoints/best_model/model.safetensors",
        '["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT]',
        "9c5605d08e4f63e223d7b5edf9c6c8d2acad88ddfe022e5b27c098588d743075",
    ):
        assert token in source, token
    assert "JOINT_IC_SLACK" not in source
    assert "cosine_refinement" not in source
    assert "last_model/model.safetensors" not in source
    module = load_module()
    assert list(module.CHECKPOINTS) == ["wc_1e5_c4_best"]
    assert module.CHECKPOINTS["wc_1e5_c4_best"]["segment"] == 21
    assert [arm["name"] for arm in module.ARMS] == [
        "prod_t065_p80_n5",
        "rank_t060_p90_n16",
    ]
    assert module.ARMS[0]["temperature"] == 0.65
    assert module.ARMS[0]["top_p"] == 0.8
    assert module.ARMS[0]["sample_count"] == 5
    assert module.ARMS[1]["temperature"] == 0.6
    assert module.ARMS[1]["top_p"] == 0.9
    assert module.ARMS[1]["sample_count"] == 16
    assert module.EXPECTED_SIGNAL_DATES == 18
    assert module.EXPECTED_SAMPLES == 92751


def test_newey_west_inflates_variance_for_positively_overlapped_series():
    module = load_module()
    rng = np.random.default_rng(0)
    innovations = rng.normal(size=200)
    overlapped = np.convolve(innovations, np.ones(10) / 10.0, mode="valid")
    naive_std = float(overlapped.std(ddof=1))
    nw = module.overlapping_icir(overlapped, lag=10)
    assert nw["newey_west_std"] > naive_std
    assert nw["newey_west_icir"] < nw["naive_icir"]


def test_industry_size_residualization_kills_pure_style_alpha():
    module = load_module()
    rows = []
    for date in ("2026-08-11", "2026-08-12"):
        for sector, size in (
            ("银行", 9),
            ("银行", 8),
            ("电子", 1),
            ("电子", 0),
        ):
            rows.append({
                "asof_date": date,
                "sector": sector,
                "size_decile": size,
                "predicted_return_d10": 0.01 * size,
                "actual_return_d10": 0.01 * size + (0.05 if sector == "银行" else 0.0),
            })
    frame = pd.DataFrame(rows)
    residual = module.residualize_column(frame, "actual_return_d10")
    assert abs(residual.mean()) < 1e-10
    raw = frame["predicted_return_d10"].corr(frame["actual_return_d10"], method="spearman")
    assert raw > 0.5
    assert np.max(np.abs(residual)) < 1e-10


def test_nonoverlap_and_bootstrap_shapes():
    module = load_module()
    ics = [0.2] * 18
    strands = module.nonoverlap_strands(ics, stride=10)
    assert strands["stride"] == 10
    assert len(strands["strands"]) == 10
    assert strands["strands"][0]["n"] == 2
    boot = module.moving_block_bootstrap(ics, block=10, reps=200, seed=1)
    assert math.isclose(boot["mean"], 0.2)
    assert boot["ci95_low"] <= boot["mean"] + 1e-12
    assert boot["mean"] <= boot["ci95_high"] + 1e-12
    assert math.isclose(boot["frac_positive"], 1.0)


def test_dual_t4_metadata_and_budget():
    source = KERNEL.read_text()
    for token in (
        "index % WORLD_SIZE == rank",
        '"CUDA_VISIBLE_DEVICES": str(rank)',
        "batch_size = max(1, EFFECTIVE_BATCH // sample_count)",
    ):
        assert token in source, token
    module = load_module()
    budget = sum(arm["sample_count"] for arm in module.ARMS) * 18 * 55.0 / module.WORLD_SIZE
    assert budget <= module.HARD_LIMIT_SECONDS, budget
    metadata = json.loads(META.read_text())
    assert metadata["id"] == "luckfu/kronos-small-0-1-c4-18d-alpha-oos"
    assert metadata["code_file"] == "c4_18d_alpha_oos.py"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == ["luckfu/a-share-120d-temporal-symbol-holdout"]
    assert metadata["kernel_sources"] == [
        "user281434/kronos-small-0-1-stage2-c2-best-wc-1e5-c4",
    ]
    assert len(metadata["id"].split("/")[1]) <= 50
