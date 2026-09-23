import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).parents[1]
KERNEL_DIR = ROOT / "finetune/kaggle_stage3_ar_vol_18d_oos"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "stage3_ar_vol_18d_oos", KERNEL_DIR / "stage3_ar_vol_18d_oos.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_and_inputs_are_locked():
    module = load_module()
    meta = json.loads((KERNEL_DIR / "kernel-metadata.json").read_text())
    assert module.ARM["temperature"] == 0.65
    assert module.ARM["top_p"] == 0.8
    assert module.ARM["sample_count"] == 5
    assert module.ARM["seed"] == 20260906
    assert module.MODEL_SHA256 == "dccd7aefed346a88a0a96166bff31d891da08a9a55452a483621b5a16128154e"
    assert len(meta["id"].split("/")[1]) <= 50
    assert meta["kernel_sources"] == [
        "smmt315/kronos-small-0-1-stage2-cosine-refinement-c2",
        "luckfu/kronos-small-0-1-s3-ar-vol-c2",
        "luckfu/kronos-small-0-1-c2-18d-alpha-oos",
    ]


def test_paired_identity_and_targets_must_match():
    module = load_module()
    baseline = pd.DataFrame({
        "identity": ["b", "a"],
        "asof_date": ["2026-08-12", "2026-08-11"],
        "symbol": ["B", "A"],
        "sector": [1, 2],
        "size_decile": [5, 6],
        "actual_return_d10": [0.1, 0.2],
    })
    candidate = baseline.iloc[::-1].copy()
    aligned, original = module.matched_frames(candidate, baseline, 2)
    assert aligned["identity"].tolist() == original["identity"].tolist() == ["a", "b"]
    candidate.loc[candidate["identity"] == "a", "actual_return_d10"] = 0.3
    with pytest.raises(RuntimeError, match="targets differ"):
        module.matched_frames(candidate, baseline, 2)
    with pytest.raises(RuntimeError, match="row count"):
        module.matched_frames(candidate, baseline, 3)
    duplicate = baseline.copy()
    duplicate["identity"] = ["a", "a"]
    with pytest.raises(RuntimeError, match="duplicate"):
        module.matched_frames(duplicate, baseline, 2)
