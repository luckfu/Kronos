"""Build Kairos Phase V tabular decision train2024→val + hist_gbm kernel.

Keeps Phase T2 git-clone vendor path. Primary = train 2024 → full val;
adds HistGradientBoostingClassifier; secondary val_temporal still reported.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "finetune/kaggle_kairos_mfe10_tabular_decision"
SOURCE = DEST / "train_tabular_decision.py"

SLUG = "kairos-mfe10-decision-tabular-phase-v"
TITLE = "Kairos MFE10 Decision Tabular Phase V"
OWNER = "user281434"
SWANLAB_RUN_ID = "kairos-mfe10-decision-tabular-phase-v-20261002"

VENDOR_FILES = [
    "modernbert_finance/build_dataset.py",
    "modernbert_finance/build_targets.py",
    "modernbert_finance/mfe10_sidecar.py",
    "modernbert_finance/ablations/_panel_io.py",
    "modernbert_finance/ablations/buy_profit_mfe_ablations.py",
    "modernbert_finance/ablations/continuous_xsection_ablations.py",
    "modernbert_finance/ablations/label_time_diagnostics.py",
    "modernbert_finance/ablations/simple_baseline.py",
]


def _sync_vendor(destination: Path) -> None:
    """Keep a local vendor mirror for offline/dev; Kaggle uses git clone.

    Package __init__.py files are written empty so a partial vendor tree does
    not pull alignment/label_shuffle (full package __init__ side effects).
    """
    vendor_root = destination / "vendor"
    if vendor_root.exists():
        shutil.rmtree(vendor_root)
    for rel in VENDOR_FILES:
        src = ROOT / rel
        if not src.is_file():
            if rel.endswith("mfe10_sidecar.py"):
                continue
            raise FileNotFoundError(rel)
        dst = vendor_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for pkg_init in (
        vendor_root / "modernbert_finance" / "__init__.py",
        vendor_root / "modernbert_finance" / "ablations" / "__init__.py",
    ):
        pkg_init.parent.mkdir(parents=True, exist_ok=True)
        pkg_init.write_text('"""Vendor stub package (submodules imported directly)."""\n')


def build(destination: Path | None = None) -> Path:
    destination = destination or DEST
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / "train_tabular_decision.py").exists():
        raise FileNotFoundError("train_tabular_decision.py missing")
    _sync_vendor(destination)
    source_bytes = (destination / "train_tabular_decision.py").read_bytes()
    metadata = {
        "id": f"{OWNER}/{SLUG}",
        "title": TITLE,
        "code_file": "train_tabular_decision.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": True,
        "dataset_sources": [
            "luckfu/a-share-120d-temporal-symbol-holdout",
            "luckfu/ashare120d-modernbert-targets",
        ],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "slug": SLUG,
        "owner": OWNER,
        "title": TITLE,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "target": "y=1{mfe10>=0.10}",
        "kind": "tabular_enet_blend_hist_gbm_train2024_to_val_v",
        "phase_s_temporal_best_delta": -0.04179349770224905,
        "gate": -0.04,
        "not_tokenizer_sequence": True,
        "not_ranking_ic": True,
        "not_22_layer": True,
        "not_tpu": True,
        "phase_t2_temporal_best_delta": -0.04179349770224927,
        "phase_u_train_to_val_best_delta": -0.027597938051234006,
        "phase_u_primary_gate_passed": False,
        "primary_protocol": "train2024_to_val",
        "secondary_protocol": "val_temporal",
        "train_window": "2024-01-01..2024-12-31",
        "train_cap": 500000,
        "recipes": [
            "logistic_C0.01",
            "enet_C0.01_l1_0.7",
            "enet_C0.01_l1_0.5",
            "hist_gbm",
            "blend_lr0.5_mlp0.5",
            "blend_lr0.7_mlp0.3",
            "blend_lr0.8_mlp0.2",
            "blend_lr0.9_mlp0.1",
        ],
        "alignment_fix": "filter_targets_by_asof_signal_window",
        "fix": "git_clone_repo_onto_sys_path_plus_vendor_fallback",
        "hypothesis": "U fail may be distant-train; test recency + tree nonlinearity",
        "swanlab_run_id": SWANLAB_RUN_ID,
        "swanlab_url": f"https://swanlab.cn/@roc_fu/finance/runs/{SWANLAB_RUN_ID}",
    }
    (destination / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return destination


if __name__ == "__main__":
    print(build())
