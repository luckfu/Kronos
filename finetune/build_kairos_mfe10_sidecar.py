"""Build Kairos short sidecar kernel for path-MFE≥10% single binary head.

Does NOT restart the old 8-head R2 recipe. Fresh single-head, short budget.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "finetune/kaggle_kairos_mfe10_sidecar/train_mfe10_sidecar.py"
DEST = ROOT / "finetune/kaggle_kairos_mfe10_sidecar"

SLUG = "kairos-mfe10-sidecar-short-phase-j-freeze-bb"
TITLE = "Kairos MFE10 Sidecar Short Phase J Freeze BB"
OWNER = "user281434"
SWANLAB_RUN_ID = "kairos-mfe10-sidecar-short-phase-j-freeze-bb-20261001"


def build(destination: Path | None = None) -> Path:
    destination = destination or DEST
    destination.mkdir(parents=True, exist_ok=True)
    code_file = "train_mfe10_sidecar.py"
    if SOURCE.resolve() != (destination / code_file).resolve():
        shutil.copy2(SOURCE, destination / code_file)
    source_bytes = (destination / code_file).read_bytes()
    metadata = {
        "id": f"{OWNER}/{SLUG}",
        "title": TITLE,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            "luckfu/a-share-120d-temporal-symbol-holdout",
            "luckfu/ashare120d-modernbert-targets",
        ],
        "competition_sources": [],
        # Fresh single-head: do NOT mount old 8-head R2 / chunk-8 weights.
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
        "target": "y=1{mfe10>=0.10} where mfe10=max(high[T+1:T+10])/close[T]-1",
        "not_close_to_close": True,
        "not_multi_head_r2": True,
        "not_old_kernel": "user281434/kairos-r2-r1-restart-lr-3e-5",
        "max_segments_this_run": 4,
        "segment_samples": 20_000,
        "gpu_budget_seconds": 5400,
        "learning_rate": 1e-4,
        "freeze_backbone": True,
        "gate_delta_vs_prior": -0.04,
        "prior_stuck_tol": 1e-3,
        "prior_stuck_patience": 2,
        "swanlab_run_id": SWANLAB_RUN_ID,
        "swanlab_url": f"https://swanlab.cn/@roc_fu/finance/runs/{SWANLAB_RUN_ID}",
        "cloud_credentials": "staging only, not tracked",
        "expected_val_pos_rate": 0.2529,
    }
    (destination / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return destination


if __name__ == "__main__":
    print(build())
