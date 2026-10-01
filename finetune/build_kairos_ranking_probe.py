"""Build Kairos short ranking probe kernel (Phase N).

Light identity + pairwise/listwise rank loss on continuous mfe10.
NOT binary mfe≥10%. NOT 22-layer R2. No TPU WIP.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "finetune/kaggle_kairos_ranking_probe/train_ranking_probe.py"
DEST = ROOT / "finetune/kaggle_kairos_ranking_probe"

SLUG = "kairos-ranking-probe-short-phase-n"
TITLE = "Kairos Ranking Probe Short Phase N"
OWNER = "user281434"
SWANLAB_RUN_ID = "kairos-ranking-probe-short-phase-n-20261001"


def build(destination: Path | None = None) -> Path:
    destination = destination or DEST
    destination.mkdir(parents=True, exist_ok=True)
    code_file = "train_ranking_probe.py"
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
        "target": "continuous mfe10 (path MFE); Rank IC / TopK vs mfe10",
        "target_mode": "mfe10_continuous",
        "not_binary_mfe10": True,
        "not_multi_head_r2": True,
        "not_22_layer_binary": True,
        "backbone_mode": "identity",
        "freeze_backbone": False,
        "loss_mode": "pairwise",
        "pairwise_min_gap": 0.005,
        "listwise_temperature": 0.05,
        "batch_size": 64,
        "max_segments_this_run": 3,
        "segment_samples": 20_000,
        "gpu_budget_seconds": 3600,
        "learning_rate": 1e-4,
        "rank_ic_bar": 0.05,
        "topk_lift_bar": 0.05,
        "swanlab_run_id": SWANLAB_RUN_ID,
        "swanlab_url": f"https://swanlab.cn/@roc_fu/finance/runs/{SWANLAB_RUN_ID}",
        "cloud_credentials": "staging only, not tracked",
        "phase_l_ridge_mfe_comb_rank_ic": 0.2734396296793431,
        "phase_m_mse_rank_ic": 0.08525129172480793,
    }
    (destination / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return destination


if __name__ == "__main__":
    print(build())
