"""Build Kairos Phase T tabular decision confirmation kernel (enet/blend)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "finetune/kaggle_kairos_mfe10_tabular_decision"
SOURCE = DEST / "train_tabular_decision.py"

SLUG = "kairos-mfe10-decision-tabular-phase-t"
TITLE = "Kairos MFE10 Decision Tabular Phase T"
OWNER = "user281434"
SWANLAB_RUN_ID = "kairos-mfe10-decision-tabular-phase-t-20261002"


def build(destination: Path | None = None) -> Path:
    destination = destination or DEST
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / "train_tabular_decision.py").exists():
        raise FileNotFoundError("train_tabular_decision.py missing")
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
        "kind": "tabular_enet_blend_official_holdout_confirm",
        "phase_s_temporal_best_delta": -0.04179349770224905,
        "gate": -0.04,
        "not_tokenizer_sequence": True,
        "not_ranking_ic": True,
        "not_22_layer": True,
        "not_tpu": True,
        "swanlab_run_id": SWANLAB_RUN_ID,
        "swanlab_url": f"https://swanlab.cn/@roc_fu/finance/runs/{SWANLAB_RUN_ID}",
    }
    (destination / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return destination


if __name__ == "__main__":
    print(build())
