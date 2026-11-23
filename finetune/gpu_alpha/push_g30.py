"""Package and push G30 (Quad-Family NonLinear + Cross-Stock Attention + 4-Signal FiLM) to Kaggle (luckfu)."""
from __future__ import annotations

import argparse
import json
import os

from short_probe_common import push_kernel

OWNER = os.environ.get("KAGGLE_OWNER", "luckfu")
SLUG = "kronos-alpha-g30c-quadfam-nonlin-xstock"
TITLE = "Kronos Alpha G30c Quadfam Nonlin Xstock"
DATASETS = [
    os.environ.get("KAGGLE_TRAIN_DATASET", "luckfu/a-share-120d-temporal-symbol-holdout"),
    os.environ.get("KAGGLE_C2_DATASET", f"{OWNER}/kronos-small-cosine-c2-best-weights"),
    os.environ.get("KAGGLE_OOS_DATASET", f"{OWNER}/a-share-120d-oos-20260717-20261023-v2"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    res = push_kernel(
        owner=OWNER,
        slug=SLUG,
        title=TITLE,
        datasets=DATASETS,
        entry_filename="g30_quadfam_nonlin_xstock_entry.py",
        work_tag="g30",
        extra_gpu_files=[
            "g2_kaggle_entry.py",
            "g4_base_kaggle_entry.py",
            "g16_pretrained_small_entry.py",
            "g24_common.py",
            "g26_mini2k_pts_alllayers_entry.py",
        ],
        push=args.push,
    )
    print(json.dumps(res, indent=2))
    if not args.push:
        print("Ready, pass --push to submit")


if __name__ == "__main__":
    main()
