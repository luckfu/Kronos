"""Package and push G27 65-day OOS scorer (reusing already-trained weights) to Kaggle (smmt315)."""
from __future__ import annotations

import argparse
import json
import os

from short_probe_common import push_kernel

OWNER = os.environ.get("KAGGLE_OWNER", "smmt315")
SLUG = "kronos-alpha-g27-oos65-score"
TITLE = "Kronos Alpha G27 OOS65 Score"
DATASETS = [
    os.environ.get("KAGGLE_OOS_DATASET", f"{OWNER}/a-share-120d-oos-20260717-20261023-v2"),
    os.environ.get("KAGGLE_G27_ARTIFACTS", f"{OWNER}/kronos-g27-trained-artifacts"),
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
        entry_filename="g27_score_oos_entry.py",
        work_tag="g27s",
        extra_gpu_files=[
            "g4_base_kaggle_entry.py",
            "g16_pretrained_small_entry.py",
            "g24_common.py",
            "g27_nonlinear_moe_lgbm_entry.py",
        ],
        push=args.push,
    )
    print(json.dumps(res, indent=2))
    if not args.push:
        print("Ready, pass --push to submit")


if __name__ == "__main__":
    main()
