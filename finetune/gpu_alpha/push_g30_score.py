"""Package and push G30c 65-day OOS scorer (reusing already-trained weights) to Kaggle (smmt315)."""
from __future__ import annotations

import argparse
import json
import os

from short_probe_common import push_kernel

OWNER = os.environ.get("KAGGLE_OWNER", "smmt315")
SLUG = "kronos-alpha-g30-oos65-score"
TITLE = "Kronos Alpha G30 OOS65 Score"
DATASETS = [
    os.environ.get("KAGGLE_C2_DATASET", f"{OWNER}/kronos-small-cosine-c2-best-weights"),
    os.environ.get("KAGGLE_OOS_DATASET", f"{OWNER}/a-share-120d-oos-20260717-20261023-v2"),
    os.environ.get("KAGGLE_G30_ARTIFACTS", f"{OWNER}/kronos-g30-trained-artifacts"),
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
        entry_filename="g30_score_oos_entry.py",
        work_tag="g30s",
        extra_gpu_files=[
            "g2_kaggle_entry.py",
            "g4_base_kaggle_entry.py",
            "g16_pretrained_small_entry.py",
            "g24_common.py",
            "g26_mini2k_pts_alllayers_entry.py",
            "g30_quadfam_nonlin_xstock_entry.py",
        ],
        push=args.push,
    )
    print(json.dumps(res, indent=2))
    if not args.push:
        print("Ready, pass --push to submit")


if __name__ == "__main__":
    main()
