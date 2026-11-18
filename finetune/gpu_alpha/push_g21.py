"""Package and push G21 (Pretrained Kronos-small 8.28M Full-Density Ultrashort) to Kaggle (user281434)."""
from __future__ import annotations

import argparse
import json
import os

from short_probe_common import push_kernel

OWNER = os.environ.get("KAGGLE_OWNER", "user281434")
SLUG = "kronos-alpha-g21-ptsmall-fulldensity"
TITLE = "Kronos Alpha G21 PTSmall Fulldensity"
DATASETS = [
    os.environ.get("KAGGLE_TRAIN_DATASET", "luckfu/a-share-120d-temporal-symbol-holdout"),
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
        entry_filename="g21_ptsmall_fulldensity_entry.py",
        work_tag="g21",
        extra_gpu_files=["g16_pretrained_small_entry.py"],
        push=args.push,
    )
    print(json.dumps(res, indent=2))
    if not args.push:
        print("Ready, pass --push to submit")


if __name__ == "__main__":
    main()
