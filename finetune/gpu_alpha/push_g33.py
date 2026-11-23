"""Package and push G33 (Kronos-base dep_layer + rich pool + gentle top-2-block FT) to Kaggle (smmt315)."""
from __future__ import annotations

import argparse
import json
import os

from short_probe_common import push_kernel

OWNER = os.environ.get("KAGGLE_OWNER", "smmt315")
SLUG = "kronos-alpha-g33-base-dep-pool-gentle-ft"
TITLE = "Kronos Alpha G33 Base Dep Pool Gentle Ft"
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
        entry_filename="g33_base_dep_pool_gentle_ft_entry.py",
        work_tag="g33",
        extra_gpu_files=[
            "g2_kaggle_entry.py",
            "g4_base_kaggle_entry.py",
            "g16_pretrained_small_entry.py",
            "g28_l45_gentle_ft_entry.py",
        ],
        push=args.push,
    )
    print(json.dumps(res, indent=2))
    if not args.push:
        print("Ready, pass --push to submit")


if __name__ == "__main__":
    main()
