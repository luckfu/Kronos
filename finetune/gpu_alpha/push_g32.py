"""Package and push G32 (Pure Zero-Shot L=45 Multi-Path AR across Mini/C2/PT-Small) to Kaggle (luckfu)."""
from __future__ import annotations

import argparse
import json
import os

from short_probe_common import push_kernel

OWNER = os.environ.get("KAGGLE_OWNER", "luckfu")
SLUG = "kronos-alpha-g32-l45-ar16-triarch"
TITLE = "Kronos Alpha G32 L45 Ar16 Triarch"
DATASETS = [
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
        entry_filename="g32_l45_ar16_triarch_entry.py",
        work_tag="g32",
        extra_gpu_files=[
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
