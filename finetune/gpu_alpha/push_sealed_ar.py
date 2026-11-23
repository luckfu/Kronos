"""Package and push Sealed Acceptance Kernel B (G32 C2 L=45 16-Path AR) to Kaggle."""
from __future__ import annotations

import argparse
import json
import os

from short_probe_common import push_kernel

OWNER = os.environ.get("KAGGLE_OWNER", "wynstonliu")
SLUG = "kronos-sealed-g32-c2l45-ar16"
TITLE = "Kronos Sealed G32 C2L45 Ar16"
DATASETS = [
    os.environ.get("KAGGLE_C2_DATASET", f"{OWNER}/kronos-small-c2-best-checkpoint"),
    os.environ.get("KAGGLE_SEALED_DATASET", f"{OWNER}/kronos-sealed-20261026-20261120-bundle"),
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
        entry_filename="sealed_g32_c2l45_ar16_entry.py",
        work_tag="sar",
        extra_gpu_files=[
            "sealed_common.py",
            "g16_pretrained_small_entry.py",
            "g24_common.py",
            "g26_mini2k_pts_alllayers_entry.py",
            "g32_l45_ar16_triarch_entry.py",
        ],
        push=args.push,
    )
    print(json.dumps(res, indent=2))
    if not args.push:
        print("Ready, pass --push to submit")


if __name__ == "__main__":
    main()
