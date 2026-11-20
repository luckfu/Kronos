"""Package and push G27 (Non-Linear Residual MLP + Top-K ListNet + Breadth-FiLM + LightGBM + Breadth-MoE) to Kaggle (user281434)."""
from __future__ import annotations

import argparse
import json
import os

from short_probe_common import push_kernel

OWNER = os.environ.get("KAGGLE_OWNER", "user281434")
SLUG = "kronos-alpha-g27-nonlinear-moe-lgbm"
TITLE = "Kronos Alpha G27 Nonlinear MoE LGBM"
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
        entry_filename="g27_nonlinear_moe_lgbm_entry.py",
        work_tag="g27",
        extra_gpu_files=["g4_base_kaggle_entry.py", "g16_pretrained_small_entry.py", "g24_common.py"],
        push=args.push,
    )
    print(json.dumps(res, indent=2))
    if not args.push:
        print("Ready, pass --push to submit")


if __name__ == "__main__":
    main()
