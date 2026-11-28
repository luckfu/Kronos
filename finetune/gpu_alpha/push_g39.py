"""Package and push G39 (3-Target Crash-Shield + Fwd5-10 Light Distillation) to Kaggle (wynstonliu)."""
from __future__ import annotations

import argparse
import json
import os

from short_probe_common import push_kernel

OWNER = os.environ.get("KAGGLE_OWNER", "wynstonliu")
SLUG = os.environ.get("KAGGLE_SLUG", "kronos-alpha-g39-tailshield-fwd-distill")
TITLE = os.environ.get("KAGGLE_TITLE", "Kronos Alpha G39 Tailshield Fwd Distill")
DATASETS = [
    os.environ.get("KAGGLE_TRAIN_DATASET", "luckfu/a-share-120d-temporal-symbol-holdout"),
    os.environ.get("KAGGLE_C2_DATASET", f"{OWNER}/kronos-small-c2-best-checkpoint"),
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
        entry_filename="g39_tailshield_fwd_light_distill_entry.py",
        work_tag="g39",
        extra_gpu_files=[
            "g2_kaggle_entry.py",
            "g4_base_kaggle_entry.py",
            "g16_pretrained_small_entry.py",
            "g24_common.py",
            "g26_mini2k_pts_alllayers_entry.py",
            "g28_l45_gentle_ft_entry.py",
            "g31_mini_mastery_c2pool_ft_entry.py",
            "g34_quad_dep_s1s2_logits_entry.py",
            "wave4_common.py",
        ],
        push=args.push,
    )
    print(json.dumps(res, indent=2))
    if not args.push:
        print("Ready, pass --push to submit")


if __name__ == "__main__":
    main()
