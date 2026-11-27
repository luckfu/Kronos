"""Package and push G38 65-day OOS scorer (reusing trained artifacts + hazard classifier) to Kaggle (user281434)."""
from __future__ import annotations

import argparse
import json
import os

from short_probe_common import push_kernel

OWNER = os.environ.get("KAGGLE_OWNER", "user281434")
SLUG = os.environ.get("KAGGLE_SLUG", "kronos-alpha-g38-oos65-score")
TITLE = os.environ.get("KAGGLE_TITLE", "Kronos Alpha G38 OOS65 Score")
DATASETS = [
    os.environ.get("KAGGLE_C2_DATASET", "luckfu/kronos-small-cosine-c2-best-weights"),
    os.environ.get("KAGGLE_OOS_DATASET", f"{OWNER}/a-share-120d-oos-20260717-20261023-v2"),
    os.environ.get("KAGGLE_G38_ARTIFACTS", f"{OWNER}/kronos-g38-trained-artifacts"),
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
        entry_filename="g38_score_oos_entry.py",
        work_tag="g38s",
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
            "g38_termstruct_hazard_net_entry.py",
        ],
        push=args.push,
    )
    print(json.dumps(res, indent=2))
    if not args.push:
        print("Ready, pass --push to submit")


if __name__ == "__main__":
    main()
