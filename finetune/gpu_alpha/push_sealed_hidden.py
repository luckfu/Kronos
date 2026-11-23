"""Package and push Sealed Acceptance Kernel A (Wave 2 + Wave 3 Hidden/Non-Linear Scorer) to Kaggle."""
from __future__ import annotations

import argparse
import json
import os

from short_probe_common import push_kernel

OWNER = os.environ.get("KAGGLE_OWNER", "wynstonliu")
SLUG = "kronos-sealed-w2w3-hidden-score-v2"
TITLE = "Kronos Sealed W2W3 Hidden Score V2"
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
        entry_filename="sealed_w2w3_hidden_score_entry.py",
        work_tag="shid",
        extra_gpu_files=[
            "sealed_common.py",
            "g2_kaggle_entry.py",
            "g4_base_kaggle_entry.py",
            "g16_pretrained_small_entry.py",
            "g24_common.py",
            "g25_base_fulldensity_l45_l52_entry.py",
            "g26_mini2k_pts_alllayers_entry.py",
            "g27_nonlinear_moe_lgbm_entry.py",
            "g28_l45_gentle_ft_entry.py",
            "g29_mini_pool_trifamily_entry.py",
            "g30_quadfam_nonlin_xstock_entry.py",
            "g31_mini_mastery_c2pool_ft_entry.py",
            "g33_base_dep_pool_gentle_ft_entry.py",
            "g34_quad_dep_s1s2_logits_entry.py",
        ],
        push=args.push,
    )
    print(json.dumps(res, indent=2))
    if not args.push:
        print("Ready, pass --push to submit")


if __name__ == "__main__":
    main()
