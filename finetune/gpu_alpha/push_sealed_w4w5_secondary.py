"""Package (dry-run by default) the December Secondary Sealed Track Wave 4+5 Scoring Kernel for Kaggle."""
from __future__ import annotations

import argparse
import json
import os

from short_probe_common import push_kernel

OWNER = os.environ.get("KAGGLE_OWNER", "wynstonliu")
SLUG = os.environ.get("KAGGLE_SLUG", "kronos-sealed-w4w5-secondary-score")
TITLE = os.environ.get("KAGGLE_TITLE", "Kronos Sealed W4W5 Secondary Score")
DATASETS = [
    os.environ.get("KAGGLE_C2_DATASET", f"{OWNER}/kronos-small-c2-best-checkpoint"),
    os.environ.get("KAGGLE_SEALED_DATASET", f"{OWNER}/kronos-sealed-dec-w4w5-bundle"),
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
        entry_filename="sealed_w4w5_secondary_score_entry.py",
        work_tag="sec45",
        extra_gpu_files=[
            "sealed_common.py",
            "sealed_w2w3_hidden_score_entry.py",
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
            "wave4_common.py",
            "g35_crash_shield_tailrisk_entry.py",
            "g36_lambdarank_topk_liquid_entry.py",
            "g37_quad_to_light_distill_entry.py",
            "g38_termstruct_hazard_net_entry.py",
            "g39_tailshield_fwd_light_distill_entry.py",
            "g40_mini_tailaware_gentle_ft_entry.py",
        ],
        push=args.push,
    )
    print(json.dumps(res, indent=2))
    if not args.push:
        print("Ready, pass --push to submit")


if __name__ == "__main__":
    main()
