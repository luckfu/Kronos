"""Package and push G19 (Full-density L=45 Tri-Arch) to Kaggle (smmt315)."""
from __future__ import annotations

import argparse
import json
import os

from short_probe_common import push_kernel

OWNER = os.environ.get("KAGGLE_OWNER", "smmt315")
SLUG = "kronos-alpha-g19-fulldensity-l45-triarch"
TITLE = "Kronos Alpha G19 Fulldensity L45 Triarch"
DATASETS = [
    os.environ.get("KAGGLE_TRAIN_DATASET", "luckfu/a-share-120d-temporal-symbol-holdout"),
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
        entry_filename="g19_fulldensity_l45_triarch_entry.py",
        work_tag="g19",
        extra_gpu_files=["g4_base_kaggle_entry.py", "g16_pretrained_small_entry.py"],
        push=args.push,
    )
    print(json.dumps(res, indent=2))
    if not args.push:
        print("Ready, pass --push to submit")


if __name__ == "__main__":
    main()
