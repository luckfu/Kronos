"""Build dual-T4 Baseline-2 generative-return OOS staging (eval-only).

Variants:
  both / prod   Seg155 forecast-best (runner kaggle_beta_v21_c1_gen_return_oos.py)
  pilot_seg9    final one-shot sealed OOS of cosine-pilot Seg9 + Seg0 (Best@475),
                prod arm only, Seg9 first (runner kaggle_beta_v21_c1_gen_return_oos_pilot.py)

Account options (defaults keep every legacy build byte-identical):
  --owner          Kaggle account that owns/runs the kernel (kernel id owner); default luckfu
  --dataset-owner  owner of the attached input datasets; default luckfu (the datasets are
                   public / shared with the krnons-train group, so other accounts can attach
                   them directly instead of copying)
"""

from __future__ import annotations

import base64
import io
import json
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "finetune/kaggle_beta_v21_c1_gen_return_oos.py"
PILOT_RUNNER = ROOT / "finetune/kaggle_beta_v21_c1_gen_return_oos_pilot.py"
SEG155_DATASETS = (
    "luckfu/a-share-120d-temporal-symbol-holdout",
    "luckfu/kronos-beta-v21-c1-seg155-forecast-best",
)
VARIANTS = {
    "both": {
        "staging": ROOT / "finetune/kaggle_beta_v21_c1_gen_return_oos_kernel",
        "kernel_id": "luckfu/kronos-beta-v21-c1-gen-return-oos",
        "title": "Kronos Beta V21 C1 Gen Return OOS",
        "output_name": "beta_v2_1_c1_gen_return_oos",
        "arm_filter": None,
        "runner": RUNNER,
        "dataset_sources": SEG155_DATASETS,
    },
    "prod": {
        "staging": ROOT / "finetune/kaggle_beta_v21_c1_gen_return_oos_prod_kernel",
        "kernel_id": "luckfu/kronos-beta-v21-c1-gen-return-oos-prod",
        "title": "Kronos Beta V21 C1 Gen Return OOS Prod",
        "output_name": "beta_v2_1_c1_gen_return_oos_prod",
        "arm_filter": ("prod_t065_p80_n5",),
        "runner": RUNNER,
        "dataset_sources": SEG155_DATASETS,
    },
    "pilot_seg9": {
        "staging": ROOT / "finetune/kaggle_beta_v21_c1_gen_return_oos_pilot_seg9_kernel",
        "kernel_id": "luckfu/kronos-beta-v21-c1-gen-return-oos-pilot-seg9",
        "title": "Kronos Beta V21 C1 Gen Return OOS Pilot Seg9",
        "output_name": "beta_v2_1_c1_gen_return_oos_pilot_seg9",
        "arm_filter": ("prod_t065_p80_n5",),
        "runner": PILOT_RUNNER,
        "dataset_sources": (
            "luckfu/a-share-120d-temporal-symbol-holdout",
            "luckfu/kronos-beta-v21-c1-cosine-pilot-best475-seg9",
        ),
    },
}

DEFAULT_OWNER = "luckfu"


def _reown(ref: str, owner: str) -> str:
    _, slug = ref.split("/", 1)
    return f"{owner}/{slug}"


def resolve_variant(name: str, owner: str = DEFAULT_OWNER,
                    dataset_owner: str = DEFAULT_OWNER) -> dict:
    """Variant with kernel id / dataset_sources moved to the given Kaggle accounts."""
    variant = dict(VARIANTS[name])
    variant["kernel_id"] = _reown(variant["kernel_id"], owner)
    variant["dataset_sources"] = tuple(
        _reown(ref, dataset_owner) for ref in variant["dataset_sources"]
    )
    return variant


FILES = (
    "finetune/evaluate_beta_v21_generative_return_oos.py",
    "finetune/evaluate_beta_v21_time_oos.py",
    "finetune/beta_v21.py",
    "finetune/dataset.py",
    "finetune/config.py",
    "finetune/asset_metadata.py",
    "finetune/validation_precision.py",
)


def build_bundle() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative in FILES:
            archive.add(ROOT / relative, arcname=Path("Kronos") / relative)
        for path in sorted((ROOT / "model").rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                archive.add(
                    path,
                    arcname=Path("Kronos/model") / path.relative_to(ROOT / "model"),
                )
        info = tarfile.TarInfo(name="Kronos/finetune/__init__.py")
        info.size = 0
        archive.addfile(info, io.BytesIO(b""))
    return buffer.getvalue()


def build_metadata(variant: dict) -> dict:
    return {
        "id": variant["kernel_id"],
        "title": variant["title"],
        "code_file": variant["runner"].name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": list(variant["dataset_sources"]),
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
        "docker_image": (
            "gcr.io/kaggle-private-byod/python@"
            "sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
        ),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="both")
    parser.add_argument("--owner", default=DEFAULT_OWNER,
                        help="Kaggle account owning the kernel (kernel id owner)")
    parser.add_argument("--dataset-owner", default=DEFAULT_OWNER,
                        help="owner of the attached input datasets")
    args = parser.parse_args()
    variant = resolve_variant(args.variant, owner=args.owner,
                              dataset_owner=args.dataset_owner)

    runner = variant["runner"]
    payload = base64.b64encode(build_bundle()).decode("ascii")
    source = runner.read_text()
    updated, count = re.subn(
        r'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n.*?\n"""',
        lambda _m: f'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n{payload}\n"""',
        source,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError("Failed to embed archive into gen-return OOS runner")
    # Keep the tracked runner generic (archive refreshed, constants untouched).
    runner.write_text(updated)
    staged = updated
    for pattern, value in (
        (r'^OUTPUT_NAME = ".*"$', f'OUTPUT_NAME = "{variant["output_name"]}"'),
        (r'^EXPERIMENT_NAME = ".*"$', f'EXPERIMENT_NAME = "{variant["output_name"]}"'),
        (r'^KERNEL_ID = ".*"$', f'KERNEL_ID = "{variant["kernel_id"]}"'),
        (r'^ARM_FILTER = .*$', f'ARM_FILTER = {variant["arm_filter"]!r}'),
    ):
        staged, n = re.subn(pattern, lambda _m, v=value: v, staged, count=1, flags=re.M)
        if n != 1:
            raise RuntimeError(f"Failed to set {pattern}")
    staging = variant["staging"]
    staging.mkdir(parents=True, exist_ok=True)
    (staging / runner.name).write_text(staged)
    (staging / "kernel-metadata.json").write_text(
        json.dumps(build_metadata(variant), indent=2) + "\n"
    )
    print(f"embedded {len(payload)} b64 chars")
    print(f"staged -> {staging}")
    print(f"kernel_id={variant['kernel_id']} arm_filter={variant['arm_filter']}")
    print(f"dataset_sources={list(variant['dataset_sources'])}")


if __name__ == "__main__":
    main()
