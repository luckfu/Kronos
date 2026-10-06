"""Build dual-T4 Baseline-2 generative-return OOS staging (Seg155, eval-only)."""

from __future__ import annotations

import base64
import io
import json
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "finetune/kaggle_beta_v21_c1_gen_return_oos.py"
VARIANTS = {
    "both": {
        "staging": ROOT / "finetune/kaggle_beta_v21_c1_gen_return_oos_kernel",
        "kernel_id": "luckfu/kronos-beta-v21-c1-gen-return-oos",
        "title": "Kronos Beta V21 C1 Gen Return OOS",
        "output_name": "beta_v2_1_c1_gen_return_oos",
        "arm_filter": None,
    },
    "prod": {
        "staging": ROOT / "finetune/kaggle_beta_v21_c1_gen_return_oos_prod_kernel",
        "kernel_id": "luckfu/kronos-beta-v21-c1-gen-return-oos-prod",
        "title": "Kronos Beta V21 C1 Gen Return OOS Prod",
        "output_name": "beta_v2_1_c1_gen_return_oos_prod",
        "arm_filter": ("prod_t065_p80_n5",),
    },
}

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
        "code_file": RUNNER.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": [
            "luckfu/a-share-120d-temporal-symbol-holdout",
            "luckfu/kronos-beta-v21-c1-seg155-forecast-best",
        ],
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
    args = parser.parse_args()
    variant = VARIANTS[args.variant]

    payload = base64.b64encode(build_bundle()).decode("ascii")
    source = RUNNER.read_text()
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
    RUNNER.write_text(updated)
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
    (staging / RUNNER.name).write_text(staged)
    (staging / "kernel-metadata.json").write_text(
        json.dumps(build_metadata(variant), indent=2) + "\n"
    )
    print(f"embedded {len(payload)} b64 chars")
    print(f"staged -> {staging}")
    print(f"kernel_id={variant['kernel_id']} arm_filter={variant['arm_filter']}")


if __name__ == "__main__":
    main()
