"""Build staging for luckfu/kronos-beta-v21-c1-forecast-cosine-pilot (dual T4).

New slug; never touches the C1 training notebook (kronos-beta-v2-1-c1-dual-t4)
or its staging. Embeds training sources (with the periodic snapshot hook) plus
the Step-1 val generative-IC eval modules so the kernel can score
Seg3/6/9/12 snapshots after training.
"""

from __future__ import annotations

import base64
import io
import json
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "finetune/kaggle_beta_v21_c1_forecast_cosine_pilot.py"
STAGING = ROOT / "finetune/kaggle_beta_v21_c1_forecast_cosine_pilot_kernel"
KERNEL_ID = "luckfu/kronos-beta-v21-c1-forecast-cosine-pilot"
FILES = (
    "finetune/train_predictor.py",
    "finetune/config.py",
    "finetune/dataset.py",
    "finetune/beta_v21.py",
    "finetune/asset_metadata.py",
    "finetune/export_last_model.py",
    "finetune/validation_precision.py",
    "finetune/drive_cleanup.py",
    "finetune/tpu_self_checks.py",
    "finetune/utils/__init__.py",
    "finetune/utils/training_utils.py",
    # Step-1 val generative-IC contract (post-training snapshot scoring).
    "finetune/val_gen_ic_driver.py",
    "finetune/evaluate_beta_v21_val_gen_ic.py",
    "finetune/evaluate_beta_v21_generative_return_oos.py",
    "finetune/evaluate_beta_v21_time_oos.py",
)


def build_bundle() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative in FILES:
            archive.add(ROOT / relative, arcname=Path("Kronos") / relative)
        for path in sorted((ROOT / "model").rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                archive.add(path, arcname=Path("Kronos/model") / path.relative_to(ROOT / "model"))
    return buffer.getvalue()


def build_metadata() -> dict:
    return {
        "id": KERNEL_ID,
        "title": "Kronos Beta V21 C1 Forecast Cosine Pilot",
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
    payload = base64.b64encode(build_bundle()).decode("ascii")
    source = RUNNER.read_text()
    staged, count = re.subn(
        r'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n.*?\n?"""',
        lambda _m: f'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n{payload}\n"""',
        source,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError("Failed to embed archive")
    STAGING.mkdir(parents=True, exist_ok=True)
    (STAGING / RUNNER.name).write_text(staged)
    (STAGING / "kernel-metadata.json").write_text(json.dumps(build_metadata(), indent=2) + "\n")
    print(f"embedded {len(payload)} b64 chars -> {STAGING}")


if __name__ == "__main__":
    main()
