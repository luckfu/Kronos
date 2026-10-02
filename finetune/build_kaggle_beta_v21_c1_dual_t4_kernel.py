"""Embed GPU training sources into the Beta v2.1 C1 dual-T4 Kaggle runner.

Kaggle clones GitHub master, which still all_gather_object-s ~123k sample-level
aux tensors on CUDA and SIGKILL'd after calibration VAL 1935/1935. Embedding
local train_predictor/config/model overlays the fix without a git commit.
GPU path must not hard-require TPU-only modules; still embed tpu_self_checks
so future TPU helpers in train_predictor do not break dual-T4.
"""

from __future__ import annotations

import base64
import io
import re
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "finetune/kaggle_beta_v21_c1_dual_t4.py"
STAGING_DIRS = (
    ROOT / "finetune/kaggle_beta_v21_c1_dual_t4_kernel",
)
FILES = (
    "finetune/train_predictor.py",
    "finetune/config.py",
    "finetune/dataset.py",
    "finetune/beta_v21.py",
    "finetune/asset_metadata.py",
    "finetune/export_last_model.py",
    "finetune/validation_precision.py",
    "finetune/drive_cleanup.py",
    "finetune/tpu_self_checks.py",  # optional on GPU; keep overlay complete for TPU helpers
    "finetune/utils/__init__.py",
    "finetune/utils/training_utils.py",
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
    return buffer.getvalue()


def main() -> None:
    if not RUNNER.is_file():
        # Staging-only source of truth: keep a working copy next to the builder.
        staging_runner = STAGING_DIRS[0] / "kaggle_beta_v21_c1_dual_t4.py"
        if staging_runner.is_file():
            shutil.copy2(staging_runner, RUNNER)
        else:
            raise SystemExit(f"Missing runner at {RUNNER}")
    payload = base64.b64encode(build_bundle()).decode("ascii")
    source = RUNNER.read_text()
    if 'EMBEDDED_KRONOS_ARCHIVE_B64 = """' not in source:
        raise RuntimeError(
            "Dual-T4 runner is missing EMBEDDED_KRONOS_ARCHIVE_B64 placeholder"
        )
    updated, count = re.subn(
        r'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n.*?\n"""',
        f'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n{payload}\n"""',
        source,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError("Dual-T4 runner archive placeholder replace failed")
    RUNNER.write_text(updated)
    for staging in STAGING_DIRS:
        staging.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RUNNER, staging / RUNNER.name)
        meta = staging / "kernel-metadata.json"
        if not meta.is_file():
            raise SystemExit(f"Missing {meta}")
    print(f"embedded {len(payload)} b64 chars into {RUNNER.name}")
    for staging in STAGING_DIRS:
        print(f"staged -> {staging / RUNNER.name}")


if __name__ == "__main__":
    main()
