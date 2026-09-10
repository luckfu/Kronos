"""Embed the TPU training sources in the C1 Kaggle runner."""

from __future__ import annotations

import base64
import io
import re
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "finetune/kaggle_beta_v21_c1_tpu.py"
STAGING = ROOT / "finetune/kaggle_beta_v21_c1_tpu_smoke_kernel"
FILES = (
    "finetune/train_predictor.py",
    "finetune/config.py",
    "finetune/dataset.py",
    "finetune/beta_v21.py",
    "finetune/asset_metadata.py",
    "finetune/export_last_model.py",
    "finetune/tpu_train_entry.py",
    "finetune/drive_cleanup.py",
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
                archive.add(path, arcname=Path("Kronos/model") / path.relative_to(ROOT / "model"))
    return buffer.getvalue()


def main() -> None:
    payload = base64.b64encode(build_bundle()).decode("ascii")
    source = RUNNER.read_text()
    updated, count = re.subn(
        r'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n.*?\n"""',
        f'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n{payload}\n"""',
        source,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError("TPU runner archive placeholder is missing")
    RUNNER.write_text(updated)
    STAGING.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUNNER, STAGING / RUNNER.name)


if __name__ == "__main__":
    main()
