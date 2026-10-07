"""Build staging for the Beta v2.1 heavy-reg / time-disjoint-val retrain (dual T4).

Reuses the forecast cosine pilot builder (same docker pin, same source bundle +
val gen-IC modules) and adds the time-disjoint val panel builder. Writes
finetune/kaggle_beta_v21_heavy_reg_time_val_kernel/{runner, kernel-metadata.json}.
Does NOT push: `kaggle kernels push` needs explicit user approval.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune"))
from build_kaggle_beta_v21_c1_forecast_cosine_pilot_kernel import (  # noqa: E402
    DOCKER_IMAGE,
    FILES as PILOT_FILES,
)

RUNNER = ROOT / "finetune/kaggle_beta_v21_heavy_reg_time_val.py"
STAGING = ROOT / "finetune/kaggle_beta_v21_heavy_reg_time_val_kernel"
FILES = PILOT_FILES + ("finetune/build_time_disjoint_val_panel.py",)


def load_runner():
    spec = importlib.util.spec_from_file_location("heavy_reg_runner_for_build", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    runner = load_runner()
    return {
        "id": runner.KERNEL_ID,
        "title": runner.KERNEL_TITLE,
        "code_file": RUNNER.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": list(runner.DATASET_SOURCES),
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
        "docker_image": DOCKER_IMAGE,
    }


def stage_source(source: str, payload: str) -> str:
    staged, count = re.subn(
        r'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n.*?\n?"""',
        lambda _m: f'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n{payload}\n"""',
        source,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError("Failed to embed archive")
    return staged


def build() -> Path:
    payload = base64.b64encode(build_bundle()).decode("ascii")
    STAGING.mkdir(parents=True, exist_ok=True)
    (STAGING / RUNNER.name).write_text(stage_source(RUNNER.read_text(), payload))
    (STAGING / "kernel-metadata.json").write_text(json.dumps(build_metadata(), indent=2) + "\n")
    print(f"embedded {len(payload)} b64 chars -> {STAGING}")
    return STAGING


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    build()


if __name__ == "__main__":
    main()
