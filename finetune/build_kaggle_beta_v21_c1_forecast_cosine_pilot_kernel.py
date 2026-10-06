"""Build staging for the C1 forecast-only cosine pilot (dual T4).

Two parents, two slugs, two staging folders (outputs never collide):

  --parent best475 (default) -> luckfu/kronos-beta-v21-c1-forecast-cosine-pilot-best475
      staging finetune/kaggle_beta_v21_c1_forecast_cosine_pilot_best475_kernel/
      parent = Beta v2.1 release Best@475 from ModelScope (same lookup as the
      val-gen-ic kernel's beta_v21_release_best475); aux heads dropped.
  --parent seg155            -> luckfu/kronos-beta-v21-c1-forecast-cosine-pilot
      staging finetune/kaggle_beta_v21_c1_forecast_cosine_pilot_kernel/
      parent = dataset luckfu/kronos-beta-v21-c1-seg155-forecast-best.

Never touches the C1 training notebook (kronos-beta-v2-1-c1-dual-t4) or its
staging. Embeds training sources (with the periodic snapshot hook) plus the
Step-1 val generative-IC eval modules so the kernel can score Seg0 and every
snapshot between training chunks.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import re
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "finetune/kaggle_beta_v21_c1_forecast_cosine_pilot.py"
STAGINGS = {
    "best475": ROOT / "finetune/kaggle_beta_v21_c1_forecast_cosine_pilot_best475_kernel",
    "seg155": ROOT / "finetune/kaggle_beta_v21_c1_forecast_cosine_pilot_kernel",
}
DEFAULT_PARENT = "best475"
DOCKER_IMAGE = (
    "gcr.io/kaggle-private-byod/python@"
    "sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
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
    "finetune/tpu_self_checks.py",
    "finetune/utils/__init__.py",
    "finetune/utils/training_utils.py",
    # Step-1 val generative-IC contract (between-chunk snapshot scoring).
    "finetune/val_gen_ic_driver.py",
    "finetune/evaluate_beta_v21_val_gen_ic.py",
    "finetune/evaluate_beta_v21_generative_return_oos.py",
    "finetune/evaluate_beta_v21_time_oos.py",
)


def load_parents() -> dict:
    spec = importlib.util.spec_from_file_location("pilot_runner_for_build", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PARENTS


def build_bundle() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative in FILES:
            archive.add(ROOT / relative, arcname=Path("Kronos") / relative)
        for path in sorted((ROOT / "model").rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                archive.add(path, arcname=Path("Kronos/model") / path.relative_to(ROOT / "model"))
    return buffer.getvalue()


def build_metadata(parent: str = DEFAULT_PARENT) -> dict:
    profile = load_parents()[parent]
    return {
        "id": profile["kernel_id"],
        "title": profile["title"],
        "code_file": RUNNER.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": list(profile["dataset_sources"]),
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
        "docker_image": DOCKER_IMAGE,
    }


def stage_source(source: str, parent: str, payload: str) -> str:
    staged, count = re.subn(
        r'^PILOT_PARENT = "[a-z0-9]+"$',
        f'PILOT_PARENT = "{parent}"',
        source,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError("Failed to pin PILOT_PARENT")
    staged, count = re.subn(
        r'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n.*?\n?"""',
        lambda _m: f'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n{payload}\n"""',
        staged,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError("Failed to embed archive")
    return staged


def build(parent: str = DEFAULT_PARENT) -> Path:
    if parent not in STAGINGS:
        raise SystemExit(f"--parent must be one of {sorted(STAGINGS)}")
    payload = base64.b64encode(build_bundle()).decode("ascii")
    staging = STAGINGS[parent]
    staging.mkdir(parents=True, exist_ok=True)
    (staging / RUNNER.name).write_text(stage_source(RUNNER.read_text(), parent, payload))
    (staging / "kernel-metadata.json").write_text(
        json.dumps(build_metadata(parent), indent=2) + "\n"
    )
    print(f"[{parent}] embedded {len(payload)} b64 chars -> {staging}")
    return staging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parent", choices=sorted(STAGINGS), default=DEFAULT_PARENT)
    args = parser.parse_args()
    build(args.parent)


if __name__ == "__main__":
    main()
