"""Build dual-T4 validation-only C2 decode + path-estimator staging (eval-only).

The embedded archive carries the current repo code (Best@475 decode, scoring, the
descriptive-analysis helpers) plus ``c2_code_e4b92bb/model``: the ``model/`` package
exactly as of commit e4b92bb, which is the source commit the Small-C2 sealed-OOS kernel
(``kaggle_c2_18d_alpha_oos``) decoded C2 with. C2 workers put it first on PYTHONPATH.

Account options (same convention as the other builders):
  --owner          Kaggle account that owns/runs the kernel (kernel id owner); default luckfu
  --dataset-owner  owner of the val-only input dataset ``kronos-val-c2-path-inputs``;
                   default wynstonliu (it only exists there: C2 weights + val_data.pkl,
                   deliberately without the sealed OOS package)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "finetune/kaggle_beta_v21_c1_val_c2_path_estimators.py"
STAGING = ROOT / "finetune/kaggle_beta_v21_c1_val_c2_path_estimators_kernel"
SLUG = "kronos-val-c2-and-path-estimators"
TITLE = "Kronos Val C2 And Path Estimators"
DATASET_SLUG = "kronos-val-c2-path-inputs"
DEFAULT_OWNER = "luckfu"
DEFAULT_DATASET_OWNER = "wynstonliu"
C2_SOURCE_COMMIT = "e4b92bb32aa47d676ebcba70b5c8427bbc404c03"
C2_CODE_DIR = "c2_code_e4b92bb"
DOCKER_IMAGE = ("gcr.io/kaggle-private-byod/python@"
                "sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461")
FILES = (
    "finetune/val_c2_path_estimators.py",
    "finetune/evaluate_beta_v21_val_gen_ic.py",
    "finetune/evaluate_beta_v21_generative_return_oos.py",
    "finetune/evaluate_beta_v21_time_oos.py",
    "finetune/analysis/beta_v21_c1_why_c2_better.py",
    "finetune/beta_v21.py",
    "finetune/dataset.py",
    "finetune/config.py",
    "finetune/asset_metadata.py",
    "finetune/validation_precision.py",
)


def old_model_files(commit: str = C2_SOURCE_COMMIT) -> dict[str, bytes]:
    """{relative path under model/: bytes} for model/ at ``commit`` (from local git)."""
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit, "model/"], cwd=ROOT, text=True
    ).split()
    files = {}
    for name in names:
        if name.endswith(".pyc") or "__pycache__" in name:
            continue
        files[name] = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
    if "model/kronos.py" not in files or "model/__init__.py" not in files:
        raise RuntimeError(f"model/ package incomplete at {commit}: {sorted(files)}")
    return files


def _add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = 0
    archive.addfile(info, io.BytesIO(data))


def build_bundle() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for relative in FILES:
            archive.add(ROOT / relative, arcname=str(Path("Kronos") / relative))
        for path in sorted((ROOT / "model").rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                archive.add(path, arcname=str(Path("Kronos/model") / path.relative_to(ROOT / "model")))
        for name, data in sorted(old_model_files().items()):
            _add_bytes(archive, f"Kronos/{C2_CODE_DIR}/{name}", data)
        _add_bytes(archive, f"Kronos/{C2_CODE_DIR}/SOURCE_COMMIT", C2_SOURCE_COMMIT.encode())
        _add_bytes(archive, "Kronos/finetune/__init__.py", b"")
        _add_bytes(archive, "Kronos/finetune/analysis/__init__.py", b"")
    return buffer.getvalue()


def build_metadata(owner: str = DEFAULT_OWNER, dataset_owner: str = DEFAULT_DATASET_OWNER) -> dict:
    return {
        "id": f"{owner}/{SLUG}",
        "title": TITLE,
        "code_file": RUNNER.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": [f"{dataset_owner}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
        "docker_image": DOCKER_IMAGE,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", default=DEFAULT_OWNER,
                        help="Kaggle account owning the kernel (kernel id owner)")
    parser.add_argument("--dataset-owner", default=DEFAULT_DATASET_OWNER,
                        help="owner of the val-only input dataset")
    args = parser.parse_args()
    payload = base64.b64encode(build_bundle()).decode("ascii")
    source = RUNNER.read_text()
    updated, count = re.subn(
        r'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n.*?\n?"""',
        lambda _m: f'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n{payload}\n"""',
        source, count=1, flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError("Failed to embed archive into runner")
    runner_text = re.sub(r"^KERNEL_ID = \".*\"$", f'KERNEL_ID = "{DEFAULT_OWNER}/{SLUG}"',
                         updated, count=1, flags=re.M)
    RUNNER.write_text(runner_text)  # tracked runner: archive refreshed, default owner
    staged, n = re.subn(r"^KERNEL_ID = \".*\"$", f'KERNEL_ID = "{args.owner}/{SLUG}"',
                        updated, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError("Failed to set KERNEL_ID")
    STAGING.mkdir(parents=True, exist_ok=True)
    (STAGING / RUNNER.name).write_text(staged)
    (STAGING / "kernel-metadata.json").write_text(
        json.dumps(build_metadata(args.owner, args.dataset_owner), indent=2) + "\n")
    print(f"embedded {len(payload)} b64 chars")
    print(f"staged -> {STAGING}")
    print(f"kernel_id={args.owner}/{SLUG} dataset={args.dataset_owner}/{DATASET_SLUG}")


if __name__ == "__main__":
    main()
