"""Kaggle entry for Base-clean Bootstrap timing segment 1 (dual T4)."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENT = "base_clean_0.1_bootstrap_timing"
OUTPUT_NAME = "base_clean_0.1_bootstrap"
RUN_ID = "base_clean_0_1_bootstrap_timing"
DATASET = "smmt315/a-share-120d-temporal-symbol-holdout"
# Matches e01 / published smmt315 data_manifest.json content hash.
DATA_SHA = "b85ccc45f7cd0e895e607e6e75c7a603427bd62a353a016742a52b2943d652cd"
TORCH_VERSION = "2.6.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"
# Update batch 32 / accum 8 / n_gpu 2 → micro 4, effective global 64
BATCH_SIZE = 32
ACCUMULATION_STEPS = 8
NPROC = 2
MAX_SEGMENTS = 1


def phase(name: str, **payload) -> None:
    print(json.dumps({"phase": name, "time": time.time(), **payload}, ensure_ascii=False), flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_swanlab_key() -> str:
    token = os.environ.get("SWANLAB_API_KEY", "").strip()
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient

        token = (UserSecretsClient().get_secret("SWANLAB_API_KEY") or "").strip()
    except Exception as exc:
        phase("swanlab_secret_unavailable", error=f"{type(exc).__name__}: {exc}")
        token = ""
    if token:
        os.environ["SWANLAB_API_KEY"] = token
    return token


def prepare_data(input_root: Path, target: Path) -> Path:
    """Materialize processed_datasets/ layout from flat or nested smmt315 mount.

    Kaggle mounts smmt315 with train/val pkl at the dataset root; the runner
    expects processed_datasets/*.pkl under KRONOS_BASE_CLEAN_DATA_ROOT. Mirror
    e01 prepare_data: verify data_manifest.json, then symlink by basename.
    """
    candidates = [
        path for path in input_root.glob("**/data_manifest.json")
        if sha256(path) == DATA_SHA
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one clean data_manifest.json ({DATA_SHA[:12]}…); "
            f"found {len(candidates)}: {candidates}"
        )
    manifest_path = candidates[0]
    source = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if target.exists():
        # Idempotent re-entry: rebuild clean tree.
        import shutil

        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    (target / "processed_datasets").mkdir(exist_ok=True)
    for entry in manifest["files"]:
        relative = Path(entry["name"])
        path = source / relative
        if not path.is_file():
            path = source / relative.name
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Clean dataset hash mismatch: {path}")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(path.resolve())
    manifest_dest = target / "data_manifest.json"
    if manifest_dest.exists() or manifest_dest.is_symlink():
        manifest_dest.unlink()
    manifest_dest.symlink_to(manifest_path.resolve())
    phase(
        "data_ready",
        dataset=DATASET,
        manifest_sha256=DATA_SHA,
        stats=manifest.get("stats"),
        split=manifest.get("split"),
        window_contract=manifest.get("window_contract"),
        data_root=str(target),
    )
    return target


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    finetune = repo / "finetune"
    if str(finetune) not in sys.path:
        sys.path.insert(0, str(finetune))

    phase(
        "started",
        experiment=EXPERIMENT,
        output=OUTPUT_NAME,
        dataset=DATASET,
        micro_bs=BATCH_SIZE // ACCUMULATION_STEPS,
        accum=ACCUMULATION_STEPS,
        n_gpu=NPROC,
        effective_global_batch=BATCH_SIZE * NPROC,
        updates_per_seg_approx=20000 // (BATCH_SIZE * NPROC),
        max_segments=MAX_SEGMENTS,
        start="NeoQuasar/Kronos-base",
        context_layer=10,
        note="HF Base clean; no Beta/V6/smce2 weights",
    )

    # Mirror e01 dual-T4 install: never bare `pip -r requirements.txt` on
    # Kaggle Python 3.13 (pandas metadata-generation-failed / source builds).
    # Pin torch + numpy/pandas binary wheels only (same recipe as smce2 e01i).
    phase(
        "install_dependencies",
        torch=TORCH_VERSION,
        numpy="2.1.3",
        pandas="2.2.3",
        note="e01-style only-binary pins; skip requirements.txt",
    )
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            f"torch=={TORCH_VERSION}",
            "--only-binary=:all:",
            "--index-url", TORCH_INDEX_URL,
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            "--index-url", "https://pypi.org/simple",
            "--only-binary=numpy,pandas,safetensors,orjson,psutil,protobuf,pyyaml,hf-xet",
            "numpy==2.1.3", "pandas==2.2.3",
            "einops==0.8.1", "huggingface_hub==0.33.1", "safetensors==0.6.2",
            "tqdm==4.67.1", "swanlab==0.10.1", "psutil", "pynvml",
        ],
        check=True,
    )

    import torch

    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(devices) != 2 or any("T4" not in name for name in devices):
        raise RuntimeError(f"Expected dual T4; found {devices}")
    phase("device_ready", devices=devices, torch=torch.__version__, cuda=torch.version.cuda)

    token = resolve_swanlab_key()
    if not token:
        phase(
            "swanlab_missing",
            note="Continuing with local logs only; set SWANLAB_API_KEY secret for remote board",
        )
        os.environ["KRONOS_REQUIRE_SWANLAB"] = "0"
    else:
        os.environ["KRONOS_REQUIRE_SWANLAB"] = os.environ.get("KRONOS_REQUIRE_SWANLAB", "0")

    for key in ("SWANLAB_PROJECT", "SWANLAB_WORKSPACE", "SWANLAB_EXPERIMENT_NAME"):
        os.environ.pop(key, None)

    input_root = Path(os.environ.get("KRONOS_KAGGLE_INPUT_ROOT", "/kaggle/input"))
    data_root = prepare_data(input_root, Path("/kaggle/working/kronos_base_clean/data"))

    os.environ.update({
        "PYTHONUNBUFFERED": "1",
        "KRONOS_BASE_CLEAN_STAGE": "bootstrap",
        "KRONOS_BASE_CLEAN_OUTPUT_NAME": OUTPUT_NAME,
        "KRONOS_BASE_CLEAN_DISABLE_AUTO_CONTINUATION": "1",
        "KRONOS_BASE_CLEAN_DATA_ROOT": str(data_root),
        "KRONOS_MAX_SEGMENTS_PER_RUN": str(MAX_SEGMENTS),
        "KRONOS_BATCH_SIZE": str(BATCH_SIZE),
        "KRONOS_ACCUMULATION_STEPS": str(ACCUMULATION_STEPS),
        "KRONOS_TORCHRUN_NPROC_PER_NODE": str(NPROC),
        "SWANLAB_PROJECT": "finance",
        "SWANLAB_WORKSPACE": "roc_fu",
        "SWANLAB_EXPERIMENT_NAME": EXPERIMENT,
        "SWANLAB_RUN_ID": RUN_ID,
    })

    phase(
        "launch_runner",
        runner="finetune/kaggle_kronos_base_clean.py",
        swanlab_experiment=EXPERIMENT,
        swanlab_run_id=RUN_ID,
        data_root=str(data_root),
    )
    subprocess.run(
        [sys.executable, "-u", str(finetune / "kaggle_kronos_base_clean.py")],
        cwd=str(repo),
        env=os.environ.copy(),
        check=True,
    )
    phase("completed", experiment=EXPERIMENT)


if __name__ == "__main__":
    main()
