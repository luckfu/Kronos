"""Kaggle TPU entry point for the predictor trainer.

Launcher choice (torch_xla 2.6-2.8, Kaggle TpuV5E8 = one host, 8 chips):

* ``xmp.spawn(nprocs=None)`` (process per chip) calls
  ``tpu.configure_topology`` and has failed on Kaggle with
  ``Expected 8 worker addresses, got 1``.
* ``pjrt.spawn_threads`` (thread per chip) works on Kaggle but initializes via
  ``_initialize_single_process``, which never calls ``xm.set_replication``.
  ``torch_xla.runtime.world_size()`` therefore stays 1, and
  ``xm.optimizer_step`` / ``xm.all_reduce`` / ``xm.all_gather`` silently become
  no-ops: every core trained an independent replica.
* ``pjrt.initialize_multiprocess`` would set replication but also calls
  ``tpu.configure_topology``, which forces ``TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1``
  (one chip per process) and would break the single 8-chip process.

So the default launcher keeps the Kaggle-proven single-process topology of
``spawn_threads`` (same ``_run_thread_per_device`` driver, same env), and only
adds what ``initialize_multiprocess`` does for multi-device processes on TPU
v2/v3: ``xm.set_replication(device, all_local_devices)``.  That makes
``runtime.world_size() == 8`` so gradients are really all-reduced.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

TRUE_VALUES = {"1", "true", "yes", "on"}

os.environ.setdefault("PJRT_DEVICE", "TPU")
# fp32 master weights: XLA_USE_BF16 / XLA_DOWNCAST_BF16 would store every fp32
# tensor (weights, AdamW moments) as bf16.  Mixed precision is bf16 autocast.
for _flag in ("XLA_USE_BF16", "XLA_DOWNCAST_BF16"):
    if os.environ.get(_flag, "").strip().lower() in TRUE_VALUES:
        raise SystemExit(
            f"{_flag}={os.environ[_flag]} is not allowed: it disables fp32 master "
            "weights. Unset it; the trainer uses torch.autocast('xla', bfloat16)."
        )

cores = int(os.getenv("KRONOS_TPU_CORES", "8"))
if cores > 1:
    os.environ["KRONOS_XLA_SINGLE_PROCESS"] = "0"
else:
    os.environ.setdefault("KRONOS_XLA_SINGLE_PROCESS", "1")

import torch_xla.distributed.xla_multiprocessing as xmp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINETUNE_ROOT = PROJECT_ROOT / "finetune"
for path in (PROJECT_ROOT, FINETUNE_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def _mp_fn(_index: int) -> None:
    os.environ["KRONOS_DEVICE"] = "xla"
    from config import Config
    from train_predictor import main

    main(Config().__dict__)


def initialize_replicated_threads(local_rank: int, local_world_size: int) -> None:
    """``_initialize_single_process`` + ``xm.set_replication`` over all devices."""
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    os.environ.setdefault("PJRT_LOCAL_PROCESS_RANK", str(local_rank))
    os.environ.setdefault("PJRT_LOCAL_PROCESS_COUNT", str(local_world_size))
    devices = xm.get_xla_supported_devices()
    device_fn = getattr(torch_xla, "device", None) or xm.xla_device
    xm.set_replication(device_fn(), devices)
    # runtime.world_size() caches its first answer; drop a stale "1".
    if getattr(xr, "_WORLD_SIZE", None) not in (None, len(devices)):
        xr._WORLD_SIZE = None
    world_size = int(xr.world_size())
    expected = int(os.getenv("KRONOS_TPU_CORES", str(len(devices))))
    print(
        "[TPU Launcher] replication devices="
        f"{len(devices)} runtime.world_size={world_size} expected={expected}",
        flush=True,
    )
    if world_size != len(devices) or world_size != expected:
        raise RuntimeError(
            "xla_world_size_check_failed: torch_xla.runtime.world_size()="
            f"{world_size}, replication devices={len(devices)}, expected={expected}"
        )


def run_replicated_threads(fn) -> None:
    from torch_xla._internal import pjrt

    pjrt._run_thread_per_device(
        local_rank=0,
        local_world_size=1,
        fn=pjrt._SpawnFn(fn),
        initializer_fn=initialize_replicated_threads,
    )


def select_launcher(cores: int, preference: str, pjrt_module) -> str:
    """Return one of: single, replicated_threads, spawn."""
    if cores == 1:
        return "single"
    preference = (preference or "auto").strip().lower()
    if preference == "spawn":
        return "spawn"
    if preference == "spawn_threads":
        raise ValueError(
            "KRONOS_TPU_LAUNCHER=spawn_threads is disabled: it leaves "
            "runtime.world_size()==1 and skips the gradient all-reduce."
        )
    has_threads = pjrt_module is not None and all(
        hasattr(pjrt_module, name) for name in ("_run_thread_per_device", "_SpawnFn")
    )
    if preference in {"auto", "replicated_threads"} and has_threads:
        return "replicated_threads"
    if preference == "replicated_threads":
        raise RuntimeError("torch_xla lacks pjrt._run_thread_per_device/_SpawnFn")
    return "spawn"


def launch() -> None:
    cores = int(os.getenv("KRONOS_TPU_CORES", "8"))
    if cores != 1 and cores != 8 and os.getenv("PJRT_DEVICE", "TPU").upper() == "TPU":
        raise ValueError(
            "KRONOS_TPU_CORES must be 1 or 8 for the Kaggle TpuV5E8 runner"
        )
    pjrt_module = None
    try:
        from torch_xla._internal import pjrt as pjrt_module
    except Exception:
        pjrt_module = None
    launcher = select_launcher(
        cores, os.getenv("KRONOS_TPU_LAUNCHER", "auto"), pjrt_module
    )
    if launcher == "single":
        xmp.spawn(_mp_fn, nprocs=1, start_method="fork")
        return
    if launcher == "replicated_threads":
        print(
            "[TPU Launcher] Using replicated thread-per-device "
            "(pjrt._run_thread_per_device + xm.set_replication)",
            flush=True,
        )
        os.environ["KRONOS_XLA_THREAD_PER_DEVICE"] = "1"
        run_replicated_threads(_mp_fn)
        return
    print("[TPU Launcher] Using xmp.spawn(nprocs=None) (process-per-device)", flush=True)
    xmp.spawn(_mp_fn, nprocs=None, start_method="fork")


if __name__ == "__main__":
    launch()
