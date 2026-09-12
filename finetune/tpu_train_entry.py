"""Kaggle TPU entry point for the predictor trainer.

Kaggle exposes one TPU host.  ``xmp.spawn(nprocs=None)`` creates one process
per device and asks PJRT to configure a multi-host topology; on Kaggle this
has failed with ``Expected 8 worker addresses, got 1``.  The multi-core path
therefore uses PJRT's one-process/thread-per-device launcher.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_BF16", "1")

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


def launch() -> None:
    cores = int(os.getenv("KRONOS_TPU_CORES", "8"))
    if cores == 1:
        xmp.spawn(_mp_fn, nprocs=1, start_method="fork")
        return
    if cores != 8:
        raise ValueError(
            "KRONOS_TPU_CORES must be 1 or 8 for the Kaggle TpuV5E8 runner"
        )

    launcher_pref = os.getenv("KRONOS_TPU_LAUNCHER", "auto").strip().lower()
    spawn_threads = None
    try:
        from torch_xla._internal import pjrt
        spawn_threads = getattr(pjrt, "spawn_threads", None)
    except Exception:
        pass

    if launcher_pref != "spawn" and spawn_threads is not None:
        print("[TPU Launcher] Using pjrt.spawn_threads (thread-per-device)", flush=True)
        os.environ["KRONOS_XLA_THREAD_PER_DEVICE"] = "1"
        spawn_threads(_mp_fn)
    else:
        print("[TPU Launcher] Using xmp.spawn(nprocs=None) (process-per-device)", flush=True)
        xmp.spawn(_mp_fn, nprocs=None, start_method="fork")


if __name__ == "__main__":
    launch()
