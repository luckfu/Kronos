"""Kaggle TPU v3-8 entry point for the predictor trainer.

Kaggle's libtpu rejects the multi-process slice init that xmp.spawn(nprocs=None)
performs on a single-host TPU VM ("TPU initialization failed: Expected 8 worker
addresses, got 1"), so the trainer must run in a single process. nprocs=1 makes
xmp.spawn call fn in-process, exercising one chip through the full pipeline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_BF16", "1")
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


if __name__ == "__main__":
    xmp.spawn(_mp_fn, nprocs=1, start_method="fork")
