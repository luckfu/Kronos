"""Kaggle TPU v3-8 entry point for the predictor trainer.

Kaggle's libtpu rejects the multi-process slice init that xmp.spawn(nprocs=None)
performs on a single-host TPU VM ("TPU initialization failed: Expected 8 worker
addresses, got 1"), so the trainer must run in a single process. nprocs=1 makes
xmp.spawn call fn in-process, exercising one chip through the full pipeline.
"""

from __future__ import annotations

import os

import torch_xla.distributed.xla_multiprocessing as xmp


def _mp_fn(_index: int) -> None:
    os.environ["KRONOS_DEVICE"] = "xla"
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("XLA_USE_BF16", "1")
    from config import Config
    from train_predictor import main

    main(Config().__dict__)


if __name__ == "__main__":
    xmp.spawn(_mp_fn, nprocs=1, start_method="fork")
