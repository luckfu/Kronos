"""TimesFM-3 official base model on the same 8-channel windows the fine-tuned model saw.

Arm B1 (see finetune/reports/20261113_timesfm3_base_vs_finetuned_prereg_cn.md): the public
``google/timesfm-3.0-pytorch`` weights with no LoRA, the same 7 numerical channels as the
fine-tuned run, and channel 7 (sector) held at exactly zero -- the base model has no sector
embedding, and zeroing it leaves the fine-tuning itself as the only difference.

Window building and the forward pass are imported from ``eval_native_merged`` unchanged; only
the weights and the sector channel differ, so the two runs are the same code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

INPUT_ROOT = Path(os.environ.get("EVAL_INPUT_ROOT", "/kaggle/input"))
OUTPUT = Path(os.environ.get("EVAL_OUTPUT", "/kaggle/working/eval_results"))
BASE_REPO = "google/timesfm-3.0-pytorch"
SEAL_AFTER = "2026-10-23"
SEGMENTS = {"A_selection_window": ("2026-08-03", "2026-09-17"), "B_new_window": ("2026-09-18", "2026-10-23"),
            "C_early": ("2026-07-17", "2026-07-31")}


def log(event: str, **fields) -> None:
    print(json.dumps({"time": time.time(), "event": event, **fields}, ensure_ascii=False, default=float), flush=True)


def main() -> None:
    import pandas as pd

    import eval_native_merged as base
    from eval_modelscope_window import build_windows

    OUTPUT.mkdir(parents=True, exist_ok=True)
    split_csv = sorted(INPUT_ROOT.glob("**/symbol_split.csv"))[0]
    names = sorted(pd.read_csv(split_csv)["sector"].dropna().unique().tolist())
    contexts, targets, sectors, dates, symbols = build_windows(
        INPUT_ROOT, {name: i for i, name in enumerate(names)}, len(names))

    device = torch.device("cuda")
    from timesfm3 import TimesFM3Forecaster
    model = TimesFM3Forecaster.from_pretrained(BASE_REPO, per_core_batch_size=512).model.to(device)
    model.eval()
    patch_len = int(model.input_patch_len)

    # Channel 7 is held at zero: an all-zero embedding table makes build_forward_inputs emit a
    # zero sector channel without touching the shared forward code.
    sector_embedding = nn.Embedding(len(names) + 5, patch_len).to(device)
    with torch.no_grad():
        sector_embedding.weight.zero_()
    if float(sector_embedding.weight.abs().max()) != 0.0:
        raise RuntimeError("sector channel is not zero")

    architecture = {name: getattr(model, name) for name in
                    ("input_patch_len", "output_patch_len", "use_stitching", "_stitching_extract_len",
                     "rolls", "use_frozen_running_stats") if hasattr(model, name)}
    log("model_ready", repo=BASE_REPO, arm="B1_zero_sector", sectors_padding=len(names) + 5,
        gpus=torch.cuda.device_count(), architecture=architecture,
        parameters=sum(p.numel() for p in model.parameters()))

    if torch.cuda.device_count() >= 2:
        attributes = {n: getattr(model, n) for n in architecture}
        model = nn.DataParallel(model)
        for name, value in attributes.items():
            setattr(model, name, value)

    base.MAX_SAFE_SECONDS = 10**9
    pred, true = base.run_full_inference(model, contexts, targets, sectors, device, sector_embedding, batch_size=512)
    if len(pred) != len(dates) or not np.isfinite(pred).all():
        raise RuntimeError(f"incomplete or non-finite inference: {len(pred)} of {len(dates)}")
    np.savez_compressed(OUTPUT / "eval_predictions_base_channel.npz", pred_ret=pred.astype(np.float32),
                        true_ret=true.astype(np.float32), dates=np.array(dates), symbols=np.array(symbols))

    date_array = np.array(dates)
    report = {"model": BASE_REPO, "arm": "B1_zero_sector", "architecture": architecture,
              "sample_count": int(len(pred)), "segments": {}}
    for name, (start, end) in SEGMENTS.items():
        mask = (date_array >= start) & (date_array <= end)
        if mask.any():
            report["segments"][name] = {"signal_start": start, "signal_end": end, "samples": int(mask.sum()),
                                        "signal_days": int(len(set(date_array[mask]))),
                                        "metrics": base.calculate_metrics(pred[mask], true[mask],
                                                                          asof_dates=list(date_array[mask]))}
    (OUTPUT / "report_base_channel.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log("finished", report=report)


if __name__ == "__main__":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "git+https://github.com/google-research/timesfm.git"])
    main()
