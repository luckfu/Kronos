"""TimesFM-3 LoRA fine-tuning preflight on Kaggle.

This deliberately checks the public package's trainability before spending a
long GPU run. The inference forecaster API is not assumed to be trainable.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import torch


def main() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "git+https://github.com/google-research/timesfm.git",
            "peft",
        ]
    )
    from timesfm3 import TimesFM3Forecaster

    forecaster = TimesFM3Forecaster.from_pretrained(
        "google/timesfm-3.0-pytorch",
        per_core_batch_size=1,
    )
    report = {
        "forecaster_class": type(forecaster).__qualname__,
        "predict_batch_signature": str(inspect.signature(forecaster.predict_batch)),
        "public_attributes": sorted(
            name for name in dir(forecaster) if not name.startswith("_")
        ),
    }
    model = getattr(forecaster, "model", None)
    report["model_class"] = type(model).__qualname__ if model is not None else None
    report["model_is_torch_module"] = isinstance(model, torch.nn.Module)
    if isinstance(model, torch.nn.Module):
        trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        report["parameter_count"] = sum(
            parameter.numel() for parameter in model.parameters()
        )
        report["trainable_parameter_count"] = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        report["named_parameter_sample"] = [
            name for name, _ in list(model.named_parameters())[:120]
        ]
        report["forward_signature"] = str(inspect.signature(model.forward))
        report["decode_signature"] = (
            str(inspect.signature(model.decode))
            if hasattr(model, "decode")
            else None
        )
        report["trainable_parameter_sample"] = trainable[:40]
    report["cuda"] = torch.cuda.is_available()
    report["torch_version"] = torch.__version__
    Path("/kaggle/working/timesfm3_lora_preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
