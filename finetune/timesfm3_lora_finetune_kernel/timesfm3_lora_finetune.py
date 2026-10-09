"""Train a TimesFM-3 PyTorch checkpoint with LoRA on price paths.

The public TimesFM-3 ``decode`` method is inference-only and wrapped in
``torch.no_grad``. This trainer therefore builds the same masked context and
horizon patches and calls ``model.forward`` directly so gradients reach LoRA.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


QUANTILE_INDEX = 4
CONTEXT_FEATURES = (0, 1, 2, 4, 5)  # open, high, low, volume, amount
DATA_FEATURES = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "size_percentile",
)


class NpzPathDataset(Dataset):
    def __init__(self, path: str | Path):
        data = np.load(path, allow_pickle=False)
        self.contexts = data["contexts"].astype(np.float32)
        self.targets = data["targets"].astype(np.float32)
        if self.contexts.ndim != 3 or self.contexts.shape[-1] != 7:
            raise ValueError("contexts must have shape (samples, context, 7)")
        if self.targets.ndim != 2:
            raise ValueError("targets must have shape (samples, horizon)")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.contexts[index]),
            torch.from_numpy(self.targets[index]),
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_npz_from_panel(panel_path: Path, output_dir: Path) -> tuple[str, str]:
    """Build the first train/val split when only the panel dataset is mounted."""
    import pandas as pd

    with panel_path.open("rb") as handle:
        panel = pickle.load(handle)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_ranges = {
        "train": ("2025-07-01", "2026-07-02"),
        "val": ("2026-07-17", "2026-08-10"),
    }
    for split, (start, end) in split_ranges.items():
        contexts, targets = [], []
        for symbol in sorted(panel):
            frame = panel[symbol].copy()
            frame.index = pd.to_datetime(frame.index)
            frame = frame.sort_index()
            if not all(column in frame.columns for column in DATA_FEATURES):
                continue
            values = frame[list(DATA_FEATURES)].astype("float32").to_numpy()
            close = values[:, 3]
            dates = frame.index[
                (frame.index >= pd.Timestamp(start))
                & (frame.index <= pd.Timestamp(end))
            ]
            for asof in dates:
                end_index = frame.index.get_loc(asof)
                if end_index < 119 or end_index + 10 >= len(frame):
                    continue
                context = values[end_index - 119 : end_index + 1]
                target = close[end_index + 1 : end_index + 11]
                if np.isfinite(context).all() and np.isfinite(target).all():
                    contexts.append(context)
                    targets.append(target)
        if not contexts:
            raise RuntimeError(f"No valid {split} samples from {panel_path}")
        np.savez_compressed(
            output_dir / f"{split}.npz",
            contexts=np.asarray(contexts, dtype=np.float32),
            targets=np.asarray(targets, dtype=np.float32),
        )
    return str(output_dir / "train.npz"), str(output_dir / "val.npz")


def build_forward_inputs(
    contexts: torch.Tensor,
    model,
    horizon: int,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Build the masked patch tensors expected by TimesFM3Torch.forward."""
    if contexts.ndim != 3 or contexts.shape[-1] != 7:
        raise ValueError("contexts must be (batch, context, 7)")
    batch, context, _ = contexts.shape
    patch_len = int(model.input_patch_len)
    ctx_padding = (patch_len - context % patch_len) % patch_len
    padded_context = context + ctx_padding
    num_context_patches = padded_context // patch_len
    if getattr(model, "use_stitching", False):
        extract_len = int(model._stitching_extract_len)
        overlap = extract_len - patch_len
        num_forecast_patches = max((horizon - overlap + patch_len - 1) // patch_len, 1)
        num_horizon_patches = num_forecast_patches + int(model.rolls) - 1
        padded_horizon = num_horizon_patches * patch_len
    else:
        padded_horizon = (horizon + int(model.output_patch_len) - 1) // int(
            model.output_patch_len
        ) * int(model.output_patch_len)
        num_horizon_patches = padded_horizon // patch_len

    close = contexts[..., 3:4].transpose(1, 2)
    covariates = contexts[..., CONTEXT_FEATURES].transpose(1, 2)
    close = F.pad(close, (ctx_padding, padded_horizon), value=0.0)
    covariates = F.pad(covariates, (ctx_padding, padded_horizon), value=0.0)
    context_mask = torch.zeros(
        batch, context + ctx_padding, dtype=torch.bool, device=contexts.device
    )
    if ctx_padding:
        context_mask[:, :ctx_padding] = True
    horizon_mask = torch.ones(
        batch, padded_horizon, dtype=torch.bool, device=contexts.device
    )
    masks = torch.cat(
        [
            context_mask,
            horizon_mask,
        ],
        dim=1,
    )
    values = torch.cat([close, covariates], dim=1)
    values = values.reshape(batch, 1 + len(CONTEXT_FEATURES), -1, patch_len)
    masks = masks[:, None, :].expand(batch, values.shape[1], -1)
    masks = masks.reshape(batch, values.shape[1], -1, patch_len)
    patch_is_target = torch.zeros(
        batch,
        values.shape[1],
        num_context_patches + num_horizon_patches,
        dtype=torch.bool,
        device=contexts.device,
    )
    # Target plus past-only covariates are all forecast-model variates. Only
    # future covariates would remain non-target here.
    patch_is_target[:, :, :] = True
    cpm_mask = torch.zeros(
        batch,
        num_context_patches + num_horizon_patches,
        dtype=torch.bool,
        device=contexts.device,
    )
    cpm_mask[:, num_context_patches:] = True
    return (
        {
            "values": values,
            "masks": masks,
            "patch_is_target": patch_is_target,
        },
        num_context_patches,
        padded_horizon,
    )


def extract_median_forecast(
    forward_output: dict[str, torch.Tensor],
    num_context_patches: int,
    horizon: int,
    model,
) -> torch.Tensor:
    logits = forward_output["logits"]
    if getattr(model, "use_stitching", False):
        forecast = logits[:, 0, num_context_patches - 1, :horizon, QUANTILE_INDEX]
    else:
        forecast = logits[:, 0, num_context_patches - 1, :, QUANTILE_INDEX]
        forecast = forecast[:, :horizon]
    return forecast


def path_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    start_close: torch.Tensor,
    return_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw = F.smooth_l1_loss(prediction, target)
    scale = start_close.clamp_min(1e-6).unsqueeze(1)
    normalized = F.smooth_l1_loss(prediction / scale, target / scale)
    pred_return = prediction / scale - 1.0
    target_return = target / scale - 1.0
    return_path = F.smooth_l1_loss(pred_return, target_return)
    total = raw + return_weight * return_path
    return total, {
        "raw_loss": float(raw.detach().cpu()),
        "normalized_loss": float(normalized.detach().cpu()),
        "return_loss": float(return_path.detach().cpu()),
    }


@torch.no_grad()
def evaluate(model, loader, device, horizon: int, return_weight: float) -> dict[str, float]:
    model.eval()
    rows = []
    for contexts, targets in loader:
        contexts, targets = contexts.to(device), targets.to(device)
        inputs, context_patches, _ = build_forward_inputs(contexts, model, horizon)
        output = model(
            inputs,
            freeze_after=context_patches - 1
            if getattr(model, "use_frozen_running_stats", False)
            else None,
            patch_cpm_mask=torch.cat(
                [
                    torch.zeros_like(
                        inputs["patch_is_target"][:, 0, :context_patches]
                    ),
                    torch.ones_like(inputs["patch_is_target"][:, 0, context_patches:]),
                ],
                dim=1,
            ),
        )
        prediction = extract_median_forecast(output, context_patches, horizon, model)
        start = contexts[:, -1, 3]
        loss, metrics = path_losses(prediction, targets, start, return_weight)
        scale = start.clamp_min(1e-6).unsqueeze(1)
        rows.append(
            {
                "loss": float(loss.detach().cpu()),
                **metrics,
                "nmae": float((prediction - targets).abs().div(scale).mean().cpu()),
                "nrmse": float(
                    torch.sqrt(((prediction - targets) / scale).square().mean()).cpu()
                ),
                "path_direction_accuracy": float(
                    (
                        torch.sign(torch.diff(prediction, dim=1))
                        == torch.sign(torch.diff(targets, dim=1))
                    )
                    .float()
                    .mean()
                    .cpu()
                ),
                "endpoint_bias": float(((prediction[:, -1] - targets[:, -1]) / start).mean().cpu()),
            }
        )
    keys = rows[0]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_a = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.lora_b = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=np.sqrt(5))
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base(inputs)
        update = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return base + update * self.scale


def attach_lora(model, rank: int, alpha: int, dropout: float):
    target_names = {
        "query_proj",
        "key_proj",
        "value_proj",
        "out_proj",
        "ff0",
        "ff1",
    }
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name not in target_names or not isinstance(child, nn.Linear):
                continue
            replacement = LoRALinear(child, rank, alpha, dropout)
            setattr(parent, child_name, replacement)
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"LoRA trainable parameters: {trainable} / {total}", flush=True)
    if trainable == 0:
        raise RuntimeError("LoRA injection matched no Linear target modules")
    return model


def train(args: argparse.Namespace) -> dict:
    from timesfm3 import TimesFM3Forecaster

    seed_everything(args.seed)
    device = torch.device(args.device)
    forecaster = TimesFM3Forecaster.from_pretrained(
        args.model,
        per_core_batch_size=args.batch_size,
    )
    model = forecaster.model.to(device)
    model = attach_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
    train_loader = DataLoader(
        NpzPathDataset(args.train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        NpzPathDataset(args.val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best = float("inf")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_rows = []
        for contexts, targets in train_loader:
            contexts, targets = contexts.to(device), targets.to(device)
            inputs, context_patches, padded_horizon = build_forward_inputs(
                contexts, model, targets.shape[1]
            )
            cpm = torch.zeros(
                contexts.shape[0],
                context_patches + padded_horizon // int(model.input_patch_len),
                dtype=torch.bool,
                device=device,
            )
            cpm[:, context_patches:] = True
            out = model(
                inputs,
                freeze_after=context_patches - 1
                if getattr(model, "use_frozen_running_stats", False)
                else None,
                patch_cpm_mask=cpm,
            )
            prediction = extract_median_forecast(
                out, context_patches, targets.shape[1], model
            )
            start = contexts[:, -1, 3]
            loss, metrics = path_losses(
                prediction, targets, start, args.return_loss_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_rows.append({"loss": float(loss.detach().cpu()), **metrics})
        val_metrics = evaluate(
            model, val_loader, device, args.horizon, args.return_loss_weight
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean([x["loss"] for x in train_rows])),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if val_metrics["nrmse"] < best:
            best = val_metrics["nrmse"]
            model.save_pretrained(output / "best_lora")
            (output / "best_metrics.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    (output / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"best_val_nrmse": best, "epochs": args.epochs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train")
    parser.add_argument("--val")
    parser.add_argument("--output", default="/kaggle/working/timesfm3_lora")
    parser.add_argument("--model", default="google/timesfm-3.0-pytorch")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--return-loss-weight", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261008)
    args = parser.parse_args()
    if args.train is None or args.val is None:
        candidates = list(Path("/kaggle/input").glob("**/train.npz"))
        if args.train is None and candidates:
            args.train = str(candidates[0])
        if args.val is None:
            val_candidates = list(Path("/kaggle/input").glob("**/val.npz"))
            if val_candidates:
                args.val = str(val_candidates[0])
    if args.train is None or args.val is None:
        panels = list(Path("/kaggle/input").glob("**/evaluation_panel.pkl"))
        if panels:
            args.train, args.val = build_npz_from_panel(
                panels[0], Path("/kaggle/working/timesfm3_dataset")
            )
    if not args.train or not args.val:
        raise FileNotFoundError(
            "Expected train.npz and val.npz under /kaggle/input or explicit --train/--val"
        )
    return args


if __name__ == "__main__":
    args = parse_args()
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "git+https://github.com/google-research/timesfm.git",
            ]
        )
    print(json.dumps(train(args), ensure_ascii=False, indent=2))
