"""Train a TimesFM-3 PyTorch checkpoint with LoRA on price paths.

The public TimesFM-3 ``decode`` method is inference-only and wrapped in
``torch.no_grad``. This trainer therefore builds the same masked context and
horizon patches and calls ``model.forward`` directly so gradients reach LoRA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

PROCESS_STARTED_AT = time.monotonic()
OUTPUT_ROOT = Path(os.environ.get("TIMESFM3_OUTPUT", "/kaggle/working/timesfm3_lora"))
LOG_LOCK = threading.Lock()
CURRENT_PHASE = "started"
PACKAGED_RUN_PLAN = None


def emit(event: str, **fields) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "phase": CURRENT_PHASE,
        **fields,
    }
    line = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    with LOG_LOCK:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        print(line, flush=True)
        for name in ("run.log", "metrics.jsonl"):
            with (OUTPUT_ROOT / name).open("a", buffering=1) as handle:
                handle.write(line + "\n")
        temporary = OUTPUT_ROOT / "progress.json.tmp"
        progress = {}
        if (OUTPUT_ROOT / "progress.json").is_file():
            progress = json.loads((OUTPUT_ROOT / "progress.json").read_text())
        progress.update(payload)
        temporary.write_text(json.dumps(progress, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(OUTPUT_ROOT / "progress.json")


def phase(name: str, **fields) -> None:
    global CURRENT_PHASE
    CURRENT_PHASE = name
    emit("phase", **fields)


def phase_heartbeat() -> None:
    while True:
        time.sleep(30)
        emit("phase_heartbeat")


def run_streamed(command: list[str]) -> None:
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=environment,
    ) as process:
        for line in process.stdout:
            with LOG_LOCK:
                print(line, end="", flush=True)
                with (OUTPUT_ROOT / "run.log").open("a", buffering=1) as handle:
                    handle.write(line)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    emit("phase", phase="started")
    threading.Thread(target=phase_heartbeat, daemon=True).start()
    phase("import_dependencies")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


QUANTILE_INDEX = 4
CONTEXT_FEATURES = (0, 1, 2, 4, 5)  # open, high, low, volume, amount
SWANLAB_API_KEY_FALLBACK = "fmEPDGk4IItxgqSZKGLi8"
SWANLAB_RUN_ID = "timesfm3-lora-finetune-v7-20261009"
SWANLAB_PROJECT = "finance"
SWANLAB_WORKSPACE = "roc_fu"
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


def gpu_metrics() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"gpu/count": 0.0}
    allocated = []
    reserved = []
    for index in range(torch.cuda.device_count()):
        allocated.append(torch.cuda.memory_allocated(index) / 1024**3)
        reserved.append(torch.cuda.memory_reserved(index) / 1024**3)
    return {
        "gpu/count": float(torch.cuda.device_count()),
        "gpu/max_allocated_gb": float(max(allocated, default=0.0)),
        "gpu/max_reserved_gb": float(max(reserved, default=0.0)),
    }


def optimizer_metrics(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        "train/learning_rate": float(
            max(group.get("lr", 0.0) for group in optimizer.param_groups)
        ),
        "train/weight_decay": float(
            max(group.get("weight_decay", 0.0) for group in optimizer.param_groups)
        ),
    }


def grad_norm(model: nn.Module) -> float:
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().float().square().sum().cpu())
    return float(total**0.5)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    horizon: int,
    return_weight: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    rows = []
    for batch_index, (contexts, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
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


def compact_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    base = model.module if isinstance(model, nn.DataParallel) else model
    required = {key for key, value in base.named_parameters() if value.requires_grad}
    required.update(key for key, _ in base.named_buffers())
    return {
        key: value.detach().cpu().clone()
        for key, value in base.state_dict().items()
        if key in required
    }


def frozen_model_sha256(model: nn.Module) -> str:
    base = model.module if isinstance(model, nn.DataParallel) else model
    digest = hashlib.sha256()
    for key, value in base.named_parameters():
        if value.requires_grad:
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def make_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    best: float,
    history: list[dict],
    *,
    metadata: dict | None = None,
) -> dict:
    base = model.module if isinstance(model, nn.DataParallel) else model
    return {
        "format": "timesfm3_trainable_compact_v2",
        "model": compact_model_state(model),
        "trainable_names": [key for key, value in base.named_parameters() if value.requires_grad],
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "best": best,
        "history": history,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
        "numpy_rng": {
            "algorithm": np.random.get_state()[0],
            "keys": torch.tensor(np.random.get_state()[1].astype(np.int64)),
            "position": np.random.get_state()[2],
            "has_gauss": np.random.get_state()[3],
            "cached_gaussian": np.random.get_state()[4],
        },
        **(metadata or {}),
    }


def restore_checkpoint(model, optimizer, state, base_sha256: str) -> None:
    if state.get("format") != "timesfm3_trainable_compact_v2":
        raise RuntimeError("Incomplete/legacy checkpoint is not a valid continuation source")
    if not state.get("validation_complete"):
        raise RuntimeError("Only a validation-complete checkpoint may be continued")
    if state.get("base_model_sha256") != base_sha256:
        raise RuntimeError("Frozen base model SHA-256 differs from checkpoint")
    base = model.module if isinstance(model, nn.DataParallel) else model
    trainable = [key for key, value in base.named_parameters() if value.requires_grad]
    if trainable != state.get("trainable_names"):
        raise RuntimeError("Trainable parameter names/order differ from checkpoint")
    required = set(compact_model_state(model))
    if required != set(state["model"]):
        raise RuntimeError("Checkpoint must contain every trained parameter and buffer")
    _, unexpected = base.load_state_dict(state["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected}")
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    random.setstate(state["python_rng"])
    rng = state["numpy_rng"]
    np.random.set_state((
        rng["algorithm"], rng["keys"].cpu().numpy().astype(np.uint32),
        rng["position"], rng["has_gauss"], rng["cached_gaussian"],
    ))


def save_checkpoint(path: Path, state: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def inherit_continuation(args: argparse.Namespace) -> None:
    if not args.continuation_slug:
        return
    candidates = []
    for path in Path("/kaggle/input").glob("**/experiment_manifest.json"):
        if args.continuation_slug not in path.parts:
            continue
        manifest = json.loads(path.read_text())
        if manifest.get("run_id") == SWANLAB_RUN_ID:
            candidates.append(path.parent)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one continuation output, found {candidates}")
    source = candidates[0]
    required = (
        "last_state.pt", "best.pt", "latest.pt", "best_metrics.json", "history.json",
        "run.log", "metrics.jsonl", "progress.json", "summary.json", "experiment_manifest.json",
    )
    if any(not (source / name).is_file() for name in required):
        raise RuntimeError("Continuation output contract is incomplete")
    startup_logs = {
        name: (OUTPUT_ROOT / name).read_text() if (OUTPUT_ROOT / name).exists() else ""
        for name in ("run.log", "metrics.jsonl")
    }
    with LOG_LOCK:
        shutil.copytree(source, args.output, dirs_exist_ok=True)
        for name, text in startup_logs.items():
            with (OUTPUT_ROOT / name).open("a", buffering=1) as handle:
                handle.write(text)
    args.resume_from = str(Path(args.output) / "last_state.pt")
    emit("continuation_inherited", source=str(source), resume_state=args.resume_from)


def validate_gpu(device) -> list[str]:
    if device.type != "cuda" or torch.cuda.device_count() != 2:
        raise RuntimeError("This experiment requires exactly two CUDA GPUs")
    names = [torch.cuda.get_device_name(index) for index in range(2)]
    if not all("T4" in name for name in names):
        raise RuntimeError(f"Expected dual T4, got {names}")
    return names


def train(args: argparse.Namespace) -> dict:
    from timesfm3 import TimesFM3Forecaster

    seed_everything(args.seed)
    device = torch.device(args.device)
    gpu_names = validate_gpu(device)
    phase("swanlab_init")
    import swanlab
    swanlab.login(
        api_key=os.environ.get("SWANLAB_API_KEY", "").strip() or SWANLAB_API_KEY_FALLBACK
    )
    swanlab_run = swanlab.init(
        project=SWANLAB_PROJECT, workspace=SWANLAB_WORKSPACE,
        experiment_name=SWANLAB_RUN_ID, id=SWANLAB_RUN_ID,
        resume="allow",
        config=vars(args), mode="cloud",
    )
    url = getattr(swanlab_run, "url", "") or getattr(swanlab_run, "web_url", "")
    if not url:
        raise RuntimeError("SwanLab did not return a cloud run URL")
    emit("swanlab_ready", run_id=SWANLAB_RUN_ID, url=url)
    phase("download_model", model=args.model)
    forecaster = TimesFM3Forecaster.from_pretrained(
        args.model,
        per_core_batch_size=args.batch_size,
    )
    model = forecaster.model.to(device)
    model = attach_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
    phase("checkpoint_preflight")
    base_sha256 = frozen_model_sha256(model)
    saved = compact_model_state(model)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trained_names = {key for key, p in model.named_parameters() if p.requires_grad}
    saved_count = sum(value.numel() for key, value in saved.items() if key in trained_names)
    if saved_count != trainable_count or not trained_names.issubset(saved):
        raise RuntimeError("Checkpoint preflight omitted trained parameters")
    emit("checkpoint_preflight_passed", trained_parameters=trainable_count,
         saved_trained_parameters=saved_count, base_model_sha256=base_sha256)
    del saved
    if args.multi_gpu and torch.cuda.device_count() >= 2:
        base_attrs = {
            name: getattr(model, name)
            for name in (
                "input_patch_len",
                "output_patch_len",
                "use_stitching",
                "_stitching_extract_len",
                "rolls",
                "use_frozen_running_stats",
            )
            if hasattr(model, name)
        }
        model = nn.DataParallel(model)
        for name, value in base_attrs.items():
            setattr(model, name, value)
        emit("multi_gpu_ready", gpu_count=2, device_ids=[0, 1], gpu_names=gpu_names)
    phase("load_dataset")
    train_generator = torch.Generator()
    train_loader = DataLoader(
        NpzPathDataset(args.train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=train_generator,
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
    global_step = 0
    started_at = time.time()
    metadata = {
        "config": vars(args),
        "base_model_sha256": base_sha256,
        "run_id": SWANLAB_RUN_ID,
        "epoch_index": 0,
        "batches_completed": 0,
        "validation_complete": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    phase("verify_data")
    data_hashes = {}
    for split, filename in (("train", args.train), ("val", args.val)):
        with Path(filename).open("rb") as handle:
            data_hashes[split] = hashlib.file_digest(handle, "sha256").hexdigest()
    metadata["data_sha256"] = data_hashes
    (output / "experiment_manifest.json").write_text(
        json.dumps({
            **metadata, "gpu_names": gpu_names, "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda, "trainable_parameters": trainable_count,
            "train_samples": len(train_loader.dataset),
            "val_candidates": len(val_loader.dataset),
            "validation_scope": "fixed leading subset, not full validation",
            "validation_samples": min(
                len(val_loader.dataset), args.val_max_batches * args.batch_size
            ),
        }, indent=2)
    )
    resume_epoch, resume_batches = 0, 0
    if args.resume_from:
        state = torch.load(args.resume_from, map_location="cpu", weights_only=True)
        if state.get("run_id") != SWANLAB_RUN_ID:
            raise RuntimeError("Resume source belongs to a different experiment")
        for key in (
            "model", "batch_size", "horizon", "learning_rate", "weight_decay",
            "return_loss_weight", "max_grad_norm", "lora_rank", "lora_alpha",
            "lora_dropout", "seed", "val_max_batches",
        ):
            if state["config"][key] != vars(args)[key]:
                raise RuntimeError(f"Resume configuration differs: {key}")
        restore_checkpoint(model, optimizer, state, base_sha256)
        if state.get("data_sha256") != data_hashes:
            raise RuntimeError("Resume data SHA-256 differs from checkpoint")
        global_step = int(state["global_step"])
        best = float(state["best"])
        history = list(state.get("history", []))
        resume_epoch, resume_batches = state["epoch_index"], state["batches_completed"]
        emit("resume_ready", resume_from=args.resume_from, global_step=global_step,
             best_val_nrmse=best)
    else:
        phase("baseline_validation")
        baseline = evaluate(
            model, val_loader, device, args.horizon,
            args.return_loss_weight, args.val_max_batches,
        )
        best = baseline["nrmse"]
        history.append({"epoch": 0, "step": 0, **{f"val_{k}": v for k, v in baseline.items()}})
        swanlab_run.log({f"validation/{key}": value for key, value in baseline.items()}, step=0)
        metadata["validation_complete"] = True
        initial_state = make_checkpoint(model, optimizer, 0, best, history, metadata=metadata)
        save_checkpoint(output / "best.pt", initial_state)
        save_checkpoint(output / "last_state.pt", initial_state)
        (output / "best_metrics.json").write_text(json.dumps(history[-1], indent=2))
        emit("baseline_validation_complete", step=0, **baseline)
        del initial_state
    initial_step = global_step
    started_at = time.time()
    stop_reason = "max_steps"
    emit("train_start", train_samples=len(train_loader.dataset),
         val_samples=len(val_loader.dataset), max_steps=args.max_steps,
         device=str(device), config=vars(args))
    for epoch in range(resume_epoch, args.epochs):
        phase("training", epoch=epoch + 1, step=global_step)
        model.train()
        train_generator.manual_seed(args.seed + epoch)
        train_rows = []
        for batch_index, (contexts, targets) in enumerate(train_loader):
            if epoch == resume_epoch and batch_index < resume_batches:
                continue
            if global_step >= args.max_steps or (
                time.monotonic() - PROCESS_STARTED_AT >=
                args.run_budget_seconds - args.finalize_reserve_seconds
            ):
                stop_reason = "max_steps" if global_step >= args.max_steps else "time_budget"
                break
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
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite training loss at step {global_step + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            unclipped_grad_norm = grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            post_clip_grad_norm = grad_norm(model)
            optimizer.step()
            global_step += 1
            metadata.update(
                epoch_index=epoch, batches_completed=batch_index + 1,
                validation_complete=False,
            )
            train_rows.append({"loss": float(loss.detach().cpu()), **metrics})
            if global_step % args.log_interval == 0:
                elapsed = max(time.time() - started_at, 1e-6)
                heartbeat = {
                    "event": "heartbeat",
                    "epoch": epoch + 1,
                    "step": global_step,
                    "train/loss": train_rows[-1]["loss"],
                    "train/raw_loss": train_rows[-1]["raw_loss"],
                    "train/normalized_loss": train_rows[-1]["normalized_loss"],
                    "train/return_loss": train_rows[-1]["return_loss"],
                    "train/unclipped_grad_norm": unclipped_grad_norm,
                    "train/post_clip_grad_norm": post_clip_grad_norm,
                    "train/steps_per_second": (global_step - initial_step) / elapsed,
                    "train/samples_per_second": (
                        (global_step - initial_step) * args.batch_size / elapsed
                    ),
                    "budget/seconds_remaining": max(
                        0.0, args.run_budget_seconds -
                        (time.monotonic() - PROCESS_STARTED_AT)
                    ),
                    "training/epoch": float(epoch + 1),
                    "training/step": float(global_step),
                    **optimizer_metrics(optimizer),
                    **gpu_metrics(),
                }
                emit("heartbeat", **{key: value for key, value in heartbeat.items() if key != "event"})
                if swanlab_run is not None:
                    swanlab_run.log(
                        {
                            key: value
                            for key, value in heartbeat.items()
                            if key != "event"
                        },
                        step=global_step,
                    )
            if global_step % args.checkpoint_interval == 0:
                state = make_checkpoint(
                    model, optimizer, global_step, best, history, metadata=metadata
                )
                save_checkpoint(output / "training_snapshot.pt", state)
                checkpoint_metrics = {
                    "checkpoint/step": float(global_step),
                    "checkpoint/saved": 1.0,
                }
                emit("checkpoint_saved", **checkpoint_metrics,
                     checkpoint_path=str(output / "training_snapshot.pt"),
                     resumable=False)
                del state
                if swanlab_run is not None:
                    swanlab_run.log(checkpoint_metrics, step=global_step)
            if global_step >= args.max_steps:
                break
        phase("validation", step=global_step, max_batches=args.val_max_batches)
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            args.horizon,
            args.return_loss_weight,
            args.val_max_batches,
        )
        row = {
            "epoch": epoch + 1,
            "step": global_step,
            "train_loss": float(np.mean([x["loss"] for x in train_rows])) if train_rows else None,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        emit("validation_complete", **row)
        if swanlab_run is not None:
            swanlab_run.log(
                {
                    **{f"validation/{key}": value for key, value in val_metrics.items()},
                    "training/epoch": float(epoch + 1),
                    "training/step": float(global_step),
                    **gpu_metrics(),
                },
                step=global_step,
            )
        if val_metrics["nrmse"] < best:
            best = val_metrics["nrmse"]
            metadata["validation_complete"] = True
            save_checkpoint(
                output / "best.pt",
                make_checkpoint(model, optimizer, global_step, best, history, metadata=metadata),
            )
            (output / "best_metrics.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        metadata["validation_complete"] = True
        if metadata["batches_completed"] == len(train_loader):
            metadata.update(epoch_index=epoch + 1, batches_completed=0)
        save_checkpoint(
            output / "last_state.pt",
            make_checkpoint(model, optimizer, global_step, best, history, metadata=metadata),
        )
        if global_step >= args.max_steps or stop_reason == "time_budget":
            break
    phase("export", step=global_step)
    (output / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_checkpoint(
        output / "latest.pt",
        make_checkpoint(model, optimizer, global_step, best, history, metadata=metadata),
    )
    exported = torch.load(output / "last_state.pt", map_location="cpu", weights_only=True)
    current = compact_model_state(model)
    if not exported["validation_complete"] or exported["global_step"] != global_step:
        raise RuntimeError("Published last_state does not describe the validated boundary")
    if set(exported["model"]) != set(current) or any(
        not torch.equal(exported["model"][key], value) for key, value in current.items()
    ):
        raise RuntimeError("Exported checkpoint differs from the validated model")
    required_outputs = (
        "best.pt", "best_metrics.json", "latest.pt", "last_state.pt",
        "history.json", "run.log", "metrics.jsonl", "progress.json", "experiment_manifest.json",
    )
    if any(not (output / name).is_file() for name in required_outputs):
        raise RuntimeError("Required output artifact is missing")
    emit("checkpoint_export_verified", step=global_step,
         trained_parameters=trainable_count, saved_trained_parameters=saved_count)
    del exported, current
    if swanlab_run is not None:
        swanlab_run.log(
            {
                "training/completed": 1.0,
                "training/final_step": float(global_step),
                "validation/best_nrmse": float(best),
                **gpu_metrics(),
            },
            step=global_step,
        )
    result = {
        "best_val_nrmse": best, "epochs": args.epochs, "global_step": global_step,
        "stop_reason": stop_reason, "run_id": SWANLAB_RUN_ID,
        "checkpoint_verified": True,
        "validation_complete": True,
        "elapsed_process_seconds": time.monotonic() - PROCESS_STARTED_AT,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    phase("swanlab_upload", step=global_step)
    swanlab.finish()
    phase("complete", **result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train")
    parser.add_argument("--val")
    parser.add_argument("--output", default=str(OUTPUT_ROOT))
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
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--run-budget-seconds", type=int, default=16200)
    parser.add_argument("--finalize-reserve-seconds", type=int, default=600)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--resume-from")
    parser.add_argument("--continuation-slug")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--swanlab", action="store_true", default=True)
    parser.add_argument("--require-swanlab", action="store_true", default=True)
    parser.add_argument("--multi-gpu", action="store_true", default=True)
    parser.add_argument("--val-max-batches", type=int, default=256)
    plan_path = Path(__file__).with_name("run-plan.json")
    plan = PACKAGED_RUN_PLAN
    if plan is None and plan_path.is_file():
        plan = json.loads(plan_path.read_text())
    if plan is not None:
        parser.set_defaults(**plan["arguments"])
    args = parser.parse_args()
    if args.output != str(OUTPUT_ROOT):
        raise ValueError("Set TIMESFM3_OUTPUT to keep logs and model output in the same tree")
    if args.auto_resume:
        raise ValueError("Automatic checkpoint scanning is unsafe; use a verified continuation output")
    phase("continuation_discovery")
    inherit_continuation(args)
    phase("discover_inputs")
    if args.train is None or args.val is None:
        candidates = list(Path("/kaggle/input").glob("**/train.npz"))
        if args.train is None and len(candidates) == 1:
            args.train = str(candidates[0])
        if args.val is None:
            val_candidates = list(Path("/kaggle/input").glob("**/val.npz"))
            if len(val_candidates) == 1:
                args.val = str(val_candidates[0])
    if args.train is None or args.val is None:
        panels = list(Path("/kaggle/input").glob("**/evaluation_panel.pkl"))
        if len(panels) == 1:
            phase("build_dataset")
            args.train, args.val = build_npz_from_panel(
                panels[0], Path("/kaggle/working/timesfm3_dataset")
            )
    if args.auto_resume and args.resume_from is None:
        checkpoints = sorted(Path("/kaggle/input").glob("**/step_*.pt"))
        if checkpoints:
            args.resume_from = str(checkpoints[-1])
    if not args.train or not args.val:
        raise FileNotFoundError(
            "Expected train.npz and val.npz under /kaggle/input or explicit --train/--val"
        )
    return args


if __name__ == "__main__":
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        phase("install_dependencies")
        run_streamed(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--progress-bar", "off",
                "git+https://github.com/google-research/timesfm.git",
                "swanlab==0.10.1",
            ]
        )
        phase("dependencies_ready")
    args = parse_args()
    train(args)
