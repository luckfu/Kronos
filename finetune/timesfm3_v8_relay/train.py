"""TimesFM-3 LoRA finetune V8: formal split shards + chunk relay.

数据: wynstonliu/timesfm3-formal-split-dataset-v1-cpu 的输出
      (由 luckfu/a-share-120d-temporal-symbol-holdout 的 train_data.pkl / val_data.pkl 切窗口得到)
接力: 每个 chunk 把 relay/ 写到 /kaggle/working; 下一个 chunk 以上一个 kernel 为
      kernel_source, 启动时自动发现 relay/state.pt 并精确续训 (模型/优化器/数据游标/RNG)。
看板: SwanLab 使用固定 run id + resume="allow", 所有 chunk 写入同一条曲线。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

STARTED = time.monotonic()
INPUT_ROOT = Path(os.environ.get("RELAY_INPUT_ROOT", "/kaggle/input"))
OUTPUT = Path(os.environ.get("RELAY_OUTPUT", "/kaggle/working/relay"))

SWANLAB_API_KEY = "fmEPDGk4IItxgqSZKGLi8"
SWANLAB_PROJECT = "finance"
SWANLAB_WORKSPACE = "roc_fu"
# 使用全新 21 位 ID，确保首段作为全新实验初始化，启用 SwanLab 官方硬件监控
SWANLAB_RUN_ID = "tfm3v8relay20261009t4"
EXPERIMENT_NAME = "timesfm3-lora-v8-relay-2xt4"

QUANTILE_INDEX = 4
CONTEXT_FEATURES = (0, 1, 2, 4, 5)  # open, high, low, volume, amount (close=3 是目标)


def gpu_metrics() -> dict[str, float]:
    import torch
    if not torch.cuda.is_available():
        return {"gpu/count": 0.0}
    res = {"gpu/count": float(torch.cuda.device_count())}
    allocated, reserved = [], []
    for i in range(torch.cuda.device_count()):
        alc = torch.cuda.memory_allocated(i) / (1024 ** 3)
        rsv = torch.cuda.memory_reserved(i) / (1024 ** 3)
        res[f"gpu/{i}_allocated_gb"] = round(alc, 3)
        allocated.append(alc)
        reserved.append(rsv)
    res["gpu/max_allocated_gb"] = round(max(allocated, default=0.0), 3)
    res["gpu/max_reserved_gb"] = round(max(reserved, default=0.0), 3)
    return res


def log(event: str, **fields) -> None:
    line = json.dumps({"time": datetime.now(timezone.utc).isoformat(), "event": event, **fields},
                      ensure_ascii=False, default=float)
    print(line, flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "run.log").open("a") as handle:
        handle.write(line + "\n")


# ----------------------------------------------------------------- 数据 (纯 numpy, 可本地测试)

def find_formal_root(root: Path) -> Path:
    hits = [p.parent for p in root.glob("**/timesfm3_formal/manifest.json")]
    if len(hits) != 1:
        raise FileNotFoundError(f"需要恰好一个 timesfm3_formal/manifest.json, 找到 {hits}")
    return hits[0]


def shard_files(formal: Path, split: str) -> list[Path]:
    manifest = json.loads((formal / "manifest.json").read_text())
    files = [formal / split / s["file"] for s in manifest["splits"][split]["artifacts"]["shards"]]
    missing = [f for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"{split} 缺少 {len(missing)} 个分片, 例如 {missing[0]}")
    return files


class ShardStream:
    """确定性分片流: 游标 (epoch, pos, offset) 可保存/恢复, 恢复后批次完全一致。"""

    def __init__(self, files: list[Path], batch_size: int, seed: int, cursor: dict | None = None):
        self.files, self.batch_size, self.seed = files, batch_size, seed
        cursor = cursor or {"epoch": 0, "pos": 0, "offset": 0}
        self.epoch, self.pos, self.offset = cursor["epoch"], cursor["pos"], cursor["offset"]
        self._loaded = None  # (epoch, pos, contexts, targets)

    def cursor(self) -> dict:
        return {"epoch": self.epoch, "pos": self.pos, "offset": self.offset}

    def _shard(self):
        key = (self.epoch, self.pos)
        if self._loaded is None or self._loaded[0] != key:
            order = np.random.default_rng([self.seed, self.epoch]).permutation(len(self.files))
            with np.load(self.files[order[self.pos]]) as data:
                contexts, targets = data["contexts"], data["targets"]
            rows = np.random.default_rng([self.seed, self.epoch, self.pos]).permutation(len(targets))
            self._loaded = (key, contexts[rows], targets[rows])
        return self._loaded[1], self._loaded[2]

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        while True:
            contexts, targets = self._shard()
            end = self.offset + self.batch_size
            if end <= len(targets):
                batch = contexts[self.offset:end], targets[self.offset:end]
                self.offset = end
                return batch
            # 余数不足一个 batch: 丢弃, 进入下一分片
            self.pos, self.offset = self.pos + 1, 0
            if self.pos >= len(self.files):
                self.epoch, self.pos = self.epoch + 1, 0


def load_val_subset(files: list[Path], samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """在全部验证分片上做固定随机抽样 (覆盖所有股票, 而非只取前几个 symbol)。"""
    contexts, targets = [], []
    for f in files:
        with np.load(f) as data:
            contexts.append(data["contexts"])
            targets.append(data["targets"])
    contexts, targets = np.concatenate(contexts), np.concatenate(targets)
    if samples and samples < len(targets):
        idx = np.sort(np.random.default_rng(seed).choice(len(targets), samples, replace=False))
        contexts, targets = contexts[idx], targets[idx]
    return contexts, targets


def find_relay_state(root: Path) -> Path | None:
    """在上一 chunk 的输出里寻找 relay/state.pt; 多个时取 global_step 最大的 (由 relay.json 记录)。"""
    best, best_step = None, -1
    for meta in root.glob("**/relay/relay.json"):
        info = json.loads(meta.read_text())
        if info.get("run_id") == SWANLAB_RUN_ID and info["global_step"] > best_step:
            best, best_step = meta.parent / "state.pt", info["global_step"]
    return best


# ----------------------------------------------------------------- 模型 (沿用 V7 的前向/损失)

def build_forward_inputs(contexts, model, horizon: int):
    import torch
    import torch.nn.functional as F

    batch, context, _ = contexts.shape
    patch_len = int(model.input_patch_len)
    ctx_padding = (patch_len - context % patch_len) % patch_len
    num_context_patches = (context + ctx_padding) // patch_len
    if getattr(model, "use_stitching", False):
        overlap = int(model._stitching_extract_len) - patch_len
        num_forecast_patches = max((horizon - overlap + patch_len - 1) // patch_len, 1)
        num_horizon_patches = num_forecast_patches + int(model.rolls) - 1
        padded_horizon = num_horizon_patches * patch_len
    else:
        out_len = int(model.output_patch_len)
        padded_horizon = (horizon + out_len - 1) // out_len * out_len
        num_horizon_patches = padded_horizon // patch_len
    close = contexts[..., 3:4].transpose(1, 2)
    covariates = contexts[..., CONTEXT_FEATURES].transpose(1, 2)
    values = F.pad(torch.cat([close, covariates], dim=1), (ctx_padding, padded_horizon), value=0.0)
    masks = torch.zeros(batch, context + ctx_padding + padded_horizon, dtype=torch.bool,
                        device=contexts.device)
    masks[:, :ctx_padding] = True
    masks[:, context + ctx_padding:] = True
    n_var = values.shape[1]
    values = values.reshape(batch, n_var, -1, patch_len)
    masks = masks[:, None, :].expand(batch, n_var, -1).reshape(batch, n_var, -1, patch_len)
    total_patches = num_context_patches + num_horizon_patches
    patch_is_target = torch.ones(batch, n_var, total_patches, dtype=torch.bool, device=contexts.device)
    cpm = torch.zeros(batch, total_patches, dtype=torch.bool, device=contexts.device)
    cpm[:, num_context_patches:] = True
    return {"values": values, "masks": masks, "patch_is_target": patch_is_target}, cpm, num_context_patches


def forecast(model, contexts, horizon: int):
    inputs, cpm, ctx_patches = build_forward_inputs(contexts, model, horizon)
    freeze = ctx_patches - 1 if getattr(model, "use_frozen_running_stats", False) else None
    logits = model(inputs, freeze_after=freeze, patch_cpm_mask=cpm)["logits"]
    return logits[:, 0, ctx_patches - 1, :, QUANTILE_INDEX][:, :horizon]


def path_loss(prediction, target, start, return_weight: float):
    import torch
    import torch.nn.functional as F

    scale = start.clamp_min(1e-3).unsqueeze(1)
    raw = F.smooth_l1_loss(prediction, target)
    ret = F.smooth_l1_loss(prediction / scale - 1.0, target / scale - 1.0)
    return raw + return_weight * ret, {
        "raw_loss": float(raw.detach().cpu()) if torch.isfinite(raw) else 0.0,
        "return_loss": float(ret.detach().cpu()) if torch.isfinite(ret) else 0.0,
    }


def attach_lora(model, rank: int, alpha: int, dropout: float):
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        def __init__(self, base: nn.Linear):
            super().__init__()
            self.base = base
            base.weight.requires_grad_(False)
            if base.bias is not None:
                base.bias.requires_grad_(False)
            kw = {"device": base.weight.device, "dtype": base.weight.dtype}
            self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, **kw))
            self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, **kw))
            nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
            self.scale, self.dropout = alpha / rank, nn.Dropout(dropout)

        def forward(self, x):
            return self.base(x) + (self.dropout(x) @ self.lora_a.T @ self.lora_b.T) * self.scale

    for p in model.parameters():
        p.requires_grad_(False)
    targets = {"query_proj", "key_proj", "value_proj", "out_proj", "ff0", "ff1"}
    for _, parent in list(model.named_modules()):
        for name, child in list(parent.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child))
    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("LoRA 没有匹配到任何 Linear 层")
    return model


def lora_state(model) -> dict:
    from torch import nn
    base = model.module if isinstance(model, nn.DataParallel) else model
    return {k: v.detach().cpu() for k, v in base.named_parameters() if v.requires_grad}


# ----------------------------------------------------------------- 训练

def evaluate(model, val, device, horizon, batch_size, return_weight) -> dict:
    import torch

    model.eval()
    contexts_all, targets_all = val
    sums, count = {"loss": 0.0, "nmae": 0.0, "nrmse_sq": 0.0, "direction_acc": 0.0}, 0
    with torch.no_grad():
        for i in range(0, len(targets_all), batch_size):
            c = torch.from_numpy(contexts_all[i:i + batch_size]).to(device)
            t = torch.from_numpy(targets_all[i:i + batch_size]).to(device)
            p = forecast(model, c, horizon)
            start = c[:, -1, 3]
            loss, _ = path_loss(p, t, start, return_weight)
            err = (p - t) / start.clamp_min(1e-6).unsqueeze(1)
            n = len(t)
            sums["loss"] += float(loss) * n
            sums["nmae"] += float(err.abs().mean()) * n
            sums["nrmse_sq"] += float(err.square().mean()) * n
            same = torch.sign(torch.diff(p, dim=1)) == torch.sign(torch.diff(t, dim=1))
            sums["direction_acc"] += float(same.float().mean()) * n
            count += n
    model.train()
    return {"loss": sums["loss"] / count, "nmae": sums["nmae"] / count,
            "nrmse": (sums["nrmse_sq"] / count) ** 0.5, "direction_acc": sums["direction_acc"] / count}


def save_state(model, optimizer, stream, global_step, best, history, args, chunk_index):
    import torch

    OUTPUT.mkdir(parents=True, exist_ok=True)
    state = {
        "run_id": SWANLAB_RUN_ID, "global_step": global_step, "best": best, "history": history,
        "lora": lora_state(model), "optimizer": optimizer.state_dict(),
        "cursor": stream.cursor(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "config": vars(args), "chunk_index": chunk_index,
    }
    tmp = OUTPUT / "state.pt.tmp"
    torch.save(state, tmp)
    tmp.replace(OUTPUT / "state.pt")
    (OUTPUT / "relay.json").write_text(json.dumps({
        "run_id": SWANLAB_RUN_ID, "global_step": global_step, "chunk_index": chunk_index,
        "cursor": stream.cursor(), "best_val_nrmse": best,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2))
    (OUTPUT / "history.json").write_text(json.dumps(history, indent=2))


def train(args) -> None:
    import swanlab
    import torch
    from torch import nn
    from timesfm3 import TimesFM3Forecaster

    LOCKED = ("model", "batch_size", "learning_rate", "weight_decay", "return_loss_weight",
              "lora_rank", "lora_alpha", "lora_dropout", "seed", "horizon")

    formal = find_formal_root(INPUT_ROOT)
    train_files, val_files = shard_files(formal, "train"), shard_files(formal, "val")
    resume_path = find_relay_state(INPUT_ROOT)
    state = torch.load(resume_path, map_location="cpu", weights_only=False) if resume_path else None
    if state:
        for key in LOCKED:  # 接力时超参必须一致, 否则曲线不可比
            if state["config"].get(key) != vars(args).get(key):
                raise RuntimeError(f"接力配置不一致: {key} {state['config'].get(key)} != {vars(args).get(key)}")
    chunk_index = state["chunk_index"] + 1 if state else 1
    log("start", chunk_index=chunk_index, resume_from=str(resume_path) if state else None,
        formal_root=str(formal), train_shards=len(train_files), val_shards=len(val_files),
        gpu_detected=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        val_mode="full" if args.val_samples == 0 else f"subset_{args.val_samples}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forecaster = TimesFM3Forecaster.from_pretrained(args.model, per_core_batch_size=args.batch_size)
    model = forecaster.model.to(device)
    model = attach_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
    if args.multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
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
        log("multi_gpu_enabled", count=torch.cuda.device_count())

    base_model = model.module if isinstance(model, nn.DataParallel) else model
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.learning_rate, weight_decay=args.weight_decay)
    stream = ShardStream(train_files, args.batch_size, args.seed, state["cursor"] if state else None)
    global_step, best, history = 0, float("inf"), []
    if state:
        missing, unexpected = base_model.load_state_dict(state["lora"], strict=False)
        if unexpected or set(state["lora"]) != set(lora_state(model)):
            raise RuntimeError("LoRA 参数与 checkpoint 不匹配")
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        global_step, best, history = state["global_step"], state["best"], state["history"]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    swanlab.login(api_key=SWANLAB_API_KEY)
    run = swanlab.init(project=SWANLAB_PROJECT, workspace=SWANLAB_WORKSPACE,
                       experiment_name=EXPERIMENT_NAME, id=SWANLAB_RUN_ID, resume="allow",
                       config={**vars(args), "trainable_params": trainable, "gpus": torch.cuda.device_count()})
    log("swanlab_ready", run_id=SWANLAB_RUN_ID, chunk_index=chunk_index, global_step=global_step)

    val = load_val_subset(val_files, args.val_samples, args.seed)
    if not state:  # 基线: 未训练的 TimesFM-3
        metrics = evaluate(model, val, device, args.horizon, args.eval_batch_size, args.return_loss_weight)
        best = metrics["nrmse"]
        history.append({"step": 0, **metrics})
        base_log = {f"val/{k}": v for k, v in metrics.items()}
        base_log.update(gpu_metrics())
        run.log(base_log, step=0)
        log("baseline", **metrics, **gpu_metrics())

    deadline = args.budget_seconds - args.reserve_seconds
    run.log({"relay/chunk_index": chunk_index}, step=max(global_step, 1))
    stop = "max_steps"
    model.train()
    tic, chunk_start_step = time.monotonic(), global_step
    while global_step < args.max_steps:
        if time.monotonic() - STARTED >= deadline:
            stop = "time_budget"
            break
        c, t = stream.next_batch()
        c, t = torch.from_numpy(c).to(device), torch.from_numpy(t).to(device)
        pred = forecast(model, c, args.horizon)
        loss, parts = path_loss(pred, t, c[:, -1, 3], args.return_loss_weight)
        if not torch.isfinite(loss):
            log("warning_skip_non_finite_loss", step=global_step + 1)
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                              args.max_grad_norm)
        if not torch.isfinite(grad):
            log("warning_skip_non_finite_grad", step=global_step + 1)
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()
        global_step += 1

        if global_step % args.log_every == 0:
            rate = (global_step - chunk_start_step) / max(time.monotonic() - tic, 1e-6)
            row = {"train/loss": float(loss), "train/grad_norm": float(grad),
                   **{f"train/{k}": v for k, v in parts.items()},
                   "train/steps_per_sec": rate, "data/epoch": stream.epoch,
                   "data/shard_pos": stream.pos, "relay/chunk_index": chunk_index,
                   **gpu_metrics()}
            run.log(row, step=global_step)
            log("train", step=global_step, **row)
        if global_step % args.eval_every == 0:
            metrics = evaluate(model, val, device, args.horizon, args.eval_batch_size, args.return_loss_weight)
            history.append({"step": global_step, **metrics})
            eval_row = {f"val/{k}": v for k, v in metrics.items()}
            eval_row.update(gpu_metrics())
            run.log(eval_row, step=global_step)
            log("val", step=global_step, **metrics, **gpu_metrics())
            if metrics["nrmse"] < best:
                best = metrics["nrmse"]
                torch.save(lora_state(model), OUTPUT / "best_lora.pt")
                (OUTPUT / "best_metrics.json").write_text(json.dumps(history[-1], indent=2))
        if global_step % args.save_every == 0:
            save_state(model, optimizer, stream, global_step, best, history, args, chunk_index)
            log("saved", step=global_step, cursor=stream.cursor())

    save_state(model, optimizer, stream, global_step, best, history, args, chunk_index)
    torch.save(lora_state(model), OUTPUT / "last_lora.pt")
    summary = {"chunk_index": chunk_index, "global_step": global_step, "stop": stop,
               "best_val_nrmse": best, "cursor": stream.cursor(),
               "next": "push 下一个 chunk 并以本 kernel 为 kernel_source" if stop == "time_budget" else "done"}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
    run.log({"relay/last_step_of_chunk": global_step, **gpu_metrics()}, step=global_step)
    log("chunk_done", **summary)
    swanlab.finish()


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="google/timesfm-3.0-pytorch")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--multi-gpu", action="store_true", default=True)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--return-loss-weight", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--val-samples", type=int, default=0)  # 0 为全量验证 (123,836 条)
    p.add_argument("--max-steps", type=int, default=200000)
    p.add_argument("--budget-seconds", type=int, default=1800)
    p.add_argument("--reserve-seconds", type=int, default=300)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=200)
    p.set_defaults(**CHUNK_CONFIG)  # push_chunk.py 注入的本 chunk 参数
    return p.parse_args(argv)


CHUNK_CONFIG: dict = {}  # 由 push_chunk.py 在打包时替换, 本行勿改格式


if __name__ == "__main__":
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "git+https://github.com/google-research/timesfm.git", "swanlab==0.10.1"])
    train(parse_args())
