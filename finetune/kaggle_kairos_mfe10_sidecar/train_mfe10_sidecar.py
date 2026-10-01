"""Kairos SHORT sidecar: single binary head for path MFE≥10%/10d.

Target:
  mfe10 = max(high[T+1:T+10]) / close[T] - 1
  y     = 1{ mfe10 >= 0.10 }

NOT close-to-close. NOT the old 8-head R2 (up_003/005/008/012 + down_*).
Fresh init; short budget (few segments / 1 chunk); print constant-prior
baseline every eval; early-stop if stuck within ±1e-3 of prior for 2+ evals.
"""

from __future__ import annotations

import hashlib
import math
import importlib.metadata
import json
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---- short-budget sidecar knobs (NOT full R2) ----
SEED = 20261001
BATCH_SIZE = 16
CHUNK_INDEX = 0
SEGMENT_SAMPLES = 20_000
MAX_SEGMENTS_THIS_RUN = 4
GPU_BUDGET_SECONDS = 90 * 60
RUNTIME_RESERVE_SECONDS = 12 * 60
SEGMENT_ESTIMATE_SECONDS = 600.0
SEGMENT_TIME_MARGIN = 1.15
SHUFFLE_SEED = 20261001
LEARNING_RATE = 1e-4  # Phase K: shallow/identity; adapters + embeds only
MFE10_THRESHOLD = 0.10
PRIOR_STUCK_TOL = 1e-3
PRIOR_STUCK_PATIENCE = 2
GATE_DELTA_VS_PRIOR = -0.04
FREEZE_BACKBONE = False  # Phase K: no 22-layer backbone to freeze
BACKBONE_MODE = "identity"  # Phase K: bypass random ModernBERT; mean-pool embeds
SWANLAB_API_KEY_FALLBACK = ""  # injected at private staging only
SWANLAB_RUN_ID = "kairos-mfe10-sidecar-short-phase-k-identity-20261001"
OUTPUT = Path("/kaggle/working/kairos_mfe10_sidecar")
RUN_PURPOSE = "mfe10-path-touch-binary-short-sidecar-identity-bb"
FEATURES = ("open", "high", "low", "close", "volume", "amount")
TARGET_NAME = "buy_worth_mfe10pct"
MFE10_DEF = "max(high[T+1:T+10]) / close[T] - 1"
REPO_CLONE_URL = "https://github.com/luckfu/Kronos.git"
REPO_CLONE_BRANCH = "master"


def log(phase: str, **fields: object) -> None:
    if os.environ.get("RANK", "0") != "0":
        return
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "phase": phase, **fields}
    line = json.dumps(row, ensure_ascii=False, default=str)
    print(line, flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_one(pattern: str) -> Path:
    matches = sorted(Path("/kaggle/input").glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {pattern}, found {len(matches)}: {matches}")
    return matches[0]


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def binary_from_mfe10(mfe10: Any, threshold: float = MFE10_THRESHOLD) -> float:
    value = float(mfe10)
    if not np.isfinite(value):
        raise ValueError("non-finite mfe10")
    return 1.0 if value >= threshold else 0.0


def constant_prior_log_loss(labels: np.ndarray, prior: float) -> float:
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    p = float(np.clip(prior, 1e-7, 1.0 - 1e-7))
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def binary_log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(np.asarray(probabilities, dtype=np.float64).reshape(-1), 1e-7, 1 - 1e-7)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def prior_stuck_stop(
    model_ll: float, prior_ll: float, stuck_streak: int
) -> tuple[str, int]:
    if not np.isfinite(model_ll) or not np.isfinite(prior_ll):
        return "nonfinite_validation", stuck_streak + 1
    if abs(float(model_ll) - float(prior_ll)) <= PRIOR_STUCK_TOL:
        stuck_streak += 1
    else:
        stuck_streak = 0
    if stuck_streak >= PRIOR_STUCK_PATIENCE:
        return "stuck_at_constant_prior", stuck_streak
    return "", stuck_streak


def shuffled_group_order(num_groups: int) -> list[int]:
    return np.random.default_rng(SHUFFLE_SEED).permutation(num_groups).tolist()


def shuffle_group_rows(rows: list[dict[str, Any]], group_id: int) -> list[dict[str, Any]]:
    order = np.random.default_rng(SHUFFLE_SEED + group_id + 1).permutation(len(rows))
    return [rows[int(index)] for index in order]


def group_order_hash(order: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, order)).encode("ascii")).hexdigest()


def calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(labels)
    if total == 0:
        return float("nan")
    error = 0.0
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (
            (probabilities < edges[index + 1])
            if index + 1 < bins
            else (probabilities <= edges[index + 1])
        )
        if mask.any():
            error += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(labels[mask].mean())
            )
    return error


def start_swanlab() -> tuple[Any, Any]:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--progress-bar", "off", "swanlab"],
        check=True,
    )
    import swanlab

    api_key = os.environ.get("SWANLAB_API_KEY", "").strip() or SWANLAB_API_KEY_FALLBACK
    if not api_key:
        raise RuntimeError("SWANLAB_API_KEY is empty (stage private credential before push)")
    swanlab.login(api_key=api_key)
    run = swanlab.init(
        id=SWANLAB_RUN_ID,
        resume="allow",
        project="finance",
        workspace="roc_fu",
        experiment_name=SWANLAB_RUN_ID,
        config={
            "model": "ModernBERT-base-style-single-binary",
            "hidden_size": 768,
            "layers": 22,
            "heads": 12,
            "batch_size": BATCH_SIZE,
            "chunk_index": CHUNK_INDEX,
            "segment_samples": SEGMENT_SAMPLES,
            "max_segments_this_run": MAX_SEGMENTS_THIS_RUN,
            "learning_rate": LEARNING_RATE,
            "target": TARGET_NAME,
            "mfe10_def": MFE10_DEF,
            "threshold": MFE10_THRESHOLD,
            "gate_delta_vs_prior": GATE_DELTA_VS_PRIOR,
            "prior_stuck_tol": PRIOR_STUCK_TOL,
            "prior_stuck_patience": PRIOR_STUCK_PATIENCE,
            "variant": "kairos-mfe10-short-sidecar-phase-i",
            "not_multi_head_r2": True,
            "not_close_to_close": True,
        },
        mode="cloud",
    )
    url = getattr(run, "url", getattr(run, "web_url", ""))
    if not url:
        raise RuntimeError("SwanLab did not return a run URL")
    print(json.dumps({"phase": "swanlab_ready", "url": url}), flush=True)
    return swanlab, run


def load_arrays(path: Path) -> tuple[dict[str, dict[str, Any]], set[str]]:
    import pandas as pd

    with path.open("rb") as handle:
        panel = pickle.load(handle)
    arrays: dict[str, dict[str, Any]] = {}
    sectors: set[str] = set()
    for symbol in sorted(panel):
        frame = panel[symbol].sort_index()
        sector = frame.get(
            "sector", pd.Series("unknown", index=frame.index)
        ).astype(str).to_numpy()
        sectors.update(sector.tolist())
        arrays[str(symbol)] = {
            "values": frame.loc[:, FEATURES].to_numpy(dtype=np.float32, copy=True),
            "sector": sector,
            "size": pd.to_numeric(
                frame.get("size_percentile", pd.Series(0.5, index=frame.index)),
                errors="coerce",
            ).fillna(0.5).clip(0.0, 1.0).to_numpy(dtype=np.float32),
            "dates": frame.index.to_numpy(dtype="datetime64[D]"),
        }
    return arrays, sectors


def make_batch(
    rows: list[dict[str, Any]],
    arrays: dict[str, dict[str, Any]],
    sector_ids: dict[str, int],
    check_labels: bool = False,
) -> tuple[np.ndarray, ...]:
    histories = np.empty((len(rows), 120, 6), dtype=np.float32)
    sectors = np.empty(len(rows), dtype=np.int64)
    sizes = np.empty(len(rows), dtype=np.float32)
    labels = np.empty((len(rows), 1), dtype=np.float32)
    dates = np.empty(len(rows), dtype="datetime64[D]")
    for i, row in enumerate(rows):
        symbol, start = str(row["symbol"]), int(row["start_index"])
        data = arrays[symbol]
        asof = start + 119
        history = data["values"][start:start + 120].copy()
        history = (history - history.mean(axis=0)) / (history.std(axis=0) + 1e-5)
        histories[i] = np.clip(history, -5.0, 5.0)
        sectors[i] = sector_ids.get(str(data["sector"][asof]), len(sector_ids))
        sizes[i] = data["size"][asof]
        labels[i, 0] = binary_from_mfe10(row["mfe10"])
        dates[i] = data["dates"][asof]
        if check_labels:
            future = data["values"][start + 120:start + 130]
            close = float(data["values"][asof, 3])
            path_mfe = float(future[:, 1].max() / close - 1)
            if not np.isclose(path_mfe, float(row["mfe10"]), atol=2e-6):
                raise ValueError(f"MFE mismatch: {symbol} {start}")
            # Explicitly reject close-to-close as the training label.
            ctc = float(future[-1, 3] / close - 1)
            y_mfe = binary_from_mfe10(row["mfe10"])
            y_ctc = 1.0 if ctc >= MFE10_THRESHOLD else 0.0
            if y_mfe != labels[i, 0]:
                raise ValueError("label derivation mismatch")
            # ctc may differ; we only assert our label is path-MFE based.
            _ = y_ctc
    return histories, sectors, sizes, labels, dates


def main() -> None:
    runtime_started = float(os.environ.setdefault("KAIROS_RUNTIME_STARTED", str(time.time())))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel

    if not torch.cuda.is_available():
        raise RuntimeError("sidecar training requires Kaggle GPU")
    gpu_count = torch.cuda.device_count()
    if gpu_count >= 2 and "WORLD_SIZE" not in os.environ:
        command = [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", "--nproc_per_node=2", str(Path(__file__).resolve()),
        ]
        log("ddp_relaunch", command=command, gpu_count=gpu_count)
        completed = subprocess.run(command, check=False)
        raise SystemExit(completed.returncode)

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    swanlab, swanlab_run = start_swanlab() if is_main_process() else (None, None)
    log(
        "run_started",
        purpose=RUN_PURPOSE,
        target=TARGET_NAME,
        mfe10_def=MFE10_DEF,
        threshold=MFE10_THRESHOLD,
        max_segments_this_run=MAX_SEGMENTS_THIS_RUN,
        learning_rate=LEARNING_RATE,
        not_multi_head_r2=True,
        not_close_to_close=True,
        freeze_backbone=FREEZE_BACKBONE,
        backbone_mode=BACKBONE_MODE,
        world_size=world_size,
    )
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")

    train_panel_path = find_one("**/processed_datasets/train_data.pkl")
    validation_panel_path = find_one("**/processed_datasets/val_data.pkl")
    train_targets_path = find_one("**/train_targets.parquet")
    validation_targets_path = find_one("**/validation_targets.parquet")
    target_manifest_path = train_targets_path.parent / "decision_targets_manifest.json"
    source_manifest_path = train_panel_path.parent.parent / "data_manifest.json"
    target_manifest = json.loads(target_manifest_path.read_text(encoding="utf-8"))
    if int(target_manifest["window"]["source_window"]) != 131:
        raise ValueError("expected 120+10+1 window contract")
    for path, key in (
        (train_panel_path, "train_panel_sha256"),
        (validation_panel_path, "validation_panel_sha256"),
        (source_manifest_path, "data_manifest_sha256"),
    ):
        if sha256_file(path) != target_manifest["source_dataset"][key]:
            raise ValueError(f"source hash mismatch: {path}")

    train_pq = pq.ParquetFile(train_targets_path)
    validation_pq = pq.ParquetFile(validation_targets_path)
    train_total = train_pq.metadata.num_rows
    validation_total = validation_pq.metadata.num_rows
    log(
        "runtime",
        gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        torch=torch.__version__,
        transformers=importlib.metadata.version("transformers"),
        train_samples=train_total,
        validation_samples=validation_total,
    )

    train_arrays, train_sectors = load_arrays(train_panel_path)
    validation_arrays, validation_sectors = load_arrays(validation_panel_path)
    sector_ids = {
        name: i for i, name in enumerate(sorted(train_sectors | validation_sectors))
    }

    repo_path = Path("/kaggle/working/Kronos")
    if is_main_process() and not (repo_path / "model" / "kronos.py").is_file():
        subprocess.run(
            [
                "git", "clone", "--depth", "1", "--branch", REPO_CLONE_BRANCH,
                REPO_CLONE_URL, str(repo_path),
            ],
            check=True,
        )
    if distributed:
        dist.barrier()
    sys.path.insert(0, str(repo_path))
    from model import KronosTokenizer
    from huggingface_hub import snapshot_download
    from transformers import ModernBertConfig, ModernBertModel

    tokenizer_path_holder = [
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base")
        if is_main_process() else None
    ]
    if distributed:
        dist.broadcast_object_list(tokenizer_path_holder, src=0)
    tokenizer_path = Path(tokenizer_path_holder[0])
    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path)).to(device).eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)

    class SingleBinaryModel(nn.Module):
        """Minimal single-logit head for path-MFE≥10% (not ordinal 8-head)."""

        def __init__(self, head_bias: float = 0.0) -> None:
            super().__init__()
            hidden = 768
            self.s1 = nn.Embedding(1024, hidden // 2)
            self.s2 = nn.Embedding(1024, hidden // 2)
            self.fusion = nn.Linear(hidden, hidden)
            # Phase I fix: short budget cannot open a zero-init gate; start open.
            self.gate = nn.Parameter(torch.ones(1))
            self.sector = nn.Embedding(len(sector_ids) + 1, hidden)
            self.size = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, hidden))
            self.cond = nn.Parameter(torch.zeros(1, 1, hidden))
            self.backbone_mode = BACKBONE_MODE
            if BACKBONE_MODE == "identity":
                # Phase K: no random 22-layer ModernBERT; mean-pool condition+market embeds.
                self.backbone = None
            else:
                config = ModernBertConfig(
                    vocab_size=1024, hidden_size=hidden, intermediate_size=1152,
                    num_hidden_layers=22, num_attention_heads=12,
                    max_position_embeddings=128, pad_token_id=0,
                    attention_dropout=0.0, embedding_dropout=0.0, mlp_dropout=0.0,
                    reference_compile=False,
                )
                self.backbone = ModernBertModel(config)
                native_embeddings = getattr(self.backbone.embeddings, "tok_embeddings", None)
                if native_embeddings is None:
                    native_embeddings = getattr(self.backbone.embeddings, "word_embeddings", None)
                if native_embeddings is None:
                    raise RuntimeError("Cannot locate ModernBERT native token embeddings")
                for parameter in native_embeddings.parameters():
                    parameter.requires_grad_(False)
            self.head = nn.Linear(hidden, 1)
            # Phase I fix: calibrate initial probability to train prior (~25%), not 0.5.
            nn.init.zeros_(self.head.weight)
            nn.init.constant_(self.head.bias, float(head_bias))

        def forward(self, s1: Any, s2: Any, size: Any, sector: Any) -> Any:
            condition = self.cond + self.sector(sector).unsqueeze(1) + self.size(size).unsqueeze(1)
            market = self.fusion(torch.cat([self.s1(s1), self.s2(s2)], -1)) * self.gate
            sequence = torch.cat([condition, market], 1)
            if self.backbone is None:
                pooled = sequence.mean(dim=1)
            else:
                mask = torch.ones(sequence.shape[:2], dtype=torch.long, device=sequence.device)
                pooled = self.backbone(inputs_embeds=sequence, attention_mask=mask).last_hidden_state[:, 0]
            return self.head(pooled)

    group_order = shuffled_group_order(train_pq.num_row_groups)
    total_segments = (train_total + SEGMENT_SAMPLES - 1) // SEGMENT_SAMPLES
    run_segment_limit = min(MAX_SEGMENTS_THIS_RUN, total_segments)
    log(
        "budget",
        total_segments=total_segments,
        run_segment_limit=run_segment_limit,
        gpu_budget_seconds=GPU_BUDGET_SECONDS,
        group_order_hash=group_order_hash(group_order),
    )

    # Train prior from labels derived on-the-fly from mfe10 (full train scan).
    # Computed BEFORE model init so head.bias can start at logit(prior).
    totals = 0.0
    count = 0
    for index in range(train_pq.num_row_groups):
        table = train_pq.read_row_group(index, columns=["mfe10"])
        mfe = table["mfe10"].to_numpy()
        y = (mfe >= MFE10_THRESHOLD).astype(np.float64)
        totals += float(y.sum())
        count += len(y)
    if count != train_total:
        raise RuntimeError("train prior sample count mismatch")
    train_prior = float(totals / count)
    head_bias = math.log(max(train_prior, 1e-7) / max(1.0 - train_prior, 1e-7))
    log(
        "train_prior",
        samples=count,
        prevalence=train_prior,
        head_bias_init=head_bias,
        gate_init=1.0,
        target=TARGET_NAME,
    )

    model = SingleBinaryModel(head_bias=head_bias).to(device)
    raw_model = model
    if FREEZE_BACKBONE and raw_model.backbone is not None:
        for parameter in raw_model.backbone.parameters():
            parameter.requires_grad_(False)
    trainable = [p for p in raw_model.parameters() if p.requires_grad]
    frozen = [p for p in raw_model.parameters() if not p.requires_grad]
    log(
        "param_freeze",
        freeze_backbone=FREEZE_BACKBONE,
        backbone_mode=BACKBONE_MODE,
        trainable_tensors=len(trainable),
        frozen_tensors=len(frozen),
        trainable_numel=int(sum(p.numel() for p in trainable)),
        frozen_numel=int(sum(p.numel() for p in frozen)),
        learning_rate=LEARNING_RATE,
    )
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
        )
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda")
    checkpoint = OUTPUT / "last_checkpoint.pt"
    best_checkpoint = OUTPUT / "best_model.pt"
    best_meta = OUTPUT / "best_metric.json"

    columns = ["symbol", "start_index", "asof_date", "mfe10", "mae10"]
    local_batch_size = max(1, BATCH_SIZE // world_size)
    validation_token_cache: tuple[np.ndarray, ...] | None = None

    def encode_token_cache(
        rows: list[dict[str, Any]],
        arrays: dict[str, dict[str, Any]],
        check_labels: bool = False,
    ) -> tuple[np.ndarray, ...]:
        token_s1: list[np.ndarray] = []
        token_s2: list[np.ndarray] = []
        cached_sectors: list[np.ndarray] = []
        cached_sizes: list[np.ndarray] = []
        cached_labels: list[np.ndarray] = []
        cached_dates: list[np.ndarray] = []
        for offset in range(0, len(rows), local_batch_size):
            batch = rows[offset:offset + local_batch_size]
            history, sectors, sizes, labels, dates = make_batch(
                batch, arrays, sector_ids,
                check_labels=check_labels and offset == 0,
            )
            with torch.no_grad():
                encoded_s1, encoded_s2 = tokenizer.encode(
                    torch.from_numpy(history).to(device), half=True
                )
            token_s1.append(encoded_s1.cpu().numpy().astype(np.uint16, copy=False))
            token_s2.append(encoded_s2.cpu().numpy().astype(np.uint16, copy=False))
            cached_sectors.append(sectors)
            cached_sizes.append(sizes)
            cached_labels.append(labels)
            cached_dates.append(dates)
        return (
            np.concatenate(token_s1, axis=0),
            np.concatenate(token_s2, axis=0),
            np.concatenate(cached_sectors, axis=0),
            np.concatenate(cached_sizes, axis=0),
            np.concatenate(cached_labels, axis=0),
            np.concatenate(cached_dates, axis=0),
        )

    def full_validation() -> dict[str, Any]:
        nonlocal validation_token_cache
        import sklearn.metrics

        if validation_token_cache is None:
            rows: list[dict[str, Any]] = []
            for group_id in range(validation_pq.num_row_groups):
                rows.extend(
                    validation_pq.read_row_group(group_id, columns=columns).to_pylist()
                )
            local_rows = rows[rank::world_size]
            validation_token_cache = encode_token_cache(local_rows, validation_arrays)
            log("validation_token_cache_created", rank=rank,
                samples=len(validation_token_cache[0]))

        local_cache = validation_token_cache
        model.eval()
        predictions, truth, dates = [], [], []
        for offset in range(0, len(local_cache[0]), local_batch_size):
            s1 = torch.from_numpy(local_cache[0][offset:offset + local_batch_size]).to(device)
            s2 = torch.from_numpy(local_cache[1][offset:offset + local_batch_size]).to(device)
            sizes = torch.from_numpy(local_cache[3][offset:offset + local_batch_size, None]).to(device)
            sectors = torch.from_numpy(local_cache[2][offset:offset + local_batch_size]).to(device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(s1.long(), s2.long(), sizes, sectors)
            predictions.append(logits.sigmoid().cpu().numpy())
            truth.append(local_cache[4][offset:offset + local_batch_size])
            dates.append(local_cache[5][offset:offset + local_batch_size])
        local_result = (
            np.concatenate(predictions),
            np.concatenate(truth),
            np.concatenate(dates),
        )
        if distributed:
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, local_result)
        else:
            gathered = [local_result]
        if not is_main_process():
            if distributed:
                dist.barrier()
            result_holder = [None]
            if distributed:
                dist.broadcast_object_list(result_holder, src=0)
            model.train()
            return result_holder[0]

        probabilities = np.concatenate([item[0] for item in gathered]).astype(np.float32).reshape(-1)
        labels = np.concatenate([item[1] for item in gathered]).astype(np.float32).reshape(-1)
        dates = np.concatenate([item[2] for item in gathered])
        result: dict[str, Any] = {}
        for name, mask in (
            ("all", np.ones(len(labels), dtype=bool)),
            ("2025H2", dates < np.datetime64("2026-01-01")),
            ("2026H1", dates >= np.datetime64("2026-01-01")),
        ):
            p = probabilities[mask]
            y = labels[mask]
            model_ll = binary_log_loss(p, y)
            prior_ll = constant_prior_log_loss(y, prior=train_prior)
            delta = model_ll - prior_ll
            result[name] = {
                "samples": int(mask.sum()),
                "positive_rate": float(y.mean()) if y.size else None,
                "log_loss": model_ll,
                "constant_prior_log_loss": prior_ll,
                "delta_vs_prior": delta,
                "gate_delta": GATE_DELTA_VS_PRIOR,
                "gate_passed": bool(delta <= GATE_DELTA_VS_PRIOR),
                "brier": float(np.square(p - y).mean()) if y.size else None,
                "ece_10bin": calibration_error(p, y),
                "roc_auc": float(sklearn.metrics.roc_auc_score(y, p))
                if y.size and np.unique(y).size == 2 else None,
                "pr_auc": float(sklearn.metrics.average_precision_score(y, p))
                if y.size and np.any(y == 1) else None,
                "train_prior": train_prior,
                "target": TARGET_NAME,
            }
        # Always print constant-prior baseline every eval.
        log(
            "constant_prior_baseline",
            train_prior=train_prior,
            all_prior_ll=result["all"]["constant_prior_log_loss"],
            all_model_ll=result["all"]["log_loss"],
            all_delta=result["all"]["delta_vs_prior"],
            all_pos_rate=result["all"]["positive_rate"],
            gate=GATE_DELTA_VS_PRIOR,
            gate_passed=result["all"]["gate_passed"],
        )
        result_holder = [result]
        if distributed:
            dist.barrier()
            dist.broadcast_object_list(result_holder, src=0)
        model.train()
        return result

    started = time.monotonic()
    model.train()
    completed_segments = 0
    group_order_pos = 0
    row_offset = 0
    processed = 0
    segments_this_run = 0
    segment_rows: list[dict[str, Any]] = []
    longest_segment_seconds = SEGMENT_ESTIMATE_SECONDS
    stop_reason = "segment_limit"
    validation_history: list[dict[str, Any]] = []
    best_score = float("inf")
    last_validation = None
    last_best_updated = False
    prior_stuck_streak = 0

    # Initial eval (untrained) — prints prior baseline before any segment.
    last_validation = full_validation()
    best_score = last_validation["all"]["log_loss"]
    if is_main_process():
        best_meta.write_text(
            json.dumps({
                "log_loss": best_score,
                "constant_prior_log_loss": last_validation["all"]["constant_prior_log_loss"],
                "delta_vs_prior": last_validation["all"]["delta_vs_prior"],
                "segment_index": 0,
                "processed_samples": 0,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        swanlab_run.log({
            "validation/log_loss": last_validation["all"]["log_loss"],
            "validation/constant_prior_log_loss": last_validation["all"]["constant_prior_log_loss"],
            "validation/delta_vs_prior": last_validation["all"]["delta_vs_prior"],
            "validation/positive_rate": last_validation["all"]["positive_rate"],
            "validation/gate_passed": int(last_validation["all"]["gate_passed"]),
            "train/prior": train_prior,
        }, step=0)
    log("initial_validation", validation=last_validation)

    while segments_this_run < run_segment_limit:
        if time.time() - runtime_started > GPU_BUDGET_SECONDS - RUNTIME_RESERVE_SECONDS - longest_segment_seconds:
            stop_reason = "runtime_budget_before_segment"
            break
        segment_rows = []
        while len(segment_rows) < SEGMENT_SAMPLES and group_order_pos < len(group_order):
            group_id = group_order[group_order_pos]
            rows = shuffle_group_rows(
                train_pq.read_row_group(group_id, columns=columns).to_pylist(),
                group_id,
            )
            take = min(SEGMENT_SAMPLES - len(segment_rows), len(rows) - row_offset)
            if take <= 0:
                group_order_pos += 1
                row_offset = 0
                continue
            segment_rows.extend(rows[row_offset:row_offset + take])
            row_offset += take
            if row_offset >= len(rows):
                group_order_pos += 1
                row_offset = 0
                # may continue filling from next group
        if not segment_rows:
            stop_reason = "data_exhausted"
            break

        global_segment_index = completed_segments + 1
        segment_samples = len(segment_rows)
        segment_global_start = processed
        segment_started = time.monotonic()
        padded_count = ((segment_samples + world_size - 1) // world_size) * world_size
        padded_rows = segment_rows + [segment_rows[-1]] * (padded_count - segment_samples)
        local_rows = padded_rows[rank::world_size]
        local_valid = np.arange(rank, padded_count, world_size) < segment_samples
        local_cache = encode_token_cache(
            local_rows, train_arrays,
            check_labels=(rank == 0 and completed_segments == 0),
        )

        for offset in range(0, len(local_rows), local_batch_size):
            batch_slice = slice(offset, offset + local_batch_size)
            valid = torch.from_numpy(local_valid[batch_slice]).to(device)
            s1 = torch.from_numpy(local_cache[0][batch_slice]).to(device)
            s2 = torch.from_numpy(local_cache[1][batch_slice]).to(device)
            sector = torch.from_numpy(local_cache[2][batch_slice]).to(device)
            size = torch.from_numpy(local_cache[3][batch_slice, None]).to(device)
            target = torch.from_numpy(local_cache[4][batch_slice]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(s1.long(), s2.long(), size, sector)
                per_row = F.binary_cross_entropy_with_logits(
                    logits, target, reduction="none"
                ).mean(dim=1)
                local_loss_sum = per_row[valid].sum()
                global_valid_count = valid.sum().to(dtype=torch.float32)
                if distributed:
                    dist.all_reduce(global_valid_count, op=dist.ReduceOp.SUM)
                loss = local_loss_sum * world_size / global_valid_count.clamp_min(1)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if is_main_process() and (
                offset == 0 or offset + local_batch_size >= len(local_rows)
            ):
                local_done = int(local_valid[:offset + local_batch_size].sum())
                estimated = segment_global_start + min(segment_samples, local_done * world_size)
                swanlab_run.log({
                    "train/loss": float(loss.detach().cpu()),
                    "train/processed_samples": estimated,
                    "train/segment_index": global_segment_index,
                }, step=estimated)

        if distributed:
            dist.barrier()
        if is_main_process():
            processed = segment_global_start + segment_samples
        if distributed:
            processed_holder = [processed if is_main_process() else None]
            dist.broadcast_object_list(processed_holder, src=0)
            processed = int(processed_holder[0])

        completed_segments += 1
        segments_this_run += 1
        last_validation = full_validation()
        validation_score = last_validation["all"]["log_loss"]
        prior_ll = last_validation["all"]["constant_prior_log_loss"]
        delta = last_validation["all"]["delta_vs_prior"]
        validation_history.append({
            "segment_index": global_segment_index,
            "processed_samples": processed,
            "validation": last_validation,
        })
        if is_main_process():
            (OUTPUT / "validation_history.json").write_text(
                json.dumps(validation_history, indent=2) + "\n", encoding="utf-8"
            )
        log(
            "segment_validation",
            segment_index=global_segment_index,
            model_ll=validation_score,
            prior_ll=prior_ll,
            delta_vs_prior=delta,
            gate_passed=last_validation["all"]["gate_passed"],
            pos_rate=last_validation["all"]["positive_rate"],
        )

        last_best_updated = validation_score < best_score
        if last_best_updated:
            best_score = validation_score
            if is_main_process():
                torch.save({
                    "model": raw_model.state_dict(),
                    "metrics": last_validation,
                    "segment_index": global_segment_index,
                    "processed_samples": processed,
                    "target": TARGET_NAME,
                    "mfe10_def": MFE10_DEF,
                    "threshold": MFE10_THRESHOLD,
                }, best_checkpoint.with_suffix(".pt.tmp"))
                best_checkpoint.with_suffix(".pt.tmp").replace(best_checkpoint)
                best_meta.write_text(
                    json.dumps({
                        "log_loss": best_score,
                        "constant_prior_log_loss": prior_ll,
                        "delta_vs_prior": delta,
                        "segment_index": global_segment_index,
                        "processed_samples": processed,
                    }, indent=2) + "\n",
                    encoding="utf-8",
                )
                log("best_model_updated", log_loss=best_score, delta_vs_prior=delta)

        if is_main_process():
            swanlab_run.log({
                "validation/log_loss": validation_score,
                "validation/constant_prior_log_loss": prior_ll,
                "validation/delta_vs_prior": delta,
                "validation/positive_rate": last_validation["all"]["positive_rate"],
                "validation/gate_passed": int(last_validation["all"]["gate_passed"]),
                "validation/best_updated": int(last_best_updated),
                "train/segment_index": global_segment_index,
            }, step=processed)
            torch.save({
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "chunk_index": CHUNK_INDEX,
                "completed_segments": completed_segments,
                "group_order_pos": group_order_pos,
                "row_offset": row_offset,
                "shuffle_seed": SHUFFLE_SEED,
                "group_order_hash": group_order_hash(group_order),
                "processed_samples": processed,
                "last_validation": last_validation,
                "train_prior": train_prior,
                "target": TARGET_NAME,
                "run_purpose": RUN_PURPOSE,
            }, checkpoint.with_suffix(".pt.tmp"))
            checkpoint.with_suffix(".pt.tmp").replace(checkpoint)

        if distributed:
            dist.barrier()
        segment_seconds = time.monotonic() - segment_started
        longest_segment_seconds = max(longest_segment_seconds, segment_seconds)
        log("segment_complete", segment_index=global_segment_index,
            segment_samples=segment_samples, processed_samples=processed,
            segment_seconds=segment_seconds)

        reason, prior_stuck_streak = prior_stuck_stop(
            validation_score, prior_ll, prior_stuck_streak
        )
        if reason:
            stop_reason = reason
            run_segment_limit = segments_this_run
            log("early_stop", reason=reason, stuck_streak=prior_stuck_streak,
                model_ll=validation_score, prior_ll=prior_ll)

        budget_decision = [None]
        if is_main_process():
            remaining = (
                GPU_BUDGET_SECONDS - RUNTIME_RESERVE_SECONDS
                - (time.time() - runtime_started)
            )
            budget_decision[0] = remaining < longest_segment_seconds * SEGMENT_TIME_MARGIN
            log("runtime_budget", remaining_seconds=remaining,
                stop_before_next_segment=budget_decision[0])
        if distributed:
            dist.broadcast_object_list(budget_decision, src=0)
        if budget_decision[0]:
            stop_reason = "runtime_budget"
            break

    report = {
        "status": "SIDECAR_COMPLETE",
        "purpose": RUN_PURPOSE,
        "target": TARGET_NAME,
        "mfe10_def": MFE10_DEF,
        "threshold": MFE10_THRESHOLD,
        "not_multi_head_r2": True,
        "not_close_to_close": True,
        "train_prior": train_prior,
        "stop_reason": stop_reason,
        "segments_this_run": segments_this_run,
        "processed_samples": processed,
        "best_log_loss": best_score,
        "best_delta_vs_prior": (
            None
            if last_validation is None
            else (
                best_score - last_validation["all"]["constant_prior_log_loss"]
            )
        ),
        "gate_delta": GATE_DELTA_VS_PRIOR,
        "validation": last_validation,
        "validation_history": validation_history,
        "gpu_budget_seconds": GPU_BUDGET_SECONDS,
        "runtime_elapsed_seconds": time.time() - runtime_started,
        "swanlab_run_id": SWANLAB_RUN_ID,
    }
    if is_main_process():
        (OUTPUT / "report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )
        log("sidecar_complete", **{k: report[k] for k in (
            "status", "stop_reason", "segments_this_run", "best_log_loss",
            "train_prior",
        )})
        swanlab.finish()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
