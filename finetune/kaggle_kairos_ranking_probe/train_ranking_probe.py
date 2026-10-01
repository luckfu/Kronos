"""Kairos SHORT ranking probe (Phase O): freeze tok embeds + shallow rank head.

Target:
  continuous path mfe10 = max(high[T+1:T+10]) / close[T] - 1
  (optional train label = within-day CS percentile rank of mfe10)

Phase O: freeze s1/s2 tokenizer embeds; train shallow (2-layer) non-identity
ModernBERT + rank head with MSE. Rank IC / TopK primary metrics.
Compares to Phase M identity MSE Rank IC≈0.085 / Phase N pairwise≈0.082.

NOT binary mfe≥10%. NOT 22-layer R2 / binary. No TPU WIP.
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

# ---- short-budget ranking probe knobs (NOT full R2 / NOT binary) ----
SEED = 20261001
BATCH_SIZE = 64  # denser same-day pairs after date sort
CHUNK_INDEX = 0
SEGMENT_SAMPLES = 20_000
MAX_SEGMENTS_THIS_RUN = 3  # short ranking probe
GPU_BUDGET_SECONDS = 60 * 60
RUNTIME_RESERVE_SECONDS = 10 * 60
SEGMENT_ESTIMATE_SECONDS = 600.0
SEGMENT_TIME_MARGIN = 1.15
SHUFFLE_SEED = 20261001
LEARNING_RATE = 1e-4
# Train label: continuous mfe10, or within-day CS percentile of mfe10.
TARGET_MODE = "mfe10_continuous"  # or "mfe_cs_rank"
# Phase O: MSE for A/B vs Phase M identity; pairwise/listwise still available.
LOSS_MODE = "mse"  # pairwise | listwise | mse
PAIRWISE_MIN_GAP = 0.005  # ignore near-ties in mfe (~0.5pp)
LISTWISE_TEMPERATURE = 0.05  # soft targets over raw mfe within day
RANK_IC_BAR = 0.05
TOPK_FRAC = 0.20
TOPK_LIFT_BAR = 0.05
# Early-stop if Rank IC mean stays near 0 for PATIENCE consecutive evals.
RANK_IC_STUCK_TOL = 0.01
RANK_IC_STUCK_PATIENCE = 2
FREEZE_BACKBONE = False  # train shallow layers (NOT freeze whole bb)
FREEZE_TOKENIZER_EMBEDS = True  # freeze s1/s2 (tokenizer-side embeds)
BACKBONE_MODE = "shallow"  # non-identity shallow ModernBERT + rank head
SHALLOW_LAYERS = 2  # NOT 22
SWANLAB_API_KEY_FALLBACK = ""  # injected at private staging only
SWANLAB_RUN_ID = "kairos-ranking-probe-short-phase-o-20261001"
OUTPUT = Path("/kaggle/working/kairos_ranking_probe")
RUN_PURPOSE = "mfe10-ranking-probe-shallow-freeze-embeds-phase-o"
FEATURES = ("open", "high", "low", "close", "volume", "amount")
TARGET_NAME = "mfe10_continuous_rank"
MFE10_DEF = "max(high[T+1:T+10]) / close[T] - 1"
REPO_CLONE_URL = "https://github.com/luckfu/Kronos.git"
REPO_CLONE_BRANCH = "master"
PHASE_L_RIDGE_MFE_COMB_RANK_IC = 0.2734396296793431
PHASE_M_MSE_RANK_IC = 0.08525129172480793
PHASE_N_PAIRWISE_RANK_IC = 0.08171161247975986


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


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 5:
        return float("nan")
    aa, bb = a[mask], b[mask]
    if float(np.std(aa)) < 1e-12 or float(np.std(bb)) < 1e-12:
        return float("nan")
    # Rank without scipy (Kaggle image may vary).
    ra = aa.argsort().argsort().astype(np.float64)
    rb = bb.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom < 1e-12:
        return float("nan")
    return float((ra * rb).sum() / denom)


def daily_rank_ic(
    dates: np.ndarray, score: np.ndarray, label: np.ndarray, *, min_names: int = 30
) -> dict[str, Any]:
    dates = np.asarray(dates)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    label = np.asarray(label, dtype=np.float64).reshape(-1)
    ics: list[float] = []
    for day in np.unique(dates):
        idx = np.where(dates == day)[0]
        if len(idx) < min_names:
            continue
        r = safe_spearman(score[idx], label[idx])
        if np.isfinite(r):
            ics.append(float(r))
    arr = np.asarray(ics, dtype=np.float64)
    return {
        "n_days": int(len(arr)),
        "mean": float(np.mean(arr)) if len(arr) else float("nan"),
        "std": float(np.std(arr)) if len(arr) else float("nan"),
        "median": float(np.median(arr)) if len(arr) else float("nan"),
    }


def topk_hit_rate(
    dates: np.ndarray,
    score: np.ndarray,
    label: np.ndarray,
    *,
    frac: float = TOPK_FRAC,
    min_names: int = 30,
) -> dict[str, Any]:
    dates = np.asarray(dates)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    label = np.asarray(label, dtype=np.float64).reshape(-1)
    hits: list[float] = []
    for day in np.unique(dates):
        idx = np.where(dates == day)[0]
        n = len(idx)
        if n < min_names:
            continue
        k = max(1, int(round(frac * n)))
        score_top = set(idx[np.argpartition(-score[idx], k - 1)[:k]].tolist())
        label_top = set(idx[np.argpartition(-label[idx], k - 1)[:k]].tolist())
        hits.append(len(score_top & label_top) / k)
    arr = np.asarray(hits, dtype=np.float64)
    chance = float(frac)
    mean = float(np.mean(arr)) if len(arr) else float("nan")
    return {
        "n_days": int(len(arr)),
        "frac": float(frac),
        "mean_hit": mean,
        "chance": chance,
        "lift_vs_chance": float(mean - chance) if np.isfinite(mean) else float("nan"),
    }


def clears_rank_gate(rank_ic_mean: float, topk_lift: float) -> bool:
    ic_ok = bool(np.isfinite(rank_ic_mean) and rank_ic_mean >= RANK_IC_BAR)
    topk_ok = bool(np.isfinite(topk_lift) and topk_lift >= TOPK_LIFT_BAR)
    return ic_ok or topk_ok


def dates_to_ids(dates: np.ndarray) -> np.ndarray:
    """Map datetime64[D] (or comparable) to dense int64 day ids."""
    arr = np.asarray(dates)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[D]").astype(np.int64)
    # fallback: hashable object / string dates → factorize
    _, inv = np.unique(arr.astype(str), return_inverse=True)
    return inv.astype(np.int64)


def same_date_pairwise_ranking_loss(
    scores,
    utilities,
    date_ids,
    *,
    minimum_gap: float = PAIRWISE_MIN_GAP,
    valid_mask=None,
):
    """Same-day pairwise logistic (softplus) on score order vs utility order."""
    import torch
    import torch.nn.functional as F

    scores = scores.reshape(-1)
    utilities = utilities.reshape(-1)
    date_ids = date_ids.reshape(-1)
    if valid_mask is not None:
        valid_mask = valid_mask.reshape(-1).to(dtype=torch.bool)
        scores = scores[valid_mask]
        utilities = utilities[valid_mask]
        date_ids = date_ids[valid_mask]
    if scores.numel() < 2:
        return scores.sum() * 0.0
    same_date = date_ids[:, None] == date_ids[None, :]
    group_sizes = same_date.sum(dim=1)
    score_delta = scores[:, None] - scores[None, :]
    utility_delta = utilities[:, None] - utilities[None, :]
    pair_mask = same_date & torch.triu(utility_delta.abs() >= minimum_gap, diagonal=1)
    pair_values = pair_mask.to(dtype=scores.dtype)
    pair_counts_by_row = pair_values.sum(dim=1)
    group_pair_counts = same_date.to(dtype=scores.dtype) @ pair_counts_by_row
    pair_losses = F.softplus(-score_delta * utility_delta.sign())
    normalized_pairs = (
        pair_losses * pair_values / group_pair_counts.clamp_min(1)[:, None]
    )
    groups_with_pairs = (group_pair_counts > 0).to(dtype=scores.dtype)
    group_count = (groups_with_pairs / group_sizes.clamp_min(1)).sum()
    grouped_loss = normalized_pairs.sum() / group_count.clamp_min(1)
    return torch.where(group_count > 0, grouped_loss, scores.sum() * 0.0)


def same_date_listwise_listnet_loss(
    scores,
    utilities,
    date_ids,
    *,
    temperature: float = LISTWISE_TEMPERATURE,
    valid_mask=None,
):
    """Same-day ListNet: CE between softmax(utilities/T) and log_softmax(scores)."""
    import torch
    import torch.nn.functional as F

    scores = scores.reshape(-1)
    utilities = utilities.reshape(-1)
    date_ids = date_ids.reshape(-1)
    if valid_mask is not None:
        valid_mask = valid_mask.reshape(-1).to(dtype=torch.bool)
        scores = scores[valid_mask]
        utilities = utilities[valid_mask]
        date_ids = date_ids[valid_mask]
    if scores.numel() < 2:
        return scores.sum() * 0.0
    unique = torch.unique(date_ids)
    losses = []
    for day in unique:
        mask = date_ids == day
        if int(mask.sum().item()) < 2:
            continue
        s = scores[mask]
        u = utilities[mask]
        # soft target over within-day mfe; temperature keeps mass on top names
        target = F.softmax(u / max(float(temperature), 1e-6), dim=0)
        log_p = F.log_softmax(s, dim=0)
        losses.append(-(target * log_p).sum())
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def ranking_batch_loss(pred, target, date_ids, valid, loss_mode: str = LOSS_MODE):
    """Dispatch mse / pairwise / listwise; returns scalar loss + tag."""
    import torch.nn.functional as F

    if loss_mode == "mse":
        per_row = F.mse_loss(pred, target, reduction="none").mean(dim=1)
        local_sum = per_row[valid].sum()
        # caller normalizes by global valid count for DDP MSE
        return local_sum, "mse", True
    utilities = target.reshape(-1)
    scores = pred.reshape(-1)
    if loss_mode == "listwise":
        loss = same_date_listwise_listnet_loss(
            scores, utilities, date_ids, temperature=LISTWISE_TEMPERATURE, valid_mask=valid
        )
        return loss, "listwise", False
    # default pairwise
    loss = same_date_pairwise_ranking_loss(
        scores, utilities, date_ids, minimum_gap=PAIRWISE_MIN_GAP, valid_mask=valid
    )
    return loss, "pairwise", False


def rank_ic_stuck_stop(rank_ic_mean: float, stuck_streak: int) -> tuple[str, int]:
    if not np.isfinite(rank_ic_mean):
        return "nonfinite_rank_ic", stuck_streak + 1
    if abs(float(rank_ic_mean)) <= RANK_IC_STUCK_TOL:
        stuck_streak += 1
    else:
        stuck_streak = 0
    if stuck_streak >= RANK_IC_STUCK_PATIENCE:
        return "stuck_near_zero_rank_ic", stuck_streak
    return "", stuck_streak


def attach_cs_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Within-day percentile ranks of mfe10 for the given row list."""
    if not rows:
        return rows
    dates = np.asarray([r["asof_date"] for r in rows])
    mfe = np.asarray([float(r["mfe10"]) for r in rows], dtype=np.float64)
    # pandas-free groupby rank
    order = np.argsort(dates, kind="mergesort")
    ranks = np.empty(len(rows), dtype=np.float64)
    i = 0
    n = len(rows)
    while i < n:
        j = i + 1
        while j < n and dates[order[j]] == dates[order[i]]:
            j += 1
        idx = order[i:j]
        vals = mfe[idx]
        # average rank percentile
        sorter = np.argsort(vals, kind="mergesort")
        pos = np.empty(len(idx), dtype=np.float64)
        pos[sorter] = np.arange(1, len(idx) + 1, dtype=np.float64)
        ranks[idx] = pos / float(len(idx))
        i = j
    out = []
    for row, rk in zip(rows, ranks):
        item = dict(row)
        item["mfe_cs_rank"] = float(rk)
        out.append(item)
    return out


def shuffled_group_order(num_groups: int) -> list[int]:
    return np.random.default_rng(SHUFFLE_SEED).permutation(num_groups).tolist()


def shuffle_group_rows(rows: list[dict[str, Any]], group_id: int) -> list[dict[str, Any]]:
    order = np.random.default_rng(SHUFFLE_SEED + group_id + 1).permutation(len(rows))
    return [rows[int(index)] for index in order]


def group_order_hash(order: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, order)).encode("ascii")).hexdigest()


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
            "model": "shallow-freeze-tok-embeds-rank-head",
            "hidden_size": 768,
            "layers": SHALLOW_LAYERS if BACKBONE_MODE == "shallow" else 0,
            "backbone_mode": BACKBONE_MODE,
            "shallow_layers": SHALLOW_LAYERS,
            "freeze_tokenizer_embeds": FREEZE_TOKENIZER_EMBEDS,
            "freeze_backbone": FREEZE_BACKBONE,
            "batch_size": BATCH_SIZE,
            "chunk_index": CHUNK_INDEX,
            "segment_samples": SEGMENT_SAMPLES,
            "max_segments_this_run": MAX_SEGMENTS_THIS_RUN,
            "learning_rate": LEARNING_RATE,
            "target": TARGET_NAME,
            "target_mode": TARGET_MODE,
            "mfe10_def": MFE10_DEF,
            "rank_ic_bar": RANK_IC_BAR,
            "topk_frac": TOPK_FRAC,
            "topk_lift_bar": TOPK_LIFT_BAR,
            "variant": "kairos-ranking-probe-short-phase-o",
            "loss_mode": LOSS_MODE,
            "pairwise_min_gap": PAIRWISE_MIN_GAP,
            "listwise_temperature": LISTWISE_TEMPERATURE,
            "not_multi_head_r2": True,
            "not_binary_mfe10": True,
            "not_22_layer_binary": True,
            "phase_l_ridge_mfe_comb_rank_ic": PHASE_L_RIDGE_MFE_COMB_RANK_IC,
            "phase_m_mse_rank_ic": PHASE_M_MSE_RANK_IC,
            "phase_n_pairwise_rank_ic": PHASE_N_PAIRWISE_RANK_IC,
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
    mfe_raw = np.empty((len(rows), 1), dtype=np.float32)
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
        mfe_val = float(row["mfe10"])
        if not np.isfinite(mfe_val):
            raise ValueError("non-finite mfe10")
        mfe_raw[i, 0] = mfe_val
        if TARGET_MODE == "mfe_cs_rank":
            labels[i, 0] = float(row.get("mfe_cs_rank", mfe_val))
        else:
            labels[i, 0] = mfe_val
        dates[i] = data["dates"][asof]
        if check_labels:
            future = data["values"][start + 120:start + 130]
            close = float(data["values"][asof, 3])
            path_mfe = float(future[:, 1].max() / close - 1)
            if not np.isclose(path_mfe, mfe_val, atol=2e-6):
                raise ValueError(f"MFE mismatch: {symbol} {start}")
    return histories, sectors, sizes, labels, mfe_raw, dates


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
        target_mode=TARGET_MODE,
        mfe10_def=MFE10_DEF,
        max_segments_this_run=MAX_SEGMENTS_THIS_RUN,
        learning_rate=LEARNING_RATE,
        not_multi_head_r2=True,
        not_binary_mfe10=True,
        not_22_layer_binary=True,
        freeze_backbone=FREEZE_BACKBONE,
        freeze_tokenizer_embeds=FREEZE_TOKENIZER_EMBEDS,
        backbone_mode=BACKBONE_MODE,
        shallow_layers=SHALLOW_LAYERS,
        loss_mode=LOSS_MODE,
        pairwise_min_gap=PAIRWISE_MIN_GAP,
        listwise_temperature=LISTWISE_TEMPERATURE,
        rank_ic_bar=RANK_IC_BAR,
        topk_lift_bar=TOPK_LIFT_BAR,
        world_size=world_size,
        phase_m_mse_rank_ic=PHASE_M_MSE_RANK_IC,
        phase_n_pairwise_rank_ic=PHASE_N_PAIRWISE_RANK_IC,
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

    class RankHeadModel(nn.Module):
        """Shallow non-identity (or identity) + single score head for ranking mfe10."""

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
            self.shallow_layers = int(SHALLOW_LAYERS)
            if BACKBONE_MODE == "identity":
                # Phase K/M/N: no ModernBERT; mean-pool condition+market embeds.
                self.backbone = None
            elif BACKBONE_MODE == "shallow":
                # Phase O: shallow non-identity ONLY (never 22-layer).
                if self.shallow_layers < 1 or self.shallow_layers > 4:
                    raise RuntimeError(
                        f"SHALLOW_LAYERS must be in 1..4 for Phase O, got {self.shallow_layers}"
                    )
                config = ModernBertConfig(
                    vocab_size=1024, hidden_size=hidden, intermediate_size=1152,
                    num_hidden_layers=self.shallow_layers, num_attention_heads=12,
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
            else:
                raise RuntimeError(
                    f"unsupported BACKBONE_MODE={BACKBONE_MODE!r}; "
                    "use identity|shallow (no 22-layer binary)"
                )
            self.head = nn.Linear(hidden, 1)
            # Ranking score: zero weight, bias at train-label mean (z later).
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

    # Train label mean/std from continuous mfe10 (full train scan) for bias init.
    totals = 0.0
    sq = 0.0
    count = 0
    for index in range(train_pq.num_row_groups):
        table = train_pq.read_row_group(index, columns=["mfe10"])
        mfe = np.asarray(table["mfe10"].to_numpy(), dtype=np.float64)
        totals += float(np.nansum(mfe))
        sq += float(np.nansum(mfe * mfe))
        count += int(np.isfinite(mfe).sum())
    if count <= 0:
        raise RuntimeError("train mfe10 scan empty")
    train_label_mean = float(totals / count)
    train_label_var = max(0.0, float(sq / count) - train_label_mean * train_label_mean)
    train_label_std = float(math.sqrt(train_label_var) + 1e-6)
    head_bias = train_label_mean if TARGET_MODE == "mfe10_continuous" else 0.5
    log(
        "train_label_stats",
        samples=count,
        mfe_mean=train_label_mean,
        mfe_std=train_label_std,
        head_bias_init=head_bias,
        target_mode=TARGET_MODE,
        target=TARGET_NAME,
    )

    model = RankHeadModel(head_bias=head_bias).to(device)
    raw_model = model
    if FREEZE_TOKENIZER_EMBEDS:
        for parameter in raw_model.s1.parameters():
            parameter.requires_grad_(False)
        for parameter in raw_model.s2.parameters():
            parameter.requires_grad_(False)
    if FREEZE_BACKBONE and raw_model.backbone is not None:
        for parameter in raw_model.backbone.parameters():
            parameter.requires_grad_(False)
    trainable = [p for p in raw_model.parameters() if p.requires_grad]
    frozen = [p for p in raw_model.parameters() if not p.requires_grad]
    log(
        "param_freeze",
        freeze_backbone=FREEZE_BACKBONE,
        freeze_tokenizer_embeds=FREEZE_TOKENIZER_EMBEDS,
        backbone_mode=BACKBONE_MODE,
        shallow_layers=SHALLOW_LAYERS if BACKBONE_MODE == "shallow" else 0,
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
        if TARGET_MODE == "mfe_cs_rank":
            rows = attach_cs_ranks(rows)
        token_s1: list[np.ndarray] = []
        token_s2: list[np.ndarray] = []
        cached_sectors: list[np.ndarray] = []
        cached_sizes: list[np.ndarray] = []
        cached_labels: list[np.ndarray] = []
        cached_mfe: list[np.ndarray] = []
        cached_dates: list[np.ndarray] = []
        for offset in range(0, len(rows), local_batch_size):
            batch = rows[offset:offset + local_batch_size]
            history, sectors, sizes, labels, mfe_raw, dates = make_batch(
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
            cached_mfe.append(mfe_raw)
            cached_dates.append(dates)
        return (
            np.concatenate(token_s1, axis=0),
            np.concatenate(token_s2, axis=0),
            np.concatenate(cached_sectors, axis=0),
            np.concatenate(cached_sizes, axis=0),
            np.concatenate(cached_labels, axis=0),
            np.concatenate(cached_mfe, axis=0),
            np.concatenate(cached_dates, axis=0),
        )

    def full_validation() -> dict[str, Any]:
        nonlocal validation_token_cache

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
        predictions, train_labels, mfe_labels, dates = [], [], [], []
        for offset in range(0, len(local_cache[0]), local_batch_size):
            s1 = torch.from_numpy(local_cache[0][offset:offset + local_batch_size]).to(device)
            s2 = torch.from_numpy(local_cache[1][offset:offset + local_batch_size]).to(device)
            sizes = torch.from_numpy(local_cache[3][offset:offset + local_batch_size, None]).to(device)
            sectors = torch.from_numpy(local_cache[2][offset:offset + local_batch_size]).to(device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                scores = model(s1.long(), s2.long(), sizes, sectors)
            predictions.append(scores.float().cpu().numpy())
            train_labels.append(local_cache[4][offset:offset + local_batch_size])
            mfe_labels.append(local_cache[5][offset:offset + local_batch_size])
            dates.append(local_cache[6][offset:offset + local_batch_size])
        local_result = (
            np.concatenate(predictions),
            np.concatenate(train_labels),
            np.concatenate(mfe_labels),
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

        scores = np.concatenate([item[0] for item in gathered]).astype(np.float64).reshape(-1)
        y_train = np.concatenate([item[1] for item in gathered]).astype(np.float64).reshape(-1)
        y_mfe = np.concatenate([item[2] for item in gathered]).astype(np.float64).reshape(-1)
        dates = np.concatenate([item[3] for item in gathered])
        result: dict[str, Any] = {}
        for name, mask in (
            ("all", np.ones(len(scores), dtype=bool)),
            ("2025H2", dates < np.datetime64("2026-01-01")),
            ("2026H1", dates >= np.datetime64("2026-01-01")),
        ):
            sc = scores[mask]
            ym = y_mfe[mask]
            yt = y_train[mask]
            dt = dates[mask]
            rank_ic = daily_rank_ic(dt, sc, ym)
            topk = topk_hit_rate(dt, sc, ym, frac=TOPK_FRAC)
            mse = float(np.mean(np.square(sc - yt))) if sc.size else float("nan")
            result[name] = {
                "samples": int(mask.sum()),
                "mse": mse,
                "pooled_spearman_vs_mfe": safe_spearman(sc, ym),
                "rank_ic": rank_ic,
                "topk": topk,
                "rank_ic_bar": RANK_IC_BAR,
                "topk_lift_bar": TOPK_LIFT_BAR,
                "gate_passed": clears_rank_gate(rank_ic["mean"], topk["lift_vs_chance"]),
                "target": TARGET_NAME,
                "target_mode": TARGET_MODE,
                "phase_l_ridge_mfe_comb_rank_ic": PHASE_L_RIDGE_MFE_COMB_RANK_IC,
            }
        log(
            "rank_ic_baseline",
            all_rank_ic_mean=result["all"]["rank_ic"]["mean"],
            all_rank_ic_n_days=result["all"]["rank_ic"]["n_days"],
            all_topk_hit=result["all"]["topk"]["mean_hit"],
            all_topk_lift=result["all"]["topk"]["lift_vs_chance"],
            all_mse=result["all"]["mse"],
            gate=RANK_IC_BAR,
            gate_passed=result["all"]["gate_passed"],
            phase_l_ridge_ic=PHASE_L_RIDGE_MFE_COMB_RANK_IC,
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
    best_score = float("-inf")  # maximize Rank IC mean
    last_validation = None
    last_best_updated = False
    rank_ic_stuck_streak = 0

    # Initial eval (untrained) — Rank IC / TopK before any segment.
    last_validation = full_validation()
    init_ic = float(last_validation["all"]["rank_ic"]["mean"])
    # Avoid Phase M NaN pollution of best_metric.json when n_days=0 at init.
    best_score = init_ic if math.isfinite(init_ic) else float("-inf")
    if is_main_process():
        best_meta.write_text(
            json.dumps({
                "rank_ic_mean": (init_ic if math.isfinite(init_ic) else None),
                "topk_lift": last_validation["all"]["topk"]["lift_vs_chance"],
                "mse": last_validation["all"]["mse"],
                "gate_passed": last_validation["all"]["gate_passed"],
                "segment_index": 0,
                "processed_samples": 0,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        swanlab_run.log({
            "validation/rank_ic_mean": last_validation["all"]["rank_ic"]["mean"],
            "validation/rank_ic_std": last_validation["all"]["rank_ic"]["std"],
            "validation/topk_hit": last_validation["all"]["topk"]["mean_hit"],
            "validation/topk_lift": last_validation["all"]["topk"]["lift_vs_chance"],
            "validation/mse": last_validation["all"]["mse"],
            "validation/gate_passed": int(last_validation["all"]["gate_passed"]),
            "validation/phase_l_ridge_ic": PHASE_L_RIDGE_MFE_COMB_RANK_IC,
            "train/label_mean": train_label_mean,
            "train/label_std": train_label_std,
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
        # Date-sort so contiguous batches share asof_date → denser ranking pairs.
        segment_rows = sorted(
            segment_rows,
            key=lambda r: (str(r.get("asof_date", "")), str(r.get("symbol", "")), int(r.get("start_index", 0))),
        )
        padded_count = ((segment_samples + world_size - 1) // world_size) * world_size
        padded_rows = segment_rows + [segment_rows[-1]] * (padded_count - segment_samples)
        local_rows = padded_rows[rank::world_size]
        local_valid = np.arange(rank, padded_count, world_size) < segment_samples
        local_cache = encode_token_cache(
            local_rows, train_arrays,
            check_labels=(rank == 0 and completed_segments == 0),
        )
        local_date_ids = dates_to_ids(local_cache[6])

        for offset in range(0, len(local_rows), local_batch_size):
            batch_slice = slice(offset, offset + local_batch_size)
            valid = torch.from_numpy(local_valid[batch_slice]).to(device)
            s1 = torch.from_numpy(local_cache[0][batch_slice]).to(device)
            s2 = torch.from_numpy(local_cache[1][batch_slice]).to(device)
            sector = torch.from_numpy(local_cache[2][batch_slice]).to(device)
            size = torch.from_numpy(local_cache[3][batch_slice, None]).to(device)
            target = torch.from_numpy(local_cache[4][batch_slice]).to(device)
            date_ids = torch.from_numpy(local_date_ids[batch_slice]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model(s1.long(), s2.long(), size, sector)
                raw_loss, loss_tag, is_mse_sum = ranking_batch_loss(
                    pred, target, date_ids, valid, LOSS_MODE
                )
                if is_mse_sum:
                    global_valid_count = valid.sum().to(dtype=torch.float32)
                    if distributed:
                        dist.all_reduce(global_valid_count, op=dist.ReduceOp.SUM)
                    loss = raw_loss * world_size / global_valid_count.clamp_min(1)
                else:
                    # Pairwise/listwise already reduced; mean across ranks for DDP.
                    loss = raw_loss
                    if distributed:
                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                        loss = loss / world_size
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
                    f"train/{loss_tag}_loss": float(loss.detach().cpu()),
                    "train/loss_mode": LOSS_MODE,
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
        validation_score = float(last_validation["all"]["rank_ic"]["mean"])
        topk_lift = float(last_validation["all"]["topk"]["lift_vs_chance"])
        mse_val = float(last_validation["all"]["mse"])
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
            rank_ic_mean=validation_score,
            topk_lift=topk_lift,
            mse=mse_val,
            gate_passed=last_validation["all"]["gate_passed"],
            phase_l_ridge_ic=PHASE_L_RIDGE_MFE_COMB_RANK_IC,
        )

        last_best_updated = bool(math.isfinite(validation_score) and validation_score > best_score)
        if last_best_updated:
            best_score = validation_score
            if is_main_process():
                torch.save({
                    "model": raw_model.state_dict(),
                    "metrics": last_validation,
                    "segment_index": global_segment_index,
                    "processed_samples": processed,
                    "target": TARGET_NAME,
                    "target_mode": TARGET_MODE,
                    "mfe10_def": MFE10_DEF,
                }, best_checkpoint.with_suffix(".pt.tmp"))
                best_checkpoint.with_suffix(".pt.tmp").replace(best_checkpoint)
                best_meta.write_text(
                    json.dumps({
                        "rank_ic_mean": best_score,
                        "topk_lift": topk_lift,
                        "mse": mse_val,
                        "gate_passed": last_validation["all"]["gate_passed"],
                        "segment_index": global_segment_index,
                        "processed_samples": processed,
                    }, indent=2) + "\n",
                    encoding="utf-8",
                )
                log("best_model_updated", rank_ic_mean=best_score, topk_lift=topk_lift)

        if is_main_process():
            swanlab_run.log({
                "validation/rank_ic_mean": validation_score,
                "validation/rank_ic_std": last_validation["all"]["rank_ic"]["std"],
                "validation/topk_hit": last_validation["all"]["topk"]["mean_hit"],
                "validation/topk_lift": topk_lift,
                "validation/mse": mse_val,
                "validation/gate_passed": int(last_validation["all"]["gate_passed"]),
                "validation/best_updated": int(last_best_updated),
                "validation/phase_l_ridge_ic": PHASE_L_RIDGE_MFE_COMB_RANK_IC,
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
                "train_label_mean": train_label_mean,
                "train_label_std": train_label_std,
                "target": TARGET_NAME,
                "target_mode": TARGET_MODE,
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

        reason, rank_ic_stuck_streak = rank_ic_stuck_stop(
            validation_score, rank_ic_stuck_streak
        )
        if reason:
            stop_reason = reason
            run_segment_limit = segments_this_run
            log("early_stop", reason=reason, stuck_streak=rank_ic_stuck_streak,
                rank_ic_mean=validation_score)

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

    best_topk = (
        None if last_validation is None
        else float(last_validation["all"]["topk"]["lift_vs_chance"])
    )
    report = {
        "status": "RANKING_PROBE_COMPLETE",
        "purpose": RUN_PURPOSE,
        "target": TARGET_NAME,
        "target_mode": TARGET_MODE,
        "mfe10_def": MFE10_DEF,
        "not_multi_head_r2": True,
        "not_binary_mfe10": True,
        "not_22_layer_binary": True,
        "backbone_mode": BACKBONE_MODE,
        "shallow_layers": SHALLOW_LAYERS if BACKBONE_MODE == "shallow" else 0,
        "freeze_backbone": FREEZE_BACKBONE,
        "freeze_tokenizer_embeds": FREEZE_TOKENIZER_EMBEDS,
        "loss_mode": LOSS_MODE,
        "pairwise_min_gap": PAIRWISE_MIN_GAP,
        "listwise_temperature": LISTWISE_TEMPERATURE,
        "phase_m_mse_rank_ic": PHASE_M_MSE_RANK_IC,
        "phase_n_pairwise_rank_ic": PHASE_N_PAIRWISE_RANK_IC,
        "train_label_mean": train_label_mean,
        "train_label_std": train_label_std,
        "stop_reason": stop_reason,
        "segments_this_run": segments_this_run,
        "processed_samples": processed,
        "best_rank_ic_mean": best_score,
        "best_topk_lift": best_topk,
        "rank_ic_bar": RANK_IC_BAR,
        "topk_lift_bar": TOPK_LIFT_BAR,
        "gate_passed": (
            None if last_validation is None
            else bool(last_validation["all"]["gate_passed"])
        ),
        "phase_l_ridge_mfe_comb_rank_ic": PHASE_L_RIDGE_MFE_COMB_RANK_IC,
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
        log("ranking_probe_complete", **{k: report[k] for k in (
            "status", "stop_reason", "segments_this_run", "best_rank_ic_mean",
            "best_topk_lift", "gate_passed",
        )})
        swanlab.finish()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
