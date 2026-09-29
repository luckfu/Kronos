"""Read-only Kairos R1 train-prior and validation discrimination audit."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SEED = 20260925
BATCH_SIZE = 16
CHUNK_INDEX = 7
CHUNK_COUNT = 8
SHUFFLE_SEED = 20260925
SWANLAB_API_KEY_FALLBACK = "fmEPDGk4IItxgqSZKGLi8"
SWANLAB_RUN_ID = "modernbert-decision-full-gated-v1"
SEGMENT_TOTAL = CHUNK_COUNT
OUTPUT = Path("/kaggle/working/modernbert_decision_full")
FEATURES = ("open", "high", "low", "close", "volume", "amount")
TARGET_COLUMNS = (
    "up_003", "up_005", "up_008", "up_012",
    "down_003", "down_005", "down_008", "down_012",
)


def log(phase: str, **fields: object) -> None:
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "phase": phase, **fields}
    line = json.dumps(row, ensure_ascii=False, default=str)
    print(line, flush=True)
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


def dashboard(state: dict[str, Any]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "progress.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    processed = int(state.get("processed_samples", 0))
    total = int(state.get("total_samples", 0))
    percent = 100 * processed / total if total else 0
    html = f"""<!doctype html><meta charset="utf-8"><title>ModernBERT full training</title>
<style>body{{font:16px system-ui;margin:32px;background:#f6f7f9;color:#17202a}}
main{{max-width:900px;margin:auto}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
.card,pre{{background:white;border:1px solid #ddd;border-radius:8px;padding:16px}}
.label{{color:#667085;font-size:13px}}.value{{font-size:24px;font-weight:650;margin-top:6px}}</style>
<main><h1>ModernBERT full training</h1><div class="grid">
<div class="card"><div class="label">Phase</div><div class="value">{state.get("phase","starting")}</div></div>
<div class="card"><div class="label">Progress</div><div class="value">{percent:.2f}%</div></div>
<div class="card"><div class="label">Samples</div><div class="value">{processed:,}/{total:,}</div></div>
</div><pre>{json.dumps(state, ensure_ascii=False, indent=2, default=str)}</pre></main>"""
    (OUTPUT / "dashboard.html").write_text(html, encoding="utf-8")


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
        mask = (
            (probabilities >= edges[index])
            & ((probabilities < edges[index + 1]) if index + 1 < bins else (probabilities <= edges[index + 1]))
        )
        if mask.any():
            error += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(labels[mask].mean()))
    return error


def start_swanlab() -> tuple[Any, Any]:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--progress-bar", "off", "swanlab"],
        check=True,
    )
    import swanlab

    api_key = os.environ.get("SWANLAB_API_KEY", "").strip() or SWANLAB_API_KEY_FALLBACK
    if not api_key:
        raise RuntimeError("SWANLAB_API_KEY is empty")
    swanlab.login(api_key=api_key)
    run = swanlab.init(
        id=SWANLAB_RUN_ID,
        resume="allow",
        project="finance",
        workspace="roc_fu",
        experiment_name=SWANLAB_RUN_ID,
        config={
            "model": "ModernBERT-base-style",
            "hidden_size": 768,
            "layers": 22,
            "heads": 12,
            "batch_size": BATCH_SIZE,
            "chunk_index": CHUNK_INDEX,
            "chunk_count": CHUNK_COUNT,
            "shuffle_seed": SHUFFLE_SEED,
            "variant": "gated",
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
    labels = np.empty((len(rows), 8), dtype=np.float32)
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
        labels[i] = [row[name] for name in TARGET_COLUMNS]
        dates[i] = data["dates"][asof]
        if check_labels:
            future = data["values"][start + 120:start + 130]
            close = float(data["values"][asof, 3])
            if not np.isclose(future[:, 1].max() / close - 1, row["mfe10"], atol=2e-6):
                raise ValueError(f"MFE mismatch: {symbol} {start}")
            if not np.isclose(future[:, 2].min() / close - 1, row["mae10"], atol=2e-6):
                raise ValueError(f"MAE mismatch: {symbol} {start}")
    return histories, sectors, sizes, labels, dates


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def binary_pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    if not np.any(labels == 1):
        return float("nan")
    return float(average_precision_score(labels, scores))


def verify_boundaries(rows: list[dict[str, Any]], arrays: dict[str, dict[str, Any]]) -> None:
    thresholds = (0.03, 0.05, 0.08, 0.12)
    for row in rows:
        data = arrays[str(row["symbol"])]
        start = int(row["start_index"])
        asof = start + 119
        assert len(data["values"][start:asof + 1]) == 120
        assert len(data["values"][asof + 1:asof + 11]) == 10
        assert data["dates"][asof] == np.datetime64(row["asof_date"], "D")
        assert data["dates"][asof] < data["dates"][asof + 1]
        assert np.array_equal(data["values"][asof], data["values"][start + 119])
        close = float(data["values"][asof, 3])
        future = data["values"][asof + 1:asof + 11]
        mfe = float(future[:, 1].max() / close - 1)
        mae = float(future[:, 2].min() / close - 1)
        assert np.isclose(mfe, row["mfe10"], atol=2e-6)
        assert np.isclose(mae, row["mae10"], atol=2e-6)
        expected = [float(mfe >= t) for t in thresholds]
        expected += [float(-mae >= t) for t in thresholds]
        actual = [float(row[name]) for name in TARGET_COLUMNS]
        assert np.array_equal(expected, actual), (row["symbol"], start, expected, actual)


def main() -> None:
    import pyarrow.parquet as pq
    import torch
    import torch.nn as nn

    if not torch.cuda.is_available():
        raise RuntimeError("R1 audit requires Kaggle GPU")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    output = Path("/kaggle/working/kairos_r1_metrics_audit")
    output.mkdir(parents=True, exist_ok=True)

    train_panel_path = find_one("**/processed_datasets/train_data.pkl")
    validation_panel_path = find_one("**/processed_datasets/val_data.pkl")
    train_targets_path = find_one("**/train_targets.parquet")
    validation_targets_path = find_one("**/validation_targets.parquet")
    train_pq = pq.ParquetFile(train_targets_path)
    validation_pq = pq.ParquetFile(validation_targets_path)
    train_arrays, train_sectors = load_arrays(train_panel_path)
    validation_arrays, validation_sectors = load_arrays(validation_panel_path)
    sector_ids = {
        name: i for i, name in enumerate(sorted(train_sectors | validation_sectors))
    }

    repo_path = Path("/kaggle/working/Kronos")
    if not (repo_path / "model" / "kronos.py").is_file():
        subprocess.run([
            "git", "clone", "--depth", "1", "--branch", "master",
            "https://github.com/luckfu/Kronos.git", str(repo_path)
        ], check=True)
    sys.path.insert(0, str(repo_path))
    from model import KronosTokenizer
    from huggingface_hub import snapshot_download
    from transformers import ModernBertConfig, ModernBertModel

    tokenizer_path = Path(snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base"))
    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path)).to(device).eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)

    class FullModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = 768
            self.s1 = nn.Embedding(1024, hidden // 2)
            self.s2 = nn.Embedding(1024, hidden // 2)
            self.fusion = nn.Linear(hidden, hidden)
            self.gate = nn.Parameter(torch.zeros(1))
            self.sector = nn.Embedding(len(sector_ids) + 1, hidden)
            self.size = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, hidden))
            self.cond = nn.Parameter(torch.zeros(1, 1, hidden))
            config = ModernBertConfig(
                vocab_size=1024, hidden_size=hidden, intermediate_size=1152,
                num_hidden_layers=22, num_attention_heads=12,
                max_position_embeddings=128, pad_token_id=0,
                attention_dropout=0.0, embedding_dropout=0.0, mlp_dropout=0.0,
                reference_compile=False,
            )
            self.backbone = ModernBertModel(config)
            self.up0, self.upd = nn.Linear(hidden, 1), nn.Linear(hidden, 3)
            self.dn0, self.dnd = nn.Linear(hidden, 1), nn.Linear(hidden, 3)

        def forward(self, s1: Any, s2: Any, size: Any, sector: Any) -> tuple[Any, Any]:
            condition = self.cond + self.sector(sector).unsqueeze(1) + self.size(size).unsqueeze(1)
            market = self.fusion(torch.cat([self.s1(s1), self.s2(s2)], -1)) * self.gate
            sequence = torch.cat([condition, market], 1)
            mask = torch.ones(sequence.shape[:2], dtype=torch.long, device=sequence.device)
            pooled = self.backbone(inputs_embeds=sequence, attention_mask=mask).last_hidden_state[:, 0]

            def ordinal(base: Any, delta: Any) -> Any:
                return torch.cat([base, base - torch.cumsum(F.softplus(delta), -1)], -1)

            return ordinal(self.up0(pooled), self.upd(pooled)), ordinal(self.dn0(pooled), self.dnd(pooled))

    model = FullModel().to(device)
    checkpoints = sorted(Path("/kaggle/input").glob("**/final_model.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected one Chunk 8 final_model.pt, found {checkpoints}")
    saved = torch.load(checkpoints[0], map_location=device)
    model.load_state_dict(saved["model"])

    columns = ["symbol", "start_index", "asof_date", "mfe10", "mae10", *TARGET_COLUMNS]
    train_positive_counts = np.zeros(len(TARGET_COLUMNS), dtype=np.int64)
    train_count = 0
    train_label_checks = 0
    for group_id in range(train_pq.num_row_groups):
        rows = train_pq.read_row_group(group_id, columns=columns)
        if group_id == 0:
            sample = rows.slice(0, min(64, rows.num_rows)).to_pylist()
            verify_boundaries(sample, train_arrays)
            make_batch(
                sample, train_arrays, sector_ids, check_labels=True
            )
            train_label_checks = len(sample)
        train_positive_counts += np.array(
            [rows.column(name).to_numpy().sum() for name in TARGET_COLUMNS], dtype=np.int64
        )
        train_count += rows.num_rows
    train_rates = train_positive_counts / train_count

    predictions, labels, dates = [], [], []
    model.eval()
    with torch.no_grad():
        for group_id in range(validation_pq.num_row_groups):
            rows = validation_pq.read_row_group(group_id, columns=columns).to_pylist()
            for offset in range(0, len(rows), BATCH_SIZE):
                batch = rows[offset:offset + BATCH_SIZE]
                if group_id == 0 and offset == 0:
                    verify_boundaries(batch, validation_arrays)
                history, sectors, sizes, batch_labels, batch_dates = make_batch(
                    batch, validation_arrays, sector_ids,
                    check_labels=(group_id == 0 and offset == 0),
                )
                s1, s2 = tokenizer.encode(
                    torch.from_numpy(history).to(device), half=True
                )
                up, down = model(
                    s1.long(), s2.long(),
                    torch.from_numpy(sizes[:, None]).to(device),
                    torch.from_numpy(sectors).to(device),
                )
                predictions.append(
                    torch.cat([up.sigmoid(), down.sigmoid()], 1).float().cpu().numpy()
                )
                labels.append(batch_labels)
                dates.append(batch_dates)
    probabilities = np.concatenate(predictions)
    labels = np.concatenate(labels)
    dates = np.concatenate(dates)
    if train_count != train_pq.metadata.num_rows or len(labels) != validation_pq.metadata.num_rows:
        raise RuntimeError("target row coverage mismatch")
    monotonic_violations = {
        "up": int(np.count_nonzero(np.diff(probabilities[:, :4], axis=1) > 1e-7)),
        "down": int(np.count_nonzero(np.diff(probabilities[:, 4:], axis=1) > 1e-7)),
    }

    metrics = {}
    for name, mask in (
        ("all", np.ones(len(labels), dtype=bool)),
        ("early_2025H2", dates < np.datetime64("2026-01-01")),
        ("late_2026H1", dates >= np.datetime64("2026-01-01")),
    ):
        per_threshold = {}
        for i, target_name in enumerate(TARGET_COLUMNS):
            p = np.clip(probabilities[mask, i], 1e-7, 1 - 1e-7)
            y = labels[mask, i]
            prior = float(train_rates[i])
            per_threshold[target_name] = {
                "positive_rate": float(y.mean()),
                "mean_probability": float(p.mean()),
                "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
                "brier": float(np.square(p - y).mean()),
                "ece_10bin": calibration_error(p, y),
                "roc_auc": binary_auc(p, y),
                "pr_auc": binary_pr_auc(p, y),
                "train_prior_log_loss": float(
                    -(y * np.log(prior) + (1 - y) * np.log(1 - prior)).mean()
                ),
                "train_prior_brier": float(np.square(prior - y).mean()),
            }
        metrics[name] = {
            "samples": int(mask.sum()),
            "macro_log_loss": float(np.mean([v["log_loss"] for v in per_threshold.values()])),
            "macro_train_prior_log_loss": float(np.mean([v["train_prior_log_loss"] for v in per_threshold.values()])),
            "macro_brier": float(np.mean([v["brier"] for v in per_threshold.values()])),
            "macro_train_prior_brier": float(np.mean([v["train_prior_brier"] for v in per_threshold.values()])),
            "per_threshold": per_threshold,
        }
    result = {
        "status": "PASS",
        "purpose": "R1 baseline and discrimination audit",
        "train_samples": int(train_count),
        "validation_samples": int(len(labels)),
        "train_positive_rates": dict(zip(TARGET_COLUMNS, train_rates.tolist())),
        "train_label_check_samples": train_label_checks,
        "validation_label_check_samples": min(BATCH_SIZE, int(validation_pq.metadata.num_rows)),
        "monotonic_violations": monotonic_violations,
        "metrics": metrics,
    }
    (output / "r1_metrics_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
