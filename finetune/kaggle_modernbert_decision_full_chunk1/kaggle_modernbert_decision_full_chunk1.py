"""Full one-epoch ModernBERT decision training with checkpointing and dashboard."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
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
CHUNK_INDEX = 0
CHUNK_COUNT = 8
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


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    import pyarrow.parquet as pq
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("full training requires Kaggle GPU")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")

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
    train_total, validation_total = train_pq.metadata.num_rows, validation_pq.metadata.num_rows
    dashboard({"phase": "loading panels", "processed_samples": 0, "total_samples": train_total})
    log("runtime", gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        gpu_count=torch.cuda.device_count(), torch=torch.__version__,
        transformers=importlib.metadata.version("transformers"),
        train_samples=train_total, validation_samples=validation_total)
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
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        log("multi_gpu_enabled", devices=torch.cuda.device_count(), mode="DataParallel")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    checkpoint = OUTPUT / "last_checkpoint.pt"
    if not checkpoint.exists():
        previous = sorted(Path("/kaggle/input").glob("**/last_checkpoint.pt"))
        if len(previous) > 1:
            raise RuntimeError(f"expected at most one previous checkpoint, found {previous}")
        if previous:
            shutil.copy2(previous[0], checkpoint)
    chunk_start = train_pq.num_row_groups * CHUNK_INDEX // CHUNK_COUNT
    chunk_end = train_pq.num_row_groups * (CHUNK_INDEX + 1) // CHUNK_COUNT
    start_group, processed = chunk_start, 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start_group = max(chunk_start, int(saved["row_group"]) + 1)
        processed = int(saved["processed_samples"])
        log("checkpoint_resumed", row_group=start_group, processed_samples=processed)

    columns = ["symbol", "start_index", "asof_date", "mfe10", "mae10", *TARGET_COLUMNS]
    started = time.monotonic()
    model.train()
    for group_id in range(start_group, chunk_end):
        rows = train_pq.read_row_group(group_id, columns=columns).to_pylist()
        for offset in range(0, len(rows), BATCH_SIZE):
            batch = rows[offset:offset + BATCH_SIZE]
            history, sectors, sizes, labels, _ = make_batch(
                batch, train_arrays, sector_ids,
                check_labels=(processed == 0 and group_id == start_group and offset == 0),
            )
            with torch.no_grad():
                s1, s2 = tokenizer.encode(torch.from_numpy(history).to(device), half=True)
            sector = torch.from_numpy(sectors).to(device)
            size = torch.from_numpy(sizes[:, None]).to(device)
            target = torch.from_numpy(labels).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                up, down = model(s1.long(), s2.long(), size, sector)
                loss = F.binary_cross_entropy_with_logits(up, target[:, :4])
                loss = loss + F.binary_cross_entropy_with_logits(down, target[:, 4:])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            processed += len(batch)
            if processed % 25_000 < len(batch):
                rate = processed / max(time.monotonic() - started, 1e-6)
                state = {"phase": "training", "processed_samples": processed,
                         "total_samples": train_total, "row_group": group_id,
                         "loss": float(loss.detach().cpu()), "samples_per_second": rate,
                         "eta_seconds": (train_total - processed) / max(rate, 1e-6)}
                log("training_progress", **{
                    key: value for key, value in state.items() if key != "phase"
                })
                dashboard(state)
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "row_group": group_id,
            "processed_samples": processed,
        }, checkpoint)
        log("checkpoint_saved", row_group=group_id, processed_samples=processed)

    if CHUNK_INDEX + 1 < CHUNK_COUNT:
        report = {
            "status": "CHUNK_COMPLETE", "chunk_index": CHUNK_INDEX,
            "chunk_count": CHUNK_COUNT, "row_group_start": chunk_start,
            "row_group_end": chunk_end, "processed_samples": processed,
            "checkpoint": str(checkpoint), "elapsed_seconds": time.monotonic() - started,
        }
        (OUTPUT / "chunk_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        dashboard({"phase": "chunk complete", "processed_samples": processed,
                   "total_samples": train_total, "chunk_index": CHUNK_INDEX,
                   "chunk_count": CHUNK_COUNT})
        log("chunk_complete", **report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return

    model.eval()
    predictions, truth, dates = [], [], []
    for group_id in range(validation_pq.num_row_groups):
        rows = validation_pq.read_row_group(group_id, columns=columns).to_pylist()
        for offset in range(0, len(rows), BATCH_SIZE):
            history, sectors, sizes, labels, batch_dates = make_batch(
                rows[offset:offset + BATCH_SIZE], validation_arrays, sector_ids
            )
            with torch.no_grad():
                s1, s2 = tokenizer.encode(torch.from_numpy(history).to(device), half=True)
                up, down = model(
                    s1.long(), s2.long(), torch.from_numpy(sizes[:, None]).to(device),
                    torch.from_numpy(sectors).to(device),
                )
            predictions.append(torch.cat([up.sigmoid(), down.sigmoid()], 1).cpu().numpy())
            truth.append(labels)
            dates.append(batch_dates)

    probabilities, labels, dates = np.concatenate(predictions), np.concatenate(truth), np.concatenate(dates)
    metrics: dict[str, Any] = {}
    for name, mask in (
        ("all", np.ones(len(labels), dtype=bool)),
        ("early_2025H2", dates < np.datetime64("2026-01-01")),
        ("late_2026H1", dates >= np.datetime64("2026-01-01")),
    ):
        values = []
        for i in range(8):
            p = np.clip(probabilities[mask, i], 1e-7, 1 - 1e-7)
            y = labels[mask, i]
            values.append({
                "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
                "brier": float(np.square(p - y).mean()),
            })
        metrics[name] = {
            "samples": int(mask.sum()),
            "macro_log_loss": float(np.mean([v["log_loss"] for v in values])),
            "macro_brier": float(np.mean([v["brier"] for v in values])),
        }
    torch.save({"model": model.state_dict(), "metrics": metrics}, OUTPUT / "best_model.pt")
    report = {
        "status": "PASS", "purpose": "full temporal training",
        "train_samples": train_total, "validation_samples": validation_total,
        "epochs": 1, "batch_size": BATCH_SIZE, "metrics": metrics,
        "checkpoint": str(OUTPUT / "best_model.pt"),
        "elapsed_seconds": time.monotonic() - started,
    }
    (OUTPUT / "full_training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dashboard({"phase": "complete", "processed_samples": processed,
               "total_samples": train_total, "metrics": metrics})
    log("full_training_complete", **report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
