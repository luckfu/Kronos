"""Small, reproducible A/B/C GPU experiment for the ModernBERT decision model."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = (20260925, 20260926, 20260927)
TRAIN_SAMPLES = 16384
VALIDATION_SAMPLES = 8192
BATCH_SIZE = 64
EPOCHS = 5
OUTPUT = Path("/kaggle/working/modernbert_decision_e1")
FEATURES = ("open", "high", "low", "close", "volume", "amount")
TARGET_COLUMNS = (
    "up_003", "up_005", "up_008", "up_012",
    "down_003", "down_005", "down_008", "down_012",
)


def log(phase: str, **fields: object) -> None:
    row = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase": phase,
        **fields,
    }
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


def sample_parquet(path: Path, sample_count: int, seed: int) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    total_rows = parquet.metadata.num_rows
    if total_rows < sample_count:
        raise ValueError(f"{path} has only {total_rows} rows")
    groups = parquet.num_row_groups
    sizes = np.asarray(
        [parquet.metadata.row_group(i).num_rows for i in range(groups)],
        dtype=np.int64,
    )
    exact = sizes * sample_count / total_rows
    take = np.floor(exact).astype(np.int64)
    remainder = sample_count - int(take.sum())
    if remainder:
        order = np.argsort(-(exact - take), kind="stable")
        take[order[:remainder]] += 1

    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    columns = [
        "symbol", "start_index", "asof_date", "mfe10", "mae10", *TARGET_COLUMNS
    ]
    for group_id, count in enumerate(take):
        if not count:
            continue
        table = parquet.read_row_group(int(group_id), columns=columns)
        indices = np.sort(
            rng.choice(table.num_rows, size=int(count), replace=False)
        )
        selected.extend(table.take(indices).to_pylist())
    if len(selected) != sample_count:
        raise RuntimeError(f"sample count mismatch: {len(selected)} != {sample_count}")
    return selected


def symbol_universe(path: Path) -> set[str]:
    import pyarrow.parquet as pq

    symbols: set[str] = set()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["symbol"], batch_size=250_000):
        symbols.update(str(value) for value in batch.column(0).to_pylist())
    return symbols


def panel_arrays(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    import pandas as pd

    with path.open("rb") as handle:
        panel = pickle.load(handle)
    if not isinstance(panel, dict):
        raise TypeError(f"{path} is not a symbol-to-DataFrame panel")

    sector_names: set[str] = set()
    for frame in panel.values():
        labels = frame.get("sector", pd.Series("unknown", index=frame.index))
        sector_names.update(labels.astype(str).unique())
    sectors = sorted(sector_names)
    sector_ids = {name: index for index, name in enumerate(sectors)}
    arrays: dict[str, dict[str, Any]] = {}
    for symbol in sorted(panel):
        frame = panel[symbol].sort_index()
        values = frame.loc[:, FEATURES].to_numpy(dtype=np.float32, copy=True)
        sector = frame.get(
            "sector", pd.Series("unknown", index=frame.index)
        ).astype(str).to_numpy()
        size = pd.to_numeric(
            frame.get("size_percentile", pd.Series(0.5, index=frame.index)),
            errors="coerce",
        ).fillna(0.5).clip(0.0, 1.0).to_numpy(dtype=np.float32)
        arrays[str(symbol)] = {
            "values": values,
            "sector": sector,
            "size": size,
            "dates": frame.index.to_numpy(dtype="datetime64[D]"),
        }
    return arrays, sector_ids


def materialize(
    rows: list[dict[str, Any]],
    arrays: dict[str, dict[str, Any]],
    sector_ids: dict[str, int],
) -> tuple[np.ndarray, ...]:
    histories = np.empty((len(rows), 120, 6), dtype=np.float32)
    sectors = np.empty(len(rows), dtype=np.int64)
    sizes = np.empty(len(rows), dtype=np.float32)
    targets = np.empty((len(rows), 8), dtype=np.float32)
    asof_dates = np.empty(len(rows), dtype="datetime64[D]")
    for i, row in enumerate(rows):
        symbol = str(row["symbol"])
        start = int(row["start_index"])
        data = arrays.get(symbol)
        if data is None or start < 0 or start + 130 >= len(data["values"]):
            raise ValueError(f"invalid sampled window: {symbol} start={start}")
        asof_position = start + 119
        asof = np.datetime_as_string(data["dates"][asof_position], unit="D")
        target_date = row["asof_date"]
        if hasattr(target_date, "isoformat"):
            target_date = target_date.isoformat()[:10]
        if asof != str(target_date)[:10]:
            raise ValueError(
                f"sample key date mismatch for {symbol} {start}: {asof} != {target_date}"
            )
        asof_dates[i] = np.datetime64(asof)
        future = data["values"][start + 120:start + 130]
        close = float(data["values"][asof_position, 3])
        mfe10 = float(future[:, 1].max() / close - 1.0)
        mae10 = float(future[:, 2].min() / close - 1.0)
        if not (
            np.isclose(mfe10, float(row["mfe10"]), atol=2e-6)
            and np.isclose(mae10, float(row["mae10"]), atol=2e-6)
        ):
            raise ValueError(f"sample labels do not match panel for {symbol} at {start}")

        history = data["values"][start:start + 120].copy()
        history = (history - history.mean(axis=0)) / (history.std(axis=0) + 1e-5)
        np.clip(history, -5.0, 5.0, out=history)
        if not np.isfinite(history).all():
            raise ValueError(f"non-finite history for {symbol} at {start}")
        histories[i] = history
        sector_name = str(data["sector"][asof_position])
        sectors[i] = sector_ids.get(sector_name, len(sector_ids))
        sizes[i] = data["size"][asof_position]
        targets[i] = [row[name] for name in TARGET_COLUMNS]
    return histories, sectors, sizes, targets, asof_dates


def build_model(mode: str, sector_count: int, torch: Any, nn: Any, F: Any, ModernBertConfig: Any, ModernBertModel: Any):
    class DecisionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = 256
            self.mode = mode
            self.s1_embed = nn.Embedding(1024, hidden // 2)
            self.s2_embed = nn.Embedding(1024, hidden // 2)
            self.market_fusion = nn.Linear(hidden, hidden)
            self.market_gate = nn.Parameter(torch.zeros(1))
            self.mask_market = nn.Parameter(torch.zeros(1, 1, hidden))
            self.sector_embed = nn.Embedding(sector_count + 1, hidden)
            self.size_encoder = nn.Sequential(
                nn.Linear(1, 32), nn.GELU(), nn.Linear(32, hidden)
            )
            self.cond_token = nn.Parameter(torch.zeros(1, 1, hidden))
            config = ModernBertConfig(
                vocab_size=1024,
                hidden_size=hidden,
                intermediate_size=512,
                num_hidden_layers=4,
                num_attention_heads=4,
                max_position_embeddings=128,
                pad_token_id=0,
                attention_dropout=0.0,
                embedding_dropout=0.0,
                mlp_dropout=0.0,
            )
            self.backbone = ModernBertModel(config)
            self.up_base = nn.Linear(hidden, 1)
            self.up_delta = nn.Linear(hidden, 3)
            self.down_base = nn.Linear(hidden, 1)
            self.down_delta = nn.Linear(hidden, 3)

        @staticmethod
        def ordinal_logits(base: Any, deltas: Any) -> Any:
            descending = base - torch.cumsum(F.softplus(deltas), dim=-1)
            return torch.cat([base, descending], dim=-1)

        def forward(self, s1: Any, s2: Any, size: Any, sector: Any):
            condition = (
                self.cond_token
                + self.sector_embed(sector).unsqueeze(1)
                + self.size_encoder(size).unsqueeze(1)
            )
            if self.mode in ("A", "A_gated"):
                market = self.market_fusion(
                    torch.cat([self.s1_embed(s1), self.s2_embed(s2)], dim=-1)
                )
                if self.mode == "A_gated":
                    market = market * self.market_gate
                sequence = torch.cat([condition, market], dim=1)
            elif self.mode == "B":
                market = self.mask_market.expand(s1.shape[0], 120, -1)
                sequence = torch.cat([condition, market], dim=1)
            else:
                sequence = condition
            mask = torch.ones(sequence.shape[:2], dtype=torch.long, device=sequence.device)
            pooled = self.backbone(
                inputs_embeds=sequence, attention_mask=mask
            ).last_hidden_state[:, 0]
            return (
                self.ordinal_logits(self.up_base(pooled), self.up_delta(pooled)),
                self.ordinal_logits(self.down_base(pooled), self.down_delta(pooled)),
            )

    return DecisionModel()


def binary_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    eps = 1e-7
    p = np.clip(probabilities.astype(np.float64), eps, 1.0 - eps)
    y = labels.astype(np.float64)
    logloss = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    brier = np.square(p - y).mean()
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        selected = (p >= lower) & (p < lower + 0.1)
        if selected.any():
            ece += selected.mean() * abs(p[selected].mean() - y[selected].mean())
    return {
        "log_loss": float(logloss),
        "brier": float(brier),
        "ece_10_bins": float(ece),
        "positive_count": int(y.sum()),
        "prevalence": float(y.mean()),
    }


def train_experiment(
    mode: str,
    seed: int,
    train_cache: tuple[np.ndarray, ...],
    validation_cache: tuple[np.ndarray, ...],
    sector_count: int,
    device: Any,
    torch: Any,
    nn: Any,
    F: Any,
    ModernBertConfig: Any,
    ModernBertModel: Any,
) -> dict[str, Any]:
    train_s1, train_s2, train_sector, train_size, train_y, _ = train_cache
    val_s1, val_s2, val_sector, val_size, val_y, val_dates = validation_cache
    model = build_model(
        mode, sector_count, torch, nn, F, ModernBertConfig, ModernBertModel
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    step_durations: list[float] = []
    model.train()
    start_time = time.monotonic()
    for epoch in range(EPOCHS):
        order = rng.permutation(len(train_y))
        for offset in range(0, len(order), BATCH_SIZE):
            indices = order[offset:offset + BATCH_SIZE]
            s1 = torch.as_tensor(train_s1[indices], device=device, dtype=torch.long)
            s2 = torch.as_tensor(train_s2[indices], device=device, dtype=torch.long)
            sector = torch.as_tensor(train_sector[indices], device=device, dtype=torch.long)
            size = torch.as_tensor(train_size[indices, None], device=device)
            target = torch.as_tensor(train_y[indices], device=device)
            optimizer.zero_grad(set_to_none=True)
            tick = time.monotonic()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                up, down = model(s1, s2, size, sector)
                loss = F.binary_cross_entropy_with_logits(up, target[:, :4])
                loss = loss + F.binary_cross_entropy_with_logits(down, target[:, 4:])
            if not torch.isfinite(loss):
                raise RuntimeError(f"{mode}: non-finite loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            step_durations.append(time.monotonic() - tick)
            losses.append(float(loss.detach().cpu()))
        log(
            "epoch_complete",
            mode=mode,
            epoch=epoch + 1,
            mean_train_loss=float(np.mean(losses)),
            steps=len(losses),
        )

    model.eval()
    predictions = []
    with torch.no_grad():
        for offset in range(0, len(val_y), BATCH_SIZE):
            sl = slice(offset, offset + BATCH_SIZE)
            s1 = torch.as_tensor(val_s1[sl], device=device, dtype=torch.long)
            s2 = torch.as_tensor(val_s2[sl], device=device, dtype=torch.long)
            sector = torch.as_tensor(val_sector[sl], device=device, dtype=torch.long)
            size = torch.as_tensor(val_size[sl, None], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                up, down = model(s1, s2, size, sector)
            predictions.append(torch.cat([up.sigmoid(), down.sigmoid()], dim=1).float().cpu().numpy())
    probabilities = np.concatenate(predictions)
    def metrics_for_mask(mask: np.ndarray) -> dict[str, Any]:
        selected_probabilities = probabilities[mask]
        selected_labels = val_y[mask]
        per_target = [
            binary_metrics(selected_probabilities[:, i], selected_labels[:, i])
            for i in range(selected_labels.shape[1])
        ]
        return {
            "samples": int(mask.sum()),
            "macro_log_loss": float(np.mean([m["log_loss"] for m in per_target])),
            "macro_brier": float(np.mean([m["brier"] for m in per_target])),
            "macro_ece_10_bins": float(np.mean([m["ece_10_bins"] for m in per_target])),
            "per_target": [
                {"target": name, **metrics}
                for name, metrics in zip(TARGET_COLUMNS, per_target)
            ],
        }

    per_target = [
        binary_metrics(probabilities[:, i], val_y[:, i])
        for i in range(val_y.shape[1])
    ]
    early_mask = val_dates < np.datetime64("2026-01-01")
    late_mask = ~early_mask
    elapsed = time.monotonic() - start_time
    result = {
        "mode": mode,
        "train_loss": float(np.mean(losses)),
        "validation": {
            "macro_log_loss": float(np.mean([m["log_loss"] for m in per_target])),
            "macro_brier": float(np.mean([m["brier"] for m in per_target])),
            "macro_ece_10_bins": float(np.mean([m["ece_10_bins"] for m in per_target])),
            "per_target": [
                {"target": name, **metrics}
                for name, metrics in zip(TARGET_COLUMNS, per_target)
            ],
            "early_2025H2": metrics_for_mask(early_mask),
            "late_2026H1": metrics_for_mask(late_mask),
        },
        "steps": len(losses),
        "median_step_seconds": float(np.median(step_durations)),
        "p95_step_seconds": float(np.quantile(step_durations, 0.95)),
        "train_and_validation_seconds": elapsed,
        "samples_per_second": float(
            (len(train_y) * EPOCHS + len(val_y)) / max(elapsed, 1e-6)
        ),
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pyarrow.parquet as pq

    if not torch.cuda.is_available():
        raise RuntimeError("E1 requires a Kaggle GPU")
    random.seed(SEEDS[0])
    np.random.seed(SEEDS[0])
    torch.manual_seed(SEEDS[0])
    torch.cuda.manual_seed_all(SEEDS[0])
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    log(
        "runtime",
        gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__,
        transformers=importlib.metadata.version("transformers"),
        seeds=list(SEEDS),
    )

    train_panel_path = find_one("**/processed_datasets/train_data.pkl")
    validation_panel_path = find_one("**/processed_datasets/val_data.pkl")
    train_targets_path = find_one("**/train_targets.parquet")
    validation_targets_path = find_one("**/validation_targets.parquet")
    targets_manifest_path = train_targets_path.parent / "decision_targets_manifest.json"
    source_manifest_path = train_panel_path.parent.parent / "data_manifest.json"
    target_manifest = json.loads(targets_manifest_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if target_manifest["source_dataset"]["dataset_id"] != (
        "luckfu/a-share-120d-temporal-symbol-holdout"
    ):
        raise ValueError("target sidecar is for a different source dataset")
    if int(target_manifest["window"]["source_window"]) != 131:
        raise ValueError("target sidecar does not use the 120+10+1 contract")
    for path, key in (
        (train_panel_path, "train_panel_sha256"),
        (validation_panel_path, "validation_panel_sha256"),
        (source_manifest_path, "data_manifest_sha256"),
    ):
        if sha256_file(path) != target_manifest["source_dataset"][key]:
            raise ValueError(f"source hash mismatch for {path}")
    if pq.ParquetFile(train_targets_path).metadata.num_rows != int(
        target_manifest["splits"]["train"]["records"]
    ):
        raise ValueError("training target row count differs from manifest")
    if pq.ParquetFile(validation_targets_path).metadata.num_rows != int(
        target_manifest["splits"]["validation"]["records"]
    ):
        raise ValueError("validation target row count differs from manifest")
    log("dataset_contract_verified")

    train_target_symbols = symbol_universe(train_targets_path)
    validation_target_symbols = symbol_universe(validation_targets_path)
    overlapping_target_symbols = train_target_symbols & validation_target_symbols
    train_rows = sample_parquet(train_targets_path, TRAIN_SAMPLES, SEEDS[0])
    validation_rows = sample_parquet(
        validation_targets_path, VALIDATION_SAMPLES, SEEDS[0] + 1
    )
    train_panel, train_sector_ids = panel_arrays(train_panel_path)
    validation_panel, _ = panel_arrays(validation_panel_path)
    panel_symbol_overlap = set(train_panel) & set(validation_panel)
    histories, sectors, sizes, labels, train_dates = materialize(
        train_rows, train_panel, train_sector_ids
    )
    val_histories, val_sectors, val_sizes, val_labels, val_dates = materialize(
        validation_rows, validation_panel, train_sector_ids
    )
    train_asof_dates = [
        str(row["asof_date"])[:10] for row in train_rows
    ]
    validation_asof_dates = [
        str(row["asof_date"])[:10] for row in validation_rows
    ]
    if max(train_asof_dates) > "2024-12-31":
        raise ValueError("sampled training labels extend beyond the training cutoff")
    if min(validation_asof_dates) < "2025-07-01":
        raise ValueError("sampled validation labels precede the validation cutoff")
    log(
        "sampled_data_ready",
        train=len(train_rows),
        validation=len(validation_rows),
        train_symbols=len(set(str(row["symbol"]) for row in train_rows)),
        validation_symbols=len(set(str(row["symbol"]) for row in validation_rows)),
        train_validation_label_symbol_overlap=len(overlapping_target_symbols),
        train_validation_panel_symbol_overlap=len(panel_symbol_overlap),
        temporal_split=True,
    )

    repo_path = Path("/kaggle/working/Kronos")
    if not (repo_path / "model" / "kronos.py").is_file():
        subprocess.run(
            [
                "git", "clone", "--depth", "1", "--branch", "master",
                "https://github.com/luckfu/Kronos.git", str(repo_path),
            ],
            check=True,
        )
    sys.path.insert(0, str(repo_path))
    from model import KronosTokenizer
    from huggingface_hub import snapshot_download
    try:
        from transformers import ModernBertConfig, ModernBertModel
    except (ImportError, RuntimeError):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
             "transformers==4.57.0"],
            check=True,
        )
        from transformers import ModernBertConfig, ModernBertModel

    tokenizer_path = Path(
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base")
    )
    tokenizer_hash = sha256_file(tokenizer_path / "model.safetensors")
    expected_hash = "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
    if tokenizer_hash != expected_hash:
        raise ValueError(f"unexpected tokenizer weights: {tokenizer_hash}")
    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path)).to(device).eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)

    cache_path = OUTPUT / "sample_token_cache.npz"
    if cache_path.exists():
        cache = np.load(cache_path)
        train_s1, train_s2 = cache["train_s1"], cache["train_s2"]
        val_s1, val_s2 = cache["val_s1"], cache["val_s2"]
    else:
        def encode_all(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            s1_parts, s2_parts = [], []
            with torch.no_grad():
                for offset in range(0, len(values), 128):
                    batch = torch.from_numpy(values[offset:offset + 128]).to(device)
                    s1, s2 = tokenizer.encode(batch.float(), half=True)
                    s1_parts.append(s1.long().cpu().numpy().astype(np.uint16))
                    s2_parts.append(s2.long().cpu().numpy().astype(np.uint16))
            return np.concatenate(s1_parts), np.concatenate(s2_parts)

        token_start = time.monotonic()
        train_s1, train_s2 = encode_all(histories)
        val_s1, val_s2 = encode_all(val_histories)
        np.savez_compressed(
            cache_path, train_s1=train_s1, train_s2=train_s2,
            val_s1=val_s1, val_s2=val_s2,
        )
        log("token_cache_ready", seconds=time.monotonic() - token_start, path=cache_path)

    train_cache = (train_s1, train_s2, sectors, sizes, labels, train_dates)
    validation_cache = (val_s1, val_s2, val_sectors, val_sizes, val_labels, val_dates)
    validation_prevalence = val_labels.mean(axis=0)
    baseline_metrics = [
        binary_metrics(
            np.full(len(val_labels), validation_prevalence[i], dtype=np.float32),
            val_labels[:, i],
        )
        for i in range(val_labels.shape[1])
    ]
    baseline = {
        "mode": "constant_validation_prevalence",
        "validation": {
            "macro_log_loss": float(np.mean([m["log_loss"] for m in baseline_metrics])),
            "macro_brier": float(np.mean([m["brier"] for m in baseline_metrics])),
            "macro_ece_10_bins": float(np.mean([m["ece_10_bins"] for m in baseline_metrics])),
            "per_target": [
                {"target": name, **metrics}
                for name, metrics in zip(TARGET_COLUMNS, baseline_metrics)
            ],
        },
    }
    results = []
    for seed in SEEDS:
        for mode in ("A_gated", "C"):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            log("experiment_started", mode=mode, seed=seed)
            result = train_experiment(
                mode, seed, train_cache, validation_cache, len(train_sector_ids),
                device, torch, nn, F, ModernBertConfig, ModernBertModel,
            )
            result["seed"] = seed
            results.append(result)
            log("experiment_complete", **result)

    report = {
        "status": "PASS",
        "purpose": "E1 small-scale GPU feasibility and input ablation",
        "data": {
            "train_samples": TRAIN_SAMPLES,
            "validation_samples": VALIDATION_SAMPLES,
            "train_target_rows": int(target_manifest["splits"]["train"]["records"]),
            "validation_target_rows": int(target_manifest["splits"]["validation"]["records"]),
            "lookback": 120,
            "future_label_days": 10,
            "source_window": 131,
            "tokenizer_sha256": tokenizer_hash,
            "train_panel_sha256": sha256_file(train_panel_path),
            "validation_panel_sha256": sha256_file(validation_panel_path),
            "train_target_symbol_count": len(train_target_symbols),
            "validation_target_symbol_count": len(validation_target_symbols),
            "train_validation_target_symbol_overlap": len(overlapping_target_symbols),
            "train_validation_panel_symbol_overlap": len(panel_symbol_overlap),
            "temporal_split": True,
            "training_labels_end": target_manifest["split_policy"]["train_signal_end"],
            "validation_labels_start": target_manifest["split_policy"][
                "validation_signal_start"
            ],
            "sector_vocabulary": sorted(train_sector_ids, key=train_sector_ids.get),
        },
        "configuration": {
            "seeds": list(SEEDS),
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "hidden_size": 256,
            "layers": 4,
            "heads": 4,
            "optimizer": "AdamW",
            "learning_rate": 1e-4,
            "thresholds": [0.03, 0.05, 0.08, 0.12],
            "modes": {
                "A_gated": "zero-gated Kronos s1/s2 tokens + size and sector condition",
                "C": "size and sector condition token only",
            },
        },
        "baseline": baseline,
        "experiments": results,
    }
    (OUTPUT / "e1_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    log("e1_passed", report=str(OUTPUT / "e1_report.json"))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
