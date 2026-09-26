"""Kaggle smoke test for ModernBERT decision data, tokenizer, and one GPU step."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


OUTPUT = Path("/kaggle/working/modernbert_decision_smoke")
OUTPUT.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT / "run.log"
REPORT_PATH = OUTPUT / "smoke_report.json"
sys.stdout.reconfigure(line_buffering=True)


def log(phase: str, **fields: object) -> None:
    row = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "phase": phase,
        **fields,
    }
    line = json.dumps(row, ensure_ascii=False, default=str)
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
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


def rows_from_spread_row_groups(path: Path, count: int) -> list[dict[str, object]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    group_count = parquet.num_row_groups
    if group_count < 1 or parquet.metadata.num_rows < count:
        raise RuntimeError(f"target parquet is empty or too small: {path}")
    group_ids = np.unique(
        np.linspace(0, group_count - 1, min(count, group_count), dtype=np.int64)
    )
    rows = []
    for group_id in group_ids:
        table = parquet.read_row_group(
            int(group_id),
            columns=[
                "symbol",
                "start_index",
                "asof_date",
                "mfe10",
                "mae10",
                "up_003",
                "up_005",
                "up_008",
                "up_012",
                "down_003",
                "down_005",
                "down_008",
                "down_012",
            ],
        )
        rows.append(table.slice(0, 1).to_pylist()[0])
    return rows


def as_date_text(value: object) -> str:
    if hasattr(value, "isoformat"):
        return str(value.isoformat())[:10]
    return str(value)[:10]


def main() -> None:
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    log("runtime", python=sys.version, torch=torch.__version__)
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is not available")
    device = torch.device("cuda")
    log(
        "gpu_ready",
        device=torch.cuda.get_device_name(0),
        cuda=torch.version.cuda,
        count=torch.cuda.device_count(),
    )

    train_panel_path = find_one("**/processed_datasets/train_data.pkl")
    validation_panel_path = find_one("**/processed_datasets/val_data.pkl")
    train_targets_path = find_one("**/train_targets.parquet")
    validation_targets_path = find_one("**/validation_targets.parquet")
    targets_manifest_path = train_targets_path.parent / "decision_targets_manifest.json"
    source_manifest_path = train_panel_path.parent.parent / "data_manifest.json"
    required = [
        targets_manifest_path,
        source_manifest_path,
        validation_panel_path,
        validation_targets_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing mounted dataset files: {missing}")
    targets_manifest = json.loads(targets_manifest_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if targets_manifest["source_dataset"]["dataset_id"] != (
        "luckfu/a-share-120d-temporal-symbol-holdout"
    ):
        raise RuntimeError("target sidecar points to a different source dataset")
    if int(targets_manifest["window"]["source_window"]) != 131:
        raise RuntimeError("target sidecar does not use the 120+10+1 contract")

    log("hash_verification_started")
    train_sha = sha256_file(train_panel_path)
    validation_sha = sha256_file(validation_panel_path)
    data_manifest_sha = sha256_file(source_manifest_path)
    expected = targets_manifest["source_dataset"]
    actual_hashes = {
        "train_panel_sha256": train_sha,
        "validation_panel_sha256": validation_sha,
        "data_manifest_sha256": data_manifest_sha,
    }
    for name, value in actual_hashes.items():
        if value != expected[name]:
            raise RuntimeError(
                f"{name} mismatch: sidecar={expected[name]} mounted={value}"
            )

    import pyarrow.parquet as pq

    train_parquet = pq.ParquetFile(train_targets_path)
    validation_parquet = pq.ParquetFile(validation_targets_path)
    train_expected_rows = int(targets_manifest["splits"]["train"]["records"])
    validation_expected_rows = int(
        targets_manifest["splits"]["validation"]["records"]
    )
    if train_parquet.metadata.num_rows != train_expected_rows:
        raise RuntimeError("training target row count does not match manifest")
    if validation_parquet.metadata.num_rows != validation_expected_rows:
        raise RuntimeError("validation target row count does not match manifest")
    log(
        "data_contract_verified",
        train_rows=train_parquet.metadata.num_rows,
        validation_rows=validation_parquet.metadata.num_rows,
        lookback=targets_manifest["window"]["lookback"],
        horizon=targets_manifest["window"]["horizon"],
        panel_hashes=actual_hashes,
    )

    log("panel_load_started")
    with train_panel_path.open("rb") as handle:
        train_panel = pickle.load(handle)
    with validation_panel_path.open("rb") as handle:
        validation_panel = pickle.load(handle)
    if not isinstance(train_panel, dict) or not isinstance(validation_panel, dict):
        raise TypeError("mounted panels must be symbol-to-DataFrame dictionaries")

    samples = rows_from_spread_row_groups(train_targets_path, count=8)
    histories = []
    sectors = []
    sizes = []
    for target in samples:
        symbol = str(target["symbol"])
        start = int(target["start_index"])
        frame = train_panel.get(symbol)
        if frame is None:
            raise RuntimeError(f"target symbol is absent from training panel: {symbol}")
        frame = frame.sort_index()
        asof_position = start + 119
        if start < 0 or asof_position + 11 >= len(frame):
            raise RuntimeError(f"invalid 131-row target window for {symbol} start={start}")
        asof = frame.index[asof_position]
        if as_date_text(asof) != as_date_text(target["asof_date"]):
            raise RuntimeError(f"target key date mismatch for {symbol} start={start}")

        future = frame.iloc[asof_position + 1:asof_position + 11]
        close = float(frame.iloc[asof_position]["close"])
        mfe10 = float(future["high"].max()) / close - 1.0
        mae10 = float(future["low"].min()) / close - 1.0
        if not np.isclose(mfe10, float(target["mfe10"]), atol=2e-6):
            raise RuntimeError(f"MFE label mismatch for {symbol} start={start}")
        if not np.isclose(mae10, float(target["mae10"]), atol=2e-6):
            raise RuntimeError(f"MAE label mismatch for {symbol} start={start}")

        history = frame.iloc[start:start + 120][
            ["open", "high", "low", "close", "volume", "amount"]
        ].to_numpy(dtype=np.float32)
        mean = history.mean(axis=0)
        std = history.std(axis=0)
        history = np.clip((history - mean) / (std + 1e-5), -5.0, 5.0)
        if not np.isfinite(history).all():
            raise RuntimeError(f"normalized history is non-finite for {symbol}")
        histories.append(history.astype(np.float32))
        asof_row = frame.iloc[asof_position]
        sectors.append(str(asof_row.get("sector", "unknown")))
        sizes.append(float(asof_row.get("size_percentile", 0.5)))

    sector_to_id = {name: idx for idx, name in enumerate(sorted(set(sectors)))}
    sector_ids = torch.tensor([sector_to_id[name] for name in sectors], dtype=torch.long)
    history_batch = torch.from_numpy(np.stack(histories)).to(device)
    size_batch = torch.tensor(sizes, dtype=torch.float32, device=device).unsqueeze(-1)
    up_target = torch.tensor(
        [[int(row[f"up_{n:03d}"]) for n in (3, 5, 8, 12)] for row in samples],
        dtype=torch.float32,
        device=device,
    )
    down_target = torch.tensor(
        [[int(row[f"down_{n:03d}"]) for n in (3, 5, 8, 12)] for row in samples],
        dtype=torch.float32,
        device=device,
    )
    sector_ids = sector_ids.to(device)

    repo_path = Path("/kaggle/working/Kronos")
    if not (repo_path / "model" / "kronos.py").is_file():
        log("tokenizer_source_clone_started")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "master",
                "https://github.com/luckfu/Kronos.git",
                str(repo_path),
            ],
            check=True,
        )
    sys.path.insert(0, str(repo_path))
    from model import KronosTokenizer

    try:
        from transformers import ModernBertConfig, ModernBertModel
    except (ImportError, RuntimeError):
        log("transformers_install_started", version="4.57.0")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--progress-bar",
                "off",
                "transformers==4.57.0",
            ],
            check=True,
        )
        from transformers import ModernBertConfig, ModernBertModel

    log("tokenizer_download_started", repo="NeoQuasar/Kronos-Tokenizer-base")
    from huggingface_hub import snapshot_download

    tokenizer_path = Path(
        snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base")
    )
    tokenizer_sha = sha256_file(tokenizer_path / "model.safetensors")
    expected_tokenizer_sha = (
        "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
    )
    if tokenizer_sha != expected_tokenizer_sha:
        raise RuntimeError(
            f"unexpected Kronos tokenizer weight hash: {tokenizer_sha}"
        )
    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path)).to(device).eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    if tokenizer.training or any(p.requires_grad for p in tokenizer.parameters()):
        raise RuntimeError("Kronos tokenizer is not frozen")

    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
        encoded = tokenizer.encode(history_batch.float(), half=True)
        s1, s2 = (part.long() for part in encoded)
    if tuple(s1.shape) != (len(samples), 120) or tuple(s2.shape) != (
        len(samples),
        120,
    ):
        raise RuntimeError(f"unexpected token shapes: {s1.shape}, {s2.shape}")
    if int(s1.min()) < 0 or int(s1.max()) >= 1024:
        raise RuntimeError("s1 IDs are outside [0, 1023]")
    if int(s2.min()) < 0 or int(s2.max()) >= 1024:
        raise RuntimeError("s2 IDs are outside [0, 1023]")
    log(
        "tokenization_verified",
        batch=len(samples),
        s1_shape=list(s1.shape),
        s2_shape=list(s2.shape),
        s1_range=[int(s1.min()), int(s1.max())],
        s2_range=[int(s2.min()), int(s2.max())],
        tokenizer_sha256=tokenizer_sha,
    )

    class DecisionSmokeModel(nn.Module):
        def __init__(self, sector_count: int) -> None:
            super().__init__()
            hidden = 256
            self.s1_embed = nn.Embedding(1024, hidden // 2)
            self.s2_embed = nn.Embedding(1024, hidden // 2)
            self.market_fusion = nn.Linear(hidden, hidden)
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
        def ordinal_logits(base: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
            lower = base - torch.cumsum(F.softplus(deltas), dim=-1)
            return torch.cat([base, lower], dim=-1)

        def forward(
            self,
            token_s1: torch.Tensor,
            token_s2: torch.Tensor,
            size: torch.Tensor,
            sector: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            market = self.market_fusion(
                torch.cat([self.s1_embed(token_s1), self.s2_embed(token_s2)], dim=-1)
            )
            condition = (
                self.cond_token
                + self.sector_embed(sector).unsqueeze(1)
                + self.size_encoder(size).unsqueeze(1)
            )
            sequence = torch.cat([condition, market], dim=1)
            attention_mask = torch.ones(
                sequence.shape[:2], dtype=torch.long, device=sequence.device
            )
            pooled = self.backbone(
                inputs_embeds=sequence,
                attention_mask=attention_mask,
            ).last_hidden_state[:, 0]
            up = self.ordinal_logits(self.up_base(pooled), self.up_delta(pooled))
            down = self.ordinal_logits(
                self.down_base(pooled), self.down_delta(pooled)
            )
            return up, down

    torch.manual_seed(20260925)
    model = DecisionSmokeModel(len(sector_to_id)).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    start = time.monotonic()
    up_logits, down_logits = model(s1, s2, size_batch, sector_ids)
    loss = F.binary_cross_entropy_with_logits(up_logits, up_target)
    loss = loss + F.binary_cross_entropy_with_logits(down_logits, down_target)
    if not torch.isfinite(loss):
        raise RuntimeError("smoke loss is non-finite")
    loss.backward()
    trainable_grad = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    if not trainable_grad:
        raise RuntimeError("ModernBERT decision model received no finite gradients")
    if any(parameter.grad is not None for parameter in tokenizer.parameters()):
        raise RuntimeError("frozen tokenizer unexpectedly received gradients")
    optimizer.step()

    up_prob = up_logits.detach().sigmoid()
    down_prob = down_logits.detach().sigmoid()
    if not torch.all(up_prob[:, :-1] >= up_prob[:, 1:]):
        raise RuntimeError("upside probabilities are not threshold-monotonic")
    if not torch.all(down_prob[:, :-1] >= down_prob[:, 1:]):
        raise RuntimeError("downside probabilities are not threshold-monotonic")
    if not (
        torch.isfinite(up_prob).all()
        and torch.isfinite(down_prob).all()
        and torch.all((up_prob >= 0) & (up_prob <= 1))
        and torch.all((down_prob >= 0) & (down_prob <= 1))
    ):
        raise RuntimeError("decision probabilities are invalid")

    report = {
        "status": "PASS",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "train_samples": len(samples),
        "target_train_rows": train_parquet.metadata.num_rows,
        "target_validation_rows": validation_parquet.metadata.num_rows,
        "window": {"lookback": 120, "future_label_days": 10, "source_rows": 131},
        "sample_keys": [
            {
                "symbol": row["symbol"],
                "start_index": int(row["start_index"]),
                "asof_date": as_date_text(row["asof_date"]),
            }
            for row in samples
        ],
        "tokenizer_sha256": tokenizer_sha,
        "modernbert": {
            "hidden_size": 256,
            "layers": 4,
            "heads": 4,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "loss": float(loss.detach().cpu()),
        "up_probability_range": [
            float(up_prob.min().cpu()),
            float(up_prob.max().cpu()),
        ],
        "down_probability_range": [
            float(down_prob.min().cpu()),
            float(down_prob.max().cpu()),
        ],
        "optimizer_step_seconds": time.monotonic() - start,
        "source_panel_sha256": actual_hashes,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log("smoke_passed", report=str(REPORT_PATH), loss=report["loss"])
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
