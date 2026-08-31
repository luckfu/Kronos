"""Independently evaluate one checkpoint on the fixed full validation set."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from dataset import QlibDataset
from model.kronos import Kronos, KronosTokenizer
from train_predictor import evaluate_validation, reset_conditioning, resolve_amp_dtype
from utils.training_utils import set_seed


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = Config().__dict__
    if not config.get("validation_full_only"):
        raise RuntimeError("KRONOS_VALIDATION_FULL_ONLY=1 is required")
    expected_samples = int(config["validation_large_samples"])
    if expected_samples <= 0:
        raise RuntimeError("Full-only evaluation requires a positive sample count")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Formal fixed validation requires CUDA")
    set_seed(config["seed"], 0)

    tokenizer_path = config["finetuned_tokenizer_path"]
    if not os.path.exists(tokenizer_path):
        tokenizer_path = config["pretrained_tokenizer_path"]
    with open(os.path.join(tokenizer_path, "config.json")) as handle:
        tokenizer_kwargs = json.load(handle)
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)
    tokenizer.eval().to(device)

    predictor_path = config["pretrained_predictor_path"]
    with open(os.path.join(predictor_path, "config.json")) as handle:
        model_kwargs = json.load(handle)
    model_kwargs.update({
        "num_sectors": int(config.get("num_sectors", 0)),
        "num_size_buckets": int(config.get("num_size_buckets", 0)),
        "context_layer": int(config.get("context_layer", 0)),
        "use_size_percentile": bool(config.get("use_size_percentile", False)),
        "size_mlp_hidden_dim": int(config.get("size_mlp_hidden_dim", 64)),
    })
    model = Kronos.from_pretrained(predictor_path, **model_kwargs)
    reset_conditioning(model, config)
    model.eval().to(device)

    dataset = QlibDataset("val")
    if len(dataset) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples:,} validation samples, found {len(dataset):,}"
        )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 2),
        pin_memory=True,
        drop_last=False,
    )
    result = {
        "schema_version": 1,
        "task": "independent_fixed_validation",
        "training_performed": False,
        "validation_mode": "full_only",
        "checkpoint": predictor_path,
        "checkpoint_model_sha256": sha256_file(
            os.path.join(predictor_path, "model.safetensors")
        ),
        "validation_manifest": config["fixed_validation_manifest_path"],
        "validation_manifest_sha256": dataset.fixed_validation_manifest_sha256,
        "initialization": {
            "sector_emb": "zero" if config.get("reset_sector_embedding") else "loaded",
            "size_conditioning": (
                "zero" if config.get("reset_size_embedding") else "loaded"
            ),
        },
        "validation_large": evaluate_validation(
            model,
            tokenizer,
            loader,
            device,
            config,
            resolve_amp_dtype(config, device),
            period_names=getattr(dataset, "validation_period_names", {}),
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
