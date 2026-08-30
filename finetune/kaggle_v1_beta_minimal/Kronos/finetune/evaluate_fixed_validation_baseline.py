"""Evaluate an initialized predictor on the signed fixed validation manifest."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    loader_kwargs = {
        "batch_size": config["batch_size"],
        "shuffle": False,
        "num_workers": config.get("num_workers", 2),
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    quick_count = int(dataset.quick_validation_count)
    quick_loader = DataLoader(Subset(dataset, range(quick_count)), **loader_kwargs)
    large_loader = DataLoader(dataset, **loader_kwargs)
    period_names = getattr(dataset, "validation_period_names", {})
    amp_dtype = resolve_amp_dtype(config, device)

    result = {
        "checkpoint": predictor_path,
        "checkpoint_model_sha256": sha256_file(
            os.path.join(predictor_path, "model.safetensors")
        ),
        "validation_manifest": config["fixed_validation_manifest_path"],
        "validation_manifest_sha256": dataset.fixed_validation_manifest_sha256,
        "initialization": {
            "sector_emb": "zero" if config.get("reset_sector_embedding") else "loaded",
            "size_conditioning": "zero" if config.get("reset_size_embedding") else "loaded",
        },
        "quick": evaluate_validation(
            model, tokenizer, quick_loader, device, config, amp_dtype,
            period_names=period_names,
        ),
        "large": evaluate_validation(
            model, tokenizer, large_loader, device, config, amp_dtype,
            period_names=period_names,
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
