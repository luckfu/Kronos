"""Offline CE-vs-Path gradient norms and cosine on fixed validation batches."""
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SOURCE_COMMIT = "f617ec0cf18406dc0be287055a6c1b5e8125c023"
OUTPUT = Path("/kaggle/working/stage3_gradient_diagnostic")
INPUT = Path("/kaggle/input")
LAMBDA_PATH = 0.05


def find_one(pattern):
    matches = list(INPUT.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clone_source():
    for attempt in range(1, 4):
        repo = Path(tempfile.mkdtemp(prefix="kronos-gradient-source-")) / "repo"
        try:
            subprocess.run([
                "git", "clone", "--depth", "8", "--branch", "master",
                "https://github.com/luckfu/Kronos.git", str(repo),
            ], check=True)
            subprocess.run(["git", "checkout", "--detach", SOURCE_COMMIT], cwd=repo, check=True)
            actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
            if actual != SOURCE_COMMIT:
                raise RuntimeError(actual)
            return repo
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * attempt)


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def group_indices(named_parameters):
    groups = {
        "all": [], "backbone": [], "dependency": [], "s1_head": [],
        "s2_head": [], "conditioning": [],
    }
    for index, (name, _) in enumerate(named_parameters):
        groups["all"].append(index)
        if name.startswith("transformer.") or name.startswith("embedding.") or name.startswith("time_emb.") or name.startswith("norm."):
            groups["backbone"].append(index)
        elif name.startswith("dep_layer."):
            groups["dependency"].append(index)
        elif name.startswith("head.proj_s1."):
            groups["s1_head"].append(index)
        elif name.startswith("head.proj_s2."):
            groups["s2_head"].append(index)
        elif name.startswith("sector_emb.") or name.startswith("size_emb.") or name.startswith("size_mlp."):
            groups["conditioning"].append(index)
    return groups


def gradient_stats(ce_grads, path_grads, groups, normalization_denominator):
    import torch
    scale = LAMBDA_PATH / max(float(normalization_denominator), 1e-6)
    result = {}
    for name, indices in groups.items():
        ce_sq = path_sq = dot = 0.0
        parameter_count = 0
        for index in indices:
            ce, path = ce_grads[index], path_grads[index]
            if ce is None and path is None:
                continue
            if ce is not None:
                ce = ce.detach().float()
                ce_sq += float(torch.sum(ce * ce))
                parameter_count += ce.numel()
            if path is not None:
                path = path.detach().float()
                path_sq += float(torch.sum(path * path))
            if ce is not None and path is not None:
                dot += float(torch.sum(ce * path))
        ce_norm = math.sqrt(ce_sq)
        raw_path_norm = math.sqrt(path_sq)
        weighted_path_norm = abs(scale) * raw_path_norm
        denom = ce_norm * raw_path_norm
        cosine = dot / denom if denom else float("nan")
        total_sq = ce_sq + scale * scale * path_sq + 2.0 * scale * dot
        result[name] = {
            "parameters_with_gradient": parameter_count,
            "ce_norm": ce_norm,
            "raw_path_norm": raw_path_norm,
            "ce_path_cosine": finite(cosine),
            "weighted_normalized_path_norm": weighted_path_norm,
            "weighted_path_to_ce_norm_ratio": finite(weighted_path_norm / ce_norm) if ce_norm else None,
            "implied_total_norm": math.sqrt(max(0.0, total_sq)),
        }
    return result


def losses(model, tokenizer, batch):
    import torch
    from finetune.stage3_path_alignment import PathAlignmentConfig, compute_path_alignment_loss

    x, stamp, sector_id, size_percentile = batch
    with torch.no_grad():
        s1, s2 = tokenizer.encode(x.float(), half=True)
    context = model.encode_context(
        s1[:, :-1], s2[:, :-1], stamp[:, :-1],
        sector_id=sector_id, size_percentile=size_percentile,
    )
    start, end = 119, 129
    target_slice = slice(120, 130)
    logits1 = model.predict_s1(context[:, start:end])
    logits2 = model.predict_s2(context, s1[:, 1:], is_causal=True)[:, start:end]
    ce = model.head.compute_loss(logits1, logits2, s1[:, target_slice], s2[:, target_slice])[0]
    raw_path, details = compute_path_alignment_loss(
        model, tokenizer, context, logits1, x[:, target_slice],
        PathAlignmentConfig(weight=1.0), None,
        position_start=start, is_causal=True,
    )
    return ce, raw_path, details


def main():
    os.environ.update(
        PYTHONUNBUFFERED="1",
        KRONOS_LOOKBACK_WINDOW="120",
        KRONOS_PREDICT_WINDOW="10",
        KRONOS_USE_SIZE_PERCENTILE="1",
        KRONOS_NUM_SIZE_BUCKETS="0",
        KRONOS_VALIDATION_SAMPLES="0",
        KRONOS_VAL_SIGNAL_START="2025-07-01",
        KRONOS_VAL_SIGNAL_END="2026-07-02",
        KRONOS_COVERAGE_SEED="20260915",
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    repo = clone_source()
    sys.path.insert(0, str(repo))

    data_manifests = [p for p in INPUT.glob("**/data_manifest.json")
                      if (p.parent / "processed_datasets/val_data.pkl").is_file()]
    if len(data_manifests) != 1:
        raise RuntimeError(f"Expected one training data package, found {data_manifests}")
    data_root = data_manifests[0].parent
    os.environ["KRONOS_DATASET_PATH"] = str(data_root / "processed_datasets")
    os.environ["KRONOS_METADATA_PATH"] = str(data_root / "asset_metadata.csv")

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from finetune.dataset import QlibDataset
    from model import Kronos, KronosTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required")
    torch.manual_seed(20260915)
    torch.cuda.manual_seed_all(20260915)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0")

    c2_file = find_one("**/small_0.1_stage2_cosine_refinement/checkpoints/best_model/model.safetensors")
    c3_best_file = find_one("**/stage3_joint_path_smoke/checkpoints/best_model/model.safetensors")
    c3_last_file = find_one("**/stage3_joint_path_smoke/checkpoints/last_model/model.safetensors")
    c3_state_file = find_one("**/stage3_joint_path_smoke/checkpoints/last_state.pt")
    tokenizer_file = find_one("**/Kronos-Tokenizer-base/model.safetensors")
    checkpoint_hashes = {
        "c2_best": sha256(c2_file),
        "c3_best": sha256(c3_best_file),
        "c3_last": sha256(c3_last_file),
    }
    state = torch.load(c3_state_file, map_location="cpu", weights_only=True)
    ema_value = float(torch.as_tensor(state["path_ema"]["value"]))

    dataset = QlibDataset("val")
    positions = np.linspace(0, len(dataset) - 1, num=256, dtype=np.int64).tolist()
    loader = DataLoader(Subset(dataset, positions), batch_size=32, shuffle=False, num_workers=0)
    fixed_batches = []
    for raw in loader:
        fixed_batches.append((
            raw[0].to(device), raw[1].to(device), raw[2].to(device), raw[4].to(device),
        ))
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_file.parent).to(device).eval()

    model_paths = {"c2_best_segment_179": c2_file.parent, "c3_last_segment_15": c3_last_file.parent}
    if checkpoint_hashes["c3_best"] != checkpoint_hashes["c3_last"]:
        model_paths["c3_best"] = c3_best_file.parent
    results = {}
    for label, path in model_paths.items():
        model = Kronos.from_pretrained(
            path, num_sectors=86, num_size_buckets=0, context_layer=6,
            use_size_percentile=True, size_mlp_hidden_dim=64,
        ).to(device).eval()
        named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        params = [parameter for _, parameter in named]
        groups = group_indices(named)
        batch_rows = []
        for batch_index, batch in enumerate(fixed_batches):
            ce, raw_path, details = losses(model, tokenizer, batch)
            ce_grads = torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True)
            path_grads = torch.autograd.grad(raw_path, params, allow_unused=True)
            batch_rows.append({
                "batch_index": batch_index,
                "samples": int(batch[0].shape[0]),
                "token_ce": float(ce.detach()),
                "raw_path_huber": float(raw_path.detach()),
                "path_ema_denominator": ema_value,
                "groups": gradient_stats(ce_grads, path_grads, groups, ema_value),
            })
            print({"model": label, "batch": batch_index, "ce": float(ce), "path": float(raw_path)}, flush=True)
        aggregate = {}
        for group in groups:
            keys = ["ce_norm", "raw_path_norm", "ce_path_cosine",
                    "weighted_normalized_path_norm", "weighted_path_to_ce_norm_ratio", "implied_total_norm"]
            aggregate[group] = {
                key: float(np.mean([row["groups"][group][key] for row in batch_rows
                                    if row["groups"][group][key] is not None]))
                for key in keys
            }
        results[label] = {"batches": batch_rows, "mean_across_batches": aggregate}
        del model
        torch.cuda.empty_cache()

    report = {
        "status": "complete",
        "training_performed": False,
        "source_commit": SOURCE_COMMIT,
        "lambda_path": LAMBDA_PATH,
        "path_ema_denominator": ema_value,
        "normalization_note": "All weighted Path norms use the frozen C3-last EMA denominator; cosine is invariant to positive scalar normalization.",
        "dropout": "disabled (eval mode)",
        "fixed_validation_samples": len(positions),
        "fixed_batch_size": 32,
        "checkpoint_sha256": checkpoint_hashes,
        "c3_best_equals_last": checkpoint_hashes["c3_best"] == checkpoint_hashes["c3_last"],
        "results": results,
    }
    (OUTPUT / "gradient_diagnostic.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
