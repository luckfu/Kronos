"""Offline 2x2 identity decomposition on frozen Small bootfinal — as run.

PROVENANCE (2026-10-27). Verbatim launch script from the payload of
``wynstonliu/kronos-small-bootfinal-cond-2x2`` (v1, COMPLETE).
Payload zip sha256 = a768f10ee377576f6c97455c5d52081d4ba381c6dfd1ff1f18cb1f57fb501ba3.
It mounts ``wynstonliu/kronos-small-0-1-bootstrap-final`` -> ``checkpoints/best_model``, injects at
context_layer=6, and measures the Δ=0.01669 point. The three code files in
``payload_snapshot/`` are byte-identical across all four kernels.

Runs only against ``payload_snapshot/`` (see README).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENT = "small_boot_cond_identity_ablation"
OUTPUT_NAME = "small_c2_cond_identity_ablation"
DATASET = "smmt315/a-share-120d-temporal-symbol-holdout"
C2_DATASET = "wynstonliu/kronos-small-0-1-bootstrap-final"
DATA_SHA = "b85ccc45f7cd0e895e607e6e75c7a603427bd62a353a016742a52b2943d652cd"
C2_SHA = None  # resolved at runtime from the mounted kernel output
TOKENIZER_SHA = "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"
TORCH_VERSION = "2.6.0"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu124"
VAL_BATCH = 64
TRUNK_PROBE_BATCHES = 32
# Dual-gate references (same as Base margin smoke).
TRUNK_RATIO_PASS = 0.02
FULL_SHUFFLED_BAND = 1.0e-5  # "leave ~1e-5 by ~1 order" → |Δ| ≳ 1e-4


def phase(name: str, **payload) -> None:
    print(json.dumps({"phase": name, "time": time.time(), **payload}, ensure_ascii=False), flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_data(input_root: Path, target: Path) -> Path:
    candidates = [
        path for path in input_root.glob("**/data_manifest.json")
        if sha256(path) == DATA_SHA
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one clean data_manifest.json ({DATA_SHA[:12]}…); "
            f"found {len(candidates)}: {candidates}"
        )
    manifest_path = candidates[0]
    source = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if target.exists():
        import shutil

        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    (target / "processed_datasets").mkdir(exist_ok=True)
    for entry in manifest["files"]:
        relative = Path(entry["name"])
        path = source / relative
        if not path.is_file():
            path = source / relative.name
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"Clean dataset hash mismatch: {path}")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(path.resolve())
        # Flat Kaggle mounts (train_data.pkl at root) still need the
        # processed_datasets/ layout Config.dataset_path expects.
        if relative.name in {"train_data.pkl", "val_data.pkl"}:
            flat_dest = target / "processed_datasets" / relative.name
            if flat_dest.resolve() != destination.resolve():
                if flat_dest.exists() or flat_dest.is_symlink():
                    flat_dest.unlink()
                flat_dest.symlink_to(path.resolve())
    for name in ("train_data.pkl", "val_data.pkl", "asset_metadata.csv"):
        # asset_metadata lives at target root; pkls under processed_datasets.
        check = (
            target / "processed_datasets" / name
            if name.endswith(".pkl")
            else target / name
        )
        if not (check.is_file() or check.is_symlink()):
            raise RuntimeError(f"Missing {check} after prepare_data")
    manifest_dest = target / "data_manifest.json"
    if manifest_dest.exists() or manifest_dest.is_symlink():
        manifest_dest.unlink()
    manifest_dest.symlink_to(manifest_path.resolve())
    phase(
        "data_ready",
        dataset=DATASET,
        manifest_sha256=DATA_SHA,
        data_root=str(target),
    )
    return target


def resolve_c2(input_root: Path) -> Path:
    """Small 0.1 bootstrap: pick the chunk's best_model from the mounted kernel output."""
    cands = [p.parent for p in input_root.glob("**/checkpoints/best_model/model.safetensors")]
    pool = [c for c in cands if "small_0.1_bootstrap" in str(c)] or cands
    if not pool:
        raise RuntimeError(f"no best_model found under {input_root}")
    ckpt = sorted(pool)[0]
    if not (ckpt / "config.json").is_file():
        raise RuntimeError(f"missing config.json next to {ckpt}")
    return ckpt


def resolve_tokenizer(cache: Path) -> Path:
    if (cache / "model.safetensors").is_file() and sha256(cache / "model.safetensors") == TOKENIZER_SHA:
        phase("tokenizer_cache_hit", path=str(cache), sha256=TOKENIZER_SHA)
        return cache
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            repo_id=TOKENIZER_REPO,
            local_dir=str(cache),
            local_dir_use_symlinks=False,
        )
    )
    digest = sha256(path / "model.safetensors")
    if digest != TOKENIZER_SHA:
        raise RuntimeError(
            f"Tokenizer SHA mismatch: got {digest}, expected {TOKENIZER_SHA}"
        )
    phase("tokenizer_ready", path=str(path), sha256=digest, repo=TOKENIZER_REPO)
    return path


def weight_diagnostics(model) -> dict:
    import torch

    out: dict = {"has_condition_scale": model.condition_scale is not None}
    if model.condition_scale is not None:
        out["condition_scale"] = float(model.condition_scale.detach().float().item())
    else:
        out["condition_scale"] = None
    if model.sector_emb is not None:
        w = model.sector_emb.weight.detach().float()
        # Drop pad row 0 if present (num_sectors+1 rows).
        real = w[1:] if w.shape[0] == model.num_sectors + 1 else w
        mean = real.mean(dim=0, keepdim=True)
        row_norm = real.norm(dim=-1).clamp_min(1e-8)
        cos_to_mean = torch.nn.functional.cosine_similarity(real, mean.expand_as(real), dim=-1)
        # pairwise off-diag cosine (sample if large)
        n = real.shape[0]
        normalized = real / row_norm.unsqueeze(-1)
        gram = normalized @ normalized.T
        off = gram[~torch.eye(n, dtype=torch.bool, device=gram.device)]
        out["sector_emb"] = {
            "shape_full": list(w.shape),
            "shape_real": list(real.shape),
            "mean_pairwise_cos_offdiag": float(off.mean().item()),
            "mean_row_norm": float(row_norm.mean().item()),
            "mean_cos_row_to_mean": float(cos_to_mean.mean().item()),
            "frac_cos_to_mean_gt_0.95": float((cos_to_mean > 0.95).float().mean().item()),
            "frobenius_full": float(w.norm().item()),
        }
    if model.size_mlp is not None:
        device = next(model.size_mlp.parameters()).device
        dtype = next(model.size_mlp.parameters()).dtype
        pct = torch.linspace(0, 1, 101, device=device, dtype=dtype)
        known = torch.ones_like(pct)
        feats = torch.stack([pct, known], dim=-1)
        with torch.no_grad():
            outs = model.size_mlp(feats).float()
        mean = outs.mean(dim=0, keepdim=True)
        cos = torch.nn.functional.cosine_similarity(outs, mean.expand_as(outs), dim=-1)
        # pairwise across pct
        norms = outs.norm(dim=-1).clamp_min(1e-8)
        normalized = outs / norms.unsqueeze(-1)
        gram = normalized @ normalized.T
        n = outs.shape[0]
        off = gram[~torch.eye(n, dtype=torch.bool, device=gram.device)]
        out["size_mlp"] = {
            "out_shape": list(outs.shape),
            "mean_pairwise_cos_across_pct": float(off.mean().item()),
            "frac_cos_out_to_mean_gt_0.99": float((cos > 0.99).float().mean().item()),
            "mean_out_norm": float(outs.norm(dim=-1).mean().item()),
        }
    return out


def probe_trunk_ratio(model, tokenizer, loader, device, config, amp_dtype, max_batches: int) -> dict:
    """Live forward with collect_condition_stats to read trunk_ratio."""
    import torch

    model.eval()
    core = model
    core.collect_condition_stats = True
    ratios = []
    cond_norms = []
    trunk_norms = []
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch_x, batch_x_stamp = batch[0], batch[1]
            batch_sector = batch[2] if len(batch) > 2 else None
            batch_size_bucket = batch[3] if len(batch) > 3 else None
            batch_size_percentile = batch[4] if len(batch) > 4 else None
            if not config.get("use_sector_features", True):
                batch_sector = None
            if not config.get("use_size_features", True):
                batch_size_bucket = None
            if not config.get("use_size_percentile", False):
                batch_size_percentile = None
            batch_x = batch_x.to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.to(device, non_blocking=True)
            if batch_sector is not None:
                batch_sector = batch_sector.to(device, non_blocking=True)
            if batch_size_bucket is not None:
                batch_size_bucket = batch_size_bucket.to(device, non_blocking=True)
            if batch_size_percentile is not None:
                batch_size_percentile = batch_size_percentile.to(device, non_blocking=True)
            token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float16,
                enabled=amp_dtype is not None,
            ):
                _ = model(
                    token_in[0], token_in[1], batch_x_stamp[:, :-1, :],
                    sector_id=batch_sector,
                    size_bucket=batch_size_bucket,
                    size_percentile=batch_size_percentile,
                    use_teacher_forcing=True,
                    s1_targets=token_out[0],
                )
            stats = getattr(core, "last_condition_stats", None) or {}
            if "condition_trunk_norm_ratio" in stats:
                ratios.append(float(stats["condition_trunk_norm_ratio"]))
                cond_norms.append(float(stats["condition_output_norm"]))
                trunk_norms.append(float(stats["trunk_input_norm"]))
            batches += 1
            if batches >= max_batches:
                break
    core.collect_condition_stats = False
    if not ratios:
        return {"batches": batches, "trunk_ratio_median": None, "trunk_ratio_mean": None}
    ratios_sorted = sorted(ratios)
    mid = len(ratios_sorted) // 2
    median = (
        ratios_sorted[mid]
        if len(ratios_sorted) % 2
        else 0.5 * (ratios_sorted[mid - 1] + ratios_sorted[mid])
    )
    return {
        "batches": batches,
        "trunk_ratio_mean": sum(ratios) / len(ratios),
        "trunk_ratio_median": median,
        "trunk_ratio_min": min(ratios),
        "trunk_ratio_max": max(ratios),
        "condition_output_norm_mean": sum(cond_norms) / len(cond_norms),
        "trunk_input_norm_mean": sum(trunk_norms) / len(trunk_norms),
    }


def dual_gate_verdict(trunk_ratio, full_minus_shuffled: float) -> dict:
    trunk_pass = trunk_ratio is not None and float(trunk_ratio) >= TRUNK_RATIO_PASS
    abs_delta = abs(float(full_minus_shuffled))
    # ~1 order above ~1e-5 band → roughly ≥ 1e-4
    content_pass = abs_delta >= (FULL_SHUFFLED_BAND * 10.0)
    if abs_delta < FULL_SHUFFLED_BAND * 3:
        delta_label = "微弱且远未可用"
    elif not content_pass:
        delta_label = "微弱且远未可用"
    else:
        delta_label = "above_gate"
    return {
        "trunk_ratio_threshold": TRUNK_RATIO_PASS,
        "trunk_ratio_pass": bool(trunk_pass),
        "full_shuffled_band": FULL_SHUFFLED_BAND,
        "full_shuffled_abs": abs_delta,
        "content_pass": bool(content_pass),
        "dual_gate_pass": bool(trunk_pass and content_pass),
        "full_shuffled_label": delta_label,
    }


def main() -> None:
    # Repo root: when extracted from push bundle, launch lives at
    # scratch/small_c2_cond_identity_ablation/launch_*.py → parents[2] = repo.
    # Also accept being run from a checkout of the same layout.
    here = Path(__file__).resolve().parent
    repo = here.parents[1] if (here.parents[1] / "finetune").is_dir() else here.parents[2]
    if not (repo / "finetune" / "train_predictor.py").is_file():
        # Bundle layout: repo/scratch/... OR repo/finetune/... sibling
        for candidate in (here.parents[1], here.parents[2], Path("/kaggle/working/kronos_source")):
            if (candidate / "finetune" / "train_predictor.py").is_file():
                repo = candidate
                break
    finetune = repo / "finetune"
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    if str(finetune) not in sys.path:
        sys.path.insert(0, str(finetune))

    phase(
        "started",
        experiment=EXPERIMENT,
        note="Offline Full/None/Shuffled on frozen Small 0.1 Bootstrap-final best_model (cum 534, sector_emb W_F=3.52298, Δ=0.016692); bare add; no γ; no train",
        c2_sha="resolved after ckpt mount",
        data_sha=DATA_SHA,
        tokenizer_sha=TOKENIZER_SHA,
        val_batch=VAL_BATCH,
    )

    phase(
        "install_dependencies",
        torch=TORCH_VERSION,
        numpy="2.1.3",
        pandas="2.2.3",
        note="e01-style only-binary pins; skip requirements.txt",
    )
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            f"torch=={TORCH_VERSION}",
            "--only-binary=:all:",
            "--index-url", TORCH_INDEX_URL,
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "-q",
            "--index-url", "https://pypi.org/simple",
            "--only-binary=numpy,pandas,safetensors,orjson,psutil,protobuf,pyyaml,hf-xet",
            "numpy==2.1.3", "pandas==2.2.3",
            "einops==0.8.1", "huggingface_hub==0.33.1", "safetensors==0.6.2",
            "tqdm==4.67.1", "psutil",
        ],
        check=True,
    )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for full-val Cond ablation")
    device = torch.device("cuda:0")
    phase(
        "device_ready",
        device=str(device),
        gpu_name=torch.cuda.get_device_name(0),
        n_gpu=torch.cuda.device_count(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
    )

    input_root = Path(os.environ.get("KRONOS_KAGGLE_INPUT_ROOT", "/kaggle/input"))
    out_root = Path(os.environ.get("KRONOS_ABLATION_OUTPUT", "/kaggle/working/small_c2_cond_identity_ablation"))
    out_root.mkdir(parents=True, exist_ok=True)

    data_root = prepare_data(input_root, Path("/kaggle/working/kronos_holdout_data"))
    c2_path = resolve_c2(input_root)
    c2_sha = sha256(c2_path / "model.safetensors")
    tokenizer_path = resolve_tokenizer(Path("/kaggle/working/kronos_tokenizer_base"))

    # Small C2 production recipe — bare add, no γ, forecast-mode.
    os.environ.update({
        "PYTHONUNBUFFERED": "1",
        "KRONOS_DATASET_PATH": str(data_root / "processed_datasets"),
        "KRONOS_TRAIN_DATA_PATHS": str(data_root / "processed_datasets" / "train_data.pkl"),
        "KRONOS_VAL_DATA_PATHS": str(data_root / "processed_datasets" / "val_data.pkl"),
        "KRONOS_METADATA_PATH": str(data_root / "asset_metadata.csv"),
        "KRONOS_LOOKBACK_WINDOW": "120",
        "KRONOS_PREDICT_WINDOW": "10",
        "KRONOS_CONTEXT_LAYER": "6",
        "KRONOS_NUM_SECTORS": "86",
        "KRONOS_USE_SECTOR_FEATURES": "1",
        "KRONOS_USE_SIZE_FEATURES": "0",
        "KRONOS_USE_SIZE_PERCENTILE": "1",
        "KRONOS_PREDICTOR_LOSS_MODE": "forecast",
        "KRONOS_HISTORY_LOSS_WEIGHT": "0.02",
        "KRONOS_BATCH_SIZE": str(VAL_BATCH),
        "KRONOS_VAL_BATCH_SIZE": str(VAL_BATCH),
        "KRONOS_USE_AMP": "1",
        "KRONOS_USE_COMET": "0",
        "KRONOS_USE_BETA_V21_AUXILIARY": "0",
        "KRONOS_USE_CONDITION_SCALE": "0",  # bare add; C2 has no γ
        "KRONOS_PREDICTOR_PATH": str(c2_path),
        "KRONOS_TOKENIZER_PATH": str(tokenizer_path),
        "KRONOS_PRETRAINED_PREDICTOR_PATH": str(c2_path),
        "KRONOS_PRETRAINED_TOKENIZER_PATH": str(tokenizer_path),
        "KRONOS_FINETUNED_TOKENIZER_PATH": str(tokenizer_path),
        "KRONOS_SAVE_PATH": str(out_root),
        "KRONOS_PREDICTOR_SAVE_FOLDER_NAME": "offline_ablation",
    })

    from config import Config
    from dataset import QlibDataset
    from model.kronos import Kronos, KronosTokenizer
    from train_predictor import evaluate_validation

    config = Config().__dict__
    # Force paths (Config may use different env key names).
    config["pretrained_predictor_path"] = str(c2_path)
    config["pretrained_tokenizer_path"] = str(tokenizer_path)
    config["finetuned_tokenizer_path"] = str(tokenizer_path)
    config["dataset_path"] = str(data_root / "processed_datasets")
    config["use_condition_scale"] = False
    config["use_beta_v21_auxiliary"] = False
    config["use_sector_features"] = True
    config["use_size_features"] = False
    config["use_size_percentile"] = True
    config["num_sectors"] = 86
    config["context_layer"] = 6
    config["lookback_window"] = 120
    config["predict_window"] = 10
    config["predictor_loss_mode"] = "forecast"
    config["batch_size"] = VAL_BATCH
    config["val_batch_size"] = VAL_BATCH
    config["use_amp"] = True

    phase("load_tokenizer")
    with open(tokenizer_path / "config.json") as handle:
        tokenizer_kwargs = json.load(handle)
    tokenizer = KronosTokenizer.from_pretrained(str(tokenizer_path), **tokenizer_kwargs)
    tokenizer.eval().to(device)

    phase("load_model", path=str(c2_path))
    with open(c2_path / "config.json") as handle:
        model_kwargs = json.load(handle)
    model_kwargs.update({
        "num_sectors": 86,
        "num_size_buckets": int(model_kwargs.get("num_size_buckets", 0) or 0),
        "context_layer": 6,
        "use_size_percentile": True,
        "size_mlp_hidden_dim": int(model_kwargs.get("size_mlp_hidden_dim", 64) or 64),
        "use_beta_v21_auxiliary": False,
        "use_condition_scale": False,  # bare add — do not invent γ
    })
    model = Kronos.from_pretrained(str(c2_path), **model_kwargs)
    if getattr(model, "condition_scale", None) is not None:
        raise RuntimeError(
            "C2 offline ablation must stay bare-add; condition_scale unexpectedly present"
        )
    model.eval().to(device)
    phase(
        "model_ready",
        n_layers=model.n_layers,
        d_model=model.d_model,
        context_layer=model.context_layer,
        num_sectors=model.num_sectors,
        use_size_percentile=model.size_mlp is not None,
        condition_scale="disabled",
    )

    diagnostics = weight_diagnostics(model)
    (out_root / "weight_diagnostics.json").write_text(
        json.dumps({"checkpoint": str(c2_path), "sha256": c2_sha, **diagnostics}, indent=2) + "\n"
    )
    phase("weight_diagnostics", **{k: diagnostics[k] for k in ("has_condition_scale", "sector_emb", "size_mlp") if k in diagnostics})

    phase("build_val_dataset")
    valid_dataset = QlibDataset("val")
    from torch.utils.data import DataLoader

    loader = DataLoader(
        valid_dataset,
        batch_size=VAL_BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )
    phase("val_ready", samples=len(valid_dataset), batches=len(loader), batch_size=VAL_BATCH)

    amp_dtype = torch.float16
    phase("trunk_ratio_probe", max_batches=TRUNK_PROBE_BATCHES)
    trunk = probe_trunk_ratio(
        model, tokenizer, loader, device, config, amp_dtype, TRUNK_PROBE_BATCHES
    )
    phase("trunk_ratio_ready", **trunk)

    phase("ablation_start", run_condition_ablation=True)
    started = time.time()
    metrics = evaluate_validation(
        model,
        tokenizer,
        loader,
        device,
        config,
        amp_dtype,
        run_condition_ablation=True,
        period_names=getattr(valid_dataset, "validation_period_names", {}),
        rank=0,
    )
    elapsed = time.time() - started
    phase("ablation_done", seconds=elapsed, samples=metrics.get("samples"), batches=metrics.get("batches"))

    full = float(metrics["forecast_loss"])
    none = float(metrics["condition_none_forecast_loss"])
    shuffled = float(metrics["condition_shuffled_forecast_loss"])
    d_none = full - none
    d_shuf = full - shuffled
    trunk_ratio = trunk.get("trunk_ratio_median")
    gate = dual_gate_verdict(trunk_ratio, d_shuf)

    print(
        "Validation Condition Full/None/Shuffled Forecast: "
        f"{full:.8f} / {none:.8f} / {shuffled:.8f}; "
        "Delta Full-None/Full-Shuffled: "
        f"{d_none:.8e} / {d_shuf:.8e}",
        flush=True,
    )
    print(
        f"trunk_ratio_median={trunk_ratio} "
        f"condition_output_norm_mean={trunk.get('condition_output_norm_mean')} "
        f"dual_gate_pass={gate['dual_gate_pass']} "
        f"full_shuffled_label={gate['full_shuffled_label']}",
        flush=True,
    )

    # ---------------- 2x2 identity decomposition (2026-10-27) ----------------
    # Sector-only shuffle vs size-only shuffle, plus two shifts per axis for a
    # permutation null. Same tokens, same teacher forcing, same val loader.
    from train_predictor import compute_predictor_losses, to_float32

    def _roll(x, k):
        if x is None or x.shape[0] <= 1:
            return x
        return torch.roll(x, shifts=k, dims=0)

    phase("decompose_start")
    keys = ("full", "none", "s1", "s2", "z1", "z2", "b1", "b2")
    accum = {k: [0.0, 0] for k in keys}
    started2 = time.time()
    with torch.no_grad():
        for batch in loader:
            bx = batch[0].to(device, non_blocking=True)
            stamp = batch[1].to(device, non_blocking=True)
            sec = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
            pct = batch[4].to(device, non_blocking=True) if len(batch) > 4 else None
            n = int(bx.shape[0])
            t0, t1 = tokenizer.encode(bx, half=True)
            tin = [t0[:, :-1], t1[:, :-1]]
            tout = [t0[:, 1:], t1[:, 1:]]
            variants = {
                "full": (sec, pct),
                "none": (None, None),
                "s1": (_roll(sec, 1), pct),
                "s2": (_roll(sec, 2), pct),
                "z1": (sec, _roll(pct, 1)),
                "z2": (sec, _roll(pct, 2)),
                "b1": (_roll(sec, 1), _roll(pct, 1)),
                "b2": (_roll(sec, 2), _roll(pct, 2)),
            }
            for key, (s_id, s_pct) in variants.items():
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype or torch.float16,
                    enabled=amp_dtype is not None,
                ):
                    out = model(
                        tin[0], tin[1], stamp[:, :-1, :],
                        sector_id=s_id, size_bucket=None, size_percentile=s_pct,
                        use_teacher_forcing=True, s1_targets=tout[0],
                        asof_index=int(config['lookback_window']) - 1,
                    )
                logits = to_float32(out)
                ce = compute_predictor_losses(model.head, logits, tout, config)['forecast']
                accum[key][0] += float(ce) * n
                accum[key][1] += n
    ce = {k: (v[0] / v[1]) for k, v in accum.items()}
    fs_sector = ce["full"] - ce["s1"]
    fs_size = ce["full"] - ce["z1"]
    fs_both = ce["full"] - ce["b1"]
    null_sector = ce["s1"] - ce["s2"]
    null_size = ce["z1"] - ce["z2"]
    null_both = ce["b1"] - ce["b2"]
    decompose = {
        "samples": accum["full"][1],
        "ce": ce,
        "fs_sector": fs_sector,
        "fs_size": fs_size,
        "fs_both": fs_both,
        "null_sector": null_sector,
        "null_size": null_size,
        "null_both": null_both,
        "seconds": time.time() - started2,
        "parity_expected": {"full": 2.283136875152588, "none": 2.291893516540527, "both": 2.285268211364746},
    }
    (out_root / "cond_identity_2x2.json").write_text(
        json.dumps(decompose, indent=2) + "\n"
    )
    phase("decompose_done", **{k: decompose[k] for k in (
        "fs_sector", "fs_size", "fs_both", "null_sector", "null_size", "null_both", "seconds")})

    # ------------- sector-gain / size-gain dose-response sweep (2026-10-27) ----
    # Forward-only: scale the sector embedding by alpha and the size MLP output
    # by beta at the injection point, then re-measure fs_sector / fs_size.
    # Gives the CE <-> identity frontier without touching training.
    phase("sweep_start")
    sec_base = model.sector_emb.weight.detach().clone()
    size_w_base = model.size_mlp[-1].weight.detach().clone()
    size_b_base = model.size_mlp[-1].bias.detach().clone()

    def set_gain(alpha, beta):
        with torch.no_grad():
            model.sector_emb.weight.copy_(sec_base * alpha)
            model.size_mlp[-1].weight.copy_(size_w_base * beta)
            model.size_mlp[-1].bias.copy_(size_b_base * beta)

    def measure(alpha, beta):
        set_gain(alpha, beta)
        acc = {"full": 0.0, "secshuf": 0.0, "sizeshuf": 0.0}
        total = 0
        with torch.no_grad():
            for batch in loader:
                bx = batch[0].to(device, non_blocking=True)
                stamp = batch[1].to(device, non_blocking=True)
                sec = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
                pct = batch[4].to(device, non_blocking=True) if len(batch) > 4 else None
                n = int(bx.shape[0])
                t0, t1 = tokenizer.encode(bx, half=True)
                tin = [t0[:, :-1], t1[:, :-1]]
                tout = [t0[:, 1:], t1[:, 1:]]
                for key, (s_id, s_pct) in {
                    "full": (sec, pct),
                    "secshuf": (_roll(sec, 1), pct),
                    "sizeshuf": (sec, _roll(pct, 1)),
                }.items():
                    with torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype or torch.float16,
                        enabled=amp_dtype is not None,
                    ):
                        out = model(
                            tin[0], tin[1], stamp[:, :-1, :],
                            sector_id=s_id, size_bucket=None, size_percentile=s_pct,
                            use_teacher_forcing=True, s1_targets=tout[0],
                            asof_index=int(config['lookback_window']) - 1,
                        )
                    logits = to_float32(out)
                    ce = compute_predictor_losses(model.head, logits, tout, config)['forecast']
                    acc[key] += float(ce) * n
                total += n
        ce_full = acc["full"] / total
        return {
            "alpha_sector": alpha,
            "beta_size": beta,
            "ce_full": ce_full,
            "fs_sector": ce_full - acc["secshuf"] / total,
            "fs_size": ce_full - acc["sizeshuf"] / total,
            "n_samples": total,
        }

    sweep = []
    for alpha in (1.0, 1.5, 2.0, 3.0, 4.0, 8.0):
        row = measure(alpha, 1.0)
        sweep.append(row)
        phase("sweep_cell", **row)
    for beta in (0.5,):
        row = measure(1.0, beta)
        sweep.append(row)
        phase("sweep_cell", **row)
    set_gain(1.0, 1.0)
    (out_root / "sector_gain_sweep.json").write_text(
        json.dumps({"samples": sweep[-1].get("n_samples"), "rows": sweep}, indent=2) + "\n"
    )
    phase("sweep_done", cells=len(sweep))

    document = {
        "mode": "offline_cond_identity_ablation",
        "experiment": EXPERIMENT,
        "checkpoint": {
            "path": str(c2_path),
            "sha256": c2_sha,
            "segment": 534,
            "dataset": C2_DATASET,
            "recipe": "Small 0.1 Bootstrap-final best_model (cum 534, cum windows 10.68M); bare add; no γ",
        },
        "tokenizer_sha256": TOKENIZER_SHA,
        "data_manifest_sha256": DATA_SHA,
        "holdout_dataset": DATASET,
        "samples": metrics.get("samples"),
        "batches": metrics.get("batches"),
        "validation_seconds": elapsed,
        "forecast_ce": {
            "full": full,
            "none": none,
            "shuffled": shuffled,
        },
        "deltas": {
            "full_minus_none": d_none,
            "full_minus_shuffled": d_shuf,
        },
        "trunk_probe": trunk,
        "dual_gate": gate,
        "weight_diagnostics": diagnostics,
        "decompose_2x2": decompose,
        "sector_gain_sweep": sweep,
        "base_margin_fail_reference": {
            "full": 2.74523325,
            "none": 2.74663521,
            "shuffled": 2.74526328,
            "full_minus_none": -0.00140196,
            "full_minus_shuffled": -0.00003003,
            "trunk_ratio": 0.0205,
            "verdict": "dual-gate FAIL (trunk PASS; Full−Shuffled 微弱且远未可用)",
        },
        "metrics": {
            k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in metrics.items()
            if k in {
                "objective_loss", "full_sequence_loss", "history_loss",
                "forecast_loss", "weighted_forecast_loss",
                "condition_none_forecast_loss", "condition_shuffled_forecast_loss",
                "condition_full_minus_none", "condition_full_minus_shuffled",
                "samples", "batches",
            }
        },
    }
    result_path = out_root / "cond_identity_ablation_result.json"
    result_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    phase(
        "completed",
        result=str(result_path),
        full=full,
        none=none,
        shuffled=shuffled,
        full_minus_none=d_none,
        full_minus_shuffled=d_shuf,
        trunk_ratio_median=trunk_ratio,
        dual_gate_pass=gate["dual_gate_pass"],
        full_shuffled_label=gate["full_shuffled_label"],
    )


if __name__ == "__main__":
    main()
