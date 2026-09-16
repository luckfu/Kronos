"""CE-only control checkpoints on the sealed 13-date production OOS evaluator.

P0a best_model at completion equals last (best_equals_last). Seg-0 init is
therefore C2 best, not P0a best_model. Evaluated: init / seg05 / seg10 / last.
"""
import gc
import hashlib
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

SOURCE_COMMIT = "bf83ef0bcafe4d8fc6ddebf0a17a92ec929ee5cf"
OUTPUT = Path("/kaggle/working/kronos_small_0_1_stage3_ce_only_oos")
INPUT = Path("/kaggle/input")
HARD_LIMIT_SECONDS = 39600
PURPOSE = ("exploratory_only; 13-date OOS design-contaminated; "
           "production candidate requires sealed fresh-window OOS")
# last first so a timeout still closes decision tree A/B/C
MODELS = [
    ("ce_only_seg15_last", "p0a_last"),
    ("ce_only_seg00_init", "c2_best_init"),
    ("ce_only_seg10", "p0a_milestone_seg10"),
    ("ce_only_seg05", "p0a_milestone_seg05"),
]


def find_one(pattern, predicate=lambda path: True):
    matches = [path for path in INPUT.glob(pattern) if predicate(path)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command, cwd=None):
    print({"command": command, "cwd": str(cwd) if cwd else None}, flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def clone_source():
    for attempt in range(1, 4):
        repo = Path(tempfile.mkdtemp(prefix="kronos-ce-only-oos-source-")) / "repo"
        try:
            run(["git", "clone", "--depth", "8", "--branch", "master",
                 "https://github.com/luckfu/Kronos.git", str(repo)])
            run(["git", "checkout", "--detach", SOURCE_COMMIT], cwd=repo)
            actual = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()
            if actual != SOURCE_COMMIT:
                raise RuntimeError(f"Source commit mismatch: {actual}")
            return repo
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * attempt)


def metric_or_none(value):
    value = float(value)
    return None if not math.isfinite(value) else value


def summarize(frame, top_fraction=0.10):
    import numpy as np
    import pandas as pd

    by_horizon = []
    by_date_h10 = []
    for horizon in range(1, 11):
        pred_col = f"predicted_return_d{horizon}"
        actual_col = f"actual_return_d{horizon}"
        pred, actual = frame[pred_col], frame[actual_col]
        daily = []
        for date, rows in frame.groupby("asof_date", sort=True):
            rank_ic = rows[pred_col].corr(rows[actual_col], method="spearman")
            count = max(1, int(math.ceil(len(rows) * top_fraction)))
            ranked = rows.sort_values(pred_col)
            spread = ranked.tail(count)[actual_col].mean() - ranked.head(count)[actual_col].mean()
            daily.append({
                "asof_date": str(date),
                "samples": int(len(rows)),
                "rank_ic": metric_or_none(rank_ic),
                "direction_accuracy": float((np.sign(rows[pred_col]) == np.sign(rows[actual_col])).mean()),
                "top_bottom_10pct": float(spread),
            })
        daily_ic = pd.Series([row["rank_ic"] for row in daily], dtype=float).dropna()
        ic_std = float(daily_ic.std(ddof=1)) if len(daily_ic) > 1 else float("nan")
        metrics = {
            "horizon_day": horizon,
            "samples": int(len(frame)),
            "direction_accuracy": float((np.sign(pred) == np.sign(actual)).mean()),
            "mae": float((pred - actual).abs().mean()),
            "pooled_rank_ic": metric_or_none(pred.corr(actual, method="spearman")),
            "daily_rank_ic_mean": metric_or_none(daily_ic.mean()),
            "daily_rank_ic_std": metric_or_none(ic_std),
            "daily_icir": metric_or_none(daily_ic.mean() / ic_std),
            "daily_icir_annualized_sqrt252": metric_or_none(daily_ic.mean() / ic_std * math.sqrt(252)),
            "positive_ic_ratio": float((daily_ic > 0).mean()),
            "mean_top_bottom_10pct": float(np.mean([row["top_bottom_10pct"] for row in daily])),
        }
        by_horizon.append(metrics)
        if horizon == 10:
            by_date_h10 = daily
    return {
        "samples": int(len(frame)),
        "signal_dates": int(frame["asof_date"].nunique()),
        "by_horizon": by_horizon,
        "by_signal_date_d10": by_date_h10,
    }


def tensors_equal(left, right):
    if set(left) != set(right):
        return False
    for key, tensor in left.items():
        if tuple(tensor.shape) != tuple(right[key].shape):
            return False
        if not torch.equal(tensor.float(), right[key].float()):
            return False
    return True


def agreement(left, right, name):
    pred_cols = [f"predicted_return_d{h}" for h in range(1, 11)]
    merged = left[["identity"] + pred_cols].merge(
        right[["identity"] + pred_cols], on="identity",
        suffixes=("_new", "_ref"), validate="one_to_one",
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise RuntimeError(f"{name} identity coverage mismatch: {len(merged)} vs {len(left)}/{len(right)}")
    rows = []
    for horizon in range(1, 11):
        delta = (merged[f"predicted_return_d{horizon}_new"] - merged[f"predicted_return_d{horizon}_ref"]).abs()
        spearman = merged[f"predicted_return_d{horizon}_new"].corr(
            merged[f"predicted_return_d{horizon}_ref"], method="spearman"
        )
        rows.append({
            "horizon_day": horizon,
            "max_abs_delta": float(delta.max()),
            "mean_abs_delta": float(delta.mean()),
            "spearman": metric_or_none(spearman),
        })
    return {"name": name, "identities": int(len(merged)), "by_horizon": rows}


def write_summary(payload):
    (OUTPUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


def main():
    os.environ["PYTHONUNBUFFERED"] = "1"
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shards = OUTPUT / "shards"
    shards.mkdir(exist_ok=True)

    repo = clone_source()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune"))

    import pandas as pd
    from finetune.evaluate_v1_beta_checkpoints import WindowStore, evaluate_predictions, load_samples
    from model import Kronos, KronosTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required")
    device = torch.device("cuda:0")

    c2_file = find_one("**/small_0.1_stage2_cosine_refinement/checkpoints/best_model/model.safetensors")
    last_file = find_one("**/stage3_joint_path_smoke/checkpoints/last_model/model.safetensors")
    best_file = find_one("**/stage3_joint_path_smoke/checkpoints/best_model/model.safetensors")
    seg05_file = find_one("**/stage3_joint_path_smoke/checkpoints/milestone_seg05/model.safetensors")
    seg10_file = find_one("**/stage3_joint_path_smoke/checkpoints/milestone_seg10/model.safetensors")
    tokenizer_file = find_one("**/Kronos-Tokenizer-base/model.safetensors")
    c2_predictions = find_one(
        "**/kronos_small_0_1_stage2_oos/predictions.csv.gz",
        lambda path: "c2-alpha-oos-evaluation" in str(path),
    )

    manifests = []
    for path in INPUT.glob("**/evaluation_manifest.json"):
        if not ((path.parent / "evaluation_panel.pkl").is_file()
                and (path.parent / "evaluation_samples.jsonl").is_file()):
            continue
        candidate = json.loads(path.read_text())
        if candidate.get("temporal_isolation", {}).get("incremental_signal_start"):
            manifests.append(path)
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one sealed evaluation manifest, found {manifests}")
    evaluation_root = manifests[0].parent
    manifest = json.loads(manifests[0].read_text())
    isolation = manifest["temporal_isolation"]
    if not (isolation.get("strictly_after_parent_latest_training_target")
            or isolation.get("targets_strictly_after_training_target_end")):
        raise RuntimeError("Evaluation package is not temporally isolated")

    with (evaluation_root / manifest["artifacts"]["panel_file"]).open("rb") as handle:
        panel = pickle.load(handle)
    sample_groups = load_samples(evaluation_root / manifest["artifacts"]["samples_file"])
    sample_key = "incremental_future_all" if "incremental_future_all" in sample_groups else "future_all"
    start = isolation.get("incremental_signal_start", isolation.get("future_signal_start"))
    end = isolation.get("incremental_signal_end", isolation.get("future_signal_end"))
    records = [row for row in sample_groups[sample_key] if start <= row["asof_date"] <= end]
    if not records:
        raise RuntimeError("No sealed OOS records")
    dates = sorted({row["asof_date"] for row in records})

    c2_ref = pd.read_csv(c2_predictions)
    c2_ref = c2_ref[c2_ref["model"] == "c2_best_segment_179"].copy()
    if c2_ref["identity"].duplicated().any() or len(c2_ref) != 66986:
        raise RuntimeError(f"C2 reference invalid: rows={len(c2_ref)}")
    if len(c2_ref) != len(records):
        raise RuntimeError(f"C2 prediction count mismatch: {len(c2_ref)} != {len(records)}")

    c2_tensors = load_file(str(c2_file))
    last_tensors = load_file(str(last_file))
    best_tensors = load_file(str(best_file))
    best_equals_last = tensors_equal(best_tensors, last_tensors)
    if not best_equals_last:
        raise RuntimeError("P0a best_model is not torch.equal to last_model; do not treat best as last")

    weight_dirs = {
        "ce_only_seg15_last": last_file.parent,
        "ce_only_seg00_init": c2_file.parent,
        "ce_only_seg10": seg10_file.parent,
        "ce_only_seg05": seg05_file.parent,
    }
    provenance = {
        "c2_init_weights": {"path": str(c2_file), "sha256": sha256_file(c2_file)},
        "p0a_last_weights": {"path": str(last_file), "sha256": sha256_file(last_file)},
        "p0a_best_weights": {"path": str(best_file), "sha256": sha256_file(best_file),
                             "torch_equal_last": True, "tensors": len(best_tensors)},
        "p0a_milestone_seg05": {"path": str(seg05_file), "sha256": sha256_file(seg05_file)},
        "p0a_milestone_seg10": {"path": str(seg10_file), "sha256": sha256_file(seg10_file)},
        "tokenizer": {"path": str(tokenizer_file), "sha256": sha256_file(tokenizer_file)},
        "c2_reference_predictions": {"path": str(c2_predictions), "sha256": sha256_file(c2_predictions)},
        "evaluation_root": str(evaluation_root),
        "best_model_note": "P0a best_model at completion is last, not seg-0 init; init is C2 best",
    }
    print(json.dumps({"phase": "inputs_resolved", **provenance}, ensure_ascii=False), flush=True)
    del c2_tensors, last_tensors, best_tensors
    gc.collect()

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_file.parent).to(device).eval()
    store = WindowStore(panel, manifest["model_contract"]["sector_labels"])
    metrics = {}
    sanity = {}
    remaining = []
    timed_out = False

    for label, role in MODELS:
        pending = [date for date in dates if not (shards / f"{label}_{date}.csv.gz").is_file()]
        if pending:
            model = Kronos.from_pretrained(
                weight_dirs[label], num_sectors=86, num_size_buckets=0, context_layer=6,
                use_size_percentile=True, size_mlp_hidden_dim=64,
            ).to(device).eval()
            for date in pending:
                if time.time() - started > HARD_LIMIT_SECONDS:
                    remaining = [{"label": label, "role": role, "dates": [date] + pending[pending.index(date) + 1:]}]
                    remaining.extend({"label": rest_label, "role": rest_role}
                                     for rest_label, rest_role in MODELS[MODELS.index((label, role)) + 1:])
                    timed_out = True
                    break
                date_records = [row for row in records if row["asof_date"] == date]
                result = evaluate_predictions(
                    label, model, tokenizer, date_records, store, device,
                    64, 1, 20260906, True,
                )
                shard = shards / f"{label}_{date}.csv.gz"
                result.to_csv(shard, index=False, compression="gzip")
                print({"saved": str(shard), "rows": len(result), "elapsed_sec": time.time() - started}, flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            if timed_out:
                break

        parts = [pd.read_csv(path) for path in sorted(shards.glob(f"{label}_*.csv.gz"))]
        if len(parts) != len(dates):
            remaining = [{"label": label, "role": role, "have_shards": len(parts), "need": len(dates)}]
            break
        frame = pd.concat(parts, ignore_index=True)
        if len(frame) != len(records):
            raise RuntimeError(f"{label} prediction count mismatch: {len(frame)} != {len(records)}")
        metrics[label] = summarize(frame)
        (OUTPUT / f"metrics_{label}.json").write_text(
            json.dumps(metrics[label], ensure_ascii=False, indent=2) + "\n"
        )
        if label == "ce_only_seg00_init":
            sanity[label] = agreement(frame, c2_ref, "init_vs_c2_best_segment_179")
        print(json.dumps({"phase": "model_complete", "label": label, "role": role,
                          "d10": metrics[label]["by_horizon"][9]}, ensure_ascii=False), flush=True)

    payload = {
        "status": "partial" if timed_out or remaining else "complete",
        "purpose": PURPOSE,
        "oos_used": True,
        "training_performed": False,
        "source_commit": SOURCE_COMMIT,
        "sample_set": sample_key,
        "signal_start": start,
        "signal_end": end,
        "hard_limit_seconds": HARD_LIMIT_SECONDS,
        "elapsed_sec": time.time() - started,
        "model_order": [{"label": label, "role": role} for label, role in MODELS],
        "provenance": provenance,
        "sanity_policy": "identity-merge max|delta|/mean|delta|/spearman; soft report only; fp16 autocast + date seed + cross-GPU",
        "sanity": sanity,
        "metrics": metrics,
        "remaining": remaining,
        "decision_tree_note": "Compare ce_only_seg15_last D10 pooled_rank_ic to C2 0.184 / C3 0.063; exploratory only",
    }
    write_summary(payload)
    if timed_out:
        sys.exit(0)


if __name__ == "__main__":
    main()
