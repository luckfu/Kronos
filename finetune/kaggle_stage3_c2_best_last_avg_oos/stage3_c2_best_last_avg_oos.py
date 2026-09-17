"""C2 best (seg 179) x C2 last (seg 267) float32 weight averaging on the 13-date diagnostic OOS.

Zero training. Interpolates alpha in {0.5, 1.0, 0.25, 0.75} between two checkpoints
that both already carry alpha, then evaluates each on the sealed 13-date window with
the same evaluator / seed / fp16 settings used by every Stage3 OOS kernel.

alpha=0.0 (C2 best) is not re-run: the same evaluator has reproduced its D10 pooled
Rank IC 0.18440434213929857 in P0b, P0a init, P1 init and freeze-probe B init.
alpha=1.0 (C2 last) is re-run so C2 last gets a same-footing number.
"""
import gc
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SOURCE_COMMIT = "0" * 40
OUTPUT = Path("/kaggle/working/kronos_small_0_1_stage3_c2_best_last_avg_oos")
INPUT = Path("/kaggle/input")
HARD_LIMIT_SECONDS = 39600
PURPOSE = ("exploratory_only; 13-date OOS design-contaminated; "
           "production candidate requires sealed fresh-window OOS")
EXPECTED_C2_BEST_SHA = "4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a"
EXPECTED_TOKENIZER_SHA = "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
C2_BEST_SAME_FOOTING_D10_POOLED_RANK_IC = 0.18440434213929857
# 0.5 first: it is the decision-critical checkpoint. 1.0 second: C2 last same footing.
ALPHAS = [
    (0.5, "avg_alpha_050"),
    (1.0, "avg_alpha_100_c2_last"),
    (0.25, "avg_alpha_025"),
    (0.75, "avg_alpha_075"),
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
    subprocess.run(command, cwd=cwd, check=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})


def clone_source():
    for attempt in range(1, 4):
        repo = Path(tempfile.mkdtemp(prefix="kronos-c2-avg-source-")) / "repo"
        try:
            run(["git", "init", str(repo)])
            run(["git", "remote", "add", "origin", "https://github.com/luckfu/Kronos.git"], cwd=repo)
            run(["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT], cwd=repo)
            run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo)
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


def interpolate(best, last, alpha):
    import torch
    if set(best) != set(last):
        missing = sorted(set(best) ^ set(last))
        raise RuntimeError(f"State dict keys differ: {missing[:20]}")
    merged = {}
    for key, left in best.items():
        right = last[key]
        if tuple(left.shape) != tuple(right.shape):
            raise RuntimeError(f"Shape mismatch {key}: {tuple(left.shape)} vs {tuple(right.shape)}")
        tensor = (1.0 - alpha) * left.float() + alpha * right.float()
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"Non-finite interpolated tensor: {key}")
        merged[key] = tensor
    return merged


def relative_l2(best, last):
    import torch
    num = 0.0
    den = 0.0
    for key, left in best.items():
        diff = last[key].float() - left.float()
        num += float((diff * diff).sum())
        den += float((left.float() * left.float()).sum())
    return math.sqrt(num) / math.sqrt(den) if den > 0 else float("nan")


def write_checkpoint(template_dir, tensors, dest):
    from safetensors.torch import save_file
    dest.mkdir(parents=True, exist_ok=True)
    for path in template_dir.iterdir():
        if path.name == "model.safetensors" or not path.is_file():
            continue
        shutil.copy2(path, dest / path.name)
    save_file(tensors, str(dest / "model.safetensors"), metadata={"format": "pt"})


def assert_reload(path, alpha, best, last):
    import torch
    from safetensors.torch import load_file
    reloaded = load_file(str(path))
    if set(reloaded) != set(best):
        raise RuntimeError("Reloaded keys differ from C2 best")
    if alpha == 0.0:
        reference, tag = best, "C2_best_179"
    elif alpha == 1.0:
        reference, tag = last, "C2_last_267"
    else:
        return {"checked": False}
    for key, tensor in reference.items():
        if reloaded[key].dtype != tensor.float().dtype:
            raise RuntimeError(f"alpha={alpha} dtype mismatch {key}")
        if not torch.equal(reloaded[key], tensor.float()):
            raise RuntimeError(f"alpha={alpha} torch.equal failed vs {tag}: {key}")
    return {"checked": True, "equal_to": tag, "tensors": len(reloaded)}


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
    weights = OUTPUT / "weights"
    shards.mkdir(exist_ok=True)
    weights.mkdir(exist_ok=True)

    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip()
    if not any(name in gpu for name in ("P100", "T4")):
        raise RuntimeError(f"Expected P100 or T4, got {gpu}")

    repo = clone_source()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "finetune"))

    import pandas as pd
    import torch
    from safetensors.torch import load_file
    from finetune.evaluate_v1_beta_checkpoints import WindowStore, evaluate_predictions, load_samples
    from model import Kronos, KronosTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required")
    device = torch.device("cuda:0")

    in_c2_kernel = lambda path: "cosine-refinement-c2" in str(path)
    best_file = find_one(
        "**/small_0.1_stage2_cosine_refinement/checkpoints/best_model/model.safetensors", in_c2_kernel,
    )
    last_file = find_one(
        "**/small_0.1_stage2_cosine_refinement/checkpoints/last_model/model.safetensors", in_c2_kernel,
    )
    tokenizer_file = find_one("**/Kronos-Tokenizer-base/model.safetensors", in_c2_kernel)
    c2_predictions = find_one(
        "**/kronos_small_0_1_stage2_oos/predictions.csv.gz",
        lambda path: "c2-alpha-oos-evaluation" in str(path),
    )

    best_sha = sha256_file(best_file)
    last_sha = sha256_file(last_file)
    tokenizer_sha = sha256_file(tokenizer_file)
    if best_sha != EXPECTED_C2_BEST_SHA:
        raise RuntimeError(f"C2 best sha mismatch: {best_sha}")
    if tokenizer_sha != EXPECTED_TOKENIZER_SHA:
        raise RuntimeError(f"Tokenizer sha mismatch: {tokenizer_sha}")
    if last_sha == best_sha:
        raise RuntimeError("C2 last equals C2 best; nothing to average")

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

    reference = pd.read_csv(c2_predictions)
    ref_best = reference[reference["model"] == "c2_best_segment_179"].copy()
    ref_last = reference[reference["model"] == "c2_last_segment_267"].copy()
    for name, frame in (("best", ref_best), ("last", ref_last)):
        if frame["identity"].duplicated().any() or len(frame) != len(records):
            raise RuntimeError(f"C2 {name} reference invalid: rows={len(frame)} vs {len(records)}")

    best_tensors = load_file(str(best_file))
    last_tensors = load_file(str(last_file))
    drift = relative_l2(best_tensors, last_tensors)
    provenance = {
        "c2_best_weights": {"path": str(best_file), "sha256": best_sha,
                            "dtype": sorted({str(t.dtype) for t in best_tensors.values()})},
        "c2_last_weights": {"path": str(last_file), "sha256": last_sha,
                            "dtype": sorted({str(t.dtype) for t in last_tensors.values()})},
        "tokenizer": {"path": str(tokenizer_file), "sha256": tokenizer_sha},
        "c2_reference_predictions": {"path": str(c2_predictions), "sha256": sha256_file(c2_predictions)},
        "evaluation_root": str(evaluation_root),
        "best_to_last_relative_l2": drift,
        "gpu": gpu,
    }
    print(json.dumps({"phase": "inputs_resolved", **provenance}, ensure_ascii=False), flush=True)

    # Reference-footing endpoints: metrics computed on the smmt315 prediction file.
    # These are NOT same-footing with the re-runs below (different evaluator run).
    reference_footing = {
        "c2_best_segment_179": summarize(ref_best),
        "c2_last_segment_267": summarize(ref_last),
    }
    for label, value in reference_footing.items():
        (OUTPUT / f"metrics_reference_{label}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        )
    print(json.dumps({
        "phase": "reference_footing",
        "c2_best_d10": reference_footing["c2_best_segment_179"]["by_horizon"][9]["pooled_rank_ic"],
        "c2_last_d10": reference_footing["c2_last_segment_267"]["by_horizon"][9]["pooled_rank_ic"],
    }), flush=True)

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_file.parent).to(device).eval()
    store = WindowStore(panel, manifest["model_contract"]["sector_labels"])
    metrics = {}
    sanity = {}
    weight_shas = {}
    remaining = []
    timed_out = False

    for alpha, label in ALPHAS:
        dest = weights / label
        merged = interpolate(best_tensors, last_tensors, alpha)
        write_checkpoint(best_file.parent, merged, dest)
        check = assert_reload(dest / "model.safetensors", alpha, best_tensors, last_tensors)
        weight_shas[label] = sha256_file(dest / "model.safetensors")
        print(json.dumps({"phase": "interpolated", "label": label, "alpha": alpha,
                          "reload_check": check, "sha256": weight_shas[label]}), flush=True)
        del merged
        gc.collect()

        pending = [date for date in dates if not (shards / f"{label}_{date}.csv.gz").is_file()]
        if pending:
            model = Kronos.from_pretrained(
                dest, num_sectors=86, num_size_buckets=0, context_layer=6,
                use_size_percentile=True, size_mlp_hidden_dim=64,
            ).to(device).eval()
            for date in pending:
                if time.time() - started > HARD_LIMIT_SECONDS:
                    remaining = [{"label": label, "alpha": alpha,
                                  "dates": [date] + pending[pending.index(date) + 1:]}]
                    remaining.extend({"label": rest_label, "alpha": rest_alpha}
                                     for rest_alpha, rest_label in ALPHAS[ALPHAS.index((alpha, label)) + 1:])
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
            remaining = [{"label": label, "alpha": alpha, "have_shards": len(parts), "need": len(dates)}]
            break
        frame = pd.concat(parts, ignore_index=True)
        if len(frame) != len(records):
            raise RuntimeError(f"{label} prediction count mismatch: {len(frame)} != {len(records)}")
        metrics[label] = summarize(frame)
        (OUTPUT / f"metrics_{label}.json").write_text(
            json.dumps(metrics[label], ensure_ascii=False, indent=2) + "\n"
        )
        if alpha == 1.0:
            sanity[label] = agreement(frame, ref_last, "alpha1_rerun_vs_c2_last_segment_267_reference")
        print(json.dumps({"phase": "alpha_complete", "label": label, "alpha": alpha,
                          "d10": metrics[label]["by_horizon"][9]}, ensure_ascii=False), flush=True)
        # Do not ship 100MB+ interpolated weights unless they are the decision checkpoint.
        if alpha not in (0.5,):
            shutil.rmtree(dest, ignore_errors=True)

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
        "alpha_order": [{"alpha": alpha, "label": label} for alpha, label in ALPHAS],
        "alpha_definition": "weights = (1 - alpha) * C2_best_179 + alpha * C2_last_267, float32",
        "c2_best_same_footing_d10_pooled_rank_ic": C2_BEST_SAME_FOOTING_D10_POOLED_RANK_IC,
        "c2_best_same_footing_note": (
            "alpha=0 not re-run; 0.18440434213929857 reproduced by P0b alpha=0, P0a init, P1 init, "
            "freeze-probe B init with this evaluator/seed"
        ),
        "provenance": provenance,
        "interpolated_weight_sha256": weight_shas,
        "reference_footing": {
            "note": "computed from smmt315 c2-alpha-oos-evaluation predictions; different evaluator run; not same footing",
            "c2_best_segment_179_d10": reference_footing["c2_best_segment_179"]["by_horizon"][9],
            "c2_last_segment_267_d10": reference_footing["c2_last_segment_267"]["by_horizon"][9],
        },
        "reload_checks": "alpha=1 uses torch.equal on float32 tensors vs C2 last",
        "sanity_policy": "identity-merge max|delta|/mean|delta|/spearman; soft report only; fp16 autocast + date seed + cross-GPU",
        "sanity": sanity,
        "metrics": metrics,
        "remaining": remaining,
        "decision_tree_note": (
            "Compare avg_alpha_050 D10 pooled_rank_ic to C2 best same-footing 0.1844 and to "
            "avg_alpha_100_c2_last same-footing. Averaging wins only if alpha=0.5 beats both endpoints; "
            "exploratory only."
        ),
    }
    write_summary(payload)
    if timed_out:
        sys.exit(0)


if __name__ == "__main__":
    main()
