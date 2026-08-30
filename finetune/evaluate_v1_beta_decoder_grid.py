"""Historical decoder calibration followed by one locked future evaluation.

The temperature/top-p choice is made only on a fixed subset of the historical
natural set.  The future set is evaluated after the choice is locked.
"""

import argparse
import gc
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model.kronos import auto_regressive_inference
from finetune.evaluate_v1_beta_checkpoints import (
    FEATURES,
    WindowStore,
    batches,
    find_evaluation_root,
    load_model,
    load_samples,
    stack_batch,
)


def choose_records(records, date_count, seed, per_date=100):
    by_date = defaultdict(list)
    for record in records:
        by_date[record["asof_date"]].append(record)
    dates = sorted(by_date)
    if len(dates) < date_count:
        raise RuntimeError(f"Need {date_count} historical dates, found {len(dates)}")
    positions = np.linspace(0, len(dates) - 1, date_count).round().astype(int)
    selected_dates = [dates[int(position)] for position in positions]
    selected = []
    for date_index, date in enumerate(selected_dates):
        rows = by_date[date]
        rng = np.random.default_rng(seed + date_index)
        # Enough cross-sectional coverage to choose a decoder setting without
        # making the historical grid dominate the locked future evaluation.
        take = min(len(rows), per_date)
        selected.extend(rows[int(i)] for i in rng.choice(len(rows), take, replace=False))
    return selected, selected_dates


def forecast(model, tokenizer, records, store, device, temperature, top_p, sample_count, seed):
    rows = []
    by_date = defaultdict(list)
    for record in records:
        by_date[record["asof_date"]].append(record)
    close_index = FEATURES.index("close")
    for date_index, (asof_date, date_records) in enumerate(sorted(by_date.items())):
        torch.manual_seed(seed + date_index)
        np.random.seed(seed + date_index)
        for items in batches(date_records, store, 64):
            batch = stack_batch(items, device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                paths = auto_regressive_inference(
                    tokenizer, model, batch["x"][:, :120], batch["stamp"][:, :120],
                    batch["stamp"][:, 120:130], max_context=512, pred_len=10,
                    clip=5, T=temperature, top_k=0, top_p=top_p,
                    sample_count=sample_count, verbose=False,
                    sector_id=batch["sector"], size_percentile=batch["percentile"],
                )
            predicted = paths[:, -10:, close_index]
            for index, item in enumerate(items):
                scale = item["std"][close_index] + 1e-5
                close = predicted[index] * scale + item["mean"][close_index]
                last = item["x"][119, close_index] * scale + item["mean"][close_index]
                rows.append({"asof_date": item["asof_date"], "return_10d": item["return_10d"],
                             "score": float(close[-1] / last - 1)})
    return pd.DataFrame(rows)


def metrics(frame):
    per_date = []
    for date, rows in frame.groupby("asof_date"):
        ranked = rows.sort_values("score", ascending=False)
        n = max(1, len(rows) // 5)
        per_date.append({
            "rank_ic": float(rows["score"].corr(rows["return_10d"], method="spearman")),
            "top_bottom_spread": float(ranked.head(n)["return_10d"].mean() - ranked.tail(n)["return_10d"].mean()),
            "top_return": float(ranked.head(n)["return_10d"].mean()),
        })
    values = pd.DataFrame(per_date)
    market = float(frame["return_10d"].mean())
    return {"dates": len(values), "samples": len(frame), "market_return": market,
            "mean_rank_ic": float(values["rank_ic"].mean()),
            "mean_top_bottom_spread": float(values["top_bottom_spread"].mean()),
            "mean_top_return": float(values["top_return"].mean()),
            "mean_top_excess": float(values["top_return"].mean() - market),
            "per_date": per_date}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--last-model", required=True)
    parser.add_argument("--best-model", required=True)
    parser.add_argument("--last-conditioned-model", default="")
    parser.add_argument("--best-conditioned-model", default="")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    evaluation_root, manifest = find_evaluation_root(Path(args.input_root))
    groups = load_samples(evaluation_root / manifest["artifacts"]["samples_file"])
    panel = __import__("pickle").loads((evaluation_root / manifest["artifacts"]["panel_file"]).read_bytes())
    store = WindowStore(panel, manifest["model_contract"]["sector_labels"])
    historical, dates = choose_records(groups["historical_natural"], 6, args.seed)
    future_records, future_dates = choose_records(groups["future_all"], 6, args.seed + 1, per_date=3000)
    configs = [{"temperature": t, "top_p": p, "sample_count": 1}
               for t in (0.50, 0.70) for p in (0.80, 0.95)]
    device = torch.device("cuda:0")
    tokenizer = __import__("model").KronosTokenizer.from_pretrained(args.tokenizer).to(device).eval()
    model_paths = {"last1058": args.last_model, "best1058": args.best_model}
    if args.last_conditioned_model: model_paths["conditioned_last40"] = args.last_conditioned_model
    if args.best_conditioned_model: model_paths["conditioned_best18"] = args.best_conditioned_model
    historical_results = {}
    future_results = {}
    for label, model_path in model_paths.items():
        model = load_model(Path(model_path), device)
        historical_results[label] = {}
        for config in configs:
            key = f"T{config['temperature']:g}_p{config['top_p']:g}"
            historical_results[label][key] = {"configuration": config,
                "metrics": metrics(forecast(model, tokenizer, historical, store, device, **config, seed=args.seed))}
        best_key = max(historical_results[label], key=lambda key: (
            historical_results[label][key]["metrics"]["mean_top_bottom_spread"],
            historical_results[label][key]["metrics"]["mean_rank_ic"]))
        chosen = next(item for item in configs if f"T{item['temperature']:g}_p{item['top_p']:g}" == best_key)
        future = forecast(model, tokenizer, future_records, store, device, **chosen, seed=args.seed + 10000)
        future_results[label] = {"selected_key": best_key, "configuration": chosen, "metrics": metrics(future)}
        del model; gc.collect(); torch.cuda.empty_cache()
    payload = {"task": "decoder_grid_historical_select_future_lock", "training_performed": False,
               "historical_dates": dates, "future_dates": future_dates,
               "historical_samples_per_date": 100, "future_samples_per_date": 3000,
               "configs": configs, "historical": historical_results,
               "future_locked": future_results, "model_sha256": {
                   label: hashlib.sha256(Path(path).joinpath("model.safetensors").read_bytes()).hexdigest()
                   for label, path in model_paths.items()}}
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    (output / "evaluation_manifest.json").write_text(json.dumps({"historical_dates": dates, "future_selection_locked": True, "configs": configs}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload["future_locked"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
