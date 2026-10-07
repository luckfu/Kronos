"""Validation-only C2 decode + per-path terminal returns + path-estimator scoring.

Zero training. Never reads the sealed OOS package. Validation contract is the
val-gen-ic one (temporal_symbol_validation_v1, 24-date np.linspace subsample,
12,256 windows, val_data sha256 4cce31bc...19bf7).

Decode (identical to the Baseline-2 / C2 production recipe):
  model.kronos.auto_regressive_inference, return_samples=True,
  T / top_p / N from the arm, top_k=0, clip=5, max_context=512, pred_len=10,
  lookback=120, seed 20260906 reset at the start of every (model, arm, date) task,
  sector_id + size_percentile conditioning, fp16 autocast on CUDA.
Every individual sampled path's terminal return close_d10 / last_close - 1 is
saved (``path_ret_d10_XX``) so estimators can be recomputed offline.

Estimators over the N terminal path returns of one window:
  mean (= the production score predicted_return_d10), median,
  trimmed_mean (drop the single max and the single min path),
  mean_ex_max (drop the single max path).
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

LOOKBACK = 120
PREDICT = 10
DECODE_SEED = 20260906
PATH_PREFIX = "path_ret_d10_"
ESTIMATORS = ("mean", "median", "trimmed_mean", "mean_ex_max")
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)  # weight on C2's rank; 0 = ours alone
STYLE_SHRINK_LAMBDA = 0.5
STYLE_COLUMNS = ("mom5", "mom20", "mom60", "vol20", "liq20")

ARMS = {
    "prod_t065_p80_n5": {
        "name": "prod_t065_p80_n5", "sample_count": 5, "temperature": 0.65,
        "top_p": 0.8, "seed": DECODE_SEED, "protocol": "production",
    },
    "prod_t065_p80_n16": {
        "name": "prod_t065_p80_n16", "sample_count": 16, "temperature": 0.65,
        "top_p": 0.8, "seed": DECODE_SEED, "protocol": "production_decode_n16_paths",
    },
}


# --------------------------------------------------------------------------- estimators
def path_columns(frame: pd.DataFrame) -> list[str]:
    return sorted(c for c in frame.columns if c.startswith(PATH_PREFIX))


def path_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = path_columns(frame)
    if not columns:
        raise ValueError("no per-path columns")
    return frame[columns].to_numpy(dtype=np.float64)


def estimator_scores(paths: np.ndarray) -> dict[str, np.ndarray]:
    """[windows, N] terminal path returns -> {estimator: [windows]}."""
    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim != 2 or paths.shape[1] < 1:
        raise ValueError(f"paths must be [windows, N], got {paths.shape}")
    ordered = np.sort(paths, axis=1)
    n = paths.shape[1]
    mean = paths.mean(axis=1)
    return {
        "mean": mean,
        "median": np.median(paths, axis=1),
        "trimmed_mean": ordered[:, 1:-1].mean(axis=1) if n >= 3 else mean,
        "mean_ex_max": ordered[:, :-1].mean(axis=1) if n >= 2 else mean,
    }


def add_estimator_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for name, values in estimator_scores(path_matrix(frame)).items():
        out[f"est_{name}"] = values
    return out


# ------------------------------------------------------------------------------ metrics
def _rank_ic(x, y) -> float:
    x = pd.Series(np.asarray(x, dtype=np.float64))
    y = pd.Series(np.asarray(y, dtype=np.float64))
    if len(x) < 3:
        return float("nan")
    return float(x.rank().corr(y.rank()))


def _deciles(score: pd.Series, buckets: int = 10) -> pd.Series:
    return pd.qcut(score.rank(method="first"), buckets, labels=False)


def day_metrics(group: pd.DataFrame, score: str, label: str = "return_10d") -> dict:
    """Per-date metrics; decile convention = score_val_checkpoint (qcut of first-rank)."""
    out = {
        "samples": int(len(group)),
        "return10d_rank_ic": _rank_ic(group[score], group[label]),
        "utility_rank_ic": (_rank_ic(group[score], group["utility"])
                            if "utility" in group else float("nan")),
        "top_bottom_decile_return10d": float("nan"),
        "top_bottom_quintile_return10d": float("nan"),
        "top_decile_within_ic": float("nan"),
        "top_decile_excess": float("nan"),
        "d10_minus_d9": float("nan"),
        "decile_monotonicity": float("nan"),
    }
    if len(group) >= 20:
        deciles = _deciles(group[score], 10)
        means = group[label].groupby(deciles).mean()
        out["top_bottom_decile_return10d"] = float(means.iloc[-1] - means.iloc[0])
        out["top_decile_excess"] = float(means.iloc[-1] - group[label].mean())
        out["d10_minus_d9"] = float(means.iloc[-1] - means.iloc[-2])
        out["decile_monotonicity"] = _rank_ic(np.arange(len(means)), means.to_numpy())
        top = group[deciles == deciles.max()]
        out["top_decile_within_ic"] = _rank_ic(top[score], top[label])
        quint = _deciles(group[score], 5)
        qmeans = group[label].groupby(quint).mean()
        out["top_bottom_quintile_return10d"] = float(qmeans.iloc[-1] - qmeans.iloc[0])
    return out


def summarize_score(frame: pd.DataFrame, score: str, label: str = "return_10d") -> dict:
    daily = []
    for asof, group in frame.groupby("asof_date", sort=True):
        daily.append({"asof_date": str(asof), **day_metrics(group, score, label)})
    table = pd.DataFrame(daily)
    ic = table["return10d_rank_ic"].dropna()
    ic_std = float(ic.std(ddof=1)) if len(ic) > 1 else float("nan")

    def mean_of(column):
        values = table[column].dropna()
        return float(values.mean()) if len(values) else None

    return {
        "score": score,
        "samples": int(len(frame)),
        "signal_dates": int(frame["asof_date"].nunique()),
        "return10d_rank_ic_daily": float(ic.mean()) if len(ic) else None,
        "return10d_rank_ic_pooled": _rank_ic(frame[score], frame[label]),
        "return10d_rank_icir": (float(ic.mean() / ic_std)
                                if len(ic) > 1 and ic_std > 0 else None),
        "return10d_rank_ic_pos_rate": float((ic > 0).mean()) if len(ic) else None,
        "return10d_rank_ic_se": (float(ic_std / math.sqrt(len(ic)))
                                 if len(ic) > 1 else None),
        "utility_rank_ic_daily": mean_of("utility_rank_ic"),
        "top_bottom_decile_return10d": mean_of("top_bottom_decile_return10d"),
        "top_bottom_quintile_return10d": mean_of("top_bottom_quintile_return10d"),
        "top_decile_within_ic": mean_of("top_decile_within_ic"),
        "top_decile_excess": mean_of("top_decile_excess"),
        "d10_minus_d9": mean_of("d10_minus_d9"),
        "decile_monotonicity": mean_of("decile_monotonicity"),
        "by_signal_date": daily,
    }


def by_date(summary: dict, key: str = "return10d_rank_ic") -> dict:
    return {row["asof_date"]: row[key] for row in summary["by_signal_date"]
            if row.get(key) is not None and np.isfinite(row[key])}


def paired_stats(a: dict, b: dict) -> dict | None:
    """Paired by-date a - b over common dates; t with n-1 df (overlap -> optimistic)."""
    dates = sorted(set(a) & set(b))
    diffs = np.array([float(a[d]) - float(b[d]) for d in dates], dtype=np.float64)
    n = len(diffs)
    if n == 0:
        return None
    mean = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if n > 1 else None
    se = sd / math.sqrt(n) if sd is not None else None
    return {
        "n": n, "mean_diff": mean, "sd_diff": sd, "se_diff": se,
        "t": (mean / se) if se else None,
        "wins": int((diffs > 0).sum()),
    }


def score_arm(frame: pd.DataFrame) -> dict:
    """All estimators + paired comparisons vs the mean estimator (pre-registered rule 1)."""
    work = add_estimator_columns(frame)
    drift = float(np.max(np.abs(work["est_mean"] - work["predicted_return_d10"])))
    if drift > 1e-9:
        raise RuntimeError(f"path mean != predicted_return_d10 (max abs diff {drift})")
    estimators = {name: summarize_score(work, f"est_{name}") for name in ESTIMATORS}
    base = estimators["mean"]
    versus_mean = {}
    for name in ESTIMATORS:
        if name == "mean":
            continue
        ic = paired_stats(by_date(estimators[name]), by_date(base))
        within = paired_stats(by_date(estimators[name], "top_decile_within_ic"),
                              by_date(base, "top_decile_within_ic"))
        versus_mean[name] = {"daily_ic": ic, "top_decile_within_ic": within,
                             **estimator_decision(ic, estimators[name], base)}
    return {
        "samples": int(len(work)),
        "signal_dates": int(work["asof_date"].nunique()),
        "sample_count": int(len(path_columns(work))),
        "path_mean_vs_saved_max_abs_diff": drift,
        "estimators": estimators,
        "versus_mean": versus_mean,
    }


def estimator_decision(ic: dict | None, candidate: dict, base: dict) -> dict:
    """Pre-registered rule 1: dIC >= +0.01 AND paired t >= 2 AND top-decile within-IC fixed.

    "Fixed" = candidate's daily-mean within-top-decile rank IC is >= 0 and strictly
    higher than the mean estimator's.
    """
    delta = None if ic is None else ic["mean_diff"]
    t = None if ic is None else ic["t"]
    cand_within = candidate.get("top_decile_within_ic")
    base_within = base.get("top_decile_within_ic")
    gain_ok = delta is not None and delta >= 0.01
    t_ok = t is not None and t >= 2.0
    within_ok = (cand_within is not None and base_within is not None
                 and cand_within >= 0.0 and cand_within > base_within)
    return {"rule1_gain_ok": bool(gain_ok), "rule1_t_ok": bool(t_ok),
            "rule1_top_decile_fixed": bool(within_ok),
            "rule1_adopt": bool(gain_ok and t_ok and within_ok)}


# ------------------------------------------------------------------------ blend / style
def _pct_rank(values: pd.Series) -> pd.Series:
    return values.rank(pct=True)


def rank_blend(frame: pd.DataFrame, c2_col: str, ours_col: str, weight: float) -> pd.Series:
    """Per-date percentile-rank blend: weight * rank(C2) + (1 - weight) * rank(ours)."""
    grouped = frame.groupby("asof_date", sort=False)
    c2 = grouped[c2_col].transform(_pct_rank)
    ours = grouped[ours_col].transform(_pct_rank)
    return weight * c2 + (1.0 - weight) * ours


def merged_pair(c2: pd.DataFrame, ours: pd.DataFrame, c2_score="est_mean",
                ours_score="est_mean") -> pd.DataFrame:
    keys = ["symbol", "asof_date"]
    left = c2[keys + ["return_10d", "utility", c2_score]].rename(columns={c2_score: "s_c2"})
    right = ours[keys + ["return_10d", ours_score]].rename(
        columns={ours_score: "s_ours", "return_10d": "return_10d_ours"})
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(c2) or len(merged) != len(ours):
        raise RuntimeError(f"blend join lost rows: {len(merged)} vs {len(c2)}/{len(ours)}")
    if float(np.max(np.abs(merged["return_10d"] - merged["return_10d_ours"]))) > 1e-9:
        raise RuntimeError("label mismatch between C2 and ours")
    return merged.drop(columns="return_10d_ours")


def blend_grid(merged: pd.DataFrame, weights=BLEND_WEIGHTS) -> dict:
    """Pre-registered rule 2: evaluate w in {0,.25,.5,.75,1}; pick argmax validation daily IC."""
    work = merged.copy()
    summaries = {}
    for weight in weights:
        column = f"blend_w{weight:.2f}"
        work[column] = rank_blend(work, "s_c2", "s_ours", weight)
        summaries[f"{weight:.2f}"] = summarize_score(work, column)
    best = max(summaries, key=lambda k: summaries[k]["return10d_rank_ic_daily"])
    ours_alone = by_date(summaries[f"{0.0:.2f}"])
    c2_alone = by_date(summaries[f"{1.0:.2f}"])
    return {
        "weights_on_c2": [float(w) for w in weights],
        "by_weight": {k: _headline(v) | {"by_date_ic": by_date(v)} for k, v in summaries.items()},
        "chosen_weight_on_c2": float(best),
        "chosen_vs_ours_alone": paired_stats(by_date(summaries[best]), ours_alone),
        "chosen_vs_c2_alone": paired_stats(by_date(summaries[best]), c2_alone),
        "c2_minus_ours": paired_stats(c2_alone, ours_alone),
        "score_correlation_daily_spearman": float(np.nanmean([
            _rank_ic(g["s_c2"], g["s_ours"]) for _, g in merged.groupby("asof_date")])),
        "_frame": work,
    }


def _headline(summary: dict) -> dict:
    keys = ("return10d_rank_ic_daily", "return10d_rank_ic_pooled", "return10d_rank_icir",
            "return10d_rank_ic_pos_rate", "utility_rank_ic_daily",
            "top_bottom_decile_return10d", "top_decile_within_ic", "top_decile_excess",
            "d10_minus_d9")
    return {key: summary.get(key) for key in keys}


def headline(summary: dict) -> dict:
    return _headline(summary)


def load_analysis_module(repo: Path):
    """The descriptive-analysis helpers (gauss_rank, merge_small, dummies, residualize,
    build_features) so the style transform is byte-for-byte the documented one."""
    import importlib.util

    path = Path(repo) / "finetune" / "analysis" / "beta_v21_c1_why_c2_better.py"
    spec = importlib.util.spec_from_file_location("beta_v21_c1_why_c2_better", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def style_features(analysis, panel: dict, records: list[dict], labels: pd.DataFrame) -> pd.DataFrame:
    """Lookback-only features at asof (sector, size_pct, mom*, vol20, liq20) per window."""
    recs = pd.DataFrame(records)[["symbol", "asof_date", "start_index"]].merge(
        labels[["symbol", "asof_date", "return_10d"]].rename(columns={"return_10d": "ret_pkg"}),
        on=["symbol", "asof_date"], how="inner", validate="one_to_one")
    feats = analysis.build_features(panel, recs)
    if float(np.nanmax(np.abs(feats["ret_check"] - feats["ret_pkg"]))) > 1e-9:
        raise RuntimeError("style features: panel return_10d does not match labels")
    return feats.drop(columns=["ret_check", "ret_pkg"])


def style_shrink(analysis, frame: pd.DataFrame, features: pd.DataFrame, score: str,
                 lam: float = STYLE_SHRINK_LAMBDA) -> pd.Series:
    """Per date: gauss-rank score, OLS on sector(fine, merged)+size+style gauss ranks,
    keep fitted + lam * residual (lam fixed a priori at 0.5). Same as val_postproc."""
    keys = ["symbol", "asof_date"]
    work = frame[keys + [score]].merge(features, on=keys, how="left", validate="one_to_one",
                                       indicator=True)
    if (work.pop("_merge") != "both").any():
        raise RuntimeError("style features missing for some windows")
    # pandas>=3 keeps NaN through astype(str); pandas 2 (as in the analysis run) gives "nan".
    work["sector"] = work["sector"].astype(object).where(work["sector"].notna(), "nan").astype(str)
    out = pd.Series(np.nan, index=work.index)
    for _, g in work.groupby("asof_date", sort=False):
        fine = analysis.merge_small(g.sector.values)
        style = [analysis.gauss_rank(g[c].fillna(g[c].median()).values) for c in STYLE_COLUMNS]
        design = analysis.dummies(fine) + [g.size_pct.values - g.size_pct.mean()] + style
        gs = analysis.gauss_rank(g[score].values)
        resid = analysis.residualize(gs, design)
        out.loc[g.index] = (gs - resid) + lam * resid
    out.index = frame.index
    return out


def style_shrink_report(analysis, frame: pd.DataFrame, features: pd.DataFrame,
                        score: str) -> dict:
    """Rule 3: style-shrink (lambda 0.5) vs raw, paired by date, on validation."""
    work = frame.copy()
    work["_style_shrink"] = style_shrink(analysis, work, features, score).to_numpy()
    raw = summarize_score(work, score)
    shrunk = summarize_score(work, "_style_shrink")
    return {"raw": _headline(raw), "style_shrink": _headline(shrunk),
            "lambda": STYLE_SHRINK_LAMBDA,
            "shrink_minus_raw": paired_stats(by_date(shrunk), by_date(raw)),
            "by_date_ic_style_shrink": by_date(shrunk)}


# ------------------------------------------------------------------------------ decode
def decode_with_paths(arm, model, tokenizer, records, store, device, effective_batch,
                      use_amp, label, auto_regressive_inference) -> pd.DataFrame:
    """Same RNG / batching / math as evaluate_beta_v21_generative_return_oos.decode_records,
    plus every sampled path's terminal return (``path_ret_d10_XX``) and start_index."""
    import torch

    from evaluate_beta_v21_generative_return_oos import batches, stack_batch, to_daily
    from evaluate_beta_v21_time_oos import FEATURES

    sample_count = int(arm["sample_count"])
    batch_size = max(1, int(effective_batch) // sample_count)
    close_index = FEATURES.index("close")
    torch.manual_seed(int(arm["seed"]))
    np.random.seed(int(arm["seed"]))
    starts = {(str(r["symbol"]), r["asof_date"]): int(r["start_index"]) for r in records}

    rows = []
    for items in batches(records, store, batch_size):
        batch = stack_batch(items, device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=use_amp and device.type == "cuda"):
                forecast = auto_regressive_inference(
                    tokenizer, model,
                    batch["x"][:, :LOOKBACK],
                    batch["stamp"][:, :LOOKBACK],
                    batch["stamp"][:, LOOKBACK: LOOKBACK + PREDICT],
                    max_context=512, pred_len=PREDICT, clip=5,
                    T=float(arm["temperature"]), top_k=0, top_p=float(arm["top_p"]),
                    sample_count=sample_count, verbose=False,
                    sector_id=batch["sector"], size_percentile=batch["percentile"],
                    return_samples=True,
                )
        future = forecast[:, :, -PREDICT:, close_index]
        for index, item in enumerate(items):
            scale = float(item["std"][close_index]) + 1e-5
            shift = float(item["mean"][close_index])
            last_close = float(item["x"][LOOKBACK - 1, close_index]) * scale + shift
            paths = future[index].astype(np.float64) * scale + shift
            cumulative = paths / last_close - 1.0
            mean_cumulative = cumulative.mean(axis=0)
            actual = (item["x"][LOOKBACK: LOOKBACK + PREDICT, close_index].astype(np.float64)
                      * scale + shift) / last_close - 1.0
            predicted_daily = to_daily(cumulative)
            actual_daily = to_daily(actual[None, ...])[0]
            row = {
                "model": label, "arm": arm["name"], "protocol": arm["protocol"],
                "sample_count": sample_count, "temperature": float(arm["temperature"]),
                "top_p": float(arm["top_p"]), "seed": int(arm["seed"]),
                "symbol": item["symbol"], "asof_date": item["asof_date"],
                "target_date": item["target_date"],
                "start_index": starts[(str(item["symbol"]), item["asof_date"])],
                "direction": item["direction"], "return_10d": item["return_10d"],
                "utility": item["utility"], "date_id": item["date_id"],
                "sector_id": int(item["sector_id"]),
                "size_percentile": float(item["size_percentile"]),
                "predicted_return_d10": float(mean_cumulative[-1]),
                "predicted_path_vol": float(predicted_daily.std(axis=1).mean()),
                "predicted_terminal_dispersion": (
                    float(cumulative[:, -1].std(ddof=1)) if sample_count > 1 else None),
                "realized_path_vol": float(actual_daily.std()),
                "score_kind": "predicted_return_d10_generative_ar_mean",
            }
            for path in range(sample_count):
                row[f"{PATH_PREFIX}{path:02d}"] = float(cumulative[path, -1])
            for horizon in range(PREDICT):
                row[f"predicted_return_d{horizon + 1}"] = float(mean_cumulative[horizon])
                row[f"actual_return_d{horizon + 1}"] = float(actual[horizon])
            rows.append(row)
    return pd.DataFrame(rows)


def shard_name(label: str, arm_name: str, date: str) -> str:
    return f"{label}__{arm_name}__{date}.csv.gz"


def claim(claims: Path, name: str, rank: int) -> bool:
    claims.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(str(claims / (name + ".claim")), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(handle, "w") as stream:
        stream.write(json.dumps({"rank": rank, "time": time.time()}))
    return True


def worker_from_plan(plan_path: Path, rank: int, world_size: int) -> int:
    """One phase (one model x one arm): dynamic in-order date claims, shard per date."""
    import gc
    import hashlib
    import pickle

    import torch

    import model as model_package
    from model import Kronos, KronosTokenizer
    from model.kronos import auto_regressive_inference
    from evaluate_beta_v21_val_gen_ic import ValWindowStore

    plan = json.loads(Path(plan_path).read_text())
    phase = plan["phase"]
    arm = ARMS[phase["arm"]]
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif os.environ.get("KRONOS_ALLOW_CPU_WORKER") == "1":  # unit tests only
        device = torch.device("cpu")
    else:
        raise RuntimeError(f"worker {rank} has no GPU")
    model_file = Path(model_package.__file__).resolve().parent / "kronos.py"
    code_sha = hashlib.sha256(model_file.read_bytes()).hexdigest()
    with Path(plan["val_data"]).open("rb") as handle:
        panel = pickle.load(handle)
    store = ValWindowStore(panel, plan["sector_labels"])
    by_day: dict[str, list[dict]] = {}
    for record in json.loads(Path(plan["records_file"]).read_text()):
        by_day.setdefault(record["asof_date"], []).append(record)
    tokenizer = KronosTokenizer.from_pretrained(plan["tokenizer_dir"]).to(device).eval()
    model = Kronos.from_pretrained(phase["checkpoint_dir"], **phase.get("load_kwargs", {}))
    model = model.to(device).eval()
    print(json.dumps({
        "phase": "worker_model_loaded", "rank": rank, "phase_key": phase["key"],
        "label": phase["label"], "arm": arm["name"], "effective_batch": phase["effective_batch"],
        "model_code": str(model_file), "model_code_kronos_py_sha256": code_sha,
        "load_kwargs": phase.get("load_kwargs", {}),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }), flush=True)
    shards = Path(plan["shards"])
    claims = Path(plan["claims"])
    deadline = float(plan["deadline"])
    use_amp = device.type == "cuda"
    for date in plan["dates"]:
        name = shard_name(phase["label"], arm["name"], date)
        shard = shards / name
        if shard.is_file() or not claim(claims, name, rank):
            continue
        if time.time() > deadline:
            print(json.dumps({"phase": "worker_deadline", "rank": rank, "skipped_from": date}),
                  flush=True)
            break
        started = time.time()
        result = decode_with_paths(arm, model, tokenizer, by_day[date], store, device,
                                   int(phase["effective_batch"]), use_amp, phase["label"],
                                   auto_regressive_inference)
        staging = shard.with_name(shard.name + ".tmp")
        result.to_csv(staging, index=False, compression="gzip")
        staging.replace(shard)
        print(json.dumps({
            "phase": "shard_done", "rank": rank, "phase_key": phase["key"], "date": date,
            "rows": int(len(result)), "seconds": round(time.time() - started, 1),
            "return10d_rank_ic_mean": _rank_ic(result["predicted_return_d10"],
                                               result["return_10d"]),
        }), flush=True)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(json.dumps({"phase": "worker_finished", "rank": rank, "phase_key": phase["key"]}),
          flush=True)
    return 0


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) >= 3 and _sys.argv[1] == "--worker-plan":
        raise SystemExit(worker_from_plan(Path(_sys.argv[2]),
                                          int(os.environ["KRONOS_SWEEP_RANK"]),
                                          int(os.environ.get("KRONOS_SWEEP_WORLD", "2"))))
    raise SystemExit("usage: --worker-plan PLAN (launched by the kaggle runner)")
