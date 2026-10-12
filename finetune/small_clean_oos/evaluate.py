"""Frozen, paired full OOS evaluation of clean Small best and original C2 best."""

import gc
import hashlib
import json
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

FEATURES = ["open", "high", "low", "close", "volume", "amount"]
LOOKBACK, PREDICT, WINDOW = 120, 10, 130
EXPECTED_SAMPLES, EXPECTED_DATES = 175428, 34
PANEL_SHA = "be159cd89474a22dc000403137c7d7647cd2719eb325631a9f5ba701defa8118"
SAMPLES_SHA = "d9173a69f815d0eb0375cd34d38436677bd5d738163ca3911f877ab646e2cfd5"
IDENTITIES_SHA = "057f6912f783483ad43914c90d83ea4a54d299ef7400831c0a0dbf814fd665d1"
METADATA_SHA = "697cadd672d53b8fa0a990f6c5b7fba2f88cb26aad88c3b91cfcc3865d0a1e3c"
TRAIN_SHA = "c14a31fc29b3b3d1ec9ae588475a8b41eead2be801b38dc57e56c6d20f25e755"
VAL_SHA = "ce7f1d3020e5cd22565839432b5013850955a571c01a8dbf9e5bb2da3231e583"
TOKENIZER_SHA = "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"
C2_SHA = "4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a"
DECODE = dict(temperature=0.65, top_p=0.8, sample_count=5, seed=20260906,
              effective_batch=64, window_batch=12)
OUTPUT = Path(os.environ.get("SMALL_OOS_OUTPUT", "/kaggle/working/small_clean_oos"))


def phase(name, **values):
    print(json.dumps(dict(time=time.time(), phase=name, **values)), flush=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked(path, expected):
    actual = sha(path)
    if actual != expected:
        raise RuntimeError(f"SHA mismatch: {path}: {actual}")
    return actual


def unique(paths):
    paths = sorted(paths)
    if len(paths) != 1:
        raise RuntimeError(f"Expected unique input, found {paths}")
    return paths[0]


def identity(record):
    return "|".join(str(record[key]) for key in
                    ("symbol", "start_index", "asof_date", "target_date"))


def load_oos(root):
    import pandas as pd

    manifest = json.loads((root / "evaluation_manifest.json").read_text())
    for key, expected in (("panel", PANEL_SHA), ("samples", SAMPLES_SHA)):
        checked(root / manifest["artifacts"][key + "_file"], expected)
    with (root / manifest["artifacts"]["panel_file"]).open("rb") as handle:
        panel = pickle.load(handle)
    records = [json.loads(line) for line in
               (root / manifest["artifacts"]["samples_file"]).read_text().splitlines()]
    if {r["set"] for r in records} != {"incremental_future_all"}:
        raise RuntimeError("Unexpected OOS sample set")
    ids = [identity(r) for r in records]
    dates = sorted({r["asof_date"] for r in records})
    if (len(records), len(set(ids)), len(dates)) != (
            EXPECTED_SAMPLES, EXPECTED_SAMPLES, EXPECTED_DATES):
        raise RuntimeError("Incomplete or duplicate OOS samples")
    if (dates[0], dates[-1], max(r["target_date"] for r in records)) != (
            "2026-08-03", "2026-09-17", "2026-10-09"):
        raise RuntimeError("OOS date drift")
    # prepare_actual_traded_oos signs sorted identities with a trailing newline.
    digest = hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()
    if digest != IDENTITIES_SHA:
        raise RuntimeError(f"OOS identity hash drift: {digest}")
    for symbol, frame in panel.items():
        if (not isinstance(frame.index, pd.DatetimeIndex)
                or not frame.index.is_monotonic_increasing
                or frame.index.has_duplicates or (frame.volume <= 0).any()):
            raise RuntimeError(f"Invalid traded-row panel: {symbol}")
    return panel, records, manifest


def audit_training_dates(root):
    dates = {}
    for name, expected in (("train_data.pkl", TRAIN_SHA), ("val_data.pkl", VAL_SHA)):
        path = unique(Path(root).glob("**/" + name))
        checked(path, expected)
        with path.open("rb") as handle:
            panel = pickle.load(handle)
        latest = max(frame.index.max() for frame in panel.values())
        if str(latest.date()) >= "2026-08-03":
            raise RuntimeError(f"Training/selection data overlaps OOS: {name}: {latest}")
        dates[name] = str(latest.date())
        del panel
        gc.collect()
    phase("temporal_isolation_verified", max_row_dates=dates, oos_start="2026-08-03")
    return dates


class WindowStore:
    def __init__(self, panel, sector_labels):
        self.panel = panel
        self.sector_map = {value: i for i, value in enumerate(sector_labels)}

    def prepare(self, record):
        import numpy as np
        import pandas as pd

        frame = self.panel[str(record["symbol"])]
        start = int(record["start_index"])
        window = frame.iloc[start:start + WINDOW]
        if len(window) != WINDOW:
            raise RuntimeError(f"Incomplete window: {identity(record)}")
        dates = pd.DatetimeIndex(window.index)
        if (str(dates[119].date()), str(dates[129].date())) != (
                record["asof_date"], record["target_date"]):
            raise RuntimeError("Window identity drift")
        values = window[FEATURES].to_numpy(dtype=np.float32)
        raw_close = window.close.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or (raw_close <= 0).any():
            raise RuntimeError("Invalid OHLCVA")
        mean, std = values[:120].mean(0), values[:120].std(0)
        x = np.clip((values[:120] - mean) / (std + 1e-5), -5, 5)
        actual = raw_close[120:130] / raw_close[119] - 1
        if not np.isclose(actual[-1], record["return_10d"], atol=1e-8, rtol=1e-6):
            raise RuntimeError("Raw return label drift")
        percentile = float(window.size_percentile.iloc[119])
        if not np.isfinite(percentile) or not 0 <= percentile <= 1:
            raise RuntimeError("Invalid size percentile")
        sector = str(window.sector.iloc[119])
        if sector != record["sector"]:
            raise RuntimeError("Sector label drift")
        if sector not in self.sector_map and sector != "unknown":
            raise RuntimeError(f"Unmapped sector: {sector}")
        return dict(
            identity=identity(record), symbol=record["symbol"],
            asof_date=record["asof_date"], target_date=record["target_date"],
            sector=sector, size_decile=record["size_decile"],
            future_dates=[str(d.date()) for d in dates[120:130]],
            x=x.astype(np.float32), mean=mean, std=std,
            stamp=np.column_stack([dates.minute, dates.hour, dates.weekday,
                                   dates.day, dates.month]).astype(np.float32),
            last_close=raw_close[119], actual=actual,
            sector_id=self.sector_map.get(sector, 86), percentile=percentile)


def decode(records, store, model, tokenizer, device, label):
    import numpy as np
    import pandas as pd
    import torch
    from model.kronos import auto_regressive_inference

    torch.manual_seed(DECODE["seed"])
    np.random.seed(DECODE["seed"])
    rows = []
    batch_size = DECODE["window_batch"]
    for offset in range(0, len(records), batch_size):
        items = [store.prepare(r) for r in records[offset:offset + batch_size]]
        x = torch.as_tensor(np.stack([i["x"] for i in items]), device=device)
        stamp = torch.as_tensor(np.stack([i["stamp"] for i in items]), device=device)
        sectors = torch.as_tensor([i["sector_id"] for i in items], device=device)
        sizes = torch.as_tensor([i["percentile"] for i in items],
                                dtype=torch.float32, device=device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            forecasts = auto_regressive_inference(
                tokenizer, model, x, stamp[:, :120], stamp[:, 120:130],
                max_context=512, pred_len=10, clip=5, T=DECODE["temperature"],
                top_k=0, top_p=DECODE["top_p"], sample_count=5, verbose=False,
                sector_id=sectors, size_percentile=sizes, return_samples=True)
        for i, item in enumerate(items):
            predicted_close = (forecasts[i, :, -10:, 3].astype(np.float64)
                               * (float(item["std"][3]) + 1e-5)
                               + float(item["mean"][3])).mean(0)
            predicted = predicted_close / item["last_close"] - 1
            if not np.isfinite(predicted).all():
                raise RuntimeError("Nonfinite model prediction")
            row = {k: item[k] for k in ("identity", "symbol", "asof_date",
                                       "target_date", "sector", "size_decile",
                                       "last_close")}
            row["model"] = label
            for h in range(10):
                row[f"predicted_return_d{h+1}"] = predicted[h]
                row[f"actual_return_d{h+1}"] = item["actual"][h]
                row[f"label_date_d{h+1}"] = item["future_dates"][h]
            rows.append(row)
        if offset % (batch_size * 100) == 0:
            phase("prediction", model=label, date=records[0]["asof_date"],
                  windows=min(offset + batch_size, len(records)), total=len(records))
    return pd.DataFrame(rows)


def score(frame):
    import numpy as np

    predicted = frame[[f"predicted_return_d{h}" for h in range(1, 11)]].to_numpy()
    actual = frame[[f"actual_return_d{h}" for h in range(1, 11)]].to_numpy()
    error = predicted - actual
    direction = np.sign(predicted) == np.sign(actual)
    result = dict(samples=len(frame), path_direction=float(direction.mean()),
                  path_return_mae=float(np.abs(error).mean()),
                  path_return_rmse=float(np.sqrt(np.square(error).mean())),
                  horizons={})
    for h in range(10):
        result["horizons"][str(h + 1)] = dict(
            direction=float(direction[:, h].mean()),
            return_mae=float(np.abs(error[:, h]).mean()),
            return_rmse=float(np.sqrt(np.square(error[:, h]).mean())))
    result["d10_daily_rank_ic"] = float(frame.groupby("asof_date").apply(
        lambda day: day.predicted_return_d10.corr(day.actual_return_d10,
                                                method="spearman"),
        include_groups=False).mean())
    return result


def paired_comparison(c2, clean):
    import numpy as np

    c2 = c2.sort_values("identity").reset_index(drop=True)
    clean = clean.sort_values("identity").reset_index(drop=True)
    if not c2.identity.equals(clean.identity) or c2.identity.duplicated().any():
        raise RuntimeError("Unpaired evaluation")
    actual_cols = [f"actual_return_d{h}" for h in range(1, 11)]
    if not np.array_equal(c2[actual_cols].to_numpy(), clean[actual_cols].to_numpy()):
        raise RuntimeError("Paired labels differ")
    summaries = dict(c2_best=score(c2), clean_best=score(clean))
    # Resample paired consecutive signal-date blocks, preserving within-date dependence.
    dates = sorted(c2.asof_date.unique())
    daily = {label: [score(frame[frame.asof_date == day]) for day in dates]
             for label, frame in (("c2_best", c2), ("clean_best", clean))}
    rng = np.random.default_rng(20261012)
    metrics = ("path_direction", "path_return_mae", "path_return_rmse")
    delta = {m: [] for m in metrics}
    for _ in range(2000):
        indices = []
        while len(indices) < len(dates):
            start = rng.integers(0, len(dates) - 10 + 1)
            indices.extend(range(start, start + 10))
        indices = indices[:len(dates)]
        for m in metrics:
            def aggregate(label):
                weights = np.array([daily[label][i]["samples"] for i in indices])
                values = np.array([daily[label][i][m] for i in indices])
                if m.endswith("rmse"):
                    return np.sqrt(np.average(values ** 2, weights=weights))
                return np.average(values, weights=weights)
            delta[m].append(aggregate("clean_best") - aggregate("c2_best"))
    summaries["paired_delta_clean_minus_c2"] = {
        m: dict(value=summaries["clean_best"][m] - summaries["c2_best"][m],
                block_bootstrap_95ci=np.quantile(delta[m], [.025, .975]).tolist())
        for m in metrics}
    summaries["uncertainty"] = dict(signal_dates=len(dates), block_length=10,
        replicates=2000, seed=20261012, overlapping_labels=True,
        note="Only 34 time observations; prior evaluation overlap, not a new blind set.")
    return summaries


def worker(plan, rank):
    import pandas as pd
    import torch
    from model import Kronos, KronosTokenizer

    torch.set_num_threads(2)
    label = ("c2_best", "clean_best")[rank]
    device = torch.device("cuda:0")
    phase("load_dataset", model=label)
    panel, records, _ = load_oos(Path(plan["oos_root"]))
    store = WindowStore(panel, plan["sector_labels"])
    model = Kronos.from_pretrained(
        plan["models"][label]["directory"], num_sectors=86, num_size_buckets=0,
        context_layer=6, use_size_percentile=True, size_mlp_hidden_dim=64).to(device).eval()
    tokenizer = KronosTokenizer.from_pretrained(plan["tokenizer"]).to(device).eval()
    phase("models_loaded", model=label, sha256=plan["models"][label]["sha256"])
    for date in sorted({r["asof_date"] for r in records}):
        if time.time() > plan["deadline"] - 600:
            phase("planned_stop", model=label)
            break
        path = OUTPUT / "shards" / f"{label}_{date}.csv.gz"
        selected = [r for r in records if r["asof_date"] == date]
        if path.exists():
            existing = pd.read_csv(path)
            if set(existing.identity) != {identity(r) for r in selected}:
                raise RuntimeError("Invalid resumed date shard")
            continue
        began = time.time()
        frame = decode(selected, store, model, tokenizer, device, label)
        temporary = path.with_suffix(".tmp")
        frame.to_csv(temporary, index=False, compression="gzip")
        temporary.replace(path)
        phase("date_finished", model=label, date=date, windows=len(frame),
              seconds=time.time() - began, metrics=score(frame))
    phase("worker_finished", model=label)


def main():
    phase("started")
    if os.environ.get("SMALL_OOS_RANK") is not None:
        plan = json.loads((OUTPUT / "plan.json").read_text())
        worker(plan, int(os.environ["SMALL_OOS_RANK"]))
        return
    started = float(os.environ.get("SMALL_OOS_STARTED", time.time()))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "shards").mkdir(exist_ok=True)
    stop = threading.Event()
    monitor = threading.Thread(target=lambda: heartbeat(stop), daemon=True)
    monitor.start()
    try:
        parent_main(started)
    finally:
        stop.set()
        monitor.join()


def heartbeat(stop):
    while not stop.wait(30):
        phase("heartbeat", role="coordinator")


def parent_main(started):
    import pandas as pd
    import torch
    import swanlab

    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(devices) != 2 or any("T4" not in name for name in devices):
        raise RuntimeError(f"Expected dual T4: {devices}")
    root = Path("/kaggle/input")
    oos = unique(p.parent for p in root.glob("**/evaluation_manifest.json")
                 if json.loads(p.read_text()).get("artifacts", {}).get("panel_sha256") == PANEL_SHA)
    metadata = unique(p for p in root.glob("**/asset_metadata.csv")
                      if sha(p) == METADATA_SHA)
    training_max_dates = audit_training_dates(root)
    phase("load_sector_metadata")
    labels = set()
    for part in pd.read_csv(metadata, usecols=["sector"], chunksize=500000):
        labels.update(part.sector.dropna().astype(str))
    labels = sorted(labels)
    if len(labels) != 86:
        raise RuntimeError("Training sector vocabulary drift")
    panel, records, manifest = load_oos(oos)
    store = WindowStore(panel, labels)
    phase("preflight_all_windows", samples=len(records))
    for record in records:
        store.prepare(record)
    del store, panel
    gc.collect()
    models = {}
    patterns = dict(c2_best="**/small_0.1_stage2_cosine_refinement/checkpoints/best_model/model.safetensors",
                    clean_best="**/small_0.1_clean_wc_dual_t4_20261011/checkpoints/best_model/model.safetensors")
    for label, pattern in patterns.items():
        weights = unique(root.glob(pattern))
        metric = json.loads((weights.parent / "best_metric.json").read_text())
        expected_segment = 179 if label == "c2_best" else 518
        if metric["segment"] != expected_segment:
            raise RuntimeError(f"Unexpected checkpoint: {metric}")
        actual_sha = sha(weights)
        if label == "c2_best" and actual_sha != C2_SHA:
            raise RuntimeError("C2 model hash mismatch")
        if label == "clean_best":
            result = json.loads((weights.parents[2] / "summary.json").read_text())["final_result"]
            if result["status"] != "completed" or result["completed_segments"] != 520:
                raise RuntimeError("Clean training incomplete")
        models[label] = dict(directory=str(weights.parent), sha256=actual_sha, metric=metric)
    tokenizer = unique(root.glob("**/clean_small_runtime/models/Kronos-Tokenizer-base/model.safetensors"))
    checked(tokenizer, TOKENIZER_SHA)
    plan = dict(oos_root=str(oos), sector_labels=labels, models=models,
                tokenizer=str(tokenizer.parent), deadline=started + 40500,
                decode=DECODE, artifacts=manifest["artifacts"],
                identities_sha256=IDENTITIES_SHA, samples=EXPECTED_SAMPLES,
                dates=EXPECTED_DATES, source_commit=os.environ["SMALL_OOS_COMMIT"],
                training_selection_max_row_dates=training_max_dates,
                source_sha256=os.environ["SMALL_OOS_SOURCE_SHA"],
                label_semantics="next 10 actual traded rows; original unclipped closes",
                selection="preregistered training best; no OOS tuning")
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2))
    swanlab.login(api_key=os.environ["SWANLAB_API_KEY"])
    run = swanlab.init(id="smco1012", resume="allow", project="finance",
                       workspace="roc_fu", experiment_name="small_clean_vs_c2_full_oos_20261012",
                       config=plan)
    phase("dashboard_ready", run_id="smco1012", url=getattr(run, "url", None))
    phase("inputs_verified", devices=devices, torch=torch.__version__,
          cuda=torch.version.cuda, models=models, samples=EXPECTED_SAMPLES,
          dates=EXPECTED_DATES, decode=DECODE)
    processes = []
    for rank in range(2):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), SMALL_OOS_RANK=str(rank))
        processes.append(subprocess.Popen([sys.executable, "-u", __file__], env=env))
    while any(process.poll() is None for process in processes):
        failed = [process.returncode for process in processes
                  if process.poll() is not None and process.returncode != 0]
        if failed:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            break
        time.sleep(2)
    codes = [process.wait() for process in processes]
    if any(codes):
        raise RuntimeError(f"Worker failures: {codes}")
    frames = {}
    dates = sorted({r["asof_date"] for r in records})
    for label in models:
        parts = [OUTPUT / "shards" / f"{label}_{day}.csv.gz" for day in dates]
        if not all(p.exists() for p in parts):
            (OUTPUT / "summary.json").write_text(json.dumps(dict(
                status="partial", completed={key: len(list((OUTPUT / "shards").glob(f"{key}_*.csv.gz")))
                                              for key in models}, required_dates=34)))
            phase("partial_output", conclusion="no comparison until all dates complete")
            swanlab.finish()
            return
        frame = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        if len(frame) != EXPECTED_SAMPLES or set(frame.identity) != {identity(r) for r in records}:
            raise RuntimeError("Full coverage check failed")
        frames[label] = frame
    comparison = paired_comparison(frames["c2_best"], frames["clean_best"])
    comparison["status"] = "complete"
    (OUTPUT / "summary.json").write_text(json.dumps(comparison, indent=2))
    for label in models:
        swanlab.log({f"evaluation/{label}/{k}": v for k, v in comparison[label].items()
                     if isinstance(v, (int, float))})
    swanlab.finish()
    phase("evaluation_complete", comparison=comparison)


if __name__ == "__main__":
    main()
