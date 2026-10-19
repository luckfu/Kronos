"""Paired full OOS for C04 best/last against C03 best.

This reuses the frozen OOS contract in ``evaluate.py`` (same 175,428 windows,
34 signal dates, decode settings and SHA gates) but evaluates three
checkpoints and adds amplitude-bucket comparisons.  The point is to answer two
questions the aggregate natural-distribution validation cannot:

* did the second 200-segment power-sampling round (C04) improve OOS at all?
* did it improve the large-move subset it was explicitly trained on?

C04 is identified by the SHA-256 of its published ``last_state.pt`` (the exact
state C05 resumed from), so the evaluated best/last weights are tied to the
same output tree that C05 mounted.
"""

from __future__ import annotations

import gc
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from evaluate import (  # noqa: E402
    DECODE,
    EXPECTED_DATES,
    EXPECTED_SAMPLES,
    IDENTITIES_SHA,
    METADATA_SHA,
    PANEL_SHA,
    SAMPLES_SHA,
    TOKENIZER_SHA,
    WindowStore,
    audit_training_dates,
    checked,
    decode,
    identity,
    load_oos,
    phase,
    sha,
    unique,
)

OUTPUT = Path(os.environ.get("SMALL_OOS_OUTPUT", "/kaggle/working/small_clean_c04_oos"))
C03_BEST_SHA = "0837035e4cde286240b196cb2c7fda1dc671cacc2b501b00ab7f0ec38d887c9f"
C04_STATE_SHA = "578e3bc420aa94cd7345da8ecb3637a7d2156eee2d6a4143cde1bb28e0299cd2"
C03_SELECTION_LOSS = 2.2806119943659313
C04_SELECTION_LOSS = 2.281827528184304
RUN_ID = "smco1018"
EXPERIMENT = "small_clean_c04_vs_c03_full_oos_20261018"
BASELINE = "c03_best"
CANDIDATES = ("c04_best", "c04_last")
PAIRS = (("c04_best", BASELINE), ("c04_last", BASELINE), ("c04_last", "c04_best"))
BUCKETS = ("all", "abs_ge_5pct", "abs_ge_10pct", "amplitude_top_decile")
BLOCK, REPLICATES = 10, 2000
RNG_SEED = 20261012


def find_oos(root: Path) -> Path:
    return unique(
        path.parent
        for path in root.glob("**/evaluation_manifest.json")
        if json.loads(path.read_text()).get("artifacts", {}).get("panel_sha256")
        == PANEL_SHA
    )


def find_metadata(root: Path) -> Path:
    return unique(path for path in root.glob("**/asset_metadata.csv")
                  if sha(path) == METADATA_SHA)


def sector_labels(metadata: Path) -> list[str]:
    import pandas as pd

    labels = set()
    for chunk in pd.read_csv(metadata, usecols=["sector"], chunksize=500_000):
        labels.update(chunk.sector.dropna().astype(str))
    labels = sorted(labels)
    if len(labels) != 86:
        raise RuntimeError(f"Training sector vocabulary drift: {len(labels)}")
    return labels


def resolve_tokenizer(root: Path) -> Path:
    # C03 and C04 outputs both carry an identical tokenizer, so multiple
    # matching files are expected; any one of them is byte-identical.
    paths = list(root.glob("**/clean_small_runtime/models/Kronos-Tokenizer-base/model.safetensors"))
    paths += list(root.glob("**/small_power_runtime/models/Kronos-Tokenizer-base/model.safetensors"))
    matching = sorted(path for path in paths if sha(path) == TOKENIZER_SHA)
    if not matching:
        raise RuntimeError("No tokenizer matching the frozen SHA-256")
    return matching[0]


def resolve_models(root: Path) -> dict:
    c03_best = unique(root.glob(
        "**/small_0.1_clean_wc_dual_t4_20261011/checkpoints/best_model/model.safetensors"))
    c04_best = unique(root.glob(
        "**/small_0.1_clean_wc_dual_t4_20261011_power_round2/checkpoints/best_model/model.safetensors"))
    c04_last = unique(root.glob(
        "**/small_0.1_clean_wc_dual_t4_20261011_power_round2/checkpoints/last_model/model.safetensors"))
    c04_state = unique(root.glob(
        "**/small_0.1_clean_wc_dual_t4_20261011_power_round2/checkpoints/last_state.pt"))

    c03_sha = sha(c03_best)
    if c03_sha != C03_BEST_SHA:
        raise RuntimeError(f"C03 best hash mismatch: {c03_sha}")
    c03_metric = json.loads((c03_best.parent / "best_metric.json").read_text())
    if c03_metric["segment"] != 518 or not math.isclose(
            float(c03_metric["selection_loss"]), C03_SELECTION_LOSS,
            rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(f"Unexpected C03 best checkpoint: {c03_metric}")
    c03_summary = json.loads((c03_best.parents[2] / "summary.json").read_text())["final_result"]
    if c03_summary["status"] != "completed" or c03_summary["completed_segments"] != 520:
        raise RuntimeError("C03 training is not the completed 520-segment run")

    c04_best_sha = sha(c04_best)
    c04_last_sha = sha(c04_last)
    if sha(c04_state) != C04_STATE_SHA:
        raise RuntimeError("C04 published state hash mismatch")
    c04_metric = json.loads((c04_best.parent / "best_metric.json").read_text())
    if c04_metric["segment"] != 715 or not math.isclose(
            float(c04_metric["selection_loss"]), C04_SELECTION_LOSS,
            rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(f"Unexpected C04 best checkpoint: {c04_metric}")
    c04_summary = json.loads((c04_best.parents[2] / "summary.json").read_text())["final_result"]
    c04_progress = json.loads((c04_best.parents[2] / "progress.json").read_text())
    if (c04_summary["status"] != "completed" or c04_summary["completed_segments"] != 720
            or c04_progress["status"] != "completed"
            or c04_progress["current_segment"] != 720):
        raise RuntimeError("C04 training is not the completed 720-segment run")

    return {
        "c03_best": {
            "directory": str(c03_best.parent), "sha256": c03_sha, "kind": "best",
            "segment": 518, "selection_loss": C03_SELECTION_LOSS,
            "kernel": "luckfu/kronos-small-clean-wc-dual-t4-c03",
        },
        "c04_best": {
            "directory": str(c04_best.parent), "sha256": c04_best_sha, "kind": "best",
            "segment": 715, "selection_loss": C04_SELECTION_LOSS,
            "kernel": "user281434/kronos-small-clean-wc-dual-t4-c04",
        },
        "c04_last": {
            "directory": str(c04_last.parent), "sha256": c04_last_sha, "kind": "last",
            "segment": 720, "state_sha256": C04_STATE_SHA,
            "kernel": "user281434/kronos-small-clean-wc-dual-t4-c04",
        },
    }


def stat(frame) -> dict | None:
    """Same direction/MAE/RMSE semantics as evaluate.score, with guarded IC."""
    import numpy as np

    if frame is None or len(frame) == 0:
        return None
    predicted = frame[[f"predicted_return_d{h}" for h in range(1, 11)]].to_numpy(dtype=np.float64)
    actual = frame[[f"actual_return_d{h}" for h in range(1, 11)]].to_numpy(dtype=np.float64)
    error = predicted - actual
    direction = np.sign(predicted) == np.sign(actual)
    result = {
        "samples": int(len(frame)),
        "path_direction": float(direction.mean()),
        "path_return_mae": float(np.abs(error).mean()),
        "path_return_rmse": float(np.sqrt(np.square(error).mean())),
        "horizons": {},
    }
    for h in range(10):
        result["horizons"][str(h + 1)] = {
            "direction": float(direction[:, h].mean()),
            "return_mae": float(np.abs(error[:, h]).mean()),
            "return_rmse": float(np.sqrt(np.square(error[:, h]).mean())),
        }
    ics = []
    for _, day in frame.groupby("asof_date"):
        if len(day) < 3:
            continue
        value = day.predicted_return_d10.corr(day.actual_return_d10, method="spearman")
        if np.isfinite(value):
            ics.append(float(value))
    result["d10_daily_rank_ic"] = float(np.mean(ics)) if ics else None
    return result


def bucket_masks(frame) -> dict:
    import numpy as np

    actual = frame[[f"actual_return_d{h}" for h in range(1, 11)]].to_numpy(dtype=np.float64)
    amplitude = np.abs(actual).max(axis=1)
    return {
        "all": np.ones(len(frame), dtype=bool),
        "abs_ge_5pct": amplitude >= 0.05,
        "abs_ge_10pct": amplitude >= 0.10,
        "amplitude_top_decile": amplitude >= float(np.quantile(amplitude, 0.90)),
    }


def daily_stats(frame) -> dict:
    """Per-signal-date sufficient statistics for direction/MAE/RMSE."""
    import numpy as np

    days = frame.asof_date.to_numpy()
    predicted = frame[[f"predicted_return_d{h}" for h in range(1, 11)]].to_numpy(dtype=np.float64)
    actual = frame[[f"actual_return_d{h}" for h in range(1, 11)]].to_numpy(dtype=np.float64)
    error = predicted - actual
    direction = np.sign(predicted) == np.sign(actual)
    stats = {}
    for day in np.unique(days):
        selected = days == day
        stats[str(day)] = (
            int(selected.sum()),
            int(direction[selected].sum()),
            float(np.abs(error[selected]).sum()),
            float(np.square(error[selected]).sum()),
        )
    return stats


def block_indices(count: int, dates: list[str]) -> list[list[str]]:
    import numpy as np

    rng = np.random.default_rng(RNG_SEED)
    samples = []
    for _ in range(REPLICATES):
        picked: list[int] = []
        while len(picked) < count:
            start = int(rng.integers(0, max(1, count - BLOCK + 1)))
            picked.extend(range(start, start + BLOCK))
        samples.append([dates[i] for i in picked[:count]])
    return samples


def aggregate(stats: dict, days: list[str]) -> tuple[float, float, float, int] | None:
    import numpy as np

    total = sum(stats[day][0] for day in days if day in stats)
    if total == 0:
        return None
    # Each window contributes ten horizons, so direction/MAE sums must be
    # normalised by samples * horizons to stay on the same scale as stat().
    direction = sum(stats[day][1] for day in days if day in stats) / (total * 10)
    mae = sum(stats[day][2] for day in days if day in stats) / (total * 10)
    mse = sum(stats[day][3] for day in days if day in stats) / (total * 10)
    return float(direction), float(mae), float(np.sqrt(mse)), total


def paired_deltas(base_frame, candidate_frame, masks: dict, dates: list[str]) -> dict:
    import numpy as np

    blocks = block_indices(len(dates), dates)
    result = {}
    for bucket, mask in masks.items():
        base_daily = daily_stats(base_frame[mask])
        candidate_daily = daily_stats(candidate_frame[mask])
        base_point = aggregate(base_daily, dates)
        candidate_point = aggregate(candidate_daily, dates)
        if base_point is None or candidate_point is None:
            result[bucket] = None
            continue
        direction, mae, rmse = [], [], []
        for days in blocks:
            left = aggregate(base_daily, days)
            right = aggregate(candidate_daily, days)
            if left is None or right is None:
                continue
            direction.append(right[0] - left[0])
            mae.append(right[1] - left[1])
            rmse.append(right[2] - left[2])
        result[bucket] = {
            "samples": base_point[3],
            "direction": {
                "value": candidate_point[0] - base_point[0],
                "block_bootstrap_95ci": np.quantile(direction, [.025, .975]).tolist(),
            },
            "return_mae": {
                "value": candidate_point[1] - base_point[1],
                "block_bootstrap_95ci": np.quantile(mae, [.025, .975]).tolist(),
            },
            "return_rmse": {
                "value": candidate_point[2] - base_point[2],
                "block_bootstrap_95ci": np.quantile(rmse, [.025, .975]).tolist(),
            },
        }
    return result


def init_dashboard(plan: dict):
    try:
        import swanlab
        key = os.environ.get("SWANLAB_API_KEY", "").strip()
        if key:
            swanlab.login(api_key=key)
        run = swanlab.init(id=RUN_ID, resume="allow", project="finance",
                           workspace="roc_fu", experiment_name=EXPERIMENT, config=plan)
        phase("dashboard_ready", run_id=RUN_ID, url=getattr(run, "url", None))
        return swanlab
    except Exception as exc:  # dashboard is not allowed to kill a multi-hour eval
        phase("dashboard_unavailable", error=f"{type(exc).__name__}: {exc}")
        return None


def worker(plan: dict, label: str) -> None:
    import torch
    from model import Kronos, KronosTokenizer

    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    phase("load_dataset", model=label)
    panel, records, _ = load_oos(Path(plan["oos_root"]))
    store = WindowStore(panel, plan["sector_labels"])
    model = Kronos.from_pretrained(
        plan["models"][label]["directory"], num_sectors=86, num_size_buckets=0,
        context_layer=6, use_size_percentile=True, size_mlp_hidden_dim=64).to(device).eval()
    tokenizer = KronosTokenizer.from_pretrained(plan["tokenizer"]).to(device).eval()
    phase("models_loaded", model=label, sha256=plan["models"][label]["sha256"])
    for date in sorted({record["asof_date"] for record in records}):
        if time.time() > plan["deadline"] - 600:
            phase("planned_stop", model=label, date=date)
            break
        path = OUTPUT / "shards" / f"{label}_{date}.csv.gz"
        selected = [record for record in records if record["asof_date"] == date]
        if path.exists():
            import pandas as pd
            existing = pd.read_csv(path)
            if set(existing.identity) != {identity(record) for record in selected}:
                raise RuntimeError("Invalid resumed date shard")
            continue
        began = time.time()
        frame = decode(selected, store, model, tokenizer, device, label)
        temporary = path.with_suffix(".tmp")
        frame.to_csv(temporary, index=False, compression="gzip")
        temporary.replace(path)
        phase("date_finished", model=label, date=date, windows=len(frame),
              seconds=time.time() - began, metrics=stat(frame))
    phase("worker_finished", model=label)


def run_workers(labels: tuple[str, ...]) -> None:
    slots: dict[int, tuple | None] = {0: None, 1: None}
    pending = list(labels)
    while pending or any(slots.values()):
        for slot in (0, 1):
            if slots[slot] is None and pending:
                label = pending.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot),
                           SMALL_OOS_LABEL=label, SMALL_OOS_OUTPUT=str(OUTPUT))
                process = subprocess.Popen(
                    [sys.executable, "-u", str(Path(__file__).resolve())], env=env)
                slots[slot] = (process, label)
                phase("worker_started", model=label, slot=slot)
        time.sleep(2)
        for slot in (0, 1):
            entry = slots[slot]
            if entry is None or entry[0].poll() is None:
                continue
            code = entry[0].wait()
            if code:
                for other in slots.values():
                    if other is not None and other[0].poll() is None:
                        other[0].terminate()
                raise RuntimeError(f"OOS worker failed: {entry[1]} code={code}")
            phase("worker_finished", model=entry[1])
            slots[slot] = None


def parent_main(started: float) -> None:
    import numpy as np
    import pandas as pd
    import torch

    devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if len(devices) != 2 or any("T4" not in name for name in devices):
        raise RuntimeError(f"Expected dual T4: {devices}")

    root = Path("/kaggle/input")
    oos = find_oos(root)
    metadata = find_metadata(root)
    training_max_dates = audit_training_dates(root)
    labels = sector_labels(metadata)
    panel, records, manifest = load_oos(oos)
    store = WindowStore(panel, labels)
    phase("preflight_all_windows", samples=len(records))
    for index, record in enumerate(records):
        store.prepare(record)
        if index and index % 25_000 == 0:
            phase("preflight_progress", done=index)
    del store, panel
    gc.collect()

    models = resolve_models(root)
    tokenizer = resolve_tokenizer(root)
    plan = dict(
        oos_root=str(oos), sector_labels=labels, models=models,
        tokenizer=str(tokenizer.parent), deadline=started + 40_500,
        decode=DECODE, artifacts=manifest["artifacts"],
        identities_sha256=IDENTITIES_SHA, samples=EXPECTED_SAMPLES,
        dates=EXPECTED_DATES, baseline=BASELINE, candidates=list(CANDIDATES),
        pairs=[list(pair) for pair in PAIRS], buckets=list(BUCKETS),
        source_commit=os.environ["SMALL_OOS_COMMIT"],
        source_sha256=os.environ["SMALL_OOS_SOURCE_SHA"],
        training_selection_max_row_dates=training_max_dates,
        label_semantics="next 10 actual traded rows; original unclipped closes",
        selection="frozen published checkpoints; no OOS tuning",
        amplitude_definition="max(|actual_return_d1..d10|) per window",
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "shards").mkdir(exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2))

    dashboard = init_dashboard(plan)
    phase("inputs_verified", devices=devices, torch=torch.__version__,
          cuda=torch.version.cuda, models=models, samples=EXPECTED_SAMPLES,
          dates=EXPECTED_DATES, decode=DECODE)
    run_workers(tuple(models))

    frames = {}
    for label in models:
        parts = [OUTPUT / "shards" / f"{label}_{date}.csv.gz"
                 for date in sorted({record["asof_date"] for record in records})]
        if not all(part.exists() for part in parts):
            summary = dict(
                status="partial",
                completed={key: len(list((OUTPUT / "shards").glob(f"{key}_*.csv.gz")))
                           for key in models},
                required_dates=EXPECTED_DATES,
            )
            (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
            phase("partial_output", **summary)
            if dashboard is not None:
                dashboard.finish()
            return
        frame = pd.concat([pd.read_csv(part) for part in parts], ignore_index=True)
        if len(frame) != EXPECTED_SAMPLES:
            raise RuntimeError(f"Incomplete {label}: {len(frame)}")
        frames[label] = frame

    expected = {identity(record) for record in records}
    for label, frame in frames.items():
        if set(frame.identity) != expected or frame.identity.duplicated().any():
            raise RuntimeError(f"Identity coverage mismatch: {label}")
        frames[label] = frame.sort_values("identity").reset_index(drop=True)
    actual_columns = [f"actual_return_d{h}" for h in range(1, 11)]
    base_actual = frames[BASELINE][actual_columns].to_numpy(dtype=np.float64)
    for label, frame in frames.items():
        if not (frame[actual_columns].to_numpy(dtype=np.float64) == base_actual).all():
            raise RuntimeError(f"Paired labels differ: {label}")

    masks = bucket_masks(frames[BASELINE])
    scores = {label: {bucket: stat(frame[mask]) for bucket, mask in masks.items()}
              for label, frame in frames.items()}
    paired = {f"{candidate}_minus_{reference}":
              paired_deltas(frames[reference], frames[candidate], masks,
                            sorted({record["asof_date"] for record in records}))
              for candidate, reference in PAIRS}

    summary = dict(
        status="complete", baseline=BASELINE, models=models, scores=scores,
        paired_deltas=paired,
        buckets={bucket: {"samples": int(mask.sum()), "fraction": float(mask.mean())}
                 for bucket, mask in masks.items()},
        amplitude_definition="max(|actual_return_d1..d10|) per window",
        uncertainty=dict(signal_dates=EXPECTED_DATES, block_length=BLOCK,
                         replicates=REPLICATES, seed=RNG_SEED,
                         note="34 signal dates; overlapping labels; not a new blind set"),
        oos=dict(samples=EXPECTED_SAMPLES, dates=EXPECTED_DATES,
                  identities_sha256=IDENTITIES_SHA, panel_sha256=PANEL_SHA,
                  samples_sha256=SAMPLES_SHA),
        source_sha256=os.environ["SMALL_OOS_SOURCE_SHA"],
        source_commit=os.environ["SMALL_OOS_COMMIT"],
    )
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    if dashboard is not None:
        for label, buckets in scores.items():
            for bucket, value in buckets.items():
                if value is None:
                    continue
                dashboard.log({f"evaluation/{label}/{bucket}/{key}": value[key]
                               for key in ("path_direction", "path_return_mae",
                                           "path_return_rmse")})
        for pair, buckets in paired.items():
            for bucket, value in buckets.items():
                if value is None:
                    continue
                dashboard.log({f"evaluation/delta/{pair}/{bucket}/{key}": value[key]["value"]
                               for key in ("direction", "return_mae", "return_rmse")})
        dashboard.finish()
    phase("evaluation_complete", summary=summary)


def heartbeat(stop: threading.Event) -> None:
    while not stop.wait(30):
        phase("heartbeat", role="coordinator")


def main() -> None:
    label = os.environ.get("SMALL_OOS_LABEL", "").strip()
    if label:
        worker(json.loads((OUTPUT / "plan.json").read_text()), label)
        return
    phase("started")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "shards").mkdir(exist_ok=True)
    started = float(os.environ.get("SMALL_OOS_STARTED", time.time()))
    stop = threading.Event()
    monitor = threading.Thread(target=lambda: heartbeat(stop), daemon=True)
    monitor.start()
    try:
        parent_main(started)
    finally:
        stop.set()
        monitor.join()


if __name__ == "__main__":
    main()
