"""C2-best full OOS on the same frozen set, combined with the C03/C04 shards.

The C2 best checkpoint is not on Kaggle under the current account, so it is
pulled from ModelScope and pinned by SHA-256.  C03 best, C04 best and C04 last
are not re-decoded: their per-sample shards are read from the completed
``kronos-small-clean-c04-vs-c03-full-oos`` Output and paired against C2 on the
same 175,428 windows and 34 signal dates.
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

import evaluate_c04 as support  # noqa: E402
from evaluate import (  # noqa: E402
    DECODE,
    EXPECTED_DATES,
    EXPECTED_SAMPLES,
    IDENTITIES_SHA,
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
from evaluate_c04 import (  # noqa: E402
    bucket_masks,
    find_metadata,
    find_oos,
    paired_deltas,
    sector_labels,
    stat,
)

OUTPUT = Path(os.environ.get("SMALL_OOS_OUTPUT", "/kaggle/working/small_clean_c2_oos"))
C2_REPO = "luckfu/Kronos-small-0.1-Cosine-C2-Best"
C2_SHA = "4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a"
C2_SELECTION_LOSS = 2.2944018840789795
C2_SEGMENT = 179
TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"
RUN_ID = "smco1018c2"
EXPERIMENT = "small_clean_c2_vs_clean_full_oos_20261018"
BASELINE = "c2_best"
PRIOR_LABELS = ("c03_best", "c04_best", "c04_last")
SHARDS = 2


def download_modelscope(repo: str, patterns: list[str]) -> Path:
    from modelscope import snapshot_download

    path = Path(snapshot_download(repo, cache_dir="/kaggle/working/modelscope_cache",
                                  allow_patterns=patterns))
    return path


def resolve_c2() -> tuple[Path, Path]:
    root = download_modelscope(C2_REPO, ["config.json", "model.safetensors", "best_metric.json"])
    weights = unique(root.glob("**/model.safetensors"))
    actual = sha(weights)
    if actual != C2_SHA:
        raise RuntimeError(f"C2 ModelScope hash mismatch: {actual}")
    metric = json.loads(unique(root.glob("**/best_metric.json")).read_text())
    if metric["segment"] != C2_SEGMENT or not math.isclose(
            float(metric["selection_loss"]), C2_SELECTION_LOSS, rel_tol=0.0, abs_tol=1e-9):
        raise RuntimeError(f"Unexpected C2 checkpoint: {metric}")
    if json.loads(unique(root.glob("**/config.json")).read_text())["num_sectors"] != 86:
        raise RuntimeError("C2 sector vocabulary drift")

    # The original tokenizer is not on ModelScope under the NeoQuasar id; it is
    # mounted from the completed C03 Output, which carries the exact frozen bytes.
    candidates = list(Path("/kaggle/input").glob("**/Kronos-Tokenizer-base/model.safetensors"))
    matching = sorted(path for path in candidates if sha(path) == TOKENIZER_SHA)
    if not matching:
        raise RuntimeError("No frozen Kronos-Tokenizer-base found in mounted inputs")
    tokenizer_weights = matching[0]
    checked(tokenizer_weights, TOKENIZER_SHA)
    return weights.parent, tokenizer_weights.parent


def load_prior(root: Path, dates: list[str]) -> tuple[dict | None, dict | None]:
    """Read C03/C04 per-sample shards from the completed paired OOS Output."""
    try:
        plan_path = unique(root.glob("**/small_clean_c04_oos/plan.json"))
    except RuntimeError:
        phase("prior_output_missing", expected="small_clean_c04_oos/plan.json")
        return None, None
    prior_root = plan_path.parent
    plan = json.loads(plan_path.read_text())
    frames = {}
    import pandas as pd
    for label in PRIOR_LABELS:
        parts = [prior_root / "shards" / f"{label}_{date}.csv.gz" for date in dates]
        if not all(part.exists() for part in parts):
            phase("prior_shards_incomplete", model=label,
                  found=len(list((prior_root / "shards").glob(f"{label}_*.csv.gz"))))
            return None, plan
        frame = pd.concat([pd.read_csv(part) for part in parts], ignore_index=True)
        if len(frame) != EXPECTED_SAMPLES:
            raise RuntimeError(f"Prior shard coverage mismatch: {label}: {len(frame)}")
        frames[label] = frame
    phase("prior_shards_loaded", root=str(prior_root), models=sorted(frames))
    return frames, plan


def worker(plan: dict, slot: int) -> None:
    import torch
    from model import Kronos, KronosTokenizer

    torch.set_num_threads(2)
    device = torch.device("cuda:0")
    phase("load_dataset", model="c2_best", shard=slot)
    panel, records, _ = load_oos(Path(plan["oos_root"]))
    store = WindowStore(panel, plan["sector_labels"])
    model = Kronos.from_pretrained(
        plan["models"]["c2_best"]["directory"], num_sectors=86, num_size_buckets=0,
        context_layer=6, use_size_percentile=True, size_mlp_hidden_dim=64).to(device).eval()
    tokenizer = KronosTokenizer.from_pretrained(plan["tokenizer"]).to(device).eval()
    phase("models_loaded", model="c2_best", shard=slot, sha256=C2_SHA)
    dates = sorted({record["asof_date"] for record in records})[slot::SHARDS]
    for date in dates:
        if time.time() > plan["deadline"] - 600:
            phase("planned_stop", model="c2_best", shard=slot, date=date)
            break
        path = OUTPUT / "shards" / f"c2_best_{date}.csv.gz"
        selected = [record for record in records if record["asof_date"] == date]
        if path.exists():
            import pandas as pd
            existing = pd.read_csv(path)
            if set(existing.identity) != {identity(record) for record in selected}:
                raise RuntimeError("Invalid resumed date shard")
            continue
        began = time.time()
        frame = decode(selected, store, model, tokenizer, device, "c2_best")
        temporary = path.with_suffix(".tmp")
        frame.to_csv(temporary, index=False, compression="gzip")
        temporary.replace(path)
        phase("date_finished", model="c2_best", shard=slot, date=date,
              windows=len(frame), seconds=time.time() - began, metrics=stat(frame))
    phase("worker_finished", model="c2_best", shard=slot)


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
    except Exception as exc:
        phase("dashboard_unavailable", error=f"{type(exc).__name__}: {exc}")
        return None


def run_workers() -> None:
    env = dict(os.environ, SMALL_OOS_OUTPUT=str(OUTPUT))
    processes = [
        subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve())],
                         env=dict(env, CUDA_VISIBLE_DEVICES=str(slot), SMALL_OOS_SHARD=str(slot)))
        for slot in range(SHARDS)
    ]
    codes = [process.wait() for process in processes]
    if any(codes):
        raise RuntimeError(f"C2 worker failures: {codes}")


def parent_main(started: float) -> None:
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

    c2_directory, tokenizer = resolve_c2()
    dates = sorted({record["asof_date"] for record in records})
    prior_frames, prior_plan = load_prior(root, dates)
    models = {
        BASELINE: {
            "directory": str(c2_directory), "sha256": C2_SHA, "kind": "best",
            "segment": C2_SEGMENT, "selection_loss": C2_SELECTION_LOSS,
            "source": f"modelscope:{C2_REPO}",
        }
    }
    if prior_plan:
        for label in PRIOR_LABELS:
            models[label] = prior_plan["models"][label]
    plan = dict(
        oos_root=str(oos), sector_labels=labels, models=models,
        tokenizer=str(tokenizer), deadline=started + 40_500,
        decode=DECODE, artifacts=manifest["artifacts"],
        identities_sha256=IDENTITIES_SHA, samples=EXPECTED_SAMPLES,
        dates=EXPECTED_DATES, baseline=BASELINE, shards=SHARDS,
        source_commit=os.environ["SMALL_OOS_COMMIT"],
        source_sha256=os.environ["SMALL_OOS_SOURCE_SHA"],
        training_selection_max_row_dates=training_max_dates,
        label_semantics="next 10 actual traded rows; original unclipped closes",
        selection="frozen published checkpoints; no OOS tuning",
        amplitude_definition="max(|actual_return_d1..d10|) per window",
        c2_source=dict(repo=C2_REPO, sha256=C2_SHA, segment=C2_SEGMENT,
                       selection_loss=C2_SELECTION_LOSS),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "shards").mkdir(exist_ok=True)
    (OUTPUT / "plan.json").write_text(json.dumps(plan, indent=2))

    dashboard = init_dashboard(plan)
    phase("inputs_verified", devices=devices, torch=torch.__version__,
          cuda=torch.version.cuda, models=models, samples=EXPECTED_SAMPLES,
          dates=EXPECTED_DATES, decode=DECODE)
    run_workers()

    c2_parts = [OUTPUT / "shards" / f"c2_best_{date}.csv.gz" for date in dates]
    if not all(part.exists() for part in c2_parts):
        summary = dict(status="partial", completed=len(list((OUTPUT / "shards").glob("c2_best_*.csv.gz"))),
                       required_dates=EXPECTED_DATES)
        (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
        phase("partial_output", **summary)
        if dashboard is not None:
            dashboard.finish()
        return
    c2_frame = pd.concat([pd.read_csv(part) for part in c2_parts], ignore_index=True)
    if len(c2_frame) != EXPECTED_SAMPLES:
        raise RuntimeError(f"Incomplete c2_best: {len(c2_frame)}")
    frames = {BASELINE: c2_frame}
    if prior_frames:
        frames.update(prior_frames)

    expected = {identity(record) for record in records}
    for label, frame in frames.items():
        if set(frame.identity) != expected or frame.identity.duplicated().any():
            raise RuntimeError(f"Identity coverage mismatch: {label}")
        frames[label] = frame.sort_values("identity").reset_index(drop=True)
    actual_columns = [f"actual_return_d{h}" for h in range(1, 11)]
    base_actual = frames[BASELINE][actual_columns].to_numpy(dtype=float)
    for label, frame in frames.items():
        if not (frame[actual_columns].to_numpy(dtype=float) == base_actual).all():
            raise RuntimeError(f"Paired labels differ: {label}")

    masks = bucket_masks(frames[BASELINE])
    scores = {label: {bucket: stat(frame[mask]) for bucket, mask in masks.items()}
              for label, frame in frames.items()}
    pairs = [(label, BASELINE) for label in PRIOR_LABELS if label in frames]
    if "c04_best" in frames and "c03_best" in frames:
        pairs.append(("c04_best", "c03_best"))
    paired = {f"{candidate}_minus_{reference}":
              paired_deltas(frames[reference], frames[candidate], masks, dates)
              for candidate, reference in pairs}

    summary = dict(
        status="complete", baseline=BASELINE, models=models, scores=scores,
        paired_deltas=paired,
        buckets={bucket: {"samples": int(mask.sum()), "fraction": float(mask.mean())}
                 for bucket, mask in masks.items()},
        amplitude_definition="max(|actual_return_d1..d10|) per window",
        uncertainty=dict(signal_dates=EXPECTED_DATES, block_length=support.BLOCK,
                         replicates=support.REPLICATES, seed=support.RNG_SEED,
                         note="34 signal dates; overlapping labels; not a new blind set"),
        oos=dict(samples=EXPECTED_SAMPLES, dates=EXPECTED_DATES,
                  identities_sha256=IDENTITIES_SHA, panel_sha256=PANEL_SHA,
                  samples_sha256=SAMPLES_SHA),
        source_sha256=os.environ["SMALL_OOS_SOURCE_SHA"],
        source_commit=os.environ["SMALL_OOS_COMMIT"],
        c2_source=plan["c2_source"],
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
    if os.environ.get("SMALL_OOS_SHARD") is not None:
        plan = json.loads((OUTPUT / "plan.json").read_text())
        worker(plan, int(os.environ["SMALL_OOS_SHARD"]))
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
