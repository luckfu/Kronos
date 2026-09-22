"""C2 vol-gap probe: autoregressive samples vs teacher-forced mixtures.

Zero training. Last four causal-validation dates, dual T4.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

SOURCE_COMMIT = "ea8766fa37d3bb2162bc6806555551a9b8564286"
OUTPUT = Path("/kaggle/working/vol_gap_probe")
PARENT_BEST_SHA = "4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a"
TOKENIZER_SHA = "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee"


def emit(payload):
    print(json.dumps(payload), flush=True)


def run(command, cwd=None):
    subprocess.run(command, cwd=cwd, check=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clone_source():
    repo = Path("/kaggle/working/kronos_repo")
    if repo.exists():
        subprocess.run(["rm", "-rf", str(repo)], check=True)
    run(["git", "init", str(repo)])
    run(["git", "remote", "add", "origin", "https://github.com/luckfu/Kronos.git"], cwd=repo)
    run(["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT], cwd=repo)
    run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != SOURCE_COMMIT:
        raise RuntimeError(f"Source commit mismatch: {actual}")
    return repo


def find_one(root, pattern):
    matches = list(Path(root).glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern}, found {matches}")
    return matches[0]


def prepare_windows(repo):
    sys.path.insert(0, str(repo))
    os.environ.update(
        KRONOS_LOOKBACK_WINDOW="120",
        KRONOS_PREDICT_WINDOW="10",
        KRONOS_USE_SIZE_PERCENTILE="1",
        KRONOS_NUM_SIZE_BUCKETS="0",
        KRONOS_VALIDATION_SAMPLES="0",
        KRONOS_VAL_SIGNAL_START="2025-07-01",
        KRONOS_VAL_SIGNAL_END="2026-07-02",
        KRONOS_STAGE3_VOL_LOSS="1",
        KRONOS_STAGE3_RANK_LOSS="0",
    )
    root = Path("/kaggle/input")
    manifest = find_one(root, "**/data_manifest.json")
    data_root = manifest.parent
    os.environ["KRONOS_DATASET_PATH"] = str(data_root / "processed_datasets")
    os.environ["KRONOS_METADATA_PATH"] = str(data_root / "asset_metadata.csv")
    from finetune.dataset import QlibDataset
    from finetune.vol_gap_probe import _extract_windows, day_to_date, last_signal_days

    dataset = QlibDataset("val")
    days = last_signal_days(dataset.signal_date_ids, 4)
    windows = _extract_windows(dataset, days)
    del dataset
    import gc
    gc.collect()
    path = OUTPUT / "windows.npz"
    import numpy as np
    np.savez_compressed(path, **windows)
    dates = [day_to_date(day) for day in days]
    emit({"phase": "windows_ready", "dates": dates, "samples": int(len(windows["x"])), "path": str(path)})
    return path, dates


def worker(rank, repo, windows_path, model_dir, tokenizer_dir):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    from finetune.vol_gap_probe import score_shard
    from model import Kronos, KronosTokenizer

    device = torch.device("cuda:0")
    payload = np.load(windows_path)
    index = np.arange(len(payload["x"]))
    shard_index = index[index % 2 == rank]
    windows = {key: payload[key][shard_index] for key in payload.files}
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_dir).to(device).eval()
    predictor = Kronos.from_pretrained(
        model_dir, num_sectors=86, num_size_buckets=0, context_layer=6,
        use_size_percentile=True, size_mlp_hidden_dim=64,
    ).to(device).eval()
    emit({"phase": "worker_started", "rank": rank, "samples": int(len(shard_index)),
          "gpu": torch.cuda.get_device_name(0)})
    rows = score_shard(predictor, tokenizer, windows, device)
    out = OUTPUT / f"shard_{rank}.json"
    out.write_text(json.dumps(rows))
    emit({"phase": "worker_finished", "rank": rank, "rows": len(rows)})


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if os.environ.get("KRONOS_VOL_GAP_RANK"):
        worker(
            int(os.environ["KRONOS_VOL_GAP_RANK"]),
            os.environ["KRONOS_VOL_GAP_REPO"],
            os.environ["KRONOS_VOL_GAP_WINDOWS"],
            os.environ["KRONOS_VOL_GAP_MODEL"],
            os.environ["KRONOS_VOL_GAP_TOKENIZER"],
        )
        return
    emit({"phase": "started", "source_commit": SOURCE_COMMIT, "training": False})
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).strip().splitlines()
    if len(names) != 2 or any("T4" not in name for name in names):
        raise RuntimeError(f"Expected two T4 GPUs, found {names}")
    repo = clone_source()
    best = find_one("/kaggle/input", "**/small_0.1_stage2_cosine_refinement/checkpoints/best_model/model.safetensors")
    if sha256_file(best) != PARENT_BEST_SHA:
        raise RuntimeError("C2 best hash mismatch")
    tokenizer = find_one("/kaggle/input", "**/Kronos-Tokenizer-base/model.safetensors")
    if sha256_file(tokenizer) != TOKENIZER_SHA:
        raise RuntimeError("Tokenizer hash mismatch")
    windows, dates = prepare_windows(repo)
    procs = []
    for rank in (0, 1):
        env = os.environ.copy()
        env.update(
            KRONOS_VOL_GAP_RANK=str(rank),
            KRONOS_VOL_GAP_REPO=str(repo),
            KRONOS_VOL_GAP_WINDOWS=str(windows),
            KRONOS_VOL_GAP_MODEL=str(best.parent),
            KRONOS_VOL_GAP_TOKENIZER=str(tokenizer.parent),
            PYTHONPATH=str(repo),
        )
        procs.append(subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve())], env=env))
    codes = [proc.wait() for proc in procs]
    if any(codes):
        raise RuntimeError(f"Worker failed: {codes}")
    sys.path.insert(0, str(repo))
    from finetune.vol_gap_probe import summarize
    rows = []
    for rank in (0, 1):
        rows.extend(json.loads((OUTPUT / f"shard_{rank}.json").read_text()))
    summary = summarize(rows)
    summary.update(
        kernel="luckfu/kronos-small-0-1-vol-gap-probe",
        source_commit=SOURCE_COMMIT,
        checkpoint_sha256=PARENT_BEST_SHA,
        dates=dates,
        samples=len(rows),
        training_performed=False,
        precision="fp32",
    )
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
    emit({"phase": "completed", **summary})


if __name__ == "__main__":
    main()
