"""Continue Beta v1.1 Stage2 segments 11-90 from the verified C1 output."""

import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path


SOURCE_COMMIT = "b0c8ba80705949148e8d6056a3ce120bc19484b5"
TRAIN_COMMIT = "8f91e01dbf38fa138ed7a03eb058d58258b527d3"
SOURCE_SHA256 = "ee7e19de3c260a035e4c31b7b36c5a2c8d6bca00c9c7e4a370f20c8126401b40"
SOURCE_PATH = "finetune/kaggle_beta_v1_1_stage2_wc_dual_t4/kaggle_beta_v1_1_stage2_wc_dual_t4.py"
OUTPUT_NAME = "beta_v1_1_stage2_wc_dual_t4"
PARENT_SEGMENT = 10
SEGMENTS_THIS_RUN = 80
TRAIN_RUNTIME_SECONDS = 36000


def validate_parent(input_root):
    matches = sorted(input_root.glob(f"**/{OUTPUT_NAME}/checkpoints/last_state.pt"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one C1 checkpoint, found {len(matches)}")
    output_root = matches[0].parent.parent
    required = (
        "run.log", "metrics.jsonl", "progress.json", "summary.json",
        "experiment_manifest.json", "checkpoints/last_model/model.safetensors",
        "checkpoints/best_model/model.safetensors",
    )
    missing = [name for name in required if not (output_root / name).is_file()]
    if missing:
        raise RuntimeError(f"C1 output incomplete: {missing}")
    progress = json.loads((output_root / "progress.json").read_text())
    summary = json.loads((output_root / "summary.json").read_text())
    manifest = json.loads((output_root / "experiment_manifest.json").read_text())
    result = summary.get("final_result", {})
    expected = {
        "progress_segment": (progress.get("current_segment"), PARENT_SEGMENT),
        "progress_windows": (progress.get("unique_windows_covered"), 200000),
        "progress_total": (progress.get("total_segments"), 534),
        "progress_status": (progress.get("status"), "stopped"),
        "summary_segment": (result.get("completed_segments"), PARENT_SEGMENT),
        "resume_segment": (result.get("resume_segment"), PARENT_SEGMENT + 1),
        "stop_reason": (result.get("stop_reason"), "segment_limit"),
        "world_size": (summary.get("world_size"), 2),
        "parent_checkpoint": (manifest.get("parent", {}).get("checkpoint"), "Best@818"),
        "gpu": (manifest.get("training", {}).get("gpu"), "2x Tesla T4"),
        "data_manifest": (manifest.get("data", {}).get("data_manifest_sha256"),
                          "32cfcbf606dcee81c9416f9ad399ee7303b6790cf9025bf5888638f552d66598"),
        "coverage_seed": (result.get("train_selection", {}).get("coverage_seed"), 20260924),
    }
    wrong = {key: value for key, value in expected.items() if value[0] != value[1]}
    if wrong:
        raise RuntimeError(f"C1 lineage mismatch: {wrong}")
    print(json.dumps({"phase": "parent_verified", "segment": PARENT_SEGMENT,
                      "next_segment": PARENT_SEGMENT + 1,
                      "checkpoint": str(matches[0])}), flush=True)
    return output_root


def load_runner():
    url = f"https://raw.githubusercontent.com/luckfu/Kronos/{SOURCE_COMMIT}/{SOURCE_PATH}"
    with urllib.request.urlopen(url, timeout=60) as response:
        source = response.read()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Pinned C1 runner source hash mismatch")
    namespace = {"__name__": "kronos_beta_v1_1_c1_runner", "__file__": SOURCE_PATH}
    exec(compile(source, SOURCE_PATH, "exec"), namespace)
    return namespace


def pinned_clone(original_clone):
    def clone(repo):
        original_clone(repo)
        subprocess.run(["git", "-C", str(repo), "fetch", "--depth", "1",
                        "origin", TRAIN_COMMIT], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach",
                        "FETCH_HEAD"], check=True)
        actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                         text=True).strip()
        if actual != TRAIN_COMMIT:
            raise RuntimeError(f"Training source mismatch: {actual}")
        print(json.dumps({"phase": "training_source_pinned", "commit": actual}), flush=True)
        return actual
    return clone


class TimingRun:
    def __init__(self, run):
        self.run = run
        self.total_seconds = 0
        self.count = 0

    def log(self, payload, **kwargs):
        if "timing/segment_seconds" in payload:
            self.total_seconds += payload["timing/segment_seconds"]
            self.count += 1
            payload = {**payload, "timing/avg_segment_seconds": self.total_seconds / self.count}
        return self.run.log(payload, **kwargs)


def main():
    validate_parent(Path("/kaggle/input"))
    runner = load_runner()
    runner["MAX_SEGMENTS_PER_RUN"] = SEGMENTS_THIS_RUN
    runner["MAX_RUNTIME_SECONDS"] = TRAIN_RUNTIME_SECONDS
    runner["clone_repo"] = pinned_clone(runner["clone_repo"])
    original_start_swanlab = runner["start_swanlab"]

    def start_swanlab(env):
        swanlab, run = original_start_swanlab(env)
        return swanlab, TimingRun(run)

    runner["start_swanlab"] = start_swanlab
    runner["main"]()


if __name__ == "__main__":
    main()
