"""Recover a mid-scoring CANCEL_ACKNOWLEDGED chunk into a durable parent.

Kaggle's 12h hard wall can kill a dual-T4 chunk while Temporal Rank IC is
still scoring. ``last_state.pt`` at that point is already a clean segment
boundary (``resume_step == 0``), but ``last_model/``, ``summary.json``, and
``experiment_manifest.json`` were never written, so ``validate_output``
rejects the tree as a relay parent.

This module:

1. Refuses to invent metrics (best_metric / Rank IC stay as on disk).
2. Exports ``last_model`` from ``last_state`` + ``best_model`` config.
3. Rewrites ``progress.json`` / ``summary.json`` to ``stopped`` at
   ``last_state["next_epoch"]``.
4. Writes a minimal ``experiment_manifest.json`` so the next chunk can read
   ``parent_best_sha256``.

The incomplete SegN training that ran after the durable boundary is discarded
on purpose: those weights were never persisted.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_FOR_RECOVERY = (
    "progress.json",
    "metrics.jsonl",
    "small_v21_manifest.json",
    "checkpoints/last_state.pt",
    "checkpoints/best_model/model.safetensors",
    "checkpoints/best_model/config.json",
    "checkpoints/best_model/best_metric.json",
)


def _utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def inspect_incomplete_boundary(root, torch):
    """Return the durable ``next_epoch`` or raise if the tree is not recoverable."""
    root = Path(root)
    missing = [name for name in REQUIRED_FOR_RECOVERY if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete chunk is not recoverable; missing {missing}")
    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    state = torch.load(
        root / "checkpoints/last_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    boundary = int(state["next_epoch"])
    if int(state.get("resume_step", -1)) != 0:
        raise RuntimeError(
            f"last_state resume_step={state.get('resume_step')!r} is not a "
            "segment boundary; refusing recovery"
        )
    total = int(progress.get("total_segments") or 520)
    if not (0 < boundary <= total):
        raise RuntimeError(f"last_state next_epoch={boundary} is out of range")
    if not state.get("optimizer", {}).get("state"):
        raise RuntimeError("last_state has no optimizer state; refusing recovery")
    if "amp_scaler" not in state or not state.get("rng_state"):
        raise RuntimeError("last_state missing amp_scaler/rng_state; refusing recovery")
    if state.get("scheduler_type") != "warmup_constant":
        raise RuntimeError("last_state scheduler_type is not warmup_constant")
    if int(state["scheduler"]["last_epoch"]) != int(state["batch_idx_global"]):
        raise RuntimeError("scheduler last_epoch does not match batch_idx_global")
    return boundary, state, progress


def export_last_model(root, repo_root):
    """Write ``checkpoints/last_model`` from ``last_state`` + ``best_model`` config."""
    root = Path(root)
    repo_root = Path(repo_root).resolve()
    finetune_dir = repo_root / "finetune"
    for entry in (repo_root, finetune_dir):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    from export_last_model import main as export_main

    # export_last_model is a CLI; call it the same way the trainer does.
    argv = sys.argv
    try:
        sys.argv = [
            "export_last_model",
            "--repo-root", str(repo_root),
            "--output-root", str(root),
        ]
        export_main()
    finally:
        sys.argv = argv
    last = root / "checkpoints/last_model/model.safetensors"
    if not last.is_file():
        raise RuntimeError(f"export_last_model did not produce {last}")


def write_durable_sidecar(root, boundary, progress, recipe, *, chunk, parent_kernel,
                          reason):
    """Rewrite progress/summary/manifest to a stopped durable boundary."""
    root = Path(root)
    recipe = dict(recipe or {})
    if not recipe.get("parent_best_sha256"):
        raise RuntimeError(
            "recover recipe must include parent_best_sha256 "
            "(e03b best SHA lineage for later chunks)"
        )
    boundary = int(boundary)
    now = _utc_now()
    durable_progress = dict(progress)
    durable_progress.update({
        "updated_at": now,
        "status": "stopped",
        "phase": "recovered_segment_boundary",
        "current_segment": boundary,
        "completed_segments": boundary,
        "recovery_reason": str(reason),
    })
    (root / "progress.json").write_text(
        json.dumps(durable_progress, indent=2) + "\n", encoding="utf-8",
    )
    summary = {
        "start_time": progress.get("updated_at") or now,
        "save_directory": str(root),
        "world_size": int((recipe or {}).get("world_size") or 2),
        "recovered": True,
        "recovery_reason": str(reason),
        "final_result": {
            "best_val_loss": progress.get("best_val_loss"),
            "status": "stopped",
            "completed_segments": boundary,
            "current_segment": boundary,
            "total_segments": int(progress.get("total_segments") or 520),
            "recovered": True,
            "recovery_reason": str(reason),
        },
        "status": "stopped",
        "completed_segments": boundary,
        "current_segment": boundary,
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    manifest = {
        "recipe": dict(recipe or {}),
        "chunk": int(chunk),
        "parent_kernel": str(parent_kernel or ""),
        "max_runtime_seconds": None,
        "recovered": True,
        "recovery_reason": str(reason),
        "recovered_at": now,
        "recovered_boundary_next_epoch": boundary,
    }
    (root / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )


def copy_incomplete_tree(source, destination):
    """Copy the incomplete Output tree to a writable staging directory."""
    source = Path(source)
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Copy the experiment root (files + checkpoints). Status JSON noise is fine.
    shutil.copytree(source, destination)
    return destination


def recover_incomplete_chunk(
    source,
    destination,
    *,
    repo_root,
    torch,
    recipe,
    chunk,
    parent_kernel,
    reason,
):
    """Copy ``source`` -> ``destination``, export last_model, write durable sidecars.

    Returns the durable boundary (``next_epoch``).
    """
    staged = copy_incomplete_tree(source, destination)
    boundary, _state, progress = inspect_incomplete_boundary(staged, torch)
    if not (staged / "checkpoints/last_model/model.safetensors").is_file():
        export_last_model(staged, repo_root)
    write_durable_sidecar(
        staged,
        boundary,
        progress,
        recipe,
        chunk=chunk,
        parent_kernel=parent_kernel,
        reason=reason,
    )
    return boundary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Incomplete Output root")
    parser.add_argument("--destination", required=True, help="Writable recovered root")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--parent-kernel", default="")
    parser.add_argument(
        "--reason",
        default="CANCEL_ACKNOWLEDGED mid Temporal Rank IC; recovered last_state boundary",
    )
    parser.add_argument(
        "--recipe-json",
        default="",
        help="Optional recipe JSON path; otherwise a minimal recipe is written",
    )
    args = parser.parse_args(argv)
    import torch

    recipe = {}
    if args.recipe_json:
        recipe = json.loads(Path(args.recipe_json).read_text(encoding="utf-8"))
        if "recipe" in recipe and isinstance(recipe["recipe"], dict):
            recipe = recipe["recipe"]
    boundary = recover_incomplete_chunk(
        args.source,
        args.destination,
        repo_root=args.repo_root,
        torch=torch,
        recipe=recipe,
        chunk=args.chunk,
        parent_kernel=args.parent_kernel,
        reason=args.reason,
    )
    print(json.dumps({"recovered_boundary": boundary, "destination": args.destination}))


if __name__ == "__main__":
    main()
