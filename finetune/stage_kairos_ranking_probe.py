"""Stage private SwanLab credential into artifacts/ then optionally push kernel."""

from __future__ import annotations

import ast
import json
import os
import shutil
from pathlib import Path

from build_kairos_ranking_probe import (
    OWNER,
    ROOT,
    SLUG,
    SWANLAB_RUN_ID,
    TITLE,
    build,
)


def _load_swanlab_key() -> str:
    key = os.environ.get("SWANLAB_API_KEY", "").strip()
    if key:
        return key
    tree = ast.parse(
        (ROOT / "finetune/kaggle_kairos_r2_chunk1/kaggle_kairos_r2_chunk1.py").read_text()
    )
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "SWANLAB_API_KEY_FALLBACK"
        ):
            return str(ast.literal_eval(node.value))
    return ""


def stage() -> dict:
    source = build()
    key = _load_swanlab_key()
    if not key:
        raise RuntimeError("Cloud logging credential missing; refuse submission")

    import swanlab

    swanlab.login(api_key=key)
    run = swanlab.init(
        id=SWANLAB_RUN_ID,
        resume="allow",
        project="finance",
        workspace="roc_fu",
        experiment_name=SWANLAB_RUN_ID,
        mode="cloud",
        config={
            "purpose": "mfe10 ranking probe short phase M",
            "target": "continuous mfe10 Rank IC",
            "not_binary_mfe10": True,
            "not_22_layer_binary": True,
            "session_budget_seconds": 3600,
            "max_segments": 3,
        },
    )
    url = getattr(run, "url", getattr(run, "web_url", ""))
    if not url:
        raise RuntimeError("Cloud run URL unavailable")
    run.log({"preflight/cloud_logging_ready": 1})
    swanlab.finish()

    destination = ROOT / "artifacts" / "kairos_ranking_probe" / "private_staging"
    destination.mkdir(parents=True, exist_ok=True)
    os.chmod(destination, 0o700)
    tree = ast.parse((source / "train_ranking_probe.py").read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "SWANLAB_API_KEY_FALLBACK"
        ):
            node.value = ast.Constant(key)
    path = destination / "train_ranking_probe.py"
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    os.chmod(path, 0o600)
    shutil.copy2(source / "kernel-metadata.json", destination / "kernel-metadata.json")
    shutil.copy2(source / "build_manifest.json", destination / "build_manifest.json")
    preflight = {
        "url": url,
        "cloud_logging_ready": True,
        "staging": str(destination),
        "kernel_id": f"{OWNER}/{SLUG}",
        "title": TITLE,
        "swanlab_run_id": SWANLAB_RUN_ID,
        "credential_in_tracked_source": False,
        "push_command": f"kaggle kernels push -p {destination}",
    }
    (destination.parent / "cloud_preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n"
    )
    return preflight


if __name__ == "__main__":
    print(json.dumps(stage(), indent=2))
