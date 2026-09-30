"""Check cloud logging and stage private upload credentials outside Git."""
import ast
import json
import os
from pathlib import Path

import swanlab

from build_kairos_r2_restart import build, ROOT


def main():
    source = build()
    key = os.environ.get("SWANLAB_API_KEY", "").strip()
    if not key:
        # Reuse the project's already-authorized logging credential without printing it.
        tree = ast.parse((ROOT / "finetune/kaggle_kairos_r2_chunk1/kaggle_kairos_r2_chunk1.py").read_text())
        key = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == "SWANLAB_API_KEY_FALLBACK")
    if not key:
        raise RuntimeError("Cloud logging credential missing; refuse submission")
    swanlab.login(api_key=key)
    run = swanlab.init(
        id="kairos-r2-restart-20260930", resume="allow", project="finance",
        workspace="roc_fu", experiment_name="kairos-r2-restart-20260930",
        mode="cloud", config={"purpose": "formal R2 restart", "lr": 1e-4,
                             "source": "R1 final weights", "session_budget_seconds": 36000},
    )
    url = run.url
    if not url:
        raise RuntimeError("Cloud run URL unavailable")
    run.log({"preflight/cloud_logging_ready": 1})
    swanlab.finish()
    destination = ROOT / "artifacts/kairos_r2_restart_20260930/private_staging"
    destination.mkdir(parents=True, exist_ok=True)
    os.chmod(destination, 0o700)
    tree = ast.parse((source / "restart.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "SWANLAB_API_KEY_FALLBACK":
                node.value = ast.Constant(key)
    path = destination / "restart.py"
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    os.chmod(path, 0o600)
    (destination / "kernel-metadata.json").write_text((source / "kernel-metadata.json").read_text())
    (destination.parent / "cloud_preflight.json").write_text(json.dumps({
        "url": url, "cloud_logging_ready": True, "staging": str(destination),
        "credential_in_tracked_source": False,
    }, indent=2) + "\n")
    print(json.dumps({"url": url, "staging": str(destination)}))


if __name__ == "__main__":
    main()
