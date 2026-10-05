"""Build bounded LR comparisons from the fixed R2 runner, without submitting."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "finetune/kaggle_kairos_r2_chunk2/kaggle_kairos_r2_chunk2.py"
ARMS = {"1e4": 1e-4, "3e5": 3e-5, "1e5": 1e-5}


def build(arm: str, destination: Path | None = None) -> Path:
    lr = ARMS[arm]
    slug = "kairos-r2-lr-probe-0-0001" if arm == "1e4" else f"kairos-r2-lr-probe-{arm}"
    destination = destination or ROOT / "finetune" / f"kaggle_kairos_r2_lr_probe_{arm}"
    destination.mkdir(parents=True, exist_ok=True)
    tree = ast.parse(SOURCE.read_text())
    overrides = {
        "LR_PROBE": True,
        "LEARNING_RATE_OVERRIDE": lr,
        "GPU_BUDGET_SECONDS": 85 * 60,
        "RUNTIME_RESERVE_SECONDS": 15 * 60,
        "MAX_SEGMENTS_THIS_RUN": 4,
        "RUN_PURPOSE": "bounded-lr-probe-from-chunk1-segment1",
        "SWANLAB_API_KEY_FALLBACK": "",
        "SWANLAB_RUN_ID": f"kairos-lr-probe-{arm}-20260930",
    }
    replaced = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in overrides:
                node.value = ast.Constant(overrides[target.id])
                replaced.add(target.id)
    if replaced != overrides.keys():
        raise RuntimeError(f"unmatched overrides: {overrides.keys() - replaced}")
    source = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    code_file = "lr_probe.py"
    (destination / code_file).write_text(source)
    metadata = json.loads((SOURCE.parent / "kernel-metadata.json").read_text())
    metadata.update({
        "id": f"wynstonliu/{slug}",
        "title": "Kairos R2 LR Probe 0.0001" if arm == "1e4" else f"Kairos R2 LR Probe {arm}",
        "code_file": code_file,
    })
    if metadata["kernel_sources"] != ["wynstonliu/kairos-r2-modernbert-chunk-1"]:
        raise RuntimeError("LR probes must only mount the clean Chunk 1 output")
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (destination / "build_manifest.json").write_text(json.dumps({
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "generated_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "learning_rate": lr,
        "max_additional_segments": 4,
        "wall_time_budget_seconds": 5100,
        "optimizer_moments": "preserved; LR changed only after exact restore check",
        "selection_scope": "reused development validation; not independent test evidence",
    }, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    for arm_name in ARMS:
        print(build(arm_name))
