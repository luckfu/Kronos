"""Build one gated, bounded trial; publishing is a separate explicit action."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from build_kairos_lr_probe import build


def build_trial(destination: Path | None = None) -> Path:
    root = Path(__file__).resolve().parents[1]
    destination = destination or root / "finetune/kaggle_kairos_rope_fixed_trial"
    build("1e4", destination)
    path = destination / "lr_probe.py"
    tree = ast.parse(path.read_text())
    overrides = {
        "REPAIRED_TRIAL": True,
        "DIAGNOSTIC_ONLY": False,
        "GPU_BUDGET_SECONDS": 3 * 60 * 60,
        "RUNTIME_RESERVE_SECONDS": 15 * 60,
        "MAX_SEGMENTS_THIS_RUN": 12,
        "MIN_OBSERVATION_SEGMENTS": 8,
        "EARLY_STOP_PATIENCE": 4,
        "SOURCE_KERNEL_SLUG": "kairos-r2-modernbert-chunk-1",
        "REASSESS_KERNEL_SLUG": "kairos-r2-lr-probe-0-0001",
        "TOKENIZER_CODE_COMMIT": "199471183f21b0b8de073bc1547f0199ac14b81c",
        "RUN_PURPOSE": "rope-fixed-gated-bounded-observation",
        "SWANLAB_RUN_ID": "kairos-rope-fixed-bounded-trial-20260930",
    }
    replaced = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in overrides:
                node.value = ast.Constant(overrides[name])
                replaced.add(name)
    assert replaced == overrides.keys()
    source = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    path.write_text(source)
    metadata_path = destination / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update({
        "id": "wynstonliu/kairos-rope-fixed-bounded-trial",
        "title": "Kairos RoPE Fixed Bounded Trial",
        "kernel_sources": [
            "wynstonliu/kairos-r2-modernbert-chunk-1",
            "wynstonliu/kairos-r2-lr-probe-0-0001",
        ],
    })
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    manifest_path = destination / "build_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "generated_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "wall_time_budget_seconds": 10800,
        "max_additional_segments": 12,
        "minimum_segments_before_metric_patience": 8,
        "patience": 4,
        "purpose": "One repaired-RoPE acceptance then bounded training; no automatic relay",
        "source_kernel": overrides["SOURCE_KERNEL_SLUG"],
        "reassess_only_kernel": overrides["REASSESS_KERNEL_SLUG"],
        "metric_gate": "exact repeated predictions and theoretical/cross-rank RoPE; not legacy 0.67026",
    })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    print(build_trial())
