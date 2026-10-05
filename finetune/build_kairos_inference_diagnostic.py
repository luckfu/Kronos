"""Build an inference-only, 20-minute diagnostic; never submit automatically."""
import ast
import hashlib
import json
from pathlib import Path

from build_kairos_lr_probe import build

root = Path(__file__).resolve().parents[1]
destination = root / "finetune/kaggle_kairos_inference_diagnostic"
build("1e4", destination)
path = destination / "lr_probe.py"
tree = ast.parse(path.read_text())
overrides = {
    "DIAGNOSTIC_ONLY": True,
    "GPU_BUDGET_SECONDS": 1200,
    "RUN_PURPOSE": "inference-reproducibility-diagnostic",
    "SWANLAB_RUN_ID": "kairos-inference-diagnostic-20260930",
}
for node in tree.body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id in overrides:
            node.value = ast.Constant(overrides[node.targets[0].id])
source = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
path.write_text(source)
metadata_path = destination / "kernel-metadata.json"
metadata = json.loads(metadata_path.read_text())
metadata.update({
    "id": "wynstonliu/kairos-inference-diagnostic",
    "title": "Kairos Inference Diagnostic",
})
metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
manifest_path = destination / "build_manifest.json"
manifest = json.loads(manifest_path.read_text())
manifest.update({
    "generated_sha256": hashlib.sha256(source.encode()).hexdigest(),
    "wall_time_budget_seconds": 1200,
    "max_additional_segments": 0,
    "purpose": "Two full validations, fingerprints, zero optimizer steps",
})
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
print(destination)
