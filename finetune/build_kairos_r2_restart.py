"""Build the repaired R2 restart. Cloud credentials are supplied only at staging."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "finetune/kaggle_kairos_r2_chunk2/kaggle_kairos_r2_chunk2.py"


def build(destination=None):
    destination = destination or ROOT / "finetune/kaggle_kairos_r2_restart"
    destination.mkdir(parents=True, exist_ok=True)
    overrides = {
        "FRESH_R2": True, "REPAIRED_TRIAL": True, "LR_PROBE": False,
        "DIAGNOSTIC_ONLY": False, "CHUNK_INDEX": 0,
        "GPU_BUDGET_SECONDS": 36000, "RUNTIME_RESERVE_SECONDS": 1800,
        "MAX_SEGMENTS_THIS_RUN": 451,
        "SOURCE_KERNEL_SLUG": "modernbert-decision-full-chunk-8",
        "TOKENIZER_CODE_COMMIT": "199471183f21b0b8de073bc1547f0199ac14b81c",
        "RUN_PURPOSE": "r2-restart-clean-cursor-canonical-rope",
        "SWANLAB_RUN_ID": "kairos-r2-restart-20260930",
        "SWANLAB_API_KEY_FALLBACK": "",
    }
    tree = ast.parse(SOURCE.read_text())
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in overrides:
                node.value = ast.Constant(overrides[key])
                found.add(key)
    assert found == overrides.keys()
    source = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    (destination / "restart.py").write_text(source)
    metadata = json.loads((ROOT / "finetune/kaggle_kairos_r2_chunk1/kernel-metadata.json").read_text())
    metadata.update(id="wynstonliu/kairos-r2-restart-canonical", title="Kairos R2 Restart Canonical",
                    code_file="restart.py")
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (destination / "build_manifest.json").write_text(json.dumps({
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "generated_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "budget_seconds": 36000, "segment_limit_is_dataset_ceiling": True,
        "metric_patience_enabled": False, "lr": 0.0001,
        "initialization": "R1 final weights; fresh optimizer/scaler; cursor zero",
        "cloud_credentials": "staging only, not tracked",
    }, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    print(build())
