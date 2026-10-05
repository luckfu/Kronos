"""Build the repaired R2 restart. Cloud credentials are supplied only at staging."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "finetune/kaggle_kairos_r2_chunk2/kaggle_kairos_r2_chunk2.py"


PROFILES = {
    "canonical": {"lr": 1e-4, "run_id": "kairos-r2-restart-20260930",
                  "slug": "kairos-r2-restart-canonical", "title": "Kairos R2 Restart Canonical"},
    "lr3e5": {"lr": 3e-5, "run_id": "kairos-r2-r1-lr3e5-20260930",
              "slug": "kairos-r2-r1-restart-lr-3e-5", "title": "Kairos R2 R1 Restart LR 3e-5"},
}


def build(destination=None, profile="canonical"):
    config = PROFILES[profile]
    suffix = "" if profile == "canonical" else "_" + profile
    destination = destination or ROOT / ("finetune/kaggle_kairos_r2_restart" + suffix)
    destination.mkdir(parents=True, exist_ok=True)
    overrides = {
        "FRESH_R2": True, "REPAIRED_TRIAL": True, "LR_PROBE": False,
        "DIAGNOSTIC_ONLY": False, "CHUNK_INDEX": 0,
        "GPU_BUDGET_SECONDS": 36000, "RUNTIME_RESERVE_SECONDS": 1800,
        "MAX_SEGMENTS_THIS_RUN": 451,
        "SOURCE_KERNEL_SLUG": "modernbert-decision-full-chunk-8",
        "TOKENIZER_CODE_COMMIT": "199471183f21b0b8de073bc1547f0199ac14b81c",
        "RUN_PURPOSE": "r2-restart-clean-cursor-canonical-rope",
        "SWANLAB_RUN_ID": config["run_id"],
        "LEARNING_RATE_OVERRIDE": config["lr"],
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
    # Fresh R1 starts do not enter the checkpoint-restore LR override branch.
    optimizers = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == "AdamW"]
    assert len(optimizers) == 1
    lr_keywords = [kw for kw in optimizers[0].keywords if kw.arg == "lr"]
    assert len(lr_keywords) == 1
    lr_keywords[0].value = ast.Constant(config["lr"])
    source = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    (destination / "restart.py").write_text(source)
    metadata = json.loads((ROOT / "finetune/kaggle_kairos_r2_chunk1/kernel-metadata.json").read_text())
    metadata.update(id="wynstonliu/" + config["slug"], title=config["title"],
                    code_file="restart.py")
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (destination / "build_manifest.json").write_text(json.dumps({
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "generated_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "budget_seconds": 36000, "segment_limit_is_dataset_ceiling": True,
        "metric_patience_enabled": False, "lr": config["lr"],
        "profile": profile, "cloud_run_id": config["run_id"],
        "initialization": "R1 final weights; fresh optimizer/scaler; cursor zero",
        "cloud_credentials": "staging only, not tracked",
    }, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="canonical")
    print(build(profile=parser.parse_args().profile))
