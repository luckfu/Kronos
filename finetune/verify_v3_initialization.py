"""Verify that V3 starts exactly as the official unconditioned base model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import Kronos
from train_predictor import build_optimizer_groups


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    parent = Path(args.parent)
    actual_sha = sha256_file(parent / "model.safetensors")
    if actual_sha != args.parent_sha256:
        raise SystemExit(f"Parent SHA mismatch: {actual_sha} != {args.parent_sha256}")

    parent_config = json.loads((parent / "config.json").read_text())
    required_parent = {
        "num_sectors": 0,
        "num_size_buckets": 0,
        "use_size_percentile": False,
        "use_size_path": False,
    }
    normalized_parent = {
        "num_sectors": int(parent_config.get("num_sectors", 0)),
        "num_size_buckets": int(parent_config.get("num_size_buckets", 0)),
        "use_size_percentile": bool(parent_config.get("use_size_percentile", False)),
        "use_size_path": bool(parent_config.get("use_size_path", False)),
    }
    mismatches = {
        key: {"actual": normalized_parent[key], "expected": expected}
        for key, expected in required_parent.items()
        if normalized_parent[key] != expected
    }
    if mismatches:
        raise SystemExit(f"Parent config is not an unconditioned base: {mismatches}")

    v3_config = dict(parent_config)
    v3_config.update(
        num_sectors=86,
        num_size_buckets=0,
        context_layer=10,
        use_size_percentile=False,
        use_size_path=True,
        size_path_input_dim=4,
    )
    torch.manual_seed(100)
    v1 = Kronos.from_pretrained(parent, **parent_config).eval()
    torch.manual_seed(100)
    v3 = Kronos.from_pretrained(parent, **v3_config).eval()

    v1_state = v1.state_dict()
    v3_state = v3.state_dict()
    inherited_names = sorted(set(v1_state) & set(v3_state))
    changed = [name for name in inherited_names if not torch.equal(v1_state[name], v3_state[name])]
    if changed:
        raise SystemExit(f"Inherited V1.2 parameters changed: {changed[:10]}")
    new_names = sorted(set(v3_state) - set(v1_state))
    new_prefixes = ("sector_emb.", "size_path_mlp.")
    if not new_names or any(not name.startswith(new_prefixes) for name in new_names):
        raise SystemExit(f"Unexpected V3-only parameters: {new_names}")
    if torch.count_nonzero(v3.sector_emb.weight).item() != 0:
        raise SystemExit("sector_emb is not exact zero")
    if torch.count_nonzero(v3.size_path_mlp[-1].weight).item() != 0:
        raise SystemExit("size_path_mlp output weight is not exact zero")
    if torch.count_nonzero(v3.size_path_mlp[-1].bias).item() != 0:
        raise SystemExit("size_path_mlp output bias is not exact zero")

    batch, tokens = 2, 12
    torch.manual_seed(871)
    s1 = torch.randint(0, 2 ** v1.s1_bits, (batch, tokens))
    s2 = torch.randint(0, 2 ** v1.s2_bits, (batch, tokens))
    sector = torch.tensor([1, 86])
    size_path = torch.rand(batch, tokens, 4)
    with torch.no_grad():
        v1_logits = v1(
            s1, s2, use_teacher_forcing=True, s1_targets=s1,
        )
        v3_logits = v3(
            s1, s2, sector_id=sector,
            size_path=size_path, use_teacher_forcing=True, s1_targets=s1,
        )
    if not all(torch.equal(left, right) for left, right in zip(v1_logits, v3_logits)):
        raise SystemExit("V3 initial logits are not exactly equal to base logits")

    optimizer_config = {
        "predictor_learning_rate": 1e-6,
        "condition_learning_rate": 1e-5,
        "predictor_warmup_start_learning_rate": 1e-7,
        "condition_warmup_start_learning_rate": 1e-6,
        "predictor_min_learning_rate": 1e-7,
        "condition_min_learning_rate": 1e-6,
        "scheduler_min_learning_rate": 1e-7,
        "adam_weight_decay": 0.1,
        "condition_parameter_prefixes": "sector_emb.,size_path_mlp.",
    }
    groups = build_optimizer_groups(v3, optimizer_config)
    family_by_parameter = {
        id(parameter): group["family"]
        for group in groups
        for parameter in group["params"]
    }
    wrong_families = {
        name: family_by_parameter[id(parameter)]
        for name, parameter in v3.named_parameters()
        if family_by_parameter[id(parameter)]
        != ("condition" if name.startswith(new_prefixes) else "adaptation")
    }
    if wrong_families:
        raise SystemExit(f"V3 optimizer family mismatch: {wrong_families}")

    report = {
        "status": "passed",
        "parent_sha256": actual_sha,
        "parent_config_contract": required_parent,
        "inherited_parameter_tensors": len(inherited_names),
        "v3_only_parameter_tensors": new_names,
        "sector_embedding_nonzero": 0,
        "dynamic_output_weight_nonzero": 0,
        "dynamic_output_bias_nonzero": 0,
        "initial_logits_exactly_equal": True,
        "high_lr_parameter_prefixes": ["sector_emb.", "size_path_mlp."],
        "optimizer_families_verified": True,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
