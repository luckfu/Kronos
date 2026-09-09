import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file


def find_one(root, suffix):
    paths = sorted(root.glob(f"**/{suffix}"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {suffix}, found {len(paths)}: {paths}")
    return paths[0]


groups = (
    ("transformer.0-2", re.compile(r"^transformer\.[0-2]\.")),
    ("transformer.3-5", re.compile(r"^transformer\.[3-5]\.")),
    ("transformer.6", re.compile(r"^transformer\.6\.")),
    ("transformer.7", re.compile(r"^transformer\.7\.")),
    ("condition", re.compile(r"(?:^|\.)(?:sector_emb|size_emb|size_mlp)(?:\.|$)")),
    ("dep_layer", re.compile(r"^dep_layer\.")),
    ("head", re.compile(r"^head\.")),
    ("norm", re.compile(r"^norm\.")),
)


def group_for(name):
    for label, pattern in groups:
        if pattern.search(name):
            return label
    return "other"


bootstrap_root = Path("/kaggle/input/notebooks/wynstonliu/kronos-small-0-1-bootstrap-final")
chunk_root = Path("/kaggle/input/notebooks/wynstonliu/kronos-small-0-1-main-chunk-4")
initial_path = find_one(bootstrap_root, "small_0.1_bootstrap/checkpoints/best_model/model.safetensors")
current_path = find_one(chunk_root, "small_0.1_main/checkpoints/last_model/model.safetensors")
initial = load_file(str(initial_path), device="cpu")
current = load_file(str(current_path), device="cpu")
stats = defaultdict(lambda: [0.0, 0.0, 0])
rows = []
for name, current_tensor in current.items():
    if name not in initial or not current_tensor.is_floating_point():
        continue
    initial_tensor = initial[name]
    if not initial_tensor.is_floating_point():
        continue
    a = initial_tensor.float()
    b = current_tensor.float()
    initial_norm = float(torch.linalg.vector_norm(a))
    delta_norm = float(torch.linalg.vector_norm(b - a))
    count = current_tensor.numel()
    group = group_for(name)
    stats[group][0] += delta_norm ** 2
    stats[group][1] += initial_norm ** 2
    stats[group][2] += count
    rows.append((delta_norm / max(initial_norm, 1e-12), name, count))

print(json.dumps({"initial": str(initial_path), "current": str(current_path)}, ensure_ascii=False))
print("group                         params       weighted drift")
print("-" * 68)
for group, (delta_sq, initial_sq, count) in sorted(stats.items()):
    drift = (delta_sq / max(initial_sq, 1e-24)) ** 0.5
    print(f"{group:28s} {count:12,d} {drift:16.4%}")
print("\nLargest tensor drifts:")
for drift, name, count in sorted(rows, reverse=True)[:20]:
    print(f"{drift:10.4%} {count:10,d} {name}")
