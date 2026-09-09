"""Compare two Kronos predictor checkpoints without loading the model class."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import load_file


GROUPS = (
    ("transformer.0-2", re.compile(r"^transformer\.[0-2]\.")),
    ("transformer.3-5", re.compile(r"^transformer\.[3-5]\.")),
    ("transformer.6", re.compile(r"^transformer\.6\.")),
    ("transformer.7", re.compile(r"^transformer\.7\.")),
    ("condition", re.compile(r"(?:^|\.)(?:sector_emb|size_emb|size_mlp)(?:\.|$)")),
    ("dep_layer", re.compile(r"^dep_layer\.")),
    ("head", re.compile(r"^head\.")),
    ("norm", re.compile(r"^norm\.")),
)


def load_weights(directory: Path) -> dict[str, torch.Tensor]:
    path = directory / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return load_file(str(path), device="cpu")


def group_for(name: str) -> str:
    for label, pattern in GROUPS:
        if pattern.search(name):
            return label
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("initial", type=Path, help="Initial model directory")
    parser.add_argument("current", type=Path, help="Current model directory")
    args = parser.parse_args()

    initial = load_weights(args.initial)
    current = load_weights(args.current)
    stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    missing = []
    per_tensor = []

    for name, current_tensor in current.items():
        if name not in initial:
            missing.append(name)
            continue
        initial_tensor = initial[name]
        if not current_tensor.is_floating_point() or not initial_tensor.is_floating_point():
            continue
        initial_float = initial_tensor.float()
        current_float = current_tensor.float()
        initial_norm = float(torch.linalg.vector_norm(initial_float))
        delta_norm = float(torch.linalg.vector_norm(current_float - initial_float))
        count = current_tensor.numel()
        relative = delta_norm / max(initial_norm, 1e-12)
        label = group_for(name)
        item = stats[label]
        item[0] += delta_norm * delta_norm
        item[1] += initial_norm * initial_norm
        item[2] += count
        item[3] += relative * count
        per_tensor.append((relative, name, delta_norm, initial_norm, count))

    print(f"Initial: {args.initial}")
    print(f"Current: {args.current}")
    print(f"Matched floating tensors: {len(per_tensor)}")
    print(f"Current-only tensors: {len(missing)}")
    print("group                         params       weighted drift   mean tensor drift")
    print("-" * 82)
    for label, item in sorted(stats.items()):
        delta_sq, initial_sq, count, weighted_sum = item
        weighted_drift = (delta_sq / max(initial_sq, 1e-24)) ** 0.5
        mean_tensor_drift = weighted_sum / max(count, 1)
        print(f"{label:28s} {int(count):12,d} {weighted_drift:16.4%} {mean_tensor_drift:18.4%}")

    print("\nLargest tensor drifts:")
    for relative, name, delta_norm, initial_norm, count in sorted(per_tensor, reverse=True)[:20]:
        print(f"{relative:10.4%}  {count:10,d}  {name}")


if __name__ == "__main__":
    main()
