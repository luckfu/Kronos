"""Materialize model-only weights from a resumable training state."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    state_path = Path(args.state)
    config_path = Path(args.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model = state.get("model")
    if not isinstance(model, dict) or not model:
        raise SystemExit("Checkpoint does not contain a model state dict")
    tensors = {name: tensor.detach().contiguous() for name, tensor in model.items()}
    save_file(tensors, output / "model.safetensors")
    shutil.copy2(config_path, output / "config.json")
    metadata = {
        "source": "resumable training Last checkpoint",
        "source_next_epoch": state.get("next_epoch"),
        "source_best_val_loss": state.get("best_val_loss"),
        "optimizer_state_included": False,
        "scheduler_state_included": False,
    }
    (output / "README.md").write_text(
        "# Extracted Last model\n\n"
        "Model-only weights extracted from a resumable training checkpoint. "
        "A fresh optimizer and scheduler must be used for incremental training.\n",
        encoding="utf-8",
    )
    (output / "extraction.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
