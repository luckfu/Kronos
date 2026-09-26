"""Frozen Kronos tokenizer adapter for the ModernBERT decision model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from model import KronosTokenizer


class FrozenKronosTokenizer(nn.Module):
    """Encode normalized 120-day windows into frozen ``s1`` and ``s2`` IDs.

    The wrapped Kronos tokenizer is deliberately kept outside the trainable
    ModernBERT graph. Its discrete IDs are consumed by trainable embeddings,
    so the decision model can learn a new representation without changing the
    production Kronos tokenizer weights.
    """

    lookback = 120

    def __init__(self, tokenizer: nn.Module) -> None:
        super().__init__()
        if not isinstance(tokenizer, KronosTokenizer):
            raise TypeError("tokenizer must be a KronosTokenizer instance")
        if int(tokenizer.d_in) != 6:
            raise ValueError(f"expected tokenizer d_in=6, got {tokenizer.d_in}")
        if int(tokenizer.s1_bits) <= 0 or int(tokenizer.s2_bits) <= 0:
            raise ValueError("tokenizer codebooks must have positive bit widths")
        self.tokenizer = tokenizer
        self.s1_vocab_size = 2 ** int(tokenizer.s1_bits)
        self.s2_vocab_size = 2 ** int(tokenizer.s2_bits)
        self._freeze()

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "FrozenKronosTokenizer":
        """Load a Kronos tokenizer directory or Hub identifier."""

        path_text = str(path)
        config_path = Path(path_text) / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            tokenizer = KronosTokenizer(**config)
            state_path = Path(path_text) / "model.safetensors"
            if state_path.is_file():
                tokenizer = KronosTokenizer.from_pretrained(path_text)
            else:
                raise FileNotFoundError(state_path)
        else:
            tokenizer = KronosTokenizer.from_pretrained(path_text)
        return cls(tokenizer.to(device))

    def _freeze(self) -> None:
        for parameter in self.tokenizer.parameters():
            parameter.requires_grad_(False)
        self.tokenizer.eval()

    def train(self, mode: bool = True) -> "FrozenKronosTokenizer":
        # ``nn.Module.train`` recursively changes children, so restore eval
        # after the normal bookkeeping to prevent dropout from being enabled.
        super().train(mode)
        self._freeze()
        return self

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if history.ndim != 3 or tuple(history.shape[1:]) != (self.lookback, 6):
            raise ValueError(
                "history must have shape [batch, 120, 6], "
                f"got {tuple(history.shape)}"
            )
        if not torch.is_floating_point(history):
            raise TypeError("history must be a floating-point tensor")
        if not torch.isfinite(history).all():
            raise ValueError("history contains NaN or infinite values")

        history = history.to(dtype=torch.float32)
        device_type = history.device.type
        autocast_context = torch.autocast(
            device_type=device_type,
            enabled=False,
        )
        with torch.no_grad(), autocast_context:
            encoded = self.tokenizer.encode(history, half=True)

        if not isinstance(encoded, (tuple, list)) or len(encoded) != 2:
            raise RuntimeError("Kronos tokenizer did not return two codebooks")
        s1, s2 = (item.to(dtype=torch.long) for item in encoded)
        expected_shape = history.shape[:2]
        if tuple(s1.shape) != tuple(expected_shape) or tuple(s2.shape) != tuple(
            expected_shape
        ):
            raise RuntimeError(
                "Kronos tokenizer returned unexpected shapes: "
                f"{tuple(s1.shape)}, {tuple(s2.shape)}"
            )
        if (
            s1.numel()
            and (int(s1.min()) < 0 or int(s1.max()) >= self.s1_vocab_size)
        ):
            raise RuntimeError("s1 IDs are outside the configured vocabulary")
        if (
            s2.numel()
            and (int(s2.min()) < 0 or int(s2.max()) >= self.s2_vocab_size)
        ):
            raise RuntimeError("s2 IDs are outside the configured vocabulary")
        return s1, s2
