import importlib.util

import numpy as np
import torch


def load_module():
    path = "finetune/timesfm3_lora_finetune.py"
    spec = importlib.util.spec_from_file_location("timesfm3_lora_finetune", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DummyModel:
    input_patch_len = 32
    output_patch_len = 64
    use_stitching = True
    _stitching_extract_len = 64
    rolls = 2
    use_frozen_running_stats = False


def test_build_forward_inputs_masks_left_padding_and_future():
    module = load_module()
    contexts = torch.ones(2, 120, 7)
    inputs, context_patches, padded_horizon = module.build_forward_inputs(
        contexts, DummyModel(), 10
    )
    assert context_patches == 4
    assert padded_horizon == 64
    assert inputs["values"].shape == (2, 6, 6, 32)
    assert inputs["masks"].shape == (2, 6, 6, 32)
    assert inputs["patch_is_target"].shape == (2, 6, 6)
    assert inputs["masks"][:, :, 0, :8].all()
    assert not inputs["masks"][:, :, 0, 8:].any()
    assert inputs["masks"][:, :, -1].all()
    assert inputs["patch_is_target"].all()


def test_path_losses_return_normalized_and_auxiliary_metrics():
    module = load_module()
    prediction = torch.tensor([[11.0, 12.0]])
    target = torch.tensor([[10.0, 14.0]])
    total, metrics = module.path_losses(
        prediction, target, torch.tensor([10.0]), return_weight=0.1
    )
    assert np.isfinite(float(total))
    assert metrics["raw_loss"] > 0
    assert metrics["normalized_loss"] > 0
    assert metrics["return_loss"] > 0


def test_forecast_extraction_preserves_gradient_path():
    module = load_module()
    logits = torch.zeros(1, 6, 6, 64, 9)
    logits[:, 0, 3, :10, module.QUANTILE_INDEX] = 1.0
    logits.requires_grad_()
    prediction = module.extract_median_forecast(
        {"logits": logits}, 4, 10, DummyModel()
    )
    loss = prediction.square().mean()
    loss.backward()
    assert prediction.shape == (1, 10)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_dashboard_metric_helpers_are_numeric_and_complete():
    module = load_module()
    assert module.gpu_metrics()["gpu/count"] == 0.0 or module.gpu_metrics()["gpu/count"] >= 1.0
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    output = model(torch.ones(2, 3)).sum()
    output.backward()
    metrics = {
        **module.optimizer_metrics(optimizer),
        "train/grad_norm": module.grad_norm(model),
        **module.gpu_metrics(),
    }
    assert metrics["train/learning_rate"] == 1e-4
    assert metrics["train/weight_decay"] == 0.01
    assert metrics["train/grad_norm"] > 0
    assert all(isinstance(value, float) for value in metrics.values())


def test_compact_checkpoint_excludes_backbone_weights():
    module = load_module()
    model = torch.nn.Sequential(
        module.LoRALinear(torch.nn.Linear(3, 2), rank=1, alpha=1, dropout=0.0)
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-4,
    )
    checkpoint = module.make_checkpoint(model, optimizer, 7, 0.5, [])
    assert checkpoint["format"] == "timesfm3_trainable_compact_v2"
    assert checkpoint["model"]
    assert all("lora_" in key for key in checkpoint["model"])
    assert not any(key.endswith("base.weight") for key in checkpoint["model"])


def test_checkpoint_restores_non_lora_trained_weights_buffers_and_optimizer(tmp_path):
    module = load_module()

    def new_model():
        model = torch.nn.Sequential(
            module.LoRALinear(torch.nn.Linear(3, 2), rank=1, alpha=1, dropout=0.0),
            torch.nn.Linear(2, 1),
        )
        model.register_buffer("running_value", torch.tensor(4.0))
        return model

    torch.manual_seed(1)
    model = new_model()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=1e-4,
    )
    original_sha = module.frozen_model_sha256(model)
    model(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    model.running_value.fill_(7)
    metadata = {"validation_complete": True, "base_model_sha256": original_sha}
    checkpoint = module.make_checkpoint(model, optimizer, 1, 0.5, [], metadata=metadata)
    assert {"1.weight", "1.bias", "running_value"}.issubset(checkpoint["model"])
    module.save_checkpoint(tmp_path / "state.pt", checkpoint)
    state = torch.load(tmp_path / "state.pt", weights_only=True)
    torch.manual_seed(1)
    restored = new_model()
    restored_optimizer = torch.optim.AdamW(
        (p for p in restored.parameters() if p.requires_grad), lr=1e-4,
    )
    module.restore_checkpoint(restored, restored_optimizer, state, original_sha)
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key]), key
    assert torch.equal(model(torch.ones(2, 3)), restored(torch.ones(2, 3)))
    assert len(restored_optimizer.state) == len(optimizer.state)
    expected_random = torch.rand(2)
    module.restore_checkpoint(restored, restored_optimizer, state, original_sha)
    assert torch.equal(torch.rand(2), expected_random)


def test_restore_rejects_legacy_missing_parameters_and_unvalidated_states():
    import pytest

    module = load_module()
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    base_sha = module.frozen_model_sha256(model)
    with pytest.raises(RuntimeError, match="legacy"):
        module.restore_checkpoint(model, optimizer, {"format": "timesfm3_lora_compact_v1"}, base_sha)
    state = module.make_checkpoint(
        model, optimizer, 0, 0.5, [],
        metadata={"validation_complete": False, "base_model_sha256": base_sha},
    )
    with pytest.raises(RuntimeError, match="validation-complete"):
        module.restore_checkpoint(model, optimizer, state, base_sha)
    state["validation_complete"] = True
    state["model"].pop("bias")
    with pytest.raises(RuntimeError, match="every trained"):
        module.restore_checkpoint(model, optimizer, state, base_sha)


def test_timestamped_events_and_subprocess_output_are_persisted(tmp_path, monkeypatch):
    import json
    import sys

    module = load_module()
    monkeypatch.setattr(module, "OUTPUT_ROOT", tmp_path)
    module.phase("training", step=3)
    module.emit("heartbeat", step=4, **{"train/loss": 0.1})
    module.run_streamed([sys.executable, "-u", "-c", "print('child phase output', flush=True)"])
    events = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert events[-1]["phase"] == "training"
    assert events[-1]["step"] == 4
    assert "timestamp" in events[-1]
    assert "child phase output" in (tmp_path / "run.log").read_text()
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["step"] == events[-1]["step"]
    module.emit("phase_heartbeat")
    assert json.loads((tmp_path / "progress.json").read_text())["step"] == 4


def test_training_and_validation_complete_continuation_end_to_end(tmp_path, monkeypatch):
    import argparse
    import json
    import shutil
    import sys
    import types

    module = load_module()

    class TinyModel(torch.nn.Module):
        input_patch_len = 32
        use_frozen_running_stats = False

        def __init__(self):
            super().__init__()
            self.query_proj = torch.nn.Linear(7, 10)
            self.head = torch.nn.Linear(10, 10)

        def forward(self, inputs, **kwargs):
            return self.head(self.query_proj(inputs["values"].mean(dim=1)))

    class Forecaster:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return types.SimpleNamespace(model=TinyModel())

    payloads = []
    run = types.SimpleNamespace(
        url="local-test", log=lambda payload, step: payloads.append((step, payload)),
    )
    monkeypatch.setitem(sys.modules, "timesfm3", types.SimpleNamespace(TimesFM3Forecaster=Forecaster))
    monkeypatch.setitem(sys.modules, "swanlab", types.SimpleNamespace(
        login=lambda **kwargs: None, init=lambda **kwargs: run, finish=lambda: None,
    ))
    monkeypatch.setenv("SWANLAB_API_KEY", "local-test-key")
    monkeypatch.setattr(module, "validate_gpu", lambda device: ["test", "test"])
    monkeypatch.setattr(module, "build_forward_inputs", lambda contexts, model, horizon: (
        {"values": contexts, "patch_is_target": torch.ones(
            contexts.shape[0], 6, 6, dtype=torch.bool,
        )}, 4, 64,
    ))
    monkeypatch.setattr(module, "extract_median_forecast", lambda output, *args: output)
    data = tmp_path / "data.npz"
    np.savez(data, contexts=np.ones((8, 120, 7), dtype=np.float32),
             targets=np.ones((8, 10), dtype=np.float32))
    output = tmp_path / "first"
    monkeypatch.setattr(module, "OUTPUT_ROOT", output)
    args = argparse.Namespace(
        device="cpu", seed=42, model="fake", batch_size=2,
        train=str(data), val=str(data), output=str(output),
        lora_rank=1, lora_alpha=1, lora_dropout=0.05, multi_gpu=True,
        learning_rate=1e-4, weight_decay=0.01, return_loss_weight=0.1,
        horizon=10, max_grad_norm=1.0, val_max_batches=2, resume_from=None,
        max_steps=2, run_budget_seconds=1800, finalize_reserve_seconds=300,
        epochs=1, checkpoint_interval=1, log_interval=1,
    )
    assert module.train(args)["global_step"] == 2
    state = torch.load(output / "last_state.pt", weights_only=True)
    assert state["validation_complete"]
    assert state["batches_completed"] == 2
    assert {"head.weight", "head.bias"}.issubset(state["model"])
    assert not torch.load(output / "training_snapshot.pt", weights_only=True)["validation_complete"]
    assert json.loads((output / "summary.json").read_text())["global_step"] == 2
    assert all(isinstance(value, (int, float)) for _, payload in payloads for value in payload.values())
    next_output = tmp_path / "second"
    shutil.copytree(output, next_output)
    monkeypatch.setattr(module, "OUTPUT_ROOT", next_output)
    args.output = str(next_output)
    args.resume_from = str(next_output / "last_state.pt")
    args.max_steps = 4
    assert module.train(args)["global_step"] == 4
    restored = torch.load(next_output / "last_state.pt", weights_only=True)
    assert restored["validation_complete"]
    assert restored["epoch_index"] == 1
    assert restored["batches_completed"] == 0
    assert restored["optimizer"]["state"]
