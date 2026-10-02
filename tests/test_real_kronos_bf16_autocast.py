"""Real-architecture bf16-autocast smoke test for the Beta v2.1 C1 TPU path.

The v6 TPU probe crashed during Beta v2.1 validation-denominator calibration
(evaluate_validation -> Kronos.encode_context -> self_attn ->
scaled_dot_product_attention) with "Expected query, key, and value to have the
same dtype": RoPE returned float32 q/k while v stayed bfloat16 under autocast.
The earlier CPU integration harness used a toy nn.Sequential and never ran the
Kronos attention code under autocast.

These tests build the REAL KronosTokenizer + Kronos classes (random init, the
production structure: 12 layers, 16 heads, context_layer=10, 86 sectors, size
percentile MLP, Beta v2.1 auxiliary heads, 10+10 bit tokens, lookback 120 /
predict 10) and run them under ``torch.autocast('cpu', torch.bfloat16)``
through the production code in finetune/train_predictor.py:

* RoPE dtype contract (unit),
* one training step mirrored from train_model (forward, fp32 losses, backward),
* evaluate_validation exactly as the denominator calibration calls it,
* evaluate_validation with the return-path consistency decode, period
  breakdown and condition ablation,
* train_predictor.main() end to end on a tiny synthetic dataset (model loading
  from disk, auto-calibration, one training segment, validation, checkpoint).

CPU autocast casts scaled_dot_product_attention inputs to one dtype, which
masks the bug; torch_xla's autocast does not.  An autouse fixture therefore
runs SDPA with the XLA policy (autocast off for that op), so the original
RuntimeError reproduces on CPU when the RoPE fix is reverted.

Set KRONOS_SMOKE_FULL_CONFIG=1 to use the real d_model=832/ff_dim=2048
predictor widths (slow).  Run this file directly with a torch_xla install
(``python tests/test_real_kronos_bf16_autocast.py --xla``) to repeat the model
checks on an XLA CPU device under ``torch.autocast('xla', torch.bfloat16)``.
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
try:
    import pytest
except ImportError:  # the torch_xla venv runs this file directly (--xla)
    import types
    pytest = types.SimpleNamespace(mark=types.SimpleNamespace(
        parametrize=lambda *args, **kwargs: (lambda function: function)
    ))
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).parents[1]
FINETUNE = ROOT / "finetune"
for path in (ROOT, FINETUNE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model.kronos import Kronos, KronosTokenizer  # noqa: E402
from model.module import RotaryPositionalEmbedding  # noqa: E402

FULL_CONFIG = os.getenv("KRONOS_SMOKE_FULL_CONFIG", "0").strip() == "1"
LOOKBACK = 120
PREDICT = 10
WINDOW = LOOKBACK + PREDICT + 1

# Tokenizer: the real Kronos-Tokenizer-base config.
TOKENIZER_CONFIG = {
    "d_in": 6, "d_model": 256, "n_heads": 4, "ff_dim": 512,
    "n_enc_layers": 4, "n_dec_layers": 4, "ffn_dropout_p": 0.0,
    "attn_dropout_p": 0.0, "resid_dropout_p": 0.0, "s1_bits": 10,
    "s2_bits": 10, "beta": 0.05, "gamma0": 1.0, "gamma": 1.1,
    "zeta": 0.05, "group_size": 4,
}
# Predictor: the real Kronos-A-Share-Beta-V2-1 structure; widths reduced by
# default (d_model 832 -> 128, ff_dim 2048 -> 256) to keep the suite fast.
PREDICTOR_CONFIG = {
    "s1_bits": 10, "s2_bits": 10, "n_layers": 12,
    "d_model": 832 if FULL_CONFIG else 128,
    "n_heads": 16, "ff_dim": 2048 if FULL_CONFIG else 256,
    "ffn_dropout_p": 0.2, "attn_dropout_p": 0.0, "resid_dropout_p": 0.2,
    "token_dropout_p": 0.0, "learn_te": True, "num_sectors": 86,
    "num_size_buckets": 0, "context_layer": 10, "use_size_percentile": True,
    "size_mlp_hidden_dim": 64, "use_beta_v21_auxiliary": True,
}

# The C1 TPU runner environment (finetune/kaggle_beta_v21_c1_tpu.py) minus
# paths/TPU-only settings; sample counts shrunk for a CPU smoke run.
C1_ENV = {
    "KRONOS_LOOKBACK_WINDOW": str(LOOKBACK),
    "KRONOS_PREDICT_WINDOW": str(PREDICT),
    "KRONOS_CONTEXT_LAYER": "10",
    "KRONOS_NUM_SECTORS": "86",
    "KRONOS_NUM_SIZE_BUCKETS": "0",
    "KRONOS_USE_SECTOR_FEATURES": "1",
    "KRONOS_USE_SIZE_FEATURES": "0",
    "KRONOS_USE_SIZE_PERCENTILE": "1",
    "KRONOS_TRAINABLE_TRANSFORMER_LAYERS": "-1",
    "KRONOS_RESET_SECTOR_EMBEDDING": "0",
    "KRONOS_RESET_SIZE_EMBEDDING": "0",
    "KRONOS_PREDICTOR_LOSS_MODE": "forecast",
    "KRONOS_HISTORY_LOSS_WEIGHT": "0.02",
    "KRONOS_FORECAST_HORIZON_WEIGHTS": "1.364,1.364,1.364,1.136,1.136,0.909,0.909,0.682,0.682,0.455",
    "KRONOS_SCHEDULER": "warmup_constant",
    "KRONOS_SCHEDULER_WARMUP_RATIO": "0.002",
    "KRONOS_PREDICTOR_LEARNING_RATE": "3e-5",
    "KRONOS_CONDITION_LEARNING_RATE": "3e-5",
    "KRONOS_PREDICTOR_WARMUP_START_LR": "3e-6",
    "KRONOS_CONDITION_WARMUP_START_LR": "3e-6",
    "KRONOS_PREDICTOR_MIN_LR": "3e-5",
    "KRONOS_CONDITION_MIN_LR": "3e-5",
    "KRONOS_BEST_SELECTION_METRIC": "beta_v21_score",
    "KRONOS_USE_BETA_V21_AUXILIARY": "1",
    "KRONOS_BETA_V21_AUTO_CALIBRATE": "1",
    "KRONOS_BETA_V21_AUXILIARY_WARMUP_STEPS": "1000",
    "KRONOS_BETA_V21_EMA_DECAY": "0.99",
    "KRONOS_BETA_V21_CONSISTENCY_SAMPLES": "4",
    "KRONOS_COVERAGE_PASSES": "1",
    "KRONOS_EPOCHS": "1",
    "KRONOS_REQUIRE_FULL_COVERAGE": "1",
    "KRONOS_EARLY_STOPPING_PATIENCE": "0",
    "KRONOS_VALIDATION_FULL_ONLY": "1",
    "KRONOS_VALIDATION_RANK0_ONLY": "0",
    "KRONOS_VALIDATION_SAMPLES": "0",
    "KRONOS_VALIDATION_QUICK_SAMPLES": "0",
    "KRONOS_VALIDATION_LARGE_SAMPLES": "0",
    "KRONOS_VALIDATION_LARGE_INTERVAL_SEGMENTS": "1",
    "KRONOS_MAX_RUNTIME_SECONDS": "0",
    "KRONOS_EVAL_ONLY": "0",
    "KRONOS_LOG_INTERVAL": "1",
    "KRONOS_BATCH_SIZE": "4",
    "KRONOS_NUM_WORKERS": "0",
    "KRONOS_USE_AMP": "1",
    "KRONOS_AMP_DTYPE": "bfloat16",
    "KRONOS_MAX_SEGMENTS_PER_RUN": "0",
    "KRONOS_RESUME_TRAINING": "0",
}


def load_train_predictor():
    import train_predictor
    return train_predictor


def build_models(seed=0):
    torch.manual_seed(seed)
    tokenizer = KronosTokenizer(**TOKENIZER_CONFIG).eval()
    model = Kronos(**PREDICTOR_CONFIG)
    # Kronos zero-initializes the condition paths (a no-op for pretrained
    # parents); give them weights so the bf16 condition MLP is exercised.
    with torch.no_grad():
        model.sector_emb.weight.normal_(0.0, 0.02)
        model.size_mlp[-1].weight.normal_(0.0, 0.02)
    return tokenizer, model


class SyntheticBetaV21Dataset(Dataset):
    """Items shaped exactly like QlibDataset's Beta v2.1 + size-percentile path."""

    def __init__(self, count, seed, data_type="train", with_periods=False):
        from dataset import build_beta_v21_labels

        rng = np.random.default_rng(seed)
        self.items = []
        for index in range(count):
            steps = rng.normal(0.0, 0.02, size=(WINDOW, 1))
            close = 10.0 * np.exp(np.cumsum(steps, axis=0))
            open_ = close * np.exp(rng.normal(0.0, 0.005, size=(WINDOW, 1)))
            high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.01, size=(WINDOW, 1))))
            low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.01, size=(WINDOW, 1))))
            volume = rng.lognormal(10.0, 0.5, size=(WINDOW, 1))
            raw = np.concatenate([open_, high, low, close, volume, volume * close], axis=1)
            mean = raw[:LOOKBACK].mean(axis=0)
            std = raw[:LOOKBACK].std(axis=0)
            x = np.clip((raw - mean) / (std + 1e-5), -5.0, 5.0).astype(np.float32)
            days = np.arange(WINDOW) + index
            stamp = np.stack([
                np.zeros(WINDOW), np.full(WINDOW, 15.0), days % 5,
                days % 28 + 1, (days // 28) % 12 + 1,
            ], axis=1).astype(np.float32)
            labels = build_beta_v21_labels(
                raw, LOOKBACK, np.datetime64("2025-01-01") + np.timedelta64(int(index // 2), "D")
            )
            labels["feature_means"] = torch.from_numpy(mean.astype(np.float32))
            labels["feature_stds"] = torch.from_numpy(std.astype(np.float32))
            item = [
                torch.from_numpy(x), torch.from_numpy(stamp),
                torch.tensor(int(rng.integers(0, 87)), dtype=torch.long),
                torch.tensor(0, dtype=torch.long),
                torch.tensor(
                    float("nan") if index % 7 == 3 else float(rng.uniform()),
                    dtype=torch.float32,
                ),
            ]
            if with_periods:
                item.append(torch.tensor(index % 2, dtype=torch.int16))
            item.append(labels)
            self.items.append(tuple(item))
        # Attributes train_model reads from QlibDataset.
        self.data_type = data_type
        self.total_samples = count
        self.n_samples = count
        self.quick_validation_count = count
        self.validation_period_names = (
            {0: "period_a", 1: "period_b"} if with_periods else {}
        )
        self.selection_report = {"synthetic": True, "samples": count}
        self.epoch_seeds = []

    def set_epoch_seed(self, seed):
        self.epoch_seeds.append(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def c1_config(monkeypatch, tmp_path, **overrides):
    for key, value in C1_ENV.items():
        monkeypatch.setenv(key, value)
    for key in ("XLA_USE_BF16", "XLA_DOWNCAST_BF16", "KRONOS_DEVICE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KRONOS_SAVE_PATH", str(tmp_path / "outputs"))
    monkeypatch.setenv("KRONOS_PREDICTOR_SAVE_FOLDER", "c1_smoke")
    from config import Config
    config = Config().__dict__
    config.update(overrides)
    return config


def bf16_cpu_amp(config, device):
    """resolve_amp_dtype returns None on CPU; the TPU path uses bf16."""
    return torch.bfloat16 if bool(config.get("use_amp", False)) else None


def assert_finite_metrics(metrics, keys):
    for key in keys:
        assert key in metrics, key
        assert math.isfinite(float(metrics[key])), (key, metrics[key])


VALIDATION_KEYS = (
    "objective_loss", "history_loss", "forecast_loss", "weighted_forecast_loss",
    "return_loss", "barrier_loss", "ranking_loss",
)


def _xla_policy_sdpa(original):
    """SDPA with torch_xla's autocast policy.

    CPU (and CUDA) autocast list scaled_dot_product_attention as a
    lower-precision op and silently cast q/k/v to one dtype, which hides the
    v6 bug on CPU.  torch_xla's autocast does not, so mixed q/k/v dtypes reach
    the kernel and raise.  Running the call with autocast disabled reproduces
    the XLA behaviour, including the exact RuntimeError message.
    """
    def sdpa(*args, **kwargs):
        with torch.autocast(device_type="cpu", enabled=False):
            return original(*args, **kwargs)
    return sdpa


if hasattr(pytest, "fixture"):
    @pytest.fixture(autouse=True)
    def xla_sdpa_autocast_policy(monkeypatch):
        import torch.nn.functional as F
        monkeypatch.setattr(
            F, "scaled_dot_product_attention",
            _xla_policy_sdpa(F.scaled_dot_product_attention),
        )


# ------------------------------------------------------------------ unit
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rope_returns_input_dtype_under_autocast(dtype):
    rope = RotaryPositionalEmbedding(16)
    q = torch.randn(2, 4, 130, 16).to(dtype)
    k = torch.randn(2, 4, 130, 16).to(dtype)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        rq, rk = rope(q, k)
    assert rq.dtype == dtype and rk.dtype == dtype
    # The angle table itself stays float32 (precision at position ~120).
    assert rope.cos_cached.dtype == torch.float32
    # Rotation equals the float32 rotation rounded once to the input dtype.
    cos, sin = rope.cos_cached, rope.sin_cached
    q32 = q.float()
    expected = (q32 * cos + rope._rotate_half(q32) * sin).to(dtype)
    assert torch.equal(rq, expected)


def test_rope_float32_path_is_unchanged():
    torch.manual_seed(1)
    rope = RotaryPositionalEmbedding(52)
    q = torch.randn(2, 16, 130, 52)
    k = torch.randn(2, 16, 130, 52)
    rq, rk = rope(q, k)
    t = torch.arange(130).type_as(rope.inv_freq)
    freqs = torch.einsum("i,j->ij", t, rope.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]
    assert torch.equal(rq, q * cos + rope._rotate_half(q) * sin)
    assert torch.equal(rk, k * cos + rope._rotate_half(k) * sin)


def test_real_attention_modules_keep_qkv_dtype_consistent_under_autocast():
    """Every SDPA call in the real models sees one dtype for q, k and v."""
    import torch.nn.functional as F

    tokenizer, model = build_models()
    seen = []
    original = F.scaled_dot_product_attention

    def checking_sdpa(q, k, v, *args, **kwargs):
        seen.append((q.dtype, k.dtype, v.dtype))
        return original(q, k, v, *args, **kwargs)

    x = SyntheticBetaV21Dataset(2, seed=5)[0][0].unsqueeze(0)
    ids = torch.randint(0, 1024, (1, WINDOW - 1))
    stamp = torch.zeros(1, WINDOW - 1, 5)
    F.scaled_dot_product_attention = checking_sdpa
    try:
        with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            tokenizer.encode(x, half=True)
            tokenizer.decode([ids, ids], half=True)
            model(ids, ids, stamp, use_teacher_forcing=True, s1_targets=ids)
    finally:
        F.scaled_dot_product_attention = original
    # tokenizer encoder 3 + decoder 3 + predictor 12 self-attn + 1 cross-attn
    assert len(seen) == 3 + 3 + 12 + 1
    assert all(q == k == v == torch.bfloat16 for q, k, v in seen), seen


# ------------------------------------------------------ training step
def test_real_kronos_training_step_under_cpu_bf16_autocast(monkeypatch, tmp_path):
    """Mirror of the train_model inner step (tokenize fp32, predictor bf16,
    fp32 losses outside autocast, backward) on the real architecture."""
    tp = load_train_predictor()
    from beta_v21 import DetachedEMANormalizer, compose_beta_v21_objective, compute_auxiliary_losses

    config = c1_config(monkeypatch, tmp_path)
    tokenizer, model = build_models()
    model.train()
    batch = next(iter(DataLoader(SyntheticBetaV21Dataset(4, seed=1), batch_size=4)))
    batch_x, batch_x_stamp, sector, _bucket, percentile, labels = batch
    with torch.no_grad():
        token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
    token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
    token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits, auxiliary = model(
            token_in[0], token_in[1], batch_x_stamp[:, :-1, :],
            sector_id=sector, size_bucket=None, size_percentile=percentile,
            return_auxiliary=True, asof_index=LOOKBACK - 1,
        )
    assert logits[0].dtype == torch.bfloat16  # autocast really was active
    logits = tp.to_float32(logits)
    auxiliary = tp.to_float32(auxiliary)
    with torch.autocast(device_type="cpu", enabled=False):
        losses = tp.compute_predictor_losses(model.head, logits, token_out, config)
        auxiliary_losses = compute_auxiliary_losses(
            auxiliary["return"], auxiliary["barrier"], labels
        )
        loss, _normalized = compose_beta_v21_objective(
            losses["weighted_forecast"], losses["history"], auxiliary_losses,
            DetachedEMANormalizer(decay=0.99), 1, warmup_steps=1000,
        )
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(g.dtype == torch.float32 for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    # RoPE/attention gradients reach q and k projections of every layer.
    for block in model.transformer:
        assert block.self_attn.q_proj.weight.grad.abs().sum() > 0
        assert block.self_attn.k_proj.weight.grad.abs().sum() > 0


# ---------------------------------------------------- validation paths
def test_denominator_calibration_evaluate_validation_under_cpu_bf16_autocast(monkeypatch, tmp_path):
    """The exact call that crashed the v6 probe (calibration: no consistency)."""
    tp = load_train_predictor()
    config = c1_config(monkeypatch, tmp_path)
    calibration_config = dict(config)
    calibration_config["beta_v21_consistency_samples"] = 0
    tokenizer, model = build_models()
    loader = DataLoader(SyntheticBetaV21Dataset(6, seed=2, data_type="val"), batch_size=4)
    metrics = tp.evaluate_validation(
        model, tokenizer, loader, torch.device("cpu"), calibration_config,
        torch.bfloat16, run_condition_ablation=False, period_names={}, rank=0,
    )
    assert_finite_metrics(metrics, VALIDATION_KEYS)


def test_full_validation_with_return_path_decode_under_cpu_bf16_autocast(monkeypatch, tmp_path):
    """Consistency decode (auto_regressive_inference + tokenizer.decode),
    per-period breakdown and condition ablation, all under bf16 autocast."""
    tp = load_train_predictor()
    config = c1_config(monkeypatch, tmp_path)
    config["beta_v21_validation_denominators"] = "1,1,1,1,1"
    tokenizer, model = build_models()
    dataset = SyntheticBetaV21Dataset(6, seed=3, data_type="val", with_periods=True)
    loader = DataLoader(dataset, batch_size=4)
    metrics = tp.evaluate_validation(
        model, tokenizer, loader, torch.device("cpu"), config, torch.bfloat16,
        run_condition_ablation=True, period_names=dataset.validation_period_names,
        rank=0,
    )
    assert_finite_metrics(metrics, VALIDATION_KEYS + (
        "condition_none_forecast_loss", "condition_shuffled_forecast_loss",
    ))
    consistency = metrics["return_path_consistency"]
    assert consistency["samples"] == 4
    assert all(math.isfinite(v) for v in consistency["sign_agreement"])


def test_consistency_decode_tokenizes_in_float32(monkeypatch, tmp_path):
    """auto_regressive_inference runs inside the predictor autocast; the
    tokenizer it receives must encode/decode outside autocast (fp32)."""
    tp = load_train_predictor()
    config = c1_config(monkeypatch, tmp_path)
    tokenizer, model = build_models()
    autocast_states = []
    original_encode, original_decode = tokenizer.encode, tokenizer.decode

    def encode(x, half=False):
        autocast_states.append(("encode", torch.is_autocast_enabled("cpu")))
        return original_encode(x, half=half)

    def decode(x, half=False):
        autocast_states.append(("decode", torch.is_autocast_enabled("cpu")))
        return original_decode(x, half=half)

    monkeypatch.setattr(tokenizer, "encode", encode)
    monkeypatch.setattr(tokenizer, "decode", decode)
    loader = DataLoader(SyntheticBetaV21Dataset(4, seed=4, data_type="val"), batch_size=4)
    tp.evaluate_validation(
        model, tokenizer, loader, torch.device("cpu"), config, torch.bfloat16,
        period_names={}, rank=0,
    )
    assert ("decode", False) in autocast_states
    assert all(state is False for _name, state in autocast_states), autocast_states


# --------------------------------------------------------- end to end
def test_train_predictor_main_end_to_end_under_cpu_bf16_autocast(monkeypatch, tmp_path, capsys):
    """main(): load real model classes from disk, auto-calibrate Beta v2.1
    denominators, train one segment, validate (with consistency decode)."""
    tp = load_train_predictor()
    tokenizer, model = build_models()
    tokenizer_dir = tmp_path / "tokenizer"
    predictor_dir = tmp_path / "predictor"
    tokenizer.save_pretrained(tokenizer_dir)
    (tokenizer_dir / "config.json").write_text(json.dumps(TOKENIZER_CONFIG))
    model.save_pretrained(predictor_dir)
    (predictor_dir / "config.json").write_text(json.dumps(PREDICTOR_CONFIG))

    config = c1_config(
        monkeypatch, tmp_path,
        pretrained_tokenizer_path=str(tokenizer_dir),
        finetuned_tokenizer_path=str(tmp_path / "missing_tokenizer"),
        pretrained_predictor_path=str(predictor_dir),
        use_comet=False,
    )
    monkeypatch.setattr(tp, "resolve_amp_dtype", bf16_cpu_amp)
    monkeypatch.setitem(tp._DATASET_CACHE, "train", SyntheticBetaV21Dataset(8, seed=10))
    monkeypatch.setitem(
        tp._DATASET_CACHE, "val", SyntheticBetaV21Dataset(6, seed=11, data_type="val")
    )
    sdpa_dtypes = set()
    import torch.nn.functional as F
    original = F.scaled_dot_product_attention

    def checking_sdpa(q, k, v, *args, **kwargs):
        sdpa_dtypes.add((q.dtype, k.dtype, v.dtype))
        return original(q, k, v, *args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", checking_sdpa)
    tp.main(config)
    out = capsys.readouterr().out

    assert "Predictor AMP: bfloat16 autocast" in out
    assert "Calibrating fixed Beta v2.1 validation denominators" in out
    # bf16 inside the predictor autocast, fp32 for the tokenizer (which runs
    # outside autocast); never mixed within one call.
    assert all(q == k == v for q, k, v in sdpa_dtypes), sdpa_dtypes
    assert (torch.bfloat16,) * 3 in sdpa_dtypes, sdpa_dtypes
    save_dir = tmp_path / "outputs" / "c1_smoke"
    denominators = json.loads(
        (save_dir / "beta_v21_validation_denominators.json").read_text()
    )
    for key in ("path", "history", "return", "barrier", "ranking"):
        assert math.isfinite(denominators[key]) and denominators[key] > 0, key
    metrics = [
        json.loads(line)
        for line in (save_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    train_losses = [m["loss"] for m in metrics if m.get("type") == "train"]
    assert train_losses and all(math.isfinite(v) for v in train_losses), metrics
    large = [m for m in metrics if m.get("type") == "validation_large"]
    assert large, metrics
    for key in ("loss", "forecast_loss", "return_loss", "barrier_loss",
                "ranking_loss", "beta_v21_score"):
        assert math.isfinite(float(large[-1][key])), (key, large[-1])
    # The return-path consistency decode ran (auto_regressive_inference).
    assert large[-1]["return_path_consistency"]["samples"] == 4, large[-1]


# ------------------------------------------------------- torch_xla CPU
def run_xla_cpu_checks():
    """Same model checks on an XLA CPU device with torch.autocast('xla')."""
    os.environ.setdefault("PJRT_DEVICE", "CPU")
    os.environ.pop("XLA_USE_BF16", None)
    import torch_xla
    import torch_xla.core.xla_model as xm

    import train_predictor as tp
    from beta_v21 import DetachedEMANormalizer, compose_beta_v21_objective, compute_auxiliary_losses
    from config import Config

    for key, value in C1_ENV.items():
        os.environ[key] = value
    config = Config().__dict__
    device = torch_xla.device()
    tokenizer, model = build_models()
    tokenizer.to(device)
    model.to(device)

    calibration_config = dict(config, beta_v21_consistency_samples=0)
    loader = DataLoader(SyntheticBetaV21Dataset(4, seed=2, data_type="val"), batch_size=4)
    metrics = tp.evaluate_validation(
        model, tokenizer, loader, device, calibration_config, torch.bfloat16,
        period_names={}, rank=0,
    )
    assert_finite_metrics(metrics, VALIDATION_KEYS)
    print("xla_calibration_evaluate_validation_ok",
          {k: round(float(metrics[k]), 6) for k in VALIDATION_KEYS}, flush=True)

    config["beta_v21_validation_denominators"] = "1,1,1,1,1"
    metrics = tp.evaluate_validation(
        model, tokenizer, loader, device, config, torch.bfloat16,
        run_condition_ablation=True, period_names={}, rank=0,
    )
    assert_finite_metrics(metrics, VALIDATION_KEYS)
    print("xla_consistency_evaluate_validation_ok samples=",
          metrics["return_path_consistency"]["samples"], flush=True)

    model.train()
    batch = next(iter(DataLoader(SyntheticBetaV21Dataset(4, seed=1), batch_size=4)))
    batch_x, batch_x_stamp, sector, _bucket, percentile, labels = [
        {k: v.to(device) for k, v in b.items()} if isinstance(b, dict) else b.to(device)
        for b in batch
    ]
    with torch.no_grad():
        s0, s1 = tokenizer.encode(batch_x, half=True)
    with torch.autocast(device_type="xla", dtype=torch.bfloat16):
        logits, auxiliary = model(
            s0[:, :-1], s1[:, :-1], batch_x_stamp[:, :-1, :],
            sector_id=sector, size_percentile=percentile,
            return_auxiliary=True, asof_index=LOOKBACK - 1,
        )
    logits = tp.to_float32(logits)
    auxiliary = tp.to_float32(auxiliary)
    with torch.autocast(device_type="xla", enabled=False):
        losses = tp.compute_predictor_losses(model.head, logits, [s0[:, 1:], s1[:, 1:]], config)
        aux_losses = compute_auxiliary_losses(auxiliary["return"], auxiliary["barrier"], labels)
        loss, _ = compose_beta_v21_objective(
            losses["weighted_forecast"], losses["history"], aux_losses,
            DetachedEMANormalizer(decay=0.99), 1, warmup_steps=1000,
        )
    loss.backward()
    xm.mark_step()
    value = float(loss.detach().cpu())
    assert math.isfinite(value), value
    grad_ok = all(
        bool(torch.isfinite(p.grad).all().cpu())
        for p in model.parameters() if p.grad is not None
    )
    assert grad_ok
    print(f"xla_train_step_ok loss={value:.6f} param_dtypes="
          f"{sorted({str(p.dtype) for p in model.parameters()})}", flush=True)
    print("XLA_CPU_ALL_OK", flush=True)


if __name__ == "__main__":
    if "--xla" in sys.argv:
        run_xla_cpu_checks()
    else:
        sys.exit(pytest.main([__file__, "-v"]))
