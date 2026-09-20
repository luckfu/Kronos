import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL_DIR = ROOT / 'finetune/kaggle_decode_sample_sweep'
KERNEL = KERNEL_DIR / 'decode_sample_sweep.py'
META = KERNEL_DIR / 'kernel-metadata.json'


def load_module():
    spec = importlib.util.spec_from_file_location('decode_sample_sweep', KERNEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sweep_decodes_without_training_and_pins_both_checkpoints():
    source = KERNEL.read_text()
    for token in (
        '"training_performed": False',
        'return_samples=True',
        '["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT]',
        'TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"',
        'small_0.1_stage2_cosine_refinement/checkpoints/best_model/model.safetensors',
        'small_0.1_stage2_c2_best_wc_1e5_c4/checkpoints/best_model/model.safetensors',
    ):
        assert token in source, token
    assert 'last_model/model.safetensors' not in source
    assert "'--branch', 'master'" not in source
    module = load_module()
    assert module.CHECKPOINTS['c2_best']['sha256'] == (
        '4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a'
    )
    assert module.CHECKPOINTS['wc_1e5_c4_best']['sha256'] == (
        '9c5605d08e4f63e223d7b5edf9c6c8d2acad88ddfe022e5b27c098588d743075'
    )
    assert module.CHECKPOINTS['c2_best']['baseline_d10_pooled_rank_ic'] > (
        module.CHECKPOINTS['wc_1e5_c4_best']['baseline_d10_pooled_rank_ic']
    )
    assert module.CHECKPOINTS['c2_best']['val_forecast_ce'] > (
        module.CHECKPOINTS['wc_1e5_c4_best']['val_forecast_ce']
    )


def test_arms_cover_the_attenuation_curve_and_a_noise_floor():
    module = load_module()
    names = [arm['name'] for arm in module.ARMS]
    for key in module.CHECKPOINTS:
        for suffix in ('n1_s1', 'n1_s2', 'n4', 'n16', 'n8_honest'):
            assert f'{key}_{suffix}' in names, f'{key}_{suffix}'
    seeds = {arm['name']: arm['seed'] for arm in module.ARMS}
    assert seeds['c2_best_n1_s1'] != seeds['c2_best_n1_s2']
    for arm in module.ARMS:
        if arm['name'].endswith('honest'):
            assert arm['temperature'] == 1.0 and arm['top_p'] == 1.0
        else:
            assert arm['temperature'] == 0.6 and arm['top_p'] == 0.9
    # A timeout must still leave the attenuation curve, so the noise-floor arms
    # run first and the volatility arms run last.
    assert all(arm['sample_count'] == 1 for arm in module.ARMS[:4])
    assert all(arm['name'].endswith('honest') for arm in module.ARMS[-2:])
    curve = [arm['sample_count'] for arm in module.ARMS[:8]]
    assert curve == sorted(curve)
    # Both checkpoints advance together, so a partial run still compares them.
    for left, right in zip(module.ARMS[::2], module.ARMS[1::2]):
        assert left['checkpoint'] != right['checkpoint']
        assert left['sample_count'] == right['sample_count']
    # 13 dates at ~55 s per sample-unit per GPU, split across two GPUs.
    budget = sum(arm['sample_count'] for arm in module.ARMS) * 13 * 55.0 / module.WORLD_SIZE
    assert budget <= module.HARD_LIMIT_SECONDS, budget


def test_daily_returns_and_attenuation_fit_are_correct():
    module = load_module()
    import numpy as np

    cumulative = np.array([0.01, 0.0201, 0.030301])
    assert np.allclose(module.to_daily(cumulative), [0.01, 0.01, 0.01])

    ic_infinity, ratio = 0.25, 2.0
    points = [(n, ic_infinity / math.sqrt(1.0 + ratio / n)) for n in (1, 4, 16)]
    fitted = module.attenuation_fit(points)
    assert math.isclose(fitted['ic_infinity'], ic_infinity, rel_tol=1e-6)
    assert math.isclose(fitted['noise_to_signal_variance_ratio'], ratio, rel_tol=1e-6)
    assert module.attenuation_fit([(1, 0.18)]) is None


def test_workers_split_by_task_without_changing_results():
    source = KERNEL.read_text()
    for token in (
        'index % WORLD_SIZE == rank',
        'torch.manual_seed(arm["seed"])',
        'batch_size = max(1, EFFECTIVE_BATCH // sample_count)',
        '"CUDA_VISIBLE_DEVICES": str(rank)',
        'os.replace(staging, shard)',
    ):
        assert token in source, token
    module = load_module()
    assert module.WORLD_SIZE == 2
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'luckfu/kronos-small-0-1-decode-sample-sweep'
    assert metadata['code_file'] == 'decode_sample_sweep.py'
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['kernel_sources'] == [
        'smmt315/kronos-small-0-1-stage2-cosine-refinement-c2',
        'user281434/kronos-small-0-1-stage2-c2-best-wc-1e5-c4',
    ]
    assert len(metadata['id'].split('/')[1]) <= 50
