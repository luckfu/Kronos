import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL_DIR = ROOT / 'finetune/kaggle_c2_decode_temp_grid'
KERNEL = KERNEL_DIR / 'decode_temp_grid.py'
META = KERNEL_DIR / 'kernel-metadata.json'


def load_module():
    spec = importlib.util.spec_from_file_location('decode_temp_grid', KERNEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grid_is_c2_only_and_does_not_train():
    source = KERNEL.read_text()
    for token in (
        '"training_performed": False',
        'return_samples=True',
        '["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT]',
        'small_0.1_stage2_cosine_refinement/checkpoints/best_model/model.safetensors',
        'JOINT_IC_SLACK = 0.010',
    ):
        assert token in source, token
    assert 'wc_1e5_c4' not in source
    assert 'last_model/model.safetensors' not in source
    module = load_module()
    assert list(module.CHECKPOINTS) == ['c2_best']
    assert all(arm['checkpoint'] == 'c2_best' for arm in module.ARMS)
    assert module.CHECKPOINTS['c2_best']['sha256'] == (
        '4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a'
    )


def test_arms_cover_temperature_and_top_p_for_joint_ic_and_vol():
    module = load_module()
    names = [arm['name'] for arm in module.ARMS]
    assert names[0] == 'prod_t065_p80_n5'
    assert module.ARMS[0]['sample_count'] == 5
    temps = {0.50, 0.60, 0.65, 0.70, 0.80, 1.00}
    for temperature in temps:
        for top_p in (0.8, 0.9):
            tag = f"t{int(round(temperature * 100)):03d}_p{int(round(top_p * 100)):02d}_n8"
            assert tag in names, tag
    assert 't080_p100_n8' in names
    assert 't100_p100_n8' in names
    n8 = [arm for arm in module.ARMS if arm['sample_count'] == 8]
    assert len(n8) == 14
    budget = sum(arm['sample_count'] for arm in module.ARMS) * 13 * 50.0 / module.WORLD_SIZE
    assert budget <= module.HARD_LIMIT_SECONDS, budget


def test_joint_rule_keeps_ic_then_picks_calibration():
    module = load_module()
    assert module.JOINT_IC_SLACK == 0.010
    source = KERNEL.read_text()
    for token in (
        'row["d10_pooled_rank_ic"] >= ceiling - JOINT_IC_SLACK',
        'row["abs_log_calibration"]',
        '"joint_recommendation"',
        'in_joint_set": arm["sample_count"] == 8',
    ):
        assert token in source, token


def test_dual_t4_metadata():
    source = KERNEL.read_text()
    for token in (
        'index % WORLD_SIZE == rank',
        '"CUDA_VISIBLE_DEVICES": str(rank)',
        'batch_size = max(1, EFFECTIVE_BATCH // sample_count)',
    ):
        assert token in source, token
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'luckfu/kronos-small-0-1-c2-decode-temp-grid'
    assert metadata['code_file'] == 'decode_temp_grid.py'
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['kernel_sources'] == [
        'smmt315/kronos-small-0-1-stage2-cosine-refinement-c2',
    ]
    assert len(metadata['id'].split('/')[1]) <= 50
