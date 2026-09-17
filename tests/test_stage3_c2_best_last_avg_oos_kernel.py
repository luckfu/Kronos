import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = ROOT / 'finetune/kaggle_stage3_c2_best_last_avg_oos/stage3_c2_best_last_avg_oos.py'
META = ROOT / 'finetune/kaggle_stage3_c2_best_last_avg_oos/kernel-metadata.json'


def test_c2_best_last_avg_oos_kernel_averages_c2_endpoints_without_training():
    source = KERNEL.read_text()
    for token in (
        '(0.5, "avg_alpha_050")',
        '(1.0, "avg_alpha_100_c2_last")',
        '(0.25, "avg_alpha_025")',
        '(0.75, "avg_alpha_075")',
        'checkpoints/best_model/model.safetensors',
        'checkpoints/last_model/model.safetensors',
        '"c2_best_segment_179"',
        '"c2_last_segment_267"',
        'C2_BEST_SAME_FOOTING_D10_POOLED_RANK_IC = 0.18440434213929857',
        '"training_performed": False',
        '"c2-alpha-oos-evaluation"',
        '["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT]',
        'SOURCE_COMMIT',
        'EXPECTED_C2_BEST_SHA = "4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a"',
        'if last_sha == best_sha:',
        'reloaded[key].dtype != tensor.float().dtype',
    ):
        assert token in source, token
    assert "'--branch', 'master'" not in source
    assert '"--branch", "master"' not in source
    # alpha=0.5 must be evaluated first so a timeout still delivers the decision checkpoint
    assert source.index('(0.5, "avg_alpha_050")') < source.index('(1.0, "avg_alpha_100_c2_last")')
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'luckfu/kronos-small-0-1-stage3-c2-best-last-avg-oos'
    assert metadata['code_file'] == 'stage3_c2_best_last_avg_oos.py'
    assert metadata['enable_gpu'] is True
    assert metadata['machine_shape'] == 'NvidiaTeslaP100'
    assert metadata['dataset_sources'] == ['luckfu/a-share-120d-temporal-symbol-holdout']
    assert metadata['kernel_sources'] == [
        'smmt315/kronos-small-0-1-stage2-cosine-refinement-c2',
        'smmt315/kronos-small-0-1-c2-alpha-oos-evaluation',
    ]
