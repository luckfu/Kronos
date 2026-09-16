import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = ROOT / 'finetune/kaggle_stage3_freeze_probe_be_oos/stage3_freeze_probe_be_oos.py'
META = ROOT / 'finetune/kaggle_stage3_freeze_probe_be_oos/kernel-metadata.json'


def test_freeze_probe_be_oos_evaluates_b_and_e_milestones_vs_c2():
    source = KERNEL.read_text()
    for token in (
        "b_seg05_last",
        "e_seg05_last",
        "b_seg00_init",
        "b_seg03",
        "b_seg01",
        "e_seg03",
        "e_seg01",
        "freeze-probe-b",
        "freeze-probe-e",
        "c2-alpha-oos-evaluation",
            '["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT]',
        "SOURCE_COMMIT",
    ):
        assert token in source, token
    assert "'--branch', 'master'" not in source
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'luckfu/kronos-small-0-1-stage3-freeze-probe-be-oos'
    assert metadata['code_file'] == 'stage3_freeze_probe_be_oos.py'
    assert metadata['enable_gpu'] is True
    assert metadata['machine_shape'] == 'NvidiaTeslaP100'
    assert metadata['dataset_sources'] == ['luckfu/a-share-120d-temporal-symbol-holdout']
    assert metadata['kernel_sources'] == [
        'luckfu/kronos-small-0-1-stage3-freeze-probe-b',
        'luckfu/kronos-small-0-1-stage3-freeze-probe-e',
        'smmt315/kronos-small-0-1-stage2-cosine-refinement-c2',
        'smmt315/kronos-small-0-1-c2-alpha-oos-evaluation',
    ]
