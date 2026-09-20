import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = (
    ROOT / 'finetune/kaggle_stage3_c4_best_1e5_ce_only_c2_milestone_oos'
    / 'stage3_c4_best_1e5_ce_only_c2_milestone_oos.py'
)
META = ROOT / 'finetune/kaggle_stage3_c4_best_1e5_ce_only_c2_milestone_oos' / 'kernel-metadata.json'


def test_milestone_oos_covers_20_to_70_not_80():
    source = KERNEL.read_text()
    for token in (
        '("stage3_1e5_ms70", 70, 21910, 1.8225171951045631)',
        '("stage3_1e5_ms60", 60, 18780, 1.8285348677173896)',
        '("stage3_1e5_ms50", 50, 15650, 1.8355361849802163)',
        '("stage3_1e5_ms40", 40, 12520, 1.84716399290514)',
        '("stage3_1e5_ms30", 30, 9390, 1.865410650591621)',
        '("stage3_1e5_ms20", 20, 6260, 1.8965256756761861)',
        'milestone_seg{segment:02d}/model.safetensors',
        'STAGE3_1E5_SEG80_D10 = 0.15031140792229383',
        'STAGE3_1E5_SEG15_D10 = 0.11323220411413162',
        'stage3_1e5_ce_only_seg80_0.1503',
        '"training_performed": False',
        '["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT]',
        'TOKENIZER_REPO = "NeoQuasar/Kronos-Tokenizer-base"',
    ):
        assert token in source, token
    assert 'ms80' not in source
    assert 'EXPECTED_SEGMENT = 80' not in source
    assert "'--branch', 'master'" not in source
    metadata = json.loads(META.read_text())
    assert metadata['id'] == 'user281434/kronos-small-0-1-stage3-c4-best-1e5-ce-only-c2-ms-oos'
    assert metadata['kernel_sources'] == [
        'user281434/kronos-small-0-1-stage3-c4-best-1e5-ce-only-c2',
    ]
    assert metadata['machine_shape'] == 'NvidiaTeslaP100'
