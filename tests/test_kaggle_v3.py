import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_kaggle_kernel_uses_existing_dataset_and_gpu():
    metadata = json.loads(
        (ROOT / 'finetune/kaggle_v3_kernel-metadata.example.json').read_text()
    )

    assert metadata['enable_gpu'] is True
    assert metadata['enable_internet'] is True
    assert metadata['dataset_sources'] == ['luckfu/kronos-train-set-a']


def test_kaggle_benchmark_is_three_segments_without_full_coverage():
    source = (ROOT / 'finetune/kaggle_v3_benchmark.py').read_text()

    assert "env['KRONOS_EPOCHS'] = '3'" in source
    assert "env['KRONOS_COVERAGE_PASSES'] = '1'" in source
    assert "env['KRONOS_REQUIRE_FULL_COVERAGE'] = '0'" in source
    assert '/kaggle/input/kronos-train-set-a' in source
