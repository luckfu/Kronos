import json
import math
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'finetune'))

from train_predictor import segment_run_limit_reached


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


def test_chunk_limit_does_not_shorten_global_training_plan():
    total_windows = 5_233_538
    segments_per_pass = math.ceil(total_windows / 20_000)
    effective_segments = max(50, segments_per_pass * 2 + 5)

    assert segments_per_pass == 262
    assert effective_segments == 529
    assert segment_run_limit_reached(0, 180, 180)
    assert segment_run_limit_reached(180, 360, 180)
    assert not segment_run_limit_reached(360, 529, 180)

    # last_state.pt stores next_epoch=next_segment at each safe chunk boundary.
    start_segment = 180
    next_segment = 360
    assert next_segment - start_segment == 180


def test_long_kernel_restores_checkpoint_and_keeps_529_segment_plan():
    source = (ROOT / 'finetune/kaggle_v3_long.py').read_text()
    train_script = (ROOT / 'finetune/kaggle_v3_train.sh').read_text()

    assert "env['KRONOS_COVERAGE_PASSES'] = '2'" in source
    assert "env['KRONOS_EARLY_STOPPING_PATIENCE'] = '5'" in source
    assert "env['KRONOS_RESUME_TRAINING'] = '1'" in source
    assert "shutil.copytree(previous_run, target_run" in source
    assert 'KRONOS_MAX_SEGMENTS_PER_RUN' in train_script


def test_long_kernel_b_resumes_from_kernel_a_output():
    metadata = json.loads(
        (ROOT / 'finetune/kaggle_v3_long_b-metadata.json').read_text()
    )

    assert metadata['enable_gpu'] is True
    assert metadata['kernel_sources'] == [
        'luckfu/kronos-a-share-v3-p100-long-training-a'
    ]


def test_v4_ab_kernel_uses_full_data_and_preincremental_base():
    metadata = json.loads(
        (ROOT / 'finetune/kaggle_v4_ab-metadata.json').read_text()
    )
    source = (ROOT / 'finetune/kaggle_v4_ab.py').read_text()
    train_script = (ROOT / 'finetune/kaggle_v4_ab_train.sh').read_text()

    assert metadata['dataset_sources'] == [
        'luckfu/kronos-train-set-a',
        'luckfu/kronos-a-share-2026-incremental',
    ]
    assert metadata['id'] == (
        'luckfu/kronos-a-share-v4-corrected-context-a-b-training'
    )
    assert 'base_model/v3_last/model.safetensors' in source
    assert "for variant in ('recent_only', 'replay20')" in source
    assert 'KRONOS_TRAIN_SIGNAL_START="2026-01-01"' in train_script
    assert 'KRONOS_TRAIN_SIGNAL_END="2026-06-17"' in train_script
    assert 'KRONOS_VAL_SIGNAL_START="2026-06-18"' in train_script
    assert 'KRONOS_VAL_SIGNAL_END="2026-07-16"' in train_script
    assert 'KRONOS_HISTORY_REPLAY_RATIO="${REPLAY_RATIO}"' in train_script
    assert 'KRONOS_PREDICTOR_LEARNING_RATE="1e-6"' in train_script
