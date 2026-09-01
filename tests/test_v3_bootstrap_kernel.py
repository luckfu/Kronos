import runpy
from pathlib import Path


ROOT = Path(__file__).parents[1]
RUNNER = ROOT / "finetune/kaggle_v1_beta_minimal/Kronos/finetune/kaggle_v3_bootstrap_runner.py"
BUILDER = ROOT / "finetune/build_kaggle_v3_bootstrap_kernel.py"


def test_v3_bootstrap_is_isolated_and_uses_two_thousand_windows():
    source = RUNNER.read_text()

    assert 'STAGE_ID = "beta_v3_base_dynamic_size_path_bootstrap_v1"' in source
    assert '"KRONOS_VALIDATION_QUICK_SAMPLES": "2000"' in source
    assert '"KRONOS_VALIDATION_LARGE_SAMPLES": "2000"' in source
    assert '"KRONOS_USE_SIZE_PATH": "1"' in source
    assert '"KRONOS_USE_SIZE_PERCENTILE": "0"' in source
    assert '"KRONOS_PREDICTOR_LEARNING_RATE": "1e-6"' in source
    assert '"KRONOS_CONDITION_LEARNING_RATE": "1e-5"' in source
    assert '"KRONOS_MAX_SEGMENTS_PER_RUN": "10"' in source
    assert 'NeoQuasar/Kronos-base' in source
    assert 'NeoQuasar/Kronos-Tokenizer-base' in source


def test_v3_kernel_uses_dedicated_data_and_no_v2_parent_kernel():
    metadata = runpy.run_path(str(BUILDER))["METADATA"]

    assert metadata["id"] == "luckfu/kronos-beta-v3-base-dynamic-size-bootstrap"
    assert metadata["kernel_sources"] == []
    assert "luckfu/kronos-beta-v3-symbol-holdout-90-10" in metadata["dataset_sources"]
