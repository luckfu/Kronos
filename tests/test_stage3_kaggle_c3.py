import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SMOKE = ROOT / "finetune/kaggle_stage3_joint_path_smoke/stage3_joint_path_smoke.py"
C3 = ROOT / "finetune/kaggle_stage3_joint_path_c3/stage3_joint_path_c3.py"
META = ROOT / "finetune/kaggle_stage3_joint_path_c3/kernel-metadata.json"


def test_smoke_continuation_reads_env_contract_not_c1_literals():
    source = SMOKE.read_text()
    required = (
        "STAGE3_EXPECTED_PROGRESS",
        "STAGE3_MAX_RUNTIME_SECONDS",
        "STAGE3_HARD_TIMEOUT_SECONDS",
        "STAGE3_BASELINE_BEFORE_RESUME",
        "STAGE3_RESUME_STATE_SHA256",
        "int(progress['next_epoch']) + 1",
        "if BASELINE_BEFORE_RESUME:",
        "str(MAX_RUNTIME_SECONDS)",
        "completed > int(expected_progress['completed_segments'])",
    )
    for declaration in required:
        assert declaration in source
    assert "{'completed_segments': 1, 'next_epoch': 1, 'step': 313, 'status': 'completed'}" not in source
    assert "'--baseline-before-resume'] if RESUME_KERNEL else []" not in source
    assert "'--max-runtime-seconds', '10800'" not in source


def test_c3_kernel_resumes_c2_for_nine_segments_without_baselines():
    source = C3.read_text()
    required = (
        "STAGE3_RESUME_KERNEL='smmt315/kronos-small-0-1-stage3-joint-path-c2'",
        "STAGE3_TARGET_SEGMENTS='15'",
        "STAGE3_CHUNK='c3'",
        "STAGE3_MAX_RUNTIME_SECONDS='7200'",
        "STAGE3_HARD_TIMEOUT_SECONDS='9000'",
        "STAGE3_BASELINE_BEFORE_RESUME='0'",
        "'completed_segments': 6",
        "'next_epoch': 6",
        "'step': 1878",
        "next_segment=7",
        "last_segment=15",
        "resume_step=1878",
        "new_segments=9",
        "os.environ.pop('STAGE3_RESUME_STATE_SHA256', None)",
    )
    for declaration in required:
        assert declaration in source
    metadata = json.loads(META.read_text())
    assert metadata["id"] == "smmt315/kronos-small-0-1-stage3-joint-path-c3"
    assert metadata["code_file"] == "stage3_joint_path_c3.py"
    assert metadata["enable_gpu"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == ["luckfu/a-share-120d-temporal-symbol-holdout"]
    assert metadata["kernel_sources"] == [
        "smmt315/kronos-small-0-1-stage2-cosine-refinement-c2",
        "smmt315/kronos-small-0-1-stage3-joint-path-c2",
    ]
    assert "kronos-small-0-1-stage3-joint-path-smoke" not in metadata["kernel_sources"]
