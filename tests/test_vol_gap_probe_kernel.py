import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
KERNEL = ROOT / "finetune/kaggle_vol_gap_probe/vol_gap_probe.py"
META = ROOT / "finetune/kaggle_vol_gap_probe/kernel-metadata.json"


def test_vol_gap_kernel_is_eval_only_and_pinned():
    source = KERNEL.read_text()
    for token in (
        'SOURCE_COMMIT = "3b3c1c5e5357054232e896bb83032c8af2daf1ec"',
        "PARENT_BEST_SHA = \"4ee469d49522f2a155f63bbbac6ef520df47244b06a00df963123b8007b73b5a\"",
        '"training": False',
        "training_performed=False",
        "last_signal_days",
    ):
        assert token in source, token
    assert "train_stage3" not in source
    metadata = json.loads(META.read_text())
    assert metadata["id"] == "luckfu/kronos-small-0-1-vol-gap-probe"
    assert len(metadata["id"].split("/")[1]) <= 50
    assert metadata["enable_gpu"] is True
    assert metadata["dataset_sources"] == ["luckfu/a-share-120d-temporal-symbol-holdout"]
    assert metadata["kernel_sources"] == ["smmt315/kronos-small-0-1-stage2-cosine-refinement-c2"]
