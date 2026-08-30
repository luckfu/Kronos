import ast
import base64
import io
import json
import tarfile
from pathlib import Path

import pytest


RUNNER = (
    Path(__file__).parents[1]
    / "finetune"
    / "kaggle_v1_beta_best109_friend.py"
)


def load_runner_function(name):
    tree = ast.parse(RUNNER.read_text())
    node = next(node for node in tree.body if getattr(node, "name", None) == name)
    namespace = {"json": json}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(RUNNER), "exec"), namespace)
    return namespace[name]


def embedded_sources():
    tree = ast.parse(RUNNER.read_text())
    archive_text = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id == "EMBEDDED_KRONOS_ARCHIVE_B64"
    )
    with tarfile.open(
        fileobj=io.BytesIO(base64.b64decode(archive_text)), mode="r:gz"
    ) as bundle:
        return {
            name: bundle.extractfile(name).read().decode()
            for name in (
                "Kronos/finetune/config.py",
                "Kronos/finetune/train_predictor.py",
            )
        }


def test_historical_metrics_are_strictly_cut_at_best_109(tmp_path):
    copy_metrics = load_runner_function("copy_metrics_through_segment")
    source = tmp_path / "source.jsonl"
    destination = tmp_path / "destination.jsonl"
    rows = [
        {"type": "train", "segment": 108, "step": 100},
        {"type": "validation", "segment": 109, "loss": 2.435},
        {"type": "train", "segment": 110, "step": 100},
        {"type": "validation", "segment": 120, "loss": 2.5},
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    retained, discarded = copy_metrics(source, destination, 109)

    assert (retained, discarded) == (2, 2)
    copied = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [row["segment"] for row in copied] == [108, 109]


def test_invalid_historical_metrics_abort_the_branch(tmp_path):
    copy_metrics = load_runner_function("copy_metrics_through_segment")
    source = tmp_path / "source.jsonl"
    source.write_text('{"segment": 109}\nnot-json\n')

    with pytest.raises(SystemExit, match="Invalid historical metric JSON"):
        copy_metrics(source, tmp_path / "destination.jsonl", 109)


def test_runner_separates_best109_bootstrap_from_official_resume():
    source = RUNNER.read_text()

    assert "BRANCH_ORIGIN_SEGMENT = 109" in source
    assert "BRANCH_ORIGIN_OBJECTIVE = 2.4350784543960815" in source
    assert (
        'BRANCH_ORIGIN_MODEL_SHA256 = '
        '"134e33d48dcd7dd8a4b59ea1c90d94ad579ddefb151d1364b672fc76bbc27dc0"'
    ) in source
    assert 'if source_branch_marker is None:' in source
    assert 'elif str(source_branch_marker) == str(BRANCH_ORIGIN_SEGMENT):' in source
    assert 'resume_state = source_state' in source
    assert '"KRONOS_RESUME_TRAINING": "1" if resume_state is not None else "0"' in source
    assert '"KRONOS_BOOTSTRAP_COMPLETED_SEGMENTS": str(BRANCH_ORIGIN_SEGMENT)' in source
    assert '"KRONOS_BRANCH_ORIGIN_SEGMENT": str(BRANCH_ORIGIN_SEGMENT)' in source
    assert '"KRONOS_MAX_SEGMENTS_PER_RUN": "250"' in source
    assert "kronos-v1-beta-best109-official-120d-to-10d" in source


def test_embedded_trainer_supports_exact_best109_scheduler_bootstrap():
    sources = embedded_sources()
    trainer = sources["Kronos/finetune/train_predictor.py"]
    config = sources["Kronos/finetune/config.py"]

    assert "optimizer_steps_for_completed_segments" in trainer
    assert "elif bootstrap_completed_segments:" in trainer
    assert "scheduler.last_epoch = batch_idx_global" in trainer
    assert "schedule(batch_idx_global)" in trainer
    assert "Bootstrap Best segment does not match" in trainer
    assert "persist_resume_checkpoint(start_epoch)" in trainer
    assert "KRONOS_BOOTSTRAP_COMPLETED_SEGMENTS" in config
    assert "KRONOS_BOOTSTRAP_BEST_VAL_LOSS" in config
