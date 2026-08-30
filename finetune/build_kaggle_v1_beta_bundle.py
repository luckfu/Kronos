"""Rebuild the minimal v1-beta source archive embedded in the Kaggle runner."""

import argparse
import base64
import io
import re
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "kaggle_v1_beta_minimal" / "Kronos"
ARCHIVE_PATH = ROOT / "kaggle_v1_beta_minimal.tar.gz"
RUNNER_PATH = ROOT / "kaggle_v1_beta.py"
VALIDATION_ROOT = ROOT.parent / "data/a_share_full_market_v1_beta"
EXTRA_FINETUNE_FILES = (
    ROOT / "evaluate_v1_beta_checkpoints.py",
    ROOT / "evaluate_last1058_natural_stage.py",
    ROOT / "calibrate_v1_beta_checkpoint.py",
)


def include_file(path):
    return "__pycache__" not in path.parts and path.suffix != ".pyc"


def build_archive():
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(SOURCE_ROOT.rglob("*")):
            if path.is_file() and include_file(path):
                archive.add(path, arcname=Path("Kronos") / path.relative_to(SOURCE_ROOT))
        for directory, names in (
            ("balanced_validation_v1", (
                "balanced_validation_manifest.json",
                "balanced_validation_samples.jsonl",
            )),
            ("natural_validation_v1", (
                "natural_validation_manifest.json",
                "natural_validation_samples.jsonl",
            )),
        ):
            for name in names:
                path = VALIDATION_ROOT / directory / name
                if not path.is_file():
                    raise FileNotFoundError(f"Missing validation artifact: {path}")
                archive.add(path, arcname=Path("Kronos/finetune/manifests") / name)
        for path in EXTRA_FINETUNE_FILES:
            if not path.is_file():
                raise FileNotFoundError(f"Missing bundled evaluator: {path}")
            archive.add(path, arcname=Path("Kronos/finetune") / path.name)
    ARCHIVE_PATH.write_bytes(buffer.getvalue())
    return buffer.getvalue()


def embed_archive(archive_bytes, runner_path):
    encoded = base64.b64encode(archive_bytes).decode("ascii")
    source = runner_path.read_text()
    updated, replacements = re.subn(
        r'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n.*?\n"""',
        f'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n{encoded}\n"""',
        source,
        count=1,
        flags=re.DOTALL,
    )
    if replacements != 1:
        raise RuntimeError("Expected exactly one embedded archive block")
    runner_path.write_text(updated)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, default=RUNNER_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner_path = args.runner.resolve()
    archive_bytes = build_archive()
    embed_archive(archive_bytes, runner_path)
    print(
        f"Embedded {len(archive_bytes):,} bytes from {SOURCE_ROOT} into {runner_path}"
    )
