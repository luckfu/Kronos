"""Build the self-contained Kaggle v1-beta checkpoint evaluation kernel."""

import json
import py_compile
import shutil
from pathlib import Path

from build_kaggle_v1_beta_bundle import build_archive, embed_archive


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "kaggle_v1_beta_checkpoint_evaluation.py"
STAGING = ROOT / "kaggle_v1_beta_checkpoint_evaluation_kernel"
METADATA = {
    "id": "wynstonliu/kronos-v1-beta-best467-vs-last1058-evaluation",
    "title": "Kronos v1-beta Best467 vs Last1058 Evaluation",
    "code_file": RUNNER.name,
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "dataset_sources": ["wynstonliu/kronos-v1-beta-evaluation-20260826"],
    "competition_sources": [],
    "kernel_sources": ["wynstonliu/kronos-v1-beta-best109-official-chunk-5"],
}


def main():
    archive = build_archive()
    embed_archive(archive, RUNNER)
    py_compile.compile(str(RUNNER), doraise=True)
    STAGING.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUNNER, STAGING / RUNNER.name)
    (STAGING / "kernel-metadata.json").write_text(
        json.dumps(METADATA, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "archive_bytes": len(archive),
                "runner": str(RUNNER),
                "staging": str(STAGING),
                "kernel_id": METADATA["id"],
                "dataset_sources": METADATA["dataset_sources"],
                "kernel_sources": METADATA["kernel_sources"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
