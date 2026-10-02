"""Build the separate C1 base-model eval-only Kaggle staging directory.

Unlike build_kaggle_beta_v21_c1_tpu_kernel.py this never rewrites the source
runner or the main/smoke staging dirs.  It embeds the current working-tree
sources and writes an eval-only copy of the runner (KRONOS_EVAL_ONLY default
"1", 2400 s soft stop) plus its own kernel-metadata.json.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from build_kaggle_beta_v21_c1_tpu_kernel import ROOT, RUNNER, build_bundle


STAGING = ROOT / "finetune/kaggle_beta_v21_c1_eval_kernel"
MAIN_METADATA = ROOT / "finetune/kaggle_beta_v21_c1_tpu_kernel/kernel-metadata.json"
KERNEL_ID = "user281434/kronos-beta-v2-1-c1-base-eval"
KERNEL_TITLE = "Kronos Beta V2 1 C1 Base Eval"
SOFT_STOP_SECONDS = "2400"


def _sub_once(pattern: str, replacement: str, source: str) -> str:
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"Eval runner pattern not found: {pattern!r}")
    return updated


def build_eval_runner(source: str, payload: str) -> str:
    source = _sub_once(
        r'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n.*?\n"""',
        f'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n{payload}\n"""',
        source,
    )
    source = _sub_once(r'\nDEFAULT_EVAL_ONLY = "0"\n', '\nDEFAULT_EVAL_ONLY = "1"\n', source)
    source = _sub_once(
        r'\nDEFAULT_MAX_RUNTIME_SECONDS = "\d+"\n',
        f'\nDEFAULT_MAX_RUNTIME_SECONDS = "{SOFT_STOP_SECONDS}"\n',
        source,
    )
    source = _sub_once(
        r'\nEXPERIMENT_NAME = "[^"]*"\n',
        '\nEXPERIMENT_NAME = "beta_v2_1_c1_tpu_base_eval"\n',
        source,
    )
    return source


def build_metadata() -> dict:
    metadata = json.loads(MAIN_METADATA.read_text())
    metadata["id"] = KERNEL_ID
    metadata["title"] = KERNEL_TITLE
    metadata["keywords"] = sorted(set(metadata.get("keywords", [])) | {"tpu"})
    return metadata


def main() -> None:
    payload = base64.b64encode(build_bundle()).decode("ascii")
    runner = build_eval_runner(RUNNER.read_text(), payload)
    STAGING.mkdir(parents=True, exist_ok=True)
    (STAGING / RUNNER.name).write_text(runner)
    (STAGING / "kernel-metadata.json").write_text(
        json.dumps(build_metadata(), indent=2) + "\n"
    )
    print(f"Wrote {STAGING}")


if __name__ == "__main__":
    main()
