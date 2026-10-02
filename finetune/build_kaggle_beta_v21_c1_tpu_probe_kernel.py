"""Build the short C1 TPU *probe* staging directory.

Purpose: verify on real TPU v5e-8 that the fp32-master-weight / real 8-core
all-reduce fix works (startup self-checks, first-step grad sync check, a few
segments of validation) before spending a full 9 h session.

Like the base-eval builder this never rewrites the source runner or the
main/smoke staging dirs.  It embeds the current working-tree sources and writes
a copy of the runner with:

* ``DEFAULT_MAX_RUNTIME_SECONDS = "3600"`` (soft stop after the first complete
  segment past 1 h; the trainer then saves and exits cleanly),
* ``DEFAULT_OUTPUT_SUFFIX = "_probe"`` (own output tree),
* ``EXPERIMENT_NAME`` suffixed with ``_probe`` (own SwanLab run).

The kernel id is the SAME as the main kernel by default, so pushing the probe
creates a new *version* of ``user281434/kronos-beta-v2-1-c1-tpu`` and the later
formal push is simply the next version.  Set ``KRONOS_PROBE_KERNEL_ID`` to use a
separate kernel instead.
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path

from build_kaggle_beta_v21_c1_tpu_kernel import ROOT, RUNNER, build_bundle


STAGING = ROOT / "finetune/kaggle_beta_v21_c1_tpu_probe_kernel"
MAIN_METADATA = ROOT / "finetune/kaggle_beta_v21_c1_tpu_kernel/kernel-metadata.json"
SOFT_STOP_SECONDS = "3600"


def _sub_once(pattern: str, replacement: str, source: str) -> str:
    updated, count = re.subn(pattern, replacement, source, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"Probe runner pattern not found: {pattern!r}")
    return updated


def build_probe_runner(source: str, payload: str) -> str:
    source = _sub_once(
        r'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n.*?\n"""',
        f'EMBEDDED_KRONOS_ARCHIVE_B64 = """\n{payload}\n"""',
        source,
    )
    source = _sub_once(
        r'\nDEFAULT_MAX_RUNTIME_SECONDS = "\d+"\n',
        f'\nDEFAULT_MAX_RUNTIME_SECONDS = "{SOFT_STOP_SECONDS}"\n',
        source,
    )
    source = _sub_once(
        r'\nDEFAULT_OUTPUT_SUFFIX = "[^"]*"\n',
        '\nDEFAULT_OUTPUT_SUFFIX = "_probe"\n',
        source,
    )
    source, count = re.subn(
        r'\nEXPERIMENT_NAME = "([^"]*)"\n',
        lambda match: f'\nEXPERIMENT_NAME = "{match.group(1)}_probe"\n',
        source,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Probe runner EXPERIMENT_NAME pattern not found")
    return source


def build_metadata() -> dict:
    metadata = json.loads(MAIN_METADATA.read_text())
    override = os.getenv("KRONOS_PROBE_KERNEL_ID", "").strip()
    if override:
        metadata["id"] = override
        metadata["title"] = metadata["title"] + " Probe"
    return metadata


def main() -> None:
    payload = base64.b64encode(build_bundle()).decode("ascii")
    runner = build_probe_runner(RUNNER.read_text(), payload)
    STAGING.mkdir(parents=True, exist_ok=True)
    (STAGING / RUNNER.name).write_text(runner)
    (STAGING / "kernel-metadata.json").write_text(
        json.dumps(build_metadata(), indent=2) + "\n"
    )
    print(f"Wrote {STAGING}")


if __name__ == "__main__":
    main()
