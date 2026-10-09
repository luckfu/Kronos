"""Kaggle entrypoint for the TimesFM-3 close-only zero-shot benchmark."""

from __future__ import annotations

import subprocess
import sys
import shutil
from pathlib import Path


def main() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "git+https://github.com/google-research/timesfm.git",
        ]
    )
    root = Path("/kaggle/working/timesfm3_eval")
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        Path(__file__).with_name("timesfm3_zero_shot_eval.py"),
        root / "timesfm3_zero_shot_eval.py",
    )
    panel = next(
        Path("/kaggle/input").glob(
            "**/evaluation_oos_20260914/evaluation_panel.pkl"
        ),
        None,
    )
    if panel is None:
        panel = next(Path("/kaggle/input").glob("**/evaluation_panel.pkl"))
    subprocess.check_call(
        [
            sys.executable,
            str(root / "timesfm3_zero_shot_eval.py"),
            "--panel",
            str(panel),
            "--output",
            "/kaggle/working/timesfm3_zero_shot",
            "--symbols",
            "256",
            "--signal-start",
            "2026-07-17",
            "--signal-end",
            "2026-08-10",
            "--context",
            "120",
            "--horizon",
            "10",
            "--batch-size",
            "4",
        ]
    )


if __name__ == "__main__":
    main()
