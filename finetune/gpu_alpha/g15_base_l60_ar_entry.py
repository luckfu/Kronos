"""G15: Zero-shot NeoQuasar/Kronos-base L=60 16-path AR OOS evaluation."""
from __future__ import annotations

import os
import runpy
from pathlib import Path


def patch_evaluate_source_for_l60(src: str) -> str:
    """Safely rewrite 120d lookback constants/slices in evaluate_base_sealed.py to 60d."""
    replacements = [
        ("LOOKBACK, PREDICT, WINDOW = 120, 10, 130", "LOOKBACK, PREDICT, WINDOW = 60, 10, 70"),
        ("(str(dates[119].date()), str(dates[129].date()))", "(str(dates[59].date()), str(dates[69].date()))"),
        ("values[:120]", "values[:60]"),
        ("raw_close[120:130] / raw_close[119]", "raw_close[60:70] / raw_close[59]"),
        ("dates[120:130]", "dates[60:70]"),
        ("window.sector.iloc[119]", "window.sector.iloc[59]"),
        ("window.size_percentile.iloc[119]", "window.size_percentile.iloc[59]"),
        ("raw_close[119]", "raw_close[59]"),
        ("stamp[:, :120], stamp[:, 120:130]", "stamp[:, :60], stamp[:, 60:70]"),
    ]
    out = src
    for old, new in replacements:
        if old not in out:
            raise RuntimeError(f"Expected pattern missing when patching L60: {old!r}")
        out = out.replace(old, new)
    return out


def main() -> None:
    ev_path = Path(__file__).resolve().parents[2] / "finetune/oos_eval/base_clean_sealed_oos/evaluate_base_sealed.py"
    os.environ.update({
        "BASE_OOS_LABEL": "kronos_base_l60_ar16",
        "BASE_OOS_HF_REPO": "NeoQuasar/Kronos-base",
        "BASE_OOS_HF_TOKENIZER": "NeoQuasar/Kronos-Tokenizer-base",
        "BASE_OOS_CONTEXT_LAYER": "10",
        "BASE_OOS_SEED": "20261104",
        "BASE_OOS_PATHS": "16",
        "BASE_OOS_PATH_AGG": "median",
        "BASE_OOS_TEMPERATURE": "0.65",
        "BASE_OOS_TOP_P": "0.8",
        "BASE_OOS_BATCH": "16",
        "BASE_OOS_STRIDE": os.environ.get("G15_STRIDE", "2"),
        "BASE_OOS_EXPECT_SAMPLES": "335312",
        "BASE_OOS_EXPECT_DATES": "65",
        "BASE_OOS_EXPECT_FIRST": "2026-07-17",
        "BASE_OOS_EXPECT_LAST": "2026-10-23",
        "BASE_OOS_EXPECT_TARGET_END": "2026-11-06",
    })
    patched_ev = Path("/kaggle/working/g15_eval_l60.py")
    patched_ev.write_text(patch_evaluate_source_for_l60(ev_path.read_text()))
    runpy.run_path(str(patched_ev), run_name="__main__")


if __name__ == "__main__":
    main()
