"""G13: Run 16-path AR OOS eval on merged G2_FT_L120 checkpoint."""
from __future__ import annotations

import os
import runpy
from pathlib import Path


def main() -> None:
    inp = Path("/kaggle/input")
    matches = sorted(p for p in inp.rglob("model.safetensors") if "ft-l120" in p.as_posix() or "c2-ft-l120" in p.as_posix())
    if len(matches) != 1:
        raise RuntimeError(f"Expected 1 merged G2_FT_L120 model.safetensors, got {matches}")
    ev_path = Path(__file__).resolve().parents[2] / "finetune/oos_eval/base_clean_sealed_oos/evaluate_base_sealed.py"
    os.environ.update({
        "BASE_OOS_LABEL": "g2_ft_l120_ar16",
        "BASE_OOS_EXPLICIT_WEIGHTS": str(matches[0]),
        "BASE_OOS_CONTEXT_LAYER": "6",
        "BASE_OOS_SEED": "20261104",
        "BASE_OOS_PATHS": "16",
        "BASE_OOS_PATH_AGG": "median",
        "BASE_OOS_TEMPERATURE": "0.65",
        "BASE_OOS_TOP_P": "0.8",
        "BASE_OOS_STRIDE": os.environ.get("G13_STRIDE", "2"),
        "BASE_OOS_HF_TOKENIZER": "NeoQuasar/Kronos-Tokenizer-base",
        "BASE_OOS_EXPECT_SAMPLES": "335312",
        "BASE_OOS_EXPECT_DATES": "65",
        "BASE_OOS_EXPECT_FIRST": "2026-07-17",
        "BASE_OOS_EXPECT_LAST": "2026-10-23",
        "BASE_OOS_EXPECT_TARGET_END": "2026-11-06",
    })
    src = ev_path.read_text().replace(
        "matches = sorted(root.glob(WEIGHTS_GLOB))",
        'matches = [Path(os.environ["BASE_OOS_EXPLICIT_WEIGHTS"])] if os.environ.get("BASE_OOS_EXPLICIT_WEIGHTS") else sorted(root.glob(WEIGHTS_GLOB))',
    )
    patched_ev = Path("/kaggle/working/g13_eval_sealed.py")
    patched_ev.write_text(src)
    runpy.run_path(str(patched_ev), run_name="__main__")


if __name__ == "__main__":
    main()
