"""G6: Merge G2_FT_VARLEN top-block adapted weights onto C2 and run 16-path AR OOS eval."""
from __future__ import annotations

import json
import os
import runpy
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    inp = Path("/kaggle/input")
    c2_matches = sorted(p for p in inp.rglob("model.safetensors") if "kronos-small-cosine-c2-best-weights" in p.as_posix())
    g2_matches = sorted(p for p in inp.rglob("g2_top_weights.pt"))
    if len(c2_matches) != 1 or len(g2_matches) != 1:
        raise RuntimeError(f"Expected 1 C2 & 1 G2 ckpt: c2={c2_matches}, g2={g2_matches}")

    c2_dir = c2_matches[0].parent
    sd = load_file(str(c2_matches[0]))
    g2 = torch.load(g2_matches[0], map_location="cpu", weights_only=False)["g2_ft_varlen"]["state"]
    replaced = 0
    for k, v in g2.items():
        if k.startswith("model."):
            mkey = k[6:]
            if mkey not in sd:
                raise KeyError(f"Unexpected key {mkey} not in C2 state_dict")
            sd[mkey] = v.to(dtype=sd[mkey].dtype)
            replaced += 1
    if replaced == 0:
        raise RuntimeError("No weights replaced from g2_ft_varlen")

    merged_dir = Path("/kaggle/working/g6_merged_ckpt")
    merged_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(c2_dir / "config.json", merged_dir / "config.json")
    save_file(sd, str(merged_dir / "model.safetensors"))
    print(json.dumps({"phase": "merged_g6_ckpt", "replaced_tensors": replaced, "dir": str(merged_dir)}), flush=True)

    # Also evaluate inside /kaggle/working via evaluate_base_sealed.py
    ev_path = Path(__file__).resolve().parents[2] / "finetune/oos_eval/base_clean_sealed_oos/evaluate_base_sealed.py"
    os.environ.update({
        "BASE_OOS_LABEL": "g2_ft_varlen_ar16",
        "BASE_OOS_EXPLICIT_WEIGHTS": str(merged_dir / "model.safetensors"),
        "BASE_OOS_CONTEXT_LAYER": "6",
        "BASE_OOS_SEED": "20261104",
        "BASE_OOS_PATHS": "16",
        "BASE_OOS_PATH_AGG": "median",
        "BASE_OOS_TEMPERATURE": "0.65",
        "BASE_OOS_TOP_P": "0.8",
        "BASE_OOS_STRIDE": os.environ.get("G6_STRIDE", "3"),
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
    patched_ev = Path("/kaggle/working/g6_eval_sealed.py")
    patched_ev.write_text(src)
    runpy.run_path(str(patched_ev), run_name="__main__")


if __name__ == "__main__":
    main()
