"""Push the SweetSpot new-window evaluation kernel (REST; no kaggle CLI).

    KAGGLE_API_TOKEN=... KAGGLE_USERNAME=smmt315 \\
    python3 finetune/timesfm3_v11_trajectory/eval_oos/push_eval_modelscope_window.py --push

The weights are mounted from ``<owner>/tfm3-v11-sweetspot-native-weights`` (private, a copy of
the private ModelScope repo), so the kernel carries no credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "base_clean_bootstrap_relay"))

import relay_auth  # noqa: E402

SLUG = "tfm3-sweetspot-new-window-eval"
TITLE = "TFM3 SweetSpot new window eval"          # Kaggle derives the slug from the title
WEIGHTS_SLUG = "tfm3-v11-sweetspot-native-weights"
DATASETS = ["luckfu/a-share-120d-oos-20260717-20261023-v2", "smmt315/a-share-120d-temporal-symbol-holdout"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    owner = os.environ.get("KAGGLE_USERNAME", "").strip()
    if not owner:
        raise SystemExit("KAGGLE_USERNAME is required")
    datasets = [*DATASETS, f"{owner}/{WEIGHTS_SLUG}"]
    sources = {name: (HERE / name).read_text() for name in ("eval_native_merged.py", "eval_modelscope_window.py")}
    code = f'''import os, runpy, sys
from pathlib import Path
os.environ["PYTHONUNBUFFERED"] = "1"
src = Path("/kaggle/working/tfm3_src")
src.mkdir(parents=True, exist_ok=True)
for name, text in {sources!r}.items():
    (src / name).write_text(text)
sys.path.insert(0, str(src))
try:
    runpy.run_path(str(src / "eval_modelscope_window.py"), run_name="__main__")
finally:
    import shutil
    shutil.rmtree(src, ignore_errors=True)
'''
    compile(code, SLUG, "exec")
    print(json.dumps({"id": f"{owner}/{SLUG}", "datasets": datasets, "code_chars": len(code)}, indent=2))
    if not args.push:
        print(json.dumps({"status": "READY_NOT_PUSHED"}))
        return 0
    response = relay_auth.session().post(
        "https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel",
        json={"slug": f"{owner}/{SLUG}", "newTitle": TITLE, "text": code, "language": "python",
              "kernelType": "script", "isPrivate": True, "enableGpu": True, "enableInternet": True,
              "machineShape": "NvidiaTeslaT4", "datasetDataSources": datasets, "kernelDataSources": []},
        timeout=180)
    response.raise_for_status()
    print(response.text[:300])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
