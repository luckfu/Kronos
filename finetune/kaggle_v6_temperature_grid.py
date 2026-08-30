"""Kaggle GPU runner for V6 inference-parameter calibration."""
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path("/kaggle/working/Kronos")
RUNTIME = Path("/kaggle/working/kronos_v6_temperature_grid")
INPUT = Path("/kaggle/input")
SCRIPT_DIR = Path(__file__).resolve().parent


def find_one(suffix):
    matches = sorted(INPUT.rglob(suffix))
    if not matches:
        raise RuntimeError(f"No {suffix} found below {INPUT}")
    return matches[0]


if not (REPO / ".git").exists():
    subprocess.run(
        ["git", "clone", "https://github.com/luckfu/Kronos.git", str(REPO)],
        check=True,
    )

for name in ("evaluate_unseen_a_share.py", "evaluate_temperature_grid.py"):
    shutil.copy2(SCRIPT_DIR / name, REPO / "finetune" / name)

v6_mount = INPUT / "kronos-v6-segment568-last-model"
model_files = sorted(v6_mount.rglob("model.safetensors"))
if len(model_files) != 1:
    raise RuntimeError(f"Expected one V6 model in {v6_mount}, found {model_files}")
model_path = model_files[0].parent
holdout = find_one("symbol_holdout_data.pkl")
manifest = find_one("universe_manifest.csv")
print(f"MODEL={model_path}", flush=True)
print(f"HOLDOUT={holdout}", flush=True)
print(f"MANIFEST={manifest}", flush=True)

sys.path.insert(0, str(REPO))
commands = []
for top_p in (0.8, 0.9, 1.0):
    output = RUNTIME / f"top_p_{top_p:g}"
    commands.append([
        sys.executable,
        "-m",
        "finetune.evaluate_temperature_grid",
        "--model", str(model_path),
        "--holdout", str(holdout),
        "--manifest", str(manifest),
        "--output-dir", str(output),
        "--signal-start", "2026-01-01",
        "--signal-end", "2026-07-31",
        "--period-count", "8",
        "--lookback", "120",
        "--batch-size", "64",
        "--sample-count", "50",
        "--seed", "20260817",
        "--temperatures", "0.5,0.6,0.7,0.8,0.9",
        "--top-p", str(top_p),
    ])

for command in commands:
    subprocess.run(command, cwd=REPO, check=True)

summaries = {}
for path in sorted(RUNTIME.glob("top_p_*/summary.json")):
    payload = json.loads(path.read_text())
    summaries[path.parent.name] = payload
(RUNTIME / "combined_summary.json").write_text(
    json.dumps(summaries, indent=2, ensure_ascii=False)
)
print(f"RESULT_PATH={RUNTIME}", flush=True)
