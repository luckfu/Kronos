"""Bundle the validation Rank IC evaluator into a Kaggle script kernel."""

import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = HERE / "build"
BUILD.mkdir(parents=True, exist_ok=True)

files = sorted((ROOT / "model").glob("*.py"))
files.append(ROOT / "finetune" / "asset_metadata.py")
files.append(HERE / "evaluate_validation_rank_ic.py")

stream = io.BytesIO()
with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        archive.writestr(path.relative_to(ROOT).as_posix(), path.read_bytes())
payload = stream.getvalue()
digest = hashlib.sha256(payload).hexdigest()

code = f'''import base64, io, os, subprocess, sys, zipfile
from pathlib import Path
root = Path("/kaggle/working/src"); root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode({base64.b64encode(payload).decode()!r}))) as z:
    z.extractall(root)
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.6.0", "--index-url", "https://download.pytorch.org/whl/cu124"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==2.1.3", "pandas==2.2.3", "einops==0.8.1", "safetensors==0.6.2", "huggingface_hub==0.33.1", "tqdm==4.67.1"])
env = dict(os.environ, PYTHONPATH=str(root), PYTHONUNBUFFERED="1")
subprocess.check_call([sys.executable, "-u", str(root / "finetune/small_validation_rank_ic/evaluate_validation_rank_ic.py")], env=env)
'''
(BUILD / "run.py").write_text(code)

metadata = {
    "id": "wynstonliu/kronos-small-validation-rank-ic-20261023",
    "title": "Kronos Small Validation Rank IC 20261023",
    "code_file": "run.py",
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": [
        "smmt315/a-share-120d-temporal-symbol-holdout",
        "wynstonliu/kronos-small-c2-best-checkpoint",
        "wynstonliu/kronos-small-epoch2-best-checkpoint",
    ],
    "kernel_sources": [
        "luckfu/kronos-small-clean-wc-dual-t4-c03",
        "wynstonliu/kronos-small-clean-bigbatch-e03b",
    ],
}
(BUILD / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
(BUILD / "build_manifest.json").write_text(json.dumps(dict(
    source_sha256=digest, kernel=metadata["id"],
    files=[path.relative_to(ROOT).as_posix() for path in files],
), indent=2) + "\n")
print(json.dumps({"build": str(BUILD), "source_sha256": digest,
                  "id": metadata["id"], "files": [path.name for path in files]}, indent=2))
