"""Bundle the sealed OOS eval into a Kaggle script kernel."""
import base64, hashlib, io, json, zipfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = HERE / "build"
BUILD.mkdir(exist_ok=True)
files = sorted((ROOT / "model").glob("*.py")) + [HERE / "evaluate_sealed.py"]
stream = io.BytesIO()
with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.writestr(p.relative_to(ROOT).as_posix(), p.read_bytes())
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
subprocess.check_call([sys.executable, "-u", str(root / "finetune/small_sealed_oos/evaluate_sealed.py")], env=env)
'''
(BUILD / "run.py").write_text(code)
meta = {
    "id": "wynstonliu/kronos-small-sealed-oos-20260803-20260930",
    "title": "Kronos Small Sealed OOS 20260803 20260930",
    "code_file": "run.py", "language": "python", "kernel_type": "script",
    "is_private": True, "enable_gpu": True, "enable_internet": True,
    "machine_shape": "NvidiaTeslaT4",
    "dataset_sources": ["wynstonliu/a-share-120d-oos-20260803-20260930",
                        "wynstonliu/kronos-small-c2-best-checkpoint"],
    "kernel_sources": ["wynstonliu/kronos-small-clean-bigbatch-e03b"],
}
(BUILD / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps({"build": str(BUILD), "source_sha256": digest, "files": [p.relative_to(ROOT).as_posix() for p in files]}, indent=2))
