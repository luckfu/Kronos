"""Build (and optionally submit) the C04-vs-C03 full OOS kernel."""

import argparse
import base64
import hashlib
import io
import json
import runpy
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
KERNEL = "user281434/kronos-small-clean-c04-vs-c03-full-oos"
TITLE = "Kronos Small Clean C04 vs C03 Full OOS"


def source_bundle():
    files = sorted((ROOT / "model").glob("*.py"))
    files += [HERE / "evaluate.py", HERE / "evaluate_c04.py"]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (2026, 10, 18, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    payload = source_bundle()
    digest = hashlib.sha256(payload).hexdigest()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    credential = runpy.run_path(
        str(ROOT / "finetune/small_clean_relay/push_chunk.py"))["local_credential"]()
    if not credential:
        raise RuntimeError("Missing SwanLab credential")
    code = f'''import base64, hashlib, io, json, os, subprocess, sys, time, zipfile
from pathlib import Path
started = time.time()
print(json.dumps({{"phase": "started", "time": started}}), flush=True)
os.environ.update(PYTHONUNBUFFERED="1", SMALL_OOS_STARTED=str(started),
    SMALL_OOS_COMMIT={commit!r}, SMALL_OOS_SOURCE_SHA={digest!r},
    SMALL_OOS_OUTPUT="/kaggle/working/small_clean_c04_oos",
    SWANLAB_API_KEY={credential!r})
root = Path("/kaggle/working/small_clean_c04_oos")
root.mkdir(parents=True, exist_ok=True)
class Tee:
    def __init__(self, stream):
        self.stream = stream
        self.log = open(root / "run.log", "a", buffering=1)
    def write(self, text):
        self.log.write(text)
        return self.stream.write(text)
    def flush(self):
        self.log.flush()
        self.stream.flush()
sys.stdout = Tee(sys.stdout)
sys.stderr = Tee(sys.stderr)
print(json.dumps({{"phase": "started_persisted", "time": started}}), flush=True)
def command(args):
    print(json.dumps({{"phase": "install_started", "time": time.time(), "command": args}}), flush=True)
    child = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
    for line in child.stdout:
        print(line, end="", flush=True)
    if child.wait():
        raise RuntimeError("Dependency installation failed")
command([sys.executable, "-m", "pip", "install", "torch==2.6.0",
         "--only-binary=:all:", "--index-url", "https://download.pytorch.org/whl/cu124"])
command([sys.executable, "-m", "pip", "install", "--index-url", "https://pypi.org/simple",
         "--only-binary=numpy,pandas,safetensors,orjson,psutil,protobuf,pyyaml,hf-xet",
         "numpy==2.1.3", "pandas==2.2.3", "einops==0.8.1",
         "huggingface_hub==0.33.1", "safetensors==0.6.2", "tqdm==4.67.1", "swanlab==0.9.8",
         "scipy==1.15.3"])
payload = base64.b64decode({base64.b64encode(payload).decode()!r})
assert hashlib.sha256(payload).hexdigest() == {digest!r}
repo = Path("/kaggle/working/small_clean_c04_source")
zipfile.ZipFile(io.BytesIO(payload)).extractall(repo)
sys.path.insert(0, str(repo))
os.environ["PYTHONPATH"] = str(repo)
child = subprocess.Popen(
    [sys.executable, "-u", str(repo / "finetune/small_clean_oos/evaluate_c04.py")],
    cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in child.stdout:
    print(line, end="", flush=True)
if child.wait():
    raise RuntimeError("Evaluation failed")
'''
    build = HERE / "build_c04"
    build.mkdir(exist_ok=True)
    (build / "run.py").write_text(code)
    metadata = dict(
        id=KERNEL, title=TITLE, code_file="run.py", language="python",
        kernel_type="script", is_private=True, enable_gpu=True, enable_internet=True,
        machine_shape="NvidiaTeslaT4",
        dataset_sources=["smmt315/a-share-120d-temporal-symbol-holdout"],
        kernel_sources=["luckfu/kronos-small-clean-wc-dual-t4-c03",
                        "user281434/kronos-small-clean-wc-dual-t4-c04"],
    )
    (build / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (build / "build_manifest.json").write_text(json.dumps(dict(
        source_sha256=digest, commit=commit, kernel=KERNEL,
        models=["c03_best", "c04_best", "c04_last"]), indent=2) + "\n")
    compile(code, str(build / "run.py"), "exec")
    print(json.dumps(dict(kernel=KERNEL, commit=commit, source_sha256=digest,
                          hard_limit=43200, metadata=metadata), indent=2))
    if not args.submit:
        return
    subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(build),
                    "--timeout", "43200", "--accelerator", "NvidiaTeslaT4"], check=True)


if __name__ == "__main__":
    main()
