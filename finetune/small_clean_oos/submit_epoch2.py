"""Bundle and submit a frozen two-checkpoint evaluation; no checkpoint downloads."""

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
KERNEL = "luckfu/kronos-small-clean-epoch2-vs-c2-full-oos"


def published_file_names(api, parent):
    names, token = set(), None
    while True:
        response = api.kernels_list_files(parent, page_token=token, page_size=200)
        names.update(item.name for item in response.files)
        token = response.next_page_token
        if not token:
            return names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    files = sorted((ROOT / "model").glob("*.py")) + [HERE / "evaluate.py", HERE / "evaluate_epoch2.py"]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (2026, 10, 12, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    payload = stream.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    credential = runpy.run_path(str(ROOT / "finetune/small_clean_relay/push_chunk.py"))["local_credential"]()
    if not credential:
        raise RuntimeError("Missing SwanLab credential")
    code = f'''import base64, hashlib, io, json, os, runpy, subprocess, sys, threading, time, zipfile
from pathlib import Path
started = time.time()
print(json.dumps({{"phase": "started", "time": started}}), flush=True)
os.environ.update(PYTHONUNBUFFERED="1", SMALL_OOS_STARTED=str(started),
    SMALL_OOS_COMMIT={commit!r}, SMALL_OOS_SOURCE_SHA={digest!r},
    SWANLAB_API_KEY={credential!r})
os.environ["SMALL_OOS_OUTPUT"] = "/kaggle/working/small_clean_epoch2_oos"
root = Path("/kaggle/working/small_clean_epoch2_oos")
root.mkdir(exist_ok=True)
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
stop = threading.Event()
def heartbeat():
    while not stop.wait(30):
        print(json.dumps({{"phase": "setup_heartbeat", "time": time.time()}}), flush=True)
threading.Thread(target=heartbeat, daemon=True).start()
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
repo = Path("/kaggle/working/small_clean_oos_source")
zipfile.ZipFile(io.BytesIO(payload)).extractall(repo)
sys.path.insert(0, str(repo))
os.environ["PYTHONPATH"] = str(repo)
stop.set()
child = subprocess.Popen([sys.executable, "-u", str(repo / "finetune/small_clean_oos/evaluate_epoch2.py")],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in child.stdout:
    print(line, end="", flush=True)
if child.wait():
    raise RuntimeError("Evaluation failed")
'''
    build = HERE / "build_epoch2"
    build.mkdir(exist_ok=True)
    (build / "run.py").write_text(code)
    metadata = dict(id=KERNEL, title="Kronos Small Clean Epoch2 vs C2 Full OOS",
                    code_file="run.py", language="python", kernel_type="script",
                    is_private=True, enable_gpu=True, enable_internet=True,
                    machine_shape="NvidiaTeslaT4",
                    dataset_sources=["smmt315/a-share-120d-temporal-symbol-holdout"],
                    kernel_sources=["smmt315/kronos-small-0-1-stage2-cosine-refinement-c2",
                                    "luckfu/kronos-small-clean-epoch2-from-c03-e05"])
    (build / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    (build / "build_manifest.json").write_text(json.dumps(dict(
        source_sha256=digest, commit=commit, kernel=KERNEL,
        epoch2_sha256="68feb21bb0d2d6b385e69e95765536b25f2cad07a8efb011cd1bcdea878fda5d"), indent=2))
    compile(code, str(build / "run.py"), "exec")
    print(json.dumps(dict(kernel=KERNEL, commit=commit, source_sha256=digest,
                         hard_limit=43200, budget=40500, metadata=metadata), indent=2))
    if not args.submit:
        return
    configuration = subprocess.check_output(["kaggle", "config", "view"], text=True)
    if "- username: luckfu" not in configuration:
        raise RuntimeError("Wrong Kaggle owner")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    page = 1
    while True:
        rows = api.kernels_list(mine=True, page=page, page_size=200) or []
        if any(row.ref == KERNEL for row in rows):
            subprocess.run(["kaggle", "kernels", "status", KERNEL], check=True)
            subprocess.run(["kaggle", "kernels", "files", KERNEL, "--format", "json"], check=True)
            raise RuntimeError("Existing evaluation: inspect before resubmitting")
        if len(rows) < 200:
            break
        page += 1
    for parent in metadata["kernel_sources"]:
        status = subprocess.check_output(["kaggle", "kernels", "status", parent], text=True)
        if "COMPLETE" not in status:
            raise RuntimeError(f"Parent incomplete: {parent}")
        names = published_file_names(api, parent)
        if not any(name.endswith("model.safetensors") for name in names):
            raise RuntimeError(f"Missing parent weights: {parent}")
    subprocess.run(["kaggle", "kernels", "push", "-p", str(build),
                    "--timeout", "43200", "--accelerator", "NvidiaTeslaT4"], check=True)
    subprocess.run(["kaggle", "kernels", "status", KERNEL], check=True)


if __name__ == "__main__":
    main()
