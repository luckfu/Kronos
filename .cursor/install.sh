#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the Kronos A-share workbench.
# Safe to run repeatedly: it refreshes system packages, the Python venv,
# project dependencies, and the Beta V1.2 model weights only as needed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="$REPO_ROOT/.venv"
MODEL_DIR="$REPO_ROOT/models/a_share_v1_beta/releases/beta_v1.2"

echo "==> Ensuring system packages (python venv support)"
if ! python3 -c "import ensurepip" >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends python3-venv python3-pip
fi

echo "==> Creating/using virtualenv at .venv"
if [ ! -x "$VENV_DIR/bin/python" ]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip >/dev/null

echo "==> Installing CPU PyTorch"
# CPU-only wheel keeps the environment small and avoids CUDA downloads.
pip install --index-url https://download.pytorch.org/whl/cpu torch

echo "==> Installing project dependencies"
# Web UI + serverless inference + finetune numeric stack, plus test/runtime extras.
pip install \
  -r webui/requirements.txt \
  -r requirements-serverless.txt \
  matplotlib==3.9.3 \
  pytest \
  bcrypt \
  scipy \
  modelscope

echo "==> Downloading Beta V1.2 model weights (idempotent)"
if [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/model.safetensors" ]; then
  echo "    Model already present, skipping download."
else
  # Best-effort: keep setup succeeding even if ModelScope is unreachable.
  if modelscope download luckfu/Kronos-A-Share-Beta-V1-2 \
      --repo-type model --local-dir "$MODEL_DIR"; then
    echo "    Model download complete."
  else
    echo "    WARNING: model download failed; run this script again once network is available." >&2
  fi
fi

# The release ships the best checkpoint at the repo root, but the web UI and
# serverless service load it from a best_model/ subdirectory. Link it in place.
if [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/model.safetensors" ]; then
  echo "==> Linking best_model/ checkpoint directory"
  mkdir -p "$MODEL_DIR/best_model"
  ln -sf ../config.json "$MODEL_DIR/best_model/config.json"
  ln -sf ../model.safetensors "$MODEL_DIR/best_model/model.safetensors"
fi

echo "==> Install complete."
