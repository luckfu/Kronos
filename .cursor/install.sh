#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap for the Kronos A-share workbench.
#
# Scope: a lean CPU-only Python environment for editing code and running the
# `tests/` unit suite. Heavy training and full model inference run on Kaggle,
# NOT here, so this script does not download model weights or start servers.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="$REPO_ROOT/.venv"

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

echo "==> Installing project dependencies (web UI + serverless) and test extras"
pip install \
  -r webui/requirements.txt \
  -r requirements-serverless.txt \
  pytest \
  bcrypt \
  scipy

echo "==> Installing Kaggle CLI (training/eval runs on Kaggle)"
# The kaggle CLI/API is used to push kernels, check status, and pull outputs
# (see finetune/KAGGLE_*.md). Authenticate with KAGGLE_USERNAME + KAGGLE_KEY
# env vars or ~/.kaggle/kaggle.json; no credentials are baked in here.
pip install kaggle

echo "==> Install complete. Run the suite with: PYTHONPATH=. python -m pytest tests"
