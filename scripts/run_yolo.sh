#!/usr/bin/env bash
# YOLO Segmentation demo launcher
# - Creates/reuses Python env (conda preferred; venv fallback)
# - Installs ultralytics + deps and torch/torchvision
# - Launches a Tkinter GUI that runs YOLOv8-seg on video frames and overlays masks

set -euo pipefail

# ---- Config (overridable via env) ----
ENV_NAME=${ENV_NAME:-fastvlm}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
USE_CONDA=${USE_CONDA:-auto}   # auto|yes|no
YOLO_MODEL=${YOLO_MODEL:-yolov8s-seg.pt}
DEFAULT_VIDEO=${DEFAULT_VIDEO:-/Users/agc/Documents/output.mp4}
MODEL_DIR=${MODEL_DIR:-checkpoints/llava-fastvithd_1.5b_stage3}

# Args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --python) PYTHON_VERSION="$2"; shift 2 ;;
    --conda) USE_CONDA=yes; shift ;;
    --no-conda) USE_CONDA=no; shift ;;
    --model) YOLO_MODEL="$2"; shift 2 ;;
    --video) DEFAULT_VIDEO="$2"; shift 2 ;;
    --vlm-dir) MODEL_DIR="$2"; shift 2 ;;
    --help|-h)
      cat <<EOF
Usage: $0 [options]
  --env-name <name>     Conda env name (default: $ENV_NAME)
  --python <ver>        Python version for new env (default: $PYTHON_VERSION)
  --conda | --no-conda  Force conda or venv (default: auto)
  --model <weights>     YOLO weights or hub id (default: $YOLO_MODEL)
  --video <path>        Preload this video in GUI (default: $DEFAULT_VIDEO)
  --vlm-dir <path>      FastVLM model directory (default: $MODEL_DIR)
EOF
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"

has_cmd() { command -v "$1" >/dev/null 2>&1; }

activate_conda_env() {
  if has_cmd conda; then
    if [[ -f "$HOME/.conda/init.sh" ]]; then
      source "$HOME/.conda/init.sh"
    elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1091
      source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1091
      source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
      eval "$(conda shell.bash hook 2>/dev/null || true)"
    fi
    if conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
      echo "[env] Using existing conda env '$ENV_NAME'"
    else
      echo "[env] Creating conda env '$ENV_NAME' (python=$PYTHON_VERSION)"
      conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
    fi
    conda activate "$ENV_NAME"
    return 0
  fi
  return 1
}

activate_venv() {
  if [[ ! -d .venv ]]; then
    echo "[env] Creating venv at .venv (python=$PYTHON_VERSION)"
    if has_cmd "python$PYTHON_VERSION"; then
      "python$PYTHON_VERSION" -m venv .venv
    else
      python3 -m venv .venv
    fi
  else
    echo "[env] Using existing venv at .venv"
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
}

setup_env() {
  case "$USE_CONDA" in
    yes) activate_conda_env || { echo "[env] conda requested but not found"; exit 1; } ;;
    no)  activate_venv ;;
    auto)
      if activate_conda_env; then :; else
        echo "[env] conda not found; falling back to venv"
        activate_venv
      fi ;;
    *) echo "[env] invalid USE_CONDA: $USE_CONDA"; exit 1 ;;
  esac
}

verify_and_install() {
python3 - <<'PY'
import json, sys, platform
from importlib import metadata as im

reqs = {
  'ultralytics': ('ultralytics', '>=8.2.0,<9'),
  'opencv-python': ('opencv-python', None),
  'Pillow': ('Pillow', None),
  'numpy': ('numpy', None),
  'torch': ('torch', None),
  'torchvision': ('torchvision', None),
  # Minimal FastVLM deps (skip bitsandbytes for macOS)
  'transformers': ('transformers', '==4.48.3'),
  'tokenizers': ('tokenizers', '==0.21.0'),
  'sentencepiece': ('sentencepiece', '==0.1.99'),
  'accelerate': ('accelerate', '==1.6.0'),
  'peft': ('peft', '>=0.10.0,<0.14.0'),
  'einops': ('einops', '==0.6.1'),
  'einops-exts': ('einops-exts', '==0.0.4'),
  'timm': ('timm', '==1.0.15'),
}

missing = []
for pip_name, (dist_name, spec) in reqs.items():
  try:
    _ = im.version(dist_name)
  except im.PackageNotFoundError:
    missing.append(pip_name)

print(json.dumps({'missing': missing}))
PY
}

run_install_plan() {
  local plan_json="$1"
  python3 - "$plan_json" <<'PY'
import json, subprocess, sys, platform
plan = json.loads(sys.argv[1])
missing = set(plan.get('missing', []))

def pip_install(args):
    print("[pip]", " ".join(args))
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade'] + args)

# Ensure pip itself is reasonably current
pip_install(['pip'])

# Torch first (to let ultralytics detect it)
if 'torch' in missing or 'torchvision' in missing:
    # Default wheels work for CPU and enable MPS on macOS; CUDA users can swap manually
    pkgs = []
    if 'torch' in missing: pkgs.append('torch')
    if 'torchvision' in missing: pkgs.append('torchvision')
    pip_install(pkgs)
    missing.discard('torch'); missing.discard('torchvision')

# Core deps
rest = [p for p in ['ultralytics','opencv-python','Pillow','numpy'] if p in missing]
if rest:
    pip_install(rest)

# FastVLM deps (skip bitsandbytes on macOS)
extra = [p for p in ['transformers','tokenizers','sentencepiece','accelerate','peft','einops','einops-exts','timm'] if p in missing]
if extra:
    pip_install(extra)

print("OK")
PY
}

setup_env
PLAN_JSON=$(verify_and_install)
run_install_plan "$PLAN_JSON"

# Launch GUI with environment hints
export YOLO_MODEL
export VIDEO_PATH="$DEFAULT_VIDEO"
export MODEL_DIR
python3 scripts/yolo_video_gui.py
