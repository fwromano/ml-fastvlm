#!/usr/bin/env bash
# A smart launcher for the FastVLM demo that:
# - Creates or reuses a Python env (conda preferred, venv fallback)
# - Verifies (not blindly reinstalls) required dependencies
# - Ensures a model checkpoint is available
# - Launches the Gradio demo

set -euo pipefail

# ---- Config (can be overridden via env vars) ----
ENV_NAME=${ENV_NAME:-fastvlm}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
# Default model can be customized here; override at runtime with --model-dir or MODEL_DIR env var
MODEL_DIR=${MODEL_DIR:-checkpoints/llava-fastvithd_7b_stage3}
USE_CONDA=${USE_CONDA:-auto}   # auto|yes|no
# Default video path to prefill GUI/CLI if present
DEFAULT_VIDEO=${DEFAULT_VIDEO:-/Users/agc/Documents/output.mp4}

# Optional: pass options to override
#   --model-dir <path>   set model path
#   --env-name <name>    conda env name
#   --python <ver>       python version for new env
#   --conda|--no-conda   force conda or venv
#   --share              force Gradio share link (UI mode)
#   --no-share           disable Gradio share (UI mode)
#   --video <path>       run headless CLI on a video
#   --prompt <text>      prompt for CLI mode
#   --frames <n>         frames sampled for CLI mode
#   --gui                open a minimal desktop GUI for prompt input
#   --native-setup       export CoreML + get MLX model (Apple-native)
#   --native-video       run MLX-VLM on video (requires native setup)
# Ensure stale env doesn't force headless
unset CLI_VIDEO
unset CLI_PROMPT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)
      MODEL_DIR="$2"; shift 2 ;;
    --env-name)
      ENV_NAME="$2"; shift 2 ;;
    --python)
      PYTHON_VERSION="$2"; shift 2 ;;
    --conda)
      USE_CONDA=yes; shift ;;
    --no-conda)
      USE_CONDA=no; shift ;;
    --share)
      export DEMO_SHARE=1; shift ;;
    --no-share)
      export DEMO_SHARE=0; shift ;;
    --video)
      CLI_VIDEO="$2"; shift 2 ;;
    --prompt)
      CLI_PROMPT="$2"; shift 2 ;;
    --frames)
      CLI_FRAMES="$2"; shift 2 ;;
    --gui)
      GUI_MODE=1; shift ;;
    --native-setup)
      NATIVE_SETUP=1; shift ;;
    --native-video)
      NATIVE_VIDEO=1; shift ;;
    --help|-h)
      echo "Usage: $0 [--model-dir <dir>] [--env-name <name>] [--python <ver>] [--conda|--no-conda] [--share|--no-share] [--video <path>] [--prompt <text>] [--frames <n>] [--gui] [--native-setup] [--native-video]";
      exit 0 ;;
    *)
      echo "Unknown argument: $1" ; exit 1 ;;
  esac
done

HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"

has_cmd() { command -v "$1" >/dev/null 2>&1; }

activate_conda_env() {
  # shellcheck source=/dev/null
  if has_cmd conda; then
    # Try both bash and zsh init locations
    if [[ -f "$HOME/.conda/init.sh" ]]; then
      source "$HOME/.conda/init.sh"
    elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
      # common miniconda path
      # shellcheck disable=SC1091
      source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
      # common anaconda path
      # shellcheck disable=SC1091
      source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [[ -n "${CONDA_EXE:-}" ]] && [[ -f "$(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh" ]]; then
      # derive from CONDA_EXE
      # shellcheck disable=SC1091
      source "$(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh"
    else
      # fallback: try to eval shell hook
      eval "$(conda shell.bash hook 2>/dev/null || true)"
    fi

    # Check if env exists
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
    # Try to use requested python version, fallback to default python3
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
    yes)
      activate_conda_env || { echo "[env] conda requested but not found"; exit 1; } ;;
    no)
      activate_venv ;;
    auto)
      if activate_conda_env; then
        :
      else
        echo "[env] conda not found; falling back to venv"
        activate_venv
      fi ;;
    *) echo "[env] invalid USE_CONDA: $USE_CONDA"; exit 1 ;;
  esac
}

verify_and_install() {
  # Use a Python helper to check required distributions and decide what to install.
  # We prefer to install only missing or clearly mismatched packages.
python - "$MODEL_DIR" <<'PY'
import json, sys, re, platform
from importlib import metadata as im

# Minimal set of runtime requirements for the demo (subset of pyproject + opencv)
reqs = {
    # name_in_pip: (dist_name_for_version, version_spec or None)
    "torch": ("torch", "==2.6.0"),
    "torchvision": ("torchvision", "==0.21.0"),
    "transformers": ("transformers", "==4.48.3"),
    "tokenizers": ("tokenizers", "==0.21.0"),
    "sentencepiece": ("sentencepiece", "==0.1.99"),
    "shortuuid": ("shortuuid", None),
    "accelerate": ("accelerate", "==1.6.0"),
    "peft": ("peft", ">=0.10.0,<0.14.0"),
    # bitsandbytes is often unsupported on macOS; will be skipped on Darwin
    "bitsandbytes": ("bitsandbytes", None),
    "pydantic": ("pydantic", None),
    "markdown2": ("markdown2", None),
    "numpy": ("numpy", "==1.26.4"),
    "scikit-learn": ("scikit-learn", "==1.2.2"),
    "gradio": ("gradio", "==5.11.0"),
    "requests": ("requests", None),
    "uvicorn": ("uvicorn", None),
    "fastapi": ("fastapi", None),
    "einops": ("einops", "==0.6.1"),
    "einops-exts": ("einops-exts", "==0.0.4"),
    "timm": ("timm", "==1.0.15"),
    "coremltools": ("coremltools", "==8.2"),
    # Demo-specific
    "opencv-python": ("opencv-python", None),
    "pillow": ("Pillow", None),
}

# Skip problematic deps by platform
if platform.system() == "Darwin":
    reqs.pop("bitsandbytes", None)

def satisfies(installed: str, spec: str) -> bool:
    # Very minimal spec checker for == or simple ranges
    try:
        from packaging.version import Version
    except Exception:
        # If packaging is not installed, be lenient and skip strict checks
        return True
    v = Version(installed)
    if spec is None:
        return True
    for part in spec.split(","):
        part = part.strip()
        if part.startswith("=="):
            if installed != part[2:]:
                return False
        elif part.startswith(">="):
            if v < Version(part[2:]):
                return False
        elif part.startswith("<="):
            if v > Version(part[2:]):
                return False
        elif part.startswith(">"):
            if v <= Version(part[1:]):
                return False
        elif part.startswith("<"):
            if v >= Version(part[1:]):
                return False
        else:
            # unknown constraint, ignore
            pass
    return True

missing = []
outdated = []
for pip_name, (dist_name, spec) in reqs.items():
    try:
        ver = im.version(dist_name)
        if spec and not satisfies(ver, spec):
            outdated.append((pip_name, spec))
    except im.PackageNotFoundError:
        missing.append((pip_name, spec))

# Check if local package is installed in editable mode; if not, plan to install it
need_editable = False
try:
    dist = im.distribution("llava")
    # If installed but not editable pointing to this repo, still fine; we won't force reinstall
except im.PackageNotFoundError:
    need_editable = True

plan = {
    "missing": missing,
    "outdated": outdated,
    "need_editable": need_editable,
}
print(json.dumps(plan))
PY
}

run_install_plan() {
  local plan_json="$1"
  # Parse JSON with python (to avoid jq dependency). Pass JSON as argv[1].
  python3 - "$plan_json" <<'PY'
import json, os, sys, shlex, subprocess
if len(sys.argv) < 2:
    print("[error] Missing plan JSON argument", file=sys.stderr)
    sys.exit(1)
plan = json.loads(sys.argv[1])

def pip_install(args):
    cmd = [sys.executable, "-m", "pip", "install"] + args
    print("[pip]", " ".join(shlex.quote(c) for c in cmd))
    subprocess.check_call(cmd)

# Install/upgrade only what is necessary
to_install = []
for name, spec in plan.get("missing", []):
    to_install.append(name + (spec or ""))
for name, spec in plan.get("outdated", []):
    to_install.append(name + (spec or ""))

if to_install:
    pip_install(to_install)
else:
    print("[deps] All third-party requirements satisfied")

if plan.get("need_editable"):
    # Install the local package in editable mode
    pip_install(["-e", "."])  # matches README
else:
    print("[deps] Local package already present (llava)")

# Skip 'pip check' to avoid unrelated warnings (e.g., gradio-client)
PY
}

ensure_model() {
  if [[ -d "$MODEL_DIR" ]]; then
    echo "[model] Using existing model at $MODEL_DIR"
  else
    echo "[model] Model dir '$MODEL_DIR' not found; attempting download via get_models.sh"
    if [[ -f get_models.sh ]]; then
      bash get_models.sh || {
        echo "[warn] get_models.sh failed; you may need to download manually.";
      }
    else
      echo "[warn] get_models.sh not found; please place a model in '$MODEL_DIR' or set MODEL_DIR."
    fi

    # Try common alias mapping: fastvlm_* -> llava-fastvithd_*
    if [[ ! -d "$MODEL_DIR" ]]; then
      base="$(basename "$MODEL_DIR")"
      parent="$(dirname "$MODEL_DIR")"
      if [[ "$base" == fastvlm_* ]]; then
        alt="$parent/llava-fastvithd_${base#fastvlm_}"
        if [[ -d "$alt" ]]; then
          echo "[model] Found matching model at $alt; switching"
          MODEL_DIR="$alt"
        fi
      fi
    fi

    # If still not found, try defaulting to 1.5b stage3 if present
    if [[ ! -d "$MODEL_DIR" ]] && [[ -d checkpoints ]]; then
      cand=$(ls -1d checkpoints/llava-fastvithd_1.5b_stage3 2>/dev/null | head -n1 || true)
      if [[ -n "$cand" ]]; then
        echo "[model] Defaulting to $cand"
        MODEL_DIR="$cand"
      fi
    fi
  fi
}

launch_demo() {
  echo "[run] Launching demo with MODEL_DIR=$MODEL_DIR"
  MODEL_DIR="$MODEL_DIR" python demo_video_fastvlm.py
}

# Headless CLI runner
launch_cli() {
  echo "[run] Running headless CLI on: $CLI_VIDEO"
  local args=("--video" "$CLI_VIDEO" "--model-dir" "$MODEL_DIR")
  if [[ -n "${CLI_PROMPT:-}" ]]; then args+=("--prompt" "$CLI_PROMPT"); fi
  if [[ -n "${CLI_FRAMES:-}" ]]; then args+=("--frames" "$CLI_FRAMES"); fi
  python scripts/video_predict.py "${args[@]}"
}

# Desktop GUI runner
launch_gui() {
  echo "[run] Opening desktop GUI"
  local envs=("MODEL_DIR=$MODEL_DIR")
  if [[ -n "${CLI_VIDEO:-}" ]]; then envs+=("VIDEO_PATH=$CLI_VIDEO"); fi
  env ${envs[@]} python3 scripts/video_prompt_gui.py
}

# Native setup (Core ML + MLX)
native_setup() {
  echo "[native] Preparing Apple-native (Core ML + MLX) model"
  bash scripts/native_setup.sh || {
    echo "[error] Native setup failed. Please see logs above."; exit 1;
  }
}

# Native video runner (MLX-VLM)
native_video() {
  local native_dir=${NATIVE_DIR:-exported/fastvlm_7b_mlx}
  if [[ -z "${CLI_VIDEO:-}" && -f "$DEFAULT_VIDEO" ]]; then CLI_VIDEO="$DEFAULT_VIDEO"; fi
  if [[ -z "${CLI_VIDEO:-}" ]]; then
    echo "[error] No video path provided. Use --video <path> or set DEFAULT_VIDEO."; exit 1;
  fi
  # Ensure mlx_vlm is installed
  if ! python3 - <<'PY'
try:
    import mlx_vlm  # noqa: F401
    print('ok')
except Exception:
    raise SystemExit(1)
PY
  then
    echo "[deps] Installing mlx-vlm (MLX VLM runtime)"
    if ! python3 -m pip install --upgrade pip setuptools wheel >/dev/null 2>&1; then
      echo "[warn] Could not upgrade pip; continuing"
    fi
  if ! python3 -m pip install --no-deps 'mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git'; then
      echo "[error] Failed to install mlx_vlm automatically."
      echo "        Try: python3 -m pip install 'mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git'"
      echo "        Or follow model_export/README.md to build locally."
      exit 1
    fi
  fi

  # Ensure mlx_vlm has FastVLM support (patched)
  if ! python3 - <<'PY'
try:
    import importlib
    import mlx_vlm
    import mlx_vlm.models.fastvlm  # added by repo patch
    print('patched')
except Exception:
    raise SystemExit(1)
PY
  then
    echo "[deps] Installing patched mlx-vlm locally (with FastVLM support)"
    WORKDIR=".native"
    REPO_DIR="$WORKDIR/mlx-vlm"
    mkdir -p "$WORKDIR"
    if [[ ! -d "$REPO_DIR/.git" ]]; then
      if has_cmd git; then
        git clone https://github.com/Blaizzy/mlx-vlm.git "$REPO_DIR" || {
          echo "[error] git clone failed; cannot install patched mlx-vlm."; exit 1; }
      else
        echo "[error] git is not installed; cannot install patched mlx-vlm."; exit 1;
      fi
    fi
    if ! git -C "$REPO_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      echo "[error] $REPO_DIR is not a git repo"; exit 1;
    fi
    # Checkout commit expected by our FastVLM patch
    git -C "$REPO_DIR" fetch --all >/dev/null 2>&1 || true
    if ! git -C "$REPO_DIR" checkout 1884b551bc741f26b2d54d68fa89d4e934b9a3de; then
      echo "[error] Failed to checkout expected mlx-vlm commit"; exit 1;
    fi
    # Apply patch (idempotent)
    if ! git -C "$REPO_DIR" apply --check "$HERE/model_export/fastvlm_mlx-vlm.patch" >/dev/null 2>&1; then
      echo "[info] Patch may already be applied; continuing"
    else
      git -C "$REPO_DIR" apply "$HERE/model_export/fastvlm_mlx-vlm.patch"
    fi
    # Install editable
    python3 -m pip install -e "$REPO_DIR" --no-deps || { echo "[error] Failed to install patched mlx-vlm"; exit 1; }
    # Re-enforce numeric pins in case pip adjusted them
    PLAN_JSON=$(verify_and_install)
    run_install_plan "$PLAN_JSON"
  fi
  if [[ ! -d "$native_dir" ]]; then
    echo "[native] MLX export not found at $native_dir. Running native setup..."
    native_setup
  fi
  if [[ ! -d "$native_dir" ]]; then
    echo "[error] MLX export still missing at $native_dir after setup."; exit 1;
  fi
  # Ensure language model aliases so patched loader treats 'fastvlm' as 'qwen2'
  python3 - <<'PY'
try:
    import importlib, inspect
    import mlx_vlm.models.fastvlm.language as lang
    p = lang.__file__
    txt = open(p, 'r').read()
    needle = "def __init__(self, args: TextConfig):"
    if needle in txt and "# __alias_injected__" not in txt:
        out = []
        for line in txt.splitlines():
            out.append(line)
            if line.strip().startswith(needle):
                out.append("        # __alias_injected__")
                out.append("        if getattr(args, 'model_type', None) in ('fastvlm','llava_qwen2','llava'):")
                out.append("            args.model_type = 'qwen2'")
        open(p, 'w').write("\n".join(out))
        print("[native] Injected model_type alias in:", p)
except Exception as e:
    pass
PY
  # Patch config.json model_type to match available loader
  if [[ -f "$native_dir/config.json" ]]; then
    python3 - <<PY
import json, importlib
from pathlib import Path
cfg_path = Path(r"""$native_dir""")/"config.json"
try:
    cfg = json.loads(cfg_path.read_text())
except FileNotFoundError:
    raise SystemExit(0)
# Prefer custom fastvlm loader if present; otherwise fall back to llava
try:
    import mlx_vlm
    importlib.import_module('mlx_vlm.models.fastvlm')
    target = 'fastvlm'
except Exception:
    target = 'llava'
changed = cfg.get('model_type') != target
cfg['model_type'] = target
cfg.setdefault('image_token_index', cfg.get('image_token_index', 151646))
if changed:
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"[native] Patched config.json -> model_type={target}")
PY
  fi
  echo "[native] Using MLX model dir: $native_dir"
  PROMPT_DEFAULT=${PROMPT_DEFAULT:-$'Return ONLY minimal JSON for visible vehicles.\n\nSchema (use exactly these keys):\n{\n  "vehicles": [\n    {"id":"v1","type":"<sedan|suv|truck|van|bus|motorcycle|bicycle|unknown>","color":"<e.g., white>","notes":["<e.g., parked|moving>"]}\n  ]\n}\n\nRules:\n- Output JSON only; no prose, no code fences.\n- If no vehicles, use "vehicles": [].\n- Use ids v1, v2, …; lowercase all strings; lists ≤3 items; omit a field if you cannot infer it; ensure valid JSON.'}
  python3 scripts/mlx_video_cli.py --model "$native_dir" --video "$CLI_VIDEO" --prompt "${CLI_PROMPT:-$PROMPT_DEFAULT}" --max-new-tokens 100
}

# -------- Main --------
setup_env

echo "[deps] Verifying Python packages (minimal MLX stack)..."
verify_and_install() {
python3 - <<'PY'
import json
from importlib import metadata as im

def get_ver(dist):
    try:
        return im.version(dist)
    except im.PackageNotFoundError:
        return None

missing = []
outdated = []

# Minimal deps for native path and tooling (pin numeric stack for MLX-VLM compat)
pins = {
    "numpy": "1.26.4",
    "scipy": "1.10.1",
    "scikit-learn": "1.2.2",
    "coremltools": "8.2",
    "opencv-python": "4.10.0.84",
}

def ensure(dist, spec=None):
    v = get_ver(dist)
    if v is None:
        if spec:
            missing.append((f"{dist}=={spec}", None))
        else:
            missing.append((dist, None))
    else:
        if spec and v != spec:
            outdated.append((f"{dist}=={spec}", None))

# Pinned numerics
for d, v in pins.items():
    ensure(d, v)

# Unpinned basics
for d in ["Pillow", "mlx"]:
    ensure(d)

plan = {"missing": missing, "outdated": outdated, "need_editable": False}
print(json.dumps(plan))
PY
}

PLAN_JSON=$(verify_and_install)
run_install_plan "$PLAN_JSON"

ensure_model

# Auto-mode selection: prefer GUI (if tkinter available); otherwise fallback to CLI if a default video exists; otherwise web demo.
has_tkinter() { python3 - <<'PY'
try:
    import tkinter  # noqa: F401
    print("ok")
except Exception:
    raise SystemExit(1)
PY
}

# MLX-only behavior: require a video, use DEFAULT_VIDEO if present
if [[ -n "${NATIVE_SETUP:-}" ]]; then
  native_setup; exit 0
fi

# If no video specified, open MLX GUI (preferred)
if [[ -z "${CLI_VIDEO:-}" ]]; then
  # Try Tk; if not available, show guidance
  if has_tkinter >/dev/null 2>&1; then
    # Prepare MLX stack and launch GUI
    # Define helper inline to keep file cohesive
    launch_mlx_gui_inline() {
      # Ensure mlx-vlm + export ready
      # Reuse native_video preparation by invoking up to before running
      local native_dir=${NATIVE_DIR:-exported/fastvlm_7b_mlx}
      # Minimal prepare: ensure patched mlx_vlm and config
      # Install mlx_vlm if needed
      if ! python3 - <<'PY'
try:
    import mlx_vlm
    print('ok')
except Exception:
    raise SystemExit(1)
PY
      then
        python3 -m pip install --no-deps 'mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git' || true
      fi
      # Patch loader alias and config as in native_video
      python3 - <<'PY'
try:
    import importlib
    import mlx_vlm.models.fastvlm.language as lang
    p = lang.__file__
    txt = open(p,'r').read()
    if "# __alias_injected__" not in txt:
        needle = "def __init__(self, args: TextConfig):"
        out=[]
        for line in txt.splitlines():
            out.append(line)
            if line.strip().startswith(needle):
                out.append("        # __alias_injected__")
                out.append("        if getattr(args, 'model_type', None) in ('fastvlm','llava_qwen2','llava'):")
                out.append("            args.model_type = 'qwen2'")
        open(p,'w').write("\n".join(out))
except Exception:
    pass
PY
      if [[ -f "$native_dir/config.json" ]]; then
        python3 - <<PY
import json, importlib
from pathlib import Path
cfg_path = Path(r"""$native_dir""")/"config.json"
try:
    cfg = json.loads(cfg_path.read_text())
except FileNotFoundError:
    raise SystemExit(0)
try:
    import mlx_vlm, importlib
    importlib.import_module('mlx_vlm.models.fastvlm')
    target='fastvlm'
except Exception:
    target='llava'
cfg['model_type']=target
cfg.setdefault('image_token_index', cfg.get('image_token_index', 151646))
cfg_path.write_text(json.dumps(cfg, indent=2))
print(f"[native] Patched config.json -> model_type={target}")
PY
      fi
      # Launch GUI
      MODEL_DIR="$native_dir" python3 scripts/mlx_video_gui.py
    }
    launch_mlx_gui_inline
    exit 0
  else
    echo "[error] Tkinter GUI is unavailable. Install 'tk' in your conda env or provide --video."
    exit 1
  fi
fi

# If video provided, run MLX headless on that video
native_video
