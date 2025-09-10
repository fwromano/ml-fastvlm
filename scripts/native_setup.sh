#!/usr/bin/env bash
# Prepare a native (Core ML + MLX) FastVLM for Apple Silicon
# - Exports the vision tower to Core ML
# - Downloads (or converts) an MLX LLM for the chosen size
# - Produces an exported model folder you can use with MLX-VLM

set -euo pipefail

MODEL_SIZE=${MODEL_SIZE:-1.5b}      # 0.5b | 1.5b | 7b
HF_DIR=${HF_DIR:-checkpoints/llava-fastvithd_1.5b_stage3}
OUT_DIR=${OUT_DIR:-exported/fastvlm_${MODEL_SIZE}_mlx}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --size) MODEL_SIZE="$2"; shift 2;;
    --hf) HF_DIR="$2"; shift 2;;
    --out) OUT_DIR="$2"; shift 2;;
    --help|-h)
      echo "Usage: $0 [--size 0.5b|1.5b|7b] [--hf <hf_checkpoint_dir>] [--out <export_dir>]"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "[native] Export destination: $OUT_DIR"
mkdir -p "$OUT_DIR"

echo "[native] Step 1/2: Exporting Core ML vision tower (from $HF_DIR)"
python3 model_export/export_vision_encoder.py --model-path "$HF_DIR" || {
  echo "[warn] Core ML export failed. Ensure coremltools is installed and checkpoint exists.";
}

echo "[native] Step 2/2: Getting MLX LLM for $MODEL_SIZE"
if [[ -x app/get_pretrained_mlx_model.sh ]]; then
  bash app/get_pretrained_mlx_model.sh --model "$MODEL_SIZE" --dest "$OUT_DIR" || true
fi

# If mlx-vlm is available (and patched), also run a local conversion to ensure config model_type matches 'fastvlm'
if python3 - <<'PY'
try:
    import mlx_vlm, importlib
    importlib.import_module('mlx_vlm.models.fastvlm')
    print('ok')
except Exception:
    raise SystemExit(1)
PY
then
  echo "[native] Converting LLM with patched mlx-vlm to $OUT_DIR"
  python3 -m mlx_vlm convert --hf-path "$HF_DIR" --mlx-path "$OUT_DIR" --q-bits 8 --only-llm || {
    echo "[warn] mlx-vlm convert failed; proceeding with any downloaded model.";
  }
else
  echo "[warn] Patched mlx-vlm not present; skipping conversion."
fi

echo "[native] Done. If $OUT_DIR contains the MLX model (and .mlpackage in it), you can run native inference."
