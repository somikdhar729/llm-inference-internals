#!/usr/bin/env bash
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="/e/llama.cpp"
BENCH="$LLAMA_CPP_DIR/build/bin/Release/llama-bench.exe"
MODELS_DIR="$PROJECT_DIR/models"

declare -A MODEL_FILE=(
  [bf16]="$MODELS_DIR/qwen3-1.7b-bf16.gguf"
  [q8]="$MODELS_DIR/qwen3-1.7b-q8_0.gguf"
)
# Max depth that can plausibly fit in 6 GB; the deepest point documents the limit.
declare -A DEPTHS=(
  [bf16]="0 512 1024 2048 4096 8192 16384 24576"
  [q8]="0 512 1024 2048 4096 8192 16384 24576 32768"
)

SELECTED=("$@"); [[ ${#SELECTED[@]} -eq 0 ]] && SELECTED=(bf16 q8)

TS=$(date +%Y%m%d_%H%M%S)
OUT="$PROJECT_DIR/results/baseline_$TS"
mkdir -p "$OUT"

[[ -f "$BENCH" ]] || { echo "missing $BENCH"; exit 1; }
for k in "${SELECTED[@]}"; do
  [[ -n "${MODEL_FILE[$k]:-}" ]] || { echo "unknown model key: $k (use bf16/q8)"; exit 1; }
  [[ -f "${MODEL_FILE[$k]}" ]] || { echo "missing ${MODEL_FILE[$k]}"; exit 1; }
done

# --- environment record ---
{
  date
  nvidia-smi
  nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,memory.total,power.limit --format=csv
  echo "llama.cpp: $(git -C "$LLAMA_CPP_DIR" rev-parse HEAD)"
  echo "sysmem fallback policy: <set manually: Prefer No Sysmem Fallback>"
  for k in "${SELECTED[@]}"; do sha256sum "${MODEL_FILE[$k]}"; done
} > "$OUT/env.txt" 2>&1

# --- background GPU telemetry (1 Hz) ---
nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used \
  --format=csv -l 1 > "$OUT/gpu_telemetry.csv" 2>/dev/null &
TELEM_PID=$!
trap 'kill $TELEM_PID 2>/dev/null' EXIT

run() {  # tag, name, args...
  local tag=$1 name=$2; shift 2
  echo ">> $tag $name $*"
  "$BENCH" -m "${MODEL_FILE[$tag]}" -ngl 99 -o jsonl "$@" \
      >> "$OUT/${tag}_${name}.jsonl" 2>> "$OUT/${tag}_${name}.stderr" \
    || echo "FAILED: $tag $name $*" | tee -a "$OUT/failures.txt"
}

for tag in "${SELECTED[@]}"; do
  # Prefill, small batches: short runs are noisy -> many reps
  run "$tag" prefill -p 1,4,16,64,128,256 -n 0 -fa 0,1 -r 20
  # Prefill, large batches: compute-bound regime
  run "$tag" prefill -p 512,1024,2048,4096 -n 0 -fa 0,1 -r 5

  # Decode vs depth: one invocation per point so an OOM can't kill later runs
  for fa in 0 1; do
    for d in ${DEPTHS[$tag]}; do
      run "$tag" decode_depth -p 0 -n 128 -fa "$fa" -d "$d" -r 5
    done
  done
done

echo "done -> $OUT"
[[ -f "$OUT/failures.txt" ]] && { echo "failures:"; cat "$OUT/failures.txt"; }