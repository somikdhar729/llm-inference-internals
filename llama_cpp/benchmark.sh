#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLAMA_CPP_DIR="/e/llama.cpp"
BENCH="$LLAMA_CPP_DIR/build/bin/Release/llama-bench.exe"
MODEL="$PROJECT_DIR/models/gguf/Qwen_Qwen3-1.7B-Q4_K_M.gguf"

RESULTS_DIR="$PROJECT_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$RESULTS_DIR/benchmark_BASELINE_$TIMESTAMP.log"

mkdir -p "$RESULTS_DIR"

if [[ ! -f "$BENCH" ]]; then
    echo "Error: Benchmark executable not found at $BENCH"
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Error: Model file not found at $MODEL"
    exit 1
fi

run_benchmark(){
    echo "Command: $*"
    "$@" 2>&1
}

{
    echo "LLM Inference Benchmark Baseline(llama.cpp) - $(date)"
    echo "Model: $MODEL"
    echo "Runtime: $BENCH"
    
    echo "Prefill benchmark..."
    for prompt_tokens in 1 2 4 8 16 32 64 128 512 1024 2048; do
        run_benchmark "$BENCH" -m "$MODEL" -ngl 99 -p "$prompt_tokens" \
            -n 0 -r 3
    done

    echo "Decode context sweep"
    # n is held constant to isolate the effect of p
    for p in 1 2 4 8 16 32 64 128 512 1024 2048; do
        run_benchmark "$BENCH" -m "$MODEL" -ngl 99 -p "$p" \
            -n 128 -r 3
    done

    echo "Generation length sweep"
    # p is held constant to isolate the effect of n
    for n in 1 2 4 8 16 32 64 128 512 1024 2048; do
        echo "Prompt: 512; generation: $n"
        run_benchmark "$BENCH" -m "$MODEL" -ngl 99 -p 512 \
            -n "$n" -r 3
    done

    echo "Benchmark completed at $(date)"
} 2>&1 | tee "$LOG"

echo "Benchmark results saved to $LOG"