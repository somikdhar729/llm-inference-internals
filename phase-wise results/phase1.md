# Phase 1 — Performance Baseline (llama.cpp)

**Status:** Closed (2026-10-25). Open items listed at the end.

## Goal

Establish a reproducible llama.cpp baseline for Qwen3-1.7B on this machine, and explain decode speed against the memory-bandwidth ceiling.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB GDDR6, 192-bit, 14 Gbps |
| Spec bandwidth | 336 GB/s (memory clock 7001 MHz) |
| Power limit | 100 W (laptop TGP) |
| CPU | Intel i7-12700H |
| OS / driver | Windows, WDDM, driver 591.44, CUDA 13.1 |
| llama.cpp | `7951f601e89494f2ed12dbd4eb52e254b8401197` |
| Driver setting | CUDA Sysmem Fallback Policy = **Prefer No Sysmem Fallback** |

### Models

| File | Source | Params (as stored) | Size | SHA256 |
|---|---|---|---|---|
| `qwen3-1.7b-bf16.gguf` | Converted from `Qwen/Qwen3-1.7B` safetensors with `convert_hf_to_gguf.py --outtype bf16` | 2,031,739,904 | 4.06 GB | `5db34fe0…1d839a` |
| `qwen3-1.7b-q8_0.gguf` | Official `Qwen/Qwen3-1.7B-GGUF` | 1,720,574,976 | 1.83 GB | `061b54da…590cb1a` |

Full hashes, including the source safetensors: `models/hashes.txt`.

**Note on parameter counts:** the model has 1,720,574,976 parameters. The BF16 file stores 311,164,928 extra because Qwen's checkpoint contains a separate `lm_head.weight` (despite `tie_word_embeddings: true`), and the converter copies it as `output.weight`. The two matrices were verified bit-identical. Effect: no change in decode speed; +0.62 GB VRAM (about 5k tokens of context on this card).

## Model facts used in the analysis

From `config.json`: 28 layers, hidden size 2048, 16 Q heads, 8 KV heads, head_dim 128, intermediate size 6144, vocab 151,936, tied embeddings, QK-norm.

| Quantity | Value |
|---|---|
| Weight bytes read per decode token, BF16 | 3.441 GB |
| Weight bytes read per decode token, Q8_0 | 1.828 GB |
| KV cache per token (f16) | 114,688 B (112 KiB) |
| MLP share of per-layer weights | 75% |
| LM head share of all weights | 18% |

The KV size was confirmed exactly against llama.cpp's own allocation: 3,787,456,512 B = 114,688 B × 33,024 tokens (32,768 depth + 128 generated, padded to a multiple of 256).

## Method

- Tool: `llama-bench`, `-ngl 99`, JSON-lines output, per-repetition samples kept.
- Prefill: `-p 1…4096 -n 0`; 20 repetitions for p ≤ 256, 5 for larger.
- Decode: `-p 0 -n 128 -d <depth>`, 5 repetitions, one invocation per point so an out-of-memory failure can't abort later points.
- FA0 and FA1 (flash attention off/on) both measured.
- GPU telemetry logged at 1 Hz during each run.
- Reported values are medians.

## Results

### Decode vs depth (FA1, tok/s, median)

| Depth | BF16 | Q8_0 |
|---|---|---|
| 0 | 75.7 | 125.5 |
| 512 | 74.2 | 124.3 |
| 1024 | 73.8 | 121.1 |
| 2048 | 71.3 | 115.2 |
| 4096 | 68.0 | 105.6 |
| 8192 | 60.9 | 90.8 |
| 16384 | 51.8 | 70.7 |
| 24576 | OOM | 58.1 |
| 32768 | OOM | OOM |

### Decode model

All FA1 points for both formats are fit by:

```
time per token = 1.91 ms + bytes_read / 304 GB/s
bytes_read     = weight_bytes + 114,688 × depth
```

Maximum error: 0.7%.

Interpretation:
- llama.cpp's kernels stream memory at **304 GB/s, about 90% of spec**.
- **1.91 ms per token is fixed overhead**, not spent reading memory: 14% of decode time for BF16 and 23% for Q8_0 at depth 0. This is why Q8_0 is only 1.65× faster than BF16 instead of the 1.88× its byte ratio predicts.
- Hypothesis for Phase 4: in an Nsight Systems trace, the sum of GPU kernel time per token should be about 1.9 ms less than wall-clock time per token.

### Prefill (FA1, tok/s, median)

| Prompt tokens | BF16 | Q8_0 |
|---|---|---|
| 1 | 76 | 127 |
| 16 | 1,012 | 1,444 |
| 64 | 3,064 | 4,043 |
| 256 | 4,477 | 6,407 |
| 512 | 5,218 | 6,917 |
| 1024 | 5,218 | 6,825 |
| 4096 | 4,732 | 5,937 |

- Up to about 64 tokens, prefill is still bandwidth-bound: throughput rises almost linearly with batch size.
- It saturates around 512–1024 tokens. BF16 peak ≈ 5.2k tok/s ≈ 18 TFLOPS (at ~2 × 1.72 GFLOP/token).
- Q8_0 prefill is ~30% faster than BF16, likely because llama.cpp's quantized matmuls use INT8 tensor cores.
- Decline beyond 1024 tokens is attention cost; much steeper with FA0.
- p ≤ 64 is noisy (up to ±18% even with 20 repetitions). Treat those points as approximate.

### Flash attention

FA1 is equal or better everywhere: about 29% faster decode at 16k (Q8_0: 70.7 vs 50.2 tok/s) and needs less memory, so it fits deeper contexts. **FA1 is the reference configuration for all later comparisons.**

### Memory limits (6 GB)

| Model | FA0 max depth | FA1 max depth |
|---|---|---|
| BF16 | 8k | 16k |
| Q8_0 | 16k | 24k |

Peak VRAM observed: 5,579 MiB (BF16, FA1, 16k), 91% of the card.

## Reproducibility

| Session | Start temp | Result |
|---|---|---|
| Run 1 | 51°C | Baseline |
| Run 2 (BF16) | **67°C**, straight after another run | 8–21% slower, noisy, non-monotonic in depth. **Discarded.** |
| Run 2 (Q8_0) | 52°C | Within 1–3% of run 1 |
| Run 3 (BF16, cool) | 44°C | Within ±2% of run 1; refit 1.91 ms / 304 GB/s vs 1.94 ms / 304.6 GB/s |

**Cross-session variation: about ±2% when starting cool.**

Telemetry during run 3: the memory clock stayed at 7001 MHz throughout, while the SM clock varied from 1080 to 1972 MHz, power sat at the 100 W limit and temperature reached 84°C. Decode depends on the memory clock, which explains why it stays stable under load. Prefill depends on the SM clock, so **expect prefill to be more heat-sensitive than decode.**

## Pitfalls found

1. **Silent VRAM spill.** With the Windows default sysmem fallback policy, configurations exceeding 6 GB ran from system RAM over PCIe instead of failing. BF16 FA0 at 32k ran at 1.26 tok/s (looked stuck); BF16 FA0 at 16k ran at 39 tok/s and looked like a plausible measurement. Fixed with *Prefer No Sysmem Fallback*. Earlier spilled results were discarded.
2. **llama-bench `-p` doesn't set decode context.** `-p` and `-n` are separate tests; only `-d` puts context in the KV cache before decoding.
3. **Re-saved checkpoints.** Loading and re-saving a model with transformers 5.x rewrites `config.json` (e.g. `rope_parameters` instead of `rope_theta`). Always convert from the original files.

## Benchmark protocol (from now on)

1. Laptop plugged in, high-performance mode, nothing else running.
2. Start each session with the GPU at about 50°C or below.
3. When comparing A vs B, alternate (A, B, A, B) instead of running all A then all B.
4. Check the telemetry after each run; discard runs where the memory clock dropped.
5. Report medians; keep raw per-repetition samples.

## Open items

- **Phase 3, task 1:** measure achievable VRAM bandwidth with a standalone read kernel. Compare against llama.cpp's 304 GB/s to decide whether Phase 5 targets GEMV bandwidth or the per-token overhead.
- **Phase 4:** test the overhead hypothesis (≈1.9 ms/token of non-kernel time) with Nsight Systems.

## Files

```
results/
  baseline_<timestamp>/        raw jsonl, stderr, env.txt, gpu_telemetry.csv, failures.txt
  phase1.md                    this file
models/hashes.txt
scripts/bench.sh
```
