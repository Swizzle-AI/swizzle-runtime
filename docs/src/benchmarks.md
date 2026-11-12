# Benchmarks

Speed claims are only useful if you can verify them on your own hardware. This
page describes how Swizzle Runtime (`swizzle-rt`) is measured, how to repeat the
benchmarks, and exact observed results.

## Method

- **Same GGUF, same thread count, same machine.** Both engines run the exact
  same quantized GGUF file back-to-back with thermal rest intervals.
- **Plain throughput (`pp512`, `tg128`).** `pp512` processes a 512-token prompt
  (measuring prefill throughput and `ttft_ms`), and `tg128` generates 128 tokens
  with speculative drafting disabled (measuring decode tokens/sec and per-token
  inter-token latency `p50`, `p95`, and `p99`). Each throughput number is the
  best of the configured repetitions.
- **Drafted edit throughput (`edit256`) and exactness.** Generates 256 tokens
  from a code-renaming prompt using RAD-1 self-speculative decoding and asserts
  that drafted output matches plain decoding token-for-token (`edit_identical: true`).
- **Live HTTP parity and telemetry.** Starts `swizzle-rt serve` and verifies
  greedy `/v1/completions` output parity alongside the `X-Swizzle-Backend`,
  `X-Swizzle-TTFT-Ms`, `X-Swizzle-Tokens-Per-Sec`, and `X-Swizzle-Gen-Tokens`
  response headers.

## Reproduce it

### Head-to-head comparison harness

```sh
python3 scripts/bench_compare_engines.py \
  --swizzle target/release/swizzle-rt \
  --baseline path/to/baseline-engine \
  --model path/to/model.gguf \
  --device cuda --device cpu \
  --repetitions 5 --cooldown-secs 5 --max-temp-c 75 \
  --verify-server --out bench_results.json
```

### Standalone Swizzle Runtime benchmark

```sh
swizzle-rt bench path/to/model.gguf --device cuda --repetitions 5 --json
swizzle-rt bench path/to/model.gguf --device cpu --repetitions 5 --json
```

### Comparison against llama.cpp

```sh
python3 scripts/bench_vs_llamacpp.py model.gguf --llama-dir path/to/llama.cpp/bin -t 8
```

## Results

Measured back-to-back in WSL2 (Ubuntu 24.04) on an **AMD Ryzen 7 170** (16 logical / 8
physical cores) and **NVIDIA GeForce RTX 3050 Laptop GPU** (4 GB VRAM, `sm_86`,
CUDA 12.8, NVIDIA driver 591.84). Thermally guarded at max `75 °C` with 5-second
cooldowns (`repetitions = 5`, observed peak GPU temperature `56 °C`).

| Model | Size | Device | Test | Baseline (t/s) | Swizzle Runtime (t/s) | Delta | Swizzle TTFT / ITL (ms) | Edit Exact |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| `Qwen3-0.6B-Q8_0.gguf` | 0.60 GiB | RTX 3050 CUDA | `pp512` | 11550.31 | 11804.23 | +2.2% | TTFT 43.37 | yes |
| `Qwen3-0.6B-Q8_0.gguf` | 0.60 GiB | RTX 3050 CUDA | `tg128` | 196.84 | 209.70 | +6.5% | p50 4.84 / p95 5.61 / p99 5.88 | yes |
| `Qwen3-0.6B-Q8_0.gguf` | 0.60 GiB | RTX 3050 CUDA | `edit256` | 737.47 | 759.75 | +3.0% | - | yes |
| `Qwen3-0.6B-Q8_0.gguf` | 0.60 GiB | Ryzen 7 170 CPU | `pp512` | 240.19 | 254.90 | +6.1% | TTFT 2008.61 | yes |
| `Qwen3-0.6B-Q8_0.gguf` | 0.60 GiB | Ryzen 7 170 CPU | `tg128` | 24.76 | 24.37 | -1.6% | p50 39.75 / p95 52.90 / p99 74.11 | yes |
| `Qwen3-0.6B-Q8_0.gguf` | 0.60 GiB | Ryzen 7 170 CPU | `edit256` | 65.80 | 90.61 | +37.7% | - | yes |
| `LFM3-1.7B-Q4_K_M.gguf` | 1.02 GiB | RTX 3050 CUDA | `pp512` | 5282.85 | 4985.19 | -5.6% | TTFT 102.70 | yes |
| `LFM3-1.7B-Q4_K_M.gguf` | 1.02 GiB | RTX 3050 CUDA | `tg128` | 134.82 | 144.26 | +7.0% | p50 7.04 / p95 7.79 / p99 8.22 | yes |
| `LFM3-1.7B-Q4_K_M.gguf` | 1.02 GiB | RTX 3050 CUDA | `edit256` | 650.17 | 656.55 | +1.0% | - | yes |
| `LFM3-1.7B-Q4_K_M.gguf` | 1.02 GiB | Ryzen 7 170 CPU | `pp512` | 207.24 | 209.58 | +1.1% | TTFT 2442.95 | yes |
| `LFM3-1.7B-Q4_K_M.gguf` | 1.02 GiB | Ryzen 7 170 CPU | `tg128` | 16.15 | 18.06 | +11.8% | p50 55.85 / p95 79.38 / p99 97.39 | yes |
| `LFM3-1.7B-Q4_K_M.gguf` | 1.02 GiB | Ryzen 7 170 CPU | `edit256` | 83.87 | 82.25 | -1.9% | - | yes |

### Optimization Breakdown

1. **PR #1 — XOR Bank-Conflict-Free CUDA GEMM Tiling**: Replaced `+16B` padded
   shared-memory strides (`GEMM_ROW`) with power-of-two XOR chunk swizzling across
   `GemmTile32` / `GemmTile64` / `GemmTile128` (`-9.1%` SMEM footprint) plus
   `SWIZZLE_CUDA_ARCH` build targeting.
2. **PR #2 — Zero-Copy CPU & GPU-Resident Shortlist Screening**: Added
   `PreparedBatch` + `matmul_indexed_prepared` on CPU and `shortlist.cuh`
   inside CUDA graph execution (`Q8_0` / `Q6_K`, 128 MiB VRAM guard), boosting
   GPU `tg128` across both dense (`+6.5%`) and hybrid LFM3 (`+7.0%`) workloads.
3. **PR #3 — Lock-Light CPU Decode & Physical-Core Scheduling**: Eliminated
   per-token `Q8` workspace heap `malloc`/`free` via `SINGLE_WORKSPACE`,
   pinned single-token decode (`n = 1`) to physical cores (`for_each_decode`),
   and swapped `SessionIndex` to `FxHashMap`.
4. **PR #4 — Live HTTP Telemetry, ITL Percentiles & Clean Branding**: Exposed
   `X-Swizzle-*` headers on `/v1/chat/completions` & `/v1/completions`, added
   `ttft_ms` + `p50`/`p95`/`p99` ITL telemetry to `swizzle-rt bench`, and
   verified 4/4 live greedy `/v1/completions` parity.

*Note: Laptop CPU boost clocks fluctuate slightly across sequential runs in WSL2;
all rows above come verbatim from one contiguous 5-repetition cooled run.*
