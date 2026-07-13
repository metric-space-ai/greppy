# Historical Qwen3.5 MTP Q4_K_M Performance Snapshot

Date: 2026-07-12

This is a retained pre-contract snapshot, not current release evidence.
It used the embedded production asset, but the native prompt path processed
511 rows through batched prefill and held the final token out for decode. The
strict contract in `bench/inference_performance/` rejects that shape as PP511.
These values remain useful for regression archaeology only and must not be
presented as exact PP512 performance.

## Fixed inputs

- Model: `Qwen3.5-0.8B-MTP-Q4_K_M.gguf`
- SHA256: `d45e08ad7bb8787ae9b6f56b6915e8b44ac6e13c6b740fdc7bd591249209a72c`
- File size: `541903232` bytes
- Prompt/prefill label: legacy `PP512` (`511` native batch rows plus one decode row; not exact PP512)
- Decode: TG128
- Repetitions: 5
- Apple CPU: four performance workers
- x86 CPU: six physical P-cores, Linux CPU set `0,2,4,6,8,10`
- GPU: exactly one device
- Native Greppy engine commit: `4584a8f` (engine-identical to `6d6ab83`)
- llama.cpp macOS commit: `e3546c7948e3af463d0b401e6421d5a4c2faf565`
- llama.cpp Linux/CUDA commit: `ef2d770117db45b05aa7ecd1b0acca36370c5470`

## Median comparison

The historical engineering target was at least `1.05x` llama.cpp. This table
does not satisfy the current input and generation contracts, so none of its
ratios can pass or fail the current calibration. Greppy release acceptance is
based on production latency, semantic quality, robustness, and measured agent
outcomes rather than superiority over another inference engine.

| Platform | Device | llama PP512 | Native PP512 | Ratio | llama TG128 | Native TG128 | Ratio | Historical calibration |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Apple CPU | Apple M5, 4 P workers | 329.41 | 382.92 | 1.162x | 83.83 | 75.04 | 0.895x | FAIL: decode |
| x86 CPU | i5-13400F, 6 physical P-cores | 374.33 | 260.04 | 0.695x | 71.82 | 53.86 | 0.750x | FAIL: prefill + decode |
| Metal | Apple M5, one GPU | 3859.89 | 3650.66 | 0.946x | 52.53 | 114.52 | 2.180x | FAIL: prefill |
| CUDA | RTX A4500, GPU 0 only | 14172.20 | 14851.81 | 1.048x | 373.85 | 390.31 | 1.044x | FAIL: below 1.05x |

CUDA is faster at the median but has not passed the formal gate. Its first
native PP sample was `13295.71 tok/s`, below the llama.cpp PP median, so the
no-regression requirement also remains open.

## Raw samples

Values are tokens per second in execution order.

| Platform | Engine | PP512 samples | TG128 samples |
|---|---|---|---|
| Apple CPU | llama.cpp | 272.871, 306.144, 340.714, 333.645, 329.409 | 83.8293, 68.8692, 86.4764, 80.9310, 88.3897 |
| Apple CPU | Greppy native | 382.98, 382.92, 380.57, 383.16, 379.56 | 71.73, 75.04, 74.35, 80.37, 77.60 |
| x86 CPU | llama.cpp | 249.034, 394.425, 374.329, 388.603, 244.554 | 69.6063, 71.6765, 74.2323, 72.3838, 71.8204 |
| x86 CPU | Greppy native | 315.02, 260.04, 237.13, 246.29, 276.18 | 52.61, 53.86, 53.32, 54.32, 58.98 |
| Metal | llama.cpp | 3859.89, 4095.43, 2262.92, 3441.55, 4229.93 | 45.0718, 50.4336, 52.5327, 102.062, 128.459 |
| Metal | Greppy native | 3480.97, 3656.26, 3661.06, 3622.89, 3650.66 | 114.52, 113.24, 117.27, 114.30, 115.39 |
| CUDA | llama.cpp | 12366.0, 13850.1, 14172.2, 14397.9, 14330.6 | 372.531, 373.762, 373.849, 374.071, 373.924 |
| CUDA | Greppy native | 13295.71, 14877.70, 14856.93, 14851.81, 14785.88 | 390.31, 390.39, 390.81, 390.29, 389.91 |

The Apple and x86 hosts had observable system-load variance. Acceptance must
therefore be rerun on otherwise idle hosts and retain all raw samples. The
current failures are large enough that variance does not change their status.

## Command contract

llama.cpp CPU:

```sh
llama-bench -m "$MODEL" -p 512 -n 128 -r 5 -t "$P_CORES" -ngl 0 -o jsonl
```

On the hybrid Linux host the entire command is pinned to one hardware thread
from each physical P-core:

```sh
taskset -c 0,2,4,6,8,10 llama-bench ...
```

llama.cpp GPU:

```sh
CUDA_VISIBLE_DEVICES=0 llama-bench -m "$MODEL" -p 512 -n 128 -r 5 \
  -t 6 -ngl 99 -sm none -o jsonl
```

Greppy native CPU:

```sh
QWEN35_NATIVE_GGUF="$MODEL" \
QWEN35_NATIVE_TOKENIZER="$TOKENIZER" \
QWEN35_NATIVE_PERF_INPUT_TOKENS=512 \
QWEN35_NATIVE_PERF_OUTPUT_TOKENS=128 \
cargo test --release -p greppy-qwen35-native \
  model::cpu_perf_tests::qwen35_cpu_perf_prints_when_env_set \
  -- --exact --ignored --nocapture
```

Greppy native Metal uses `--features metal` and
`model::metal_perf_tests::qwen35_metal_perf_reports_backend_status_when_env_set`.
Greppy native CUDA uses `--features cuda` and
`model::tests::qwen35_cuda_perf_prints_when_env_set`, with
`CUDA_VISIBLE_DEVICES=0`. Development builds on the A4500 may set
`CUDA_ARCH_LIST=86`; release artifacts retain the supported architecture
matrix.

## Correctness status

Before these measurements, the current MTP asset passed:

- Metal Q4_K matvec versus CPU reference;
- Metal batched prefill versus tokenwise forward;
- Metal MTP generation versus target generation;
- CUDA production perf path on one enumerated device.

The vendored Metal source intentionally retains Greppy's fusion kernels. A
wholesale sync to current llama.cpp was rejected after it removed the embedded
MTP `kernel_concat` contract. Future upstream imports must be isolated to one
kernel family and pass the complete MTP parity suite.
