# Qwen3.5 llama.cpp Reference Baseline

Date: 2026-07-09

Model: `Qwen3.5-0.8B-Q4_K_M.gguf`

SHA256: `f5b14da98939b60bbe1019a964eba656407e1e0b64f1fe3003ff6d650e93bfec`

Metal benchmark command shape:

```sh
llama-bench -m Qwen3.5-0.8B-Q4_K_M.gguf -ngl 99 -p 512 -n 128 -r 5 -o json
```

CPU benchmark command shape:

```sh
llama-bench -m Qwen3.5-0.8B-Q4_K_M.gguf -ngl 0 -p 512 -n 128 -r 3 -o json
```

CUDA was measured on one GPU only:

```sh
llama-bench -m Qwen3.5-0.8B-Q4_K_M.gguf -dev CUDA0 -sm none -ngl 99 -p 512 -n 128 -r 5 -o json
```

| Host | Backend | Device | llama.cpp build | PP 512 input tok/s | TG 128 output tok/s |
|---|---|---|---:|---:|---:|
| Mac | CPU | Apple M5, 4 threads, `n_gpu_layers=0` | 9060 `ad0922465` | 201.59 | 61.24 |
| Mac | Metal | Apple M5 `MTL0`, `n_gpu_layers=99` | 9060 `ad0922465` | 2743.54 | 74.52 |
| gpu3 | CUDA | NVIDIA RTX A4500 `CUDA0` | 1 `ef2d770` | 14103.67 | 386.51 |

The Mac CPU and Metal rows above were remeasured sequentially on 2026-07-09.
The CPU row still initializes the Metal backend at process startup, but the JSON
evidence reports `n_gpu_layers=0`.

## Native Rust CUDA Snapshot

Date: 2026-07-09

Command shape:

```sh
QWEN35_NATIVE_GGUF=/tmp/gp-qwen-assets/Qwen3.5-0.8B-Q4_K_M.gguf \
QWEN35_NATIVE_TOKENIZER=/tmp/gp-qwen-assets/tokenizer.json \
QWEN35_NATIVE_PERF_INPUT_TOKENS=512 \
QWEN35_NATIVE_PERF_OUTPUT_TOKENS=128 \
cargo test --release -p greppy-qwen35-native --features cuda \
  qwen35_cuda_perf_prints_when_env_set -- --ignored --nocapture
```

| Host | Backend | Device | Input tokens | Input tok/s | Output tokens | Output tok/s |
|---|---|---|---:|---:|---:|---:|
| gpu3 | Native Rust CUDA | NVIDIA RTX A4500 `CUDA0` | 511 | 301.06 | 128 | 259.94 |
| gpu3 | Native Rust CUDA | NVIDIA RTX A4500 `CUDA0` | 511 | 299.50 | 128 | 259.91 |
| gpu3 | Native Rust CUDA, GPU greedy argmax | NVIDIA RTX A4500 `CUDA0` | 511 | 297.28 | 128 | 301.70 |
| gpu3 | Native Rust CUDA, GPU greedy argmax | NVIDIA RTX A4500 `CUDA0` | 511 | 300.11 | 128 | 301.81 |
| gpu3 | Native Rust CUDA, no-logits prefill + GPU greedy argmax | NVIDIA RTX A4500 `CUDA0` | 511 | 367.80 | 128 | 298.55 |
| gpu3 | Native Rust CUDA, no-logits prefill + GPU greedy argmax | NVIDIA RTX A4500 `CUDA0` | 511 | 373.50 | 128 | 298.94 |
| gpu3 | Native Rust CUDA, no-logits prefill + GPU greedy argmax | NVIDIA RTX A4500 `CUDA0` | 511 | 369.87 | 128 | 298.88 |
| gpu3 | Native Rust CUDA, cache-only final prefill layer + GPU greedy argmax | NVIDIA RTX A4500 `CUDA0` | 511 | 382.49 | 128 | 299.25 |
| gpu3 | Native Rust CUDA, cache-only final prefill layer + GPU greedy argmax | NVIDIA RTX A4500 `CUDA0` | 511 | 384.31 | 128 | 298.72 |
| gpu3 | Native Rust CUDA, cache-only final prefill layer + GPU greedy argmax | NVIDIA RTX A4500 `CUDA0` | 511 | 386.85 | 128 | 298.83 |
| gpu3 | Native Rust CUDA, Qwen add-RMSNorm fusion + cache-only final prefill layer | NVIDIA RTX A4500 `CUDA0` | 511 | 401.20 | 128 | 307.93 |
| gpu3 | Native Rust CUDA, Qwen add-RMSNorm fusion + cache-only final prefill layer | NVIDIA RTX A4500 `CUDA0` | 511 | 398.06 | 128 | 307.54 |
| gpu3 | Native Rust CUDA, Qwen add-RMSNorm fusion + cache-only final prefill layer | NVIDIA RTX A4500 `CUDA0` | 511 | 401.19 | 128 | 307.10 |
| gpu3 | Native Rust CUDA, Qwen add-RMSNorm fusion rerun | NVIDIA RTX A4500 `CUDA0` | 511 | 401.40 | 128 | 306.46 |

## Native Rust CPU Snapshot

Date: 2026-07-09

Command shape:

```sh
QWEN35_NATIVE_GGUF=.../Qwen3.5-0.8B-Q4_K_M.gguf \
QWEN35_NATIVE_TOKENIZER=.../tokenizer.json \
QWEN35_NATIVE_PERF_INPUT_TOKENS=512 \
QWEN35_NATIVE_PERF_OUTPUT_TOKENS=128 \
cargo test --release -p greppy-qwen35-native \
  qwen35_cpu_perf_prints_when_env_set -- --ignored --nocapture
```

| Host | Backend | Device | Input tokens | Input tok/s | Output tokens | Output tok/s |
|---|---|---|---:|---:|---:|---:|
| Mac | Native Rust CPU reference, no-logits prefill | Apple M5 | 511 | 6.38 | 128 | 4.04 |

## Native Rust Metal Snapshot

Date: 2026-07-09

Command shape:

```sh
QWEN35_NATIVE_GGUF=.../Qwen3.5-0.8B-Q4_K_M.gguf \
QWEN35_NATIVE_TOKENIZER=.../tokenizer.json \
cargo test --release -p greppy-qwen35-native --features metal \
  qwen35_metal_perf_reports_backend_status_when_env_set -- --ignored --nocapture
```

| Host | Backend | Device | Input tokens | Input tok/s | Output tokens | Output tok/s |
|---|---|---|---:|---:|---:|---:|
| Mac | Native Rust Metal, tokenwise forward + CPU logits argmax | Apple M5 `MTL0` | 31 | 70.71 | 4 | 75.10 |
| Mac | Native Rust Metal, tokenwise forward + CPU logits argmax | Apple M5 `MTL0` | 511 | 119.84 | 128 | 87.60 |
| Mac | Native Rust Metal, tokenwise forward + GPU greedy argmax | Apple M5 `MTL0` | 511 | 148.36 | 128 | 99.31 |
| Mac | Native Rust Metal, experimental batched prefill + GPU greedy argmax | Apple M5 `MTL0` | 511 | 392.02 | 128 | 60.65 |

Evidence output:

```text
qwen35_native_metal_perf backend=metal-q4k-forward input_tokens=511 input_s=3.444227 input_tok_s=148.36 output_tokens=128 output_s=1.288945 output_tok_s=99.31
qwen35_native_metal_perf backend=metal-q4k-forward input_tokens=511 input_s=1.303501 input_tok_s=392.02 output_tokens=128 output_s=2.110522 output_tok_s=60.65
```

## CTOX Archive Lessons Applied

Sources checked:

- `/Users/michaelwelsch/Documents/ctox.nosync/archive/qwen35_08b_metal_probe`
- `/Users/michaelwelsch/Documents/ctox.nosync/archive/qwen36_35b_a3b_q4km_metal`
- `/Users/michaelwelsch/Documents/ctox.nosync/archive/2026-05-26-worktree-cleanup/inference-experimental/qwen36_27b_q4km_cuda`
- `/Users/michaelwelsch/Documents/ctox.nosync/archive/qwen35_27b_q4km_dflash`

Transferred lessons:

- Metal prefill wins come from SIMDgroup/tensor-shaped `mul_mm`, not from
  repeating decode `mul_mv` per token. The 35B archive accepted
  `kernel_mul_mm_q4_K_f32` with `-DGGML_METAL_HAS_TENSOR -std=metal4.0`
  for N >= 32 because it uses the Apple matrix/tensor path.
- `greppy-embed-native` already builds both a base Metal 3.1 metallib and an
  optional Metal 4.0 tensor metallib, and its `op_mul_mm` dispatcher selects
  that tensor library on M5. Qwen35 has an experimental batched prefill path
  that calls it, but production `generate()` still gates that path until the
  multi-layer numeric drift is resolved or accepted explicitly.
- Metal device initialization now uses a single `OnceLock<Option<Device>>` and
  unique temp metallib names. The old race could make parallel tests overwrite
  the embedded tensor metallib temp file before `newLibraryWithURL` loaded it.
- Qwen35 now has an explicit Q4_K `mul_mm` parity test on
  `blk.0.attn_qkv.weight` with 16 activation rows:
  `cosine=0.99999994`, `rms_rel=4.345975e-4`, `max_abs=5.084038e-3`.
  This verifies the vendor SIMD/tensor matrix path against the real Qwen GGUF.
- Qwen35 Metal now has row kernels for batched prompt prefill: causal conv,
  DeltaNet scan, Q/K normalization, RoPE, KV cache writes, causal attention
  scores/softmax/values, strided gate, and strided RMSNorm. The path reaches
  392 tok/s input on M5, but a len>1 parity probe still drifts against the
  tokenwise Metal decode path (`len=2 cosine=0.91966438`), so it is not enabled
  by default in `generate()`.
- The suspected row primitives were isolated on 2026-07-09 and passed against
  CPU/sequential decode references: causal-conv rows, strided RMSNorm rows,
  and Full-Attention rows. Complete block-level diagnostics also passed with
  small per-block drift: Delta block rows vs sequential decode
  `cosine=0.99996358`, `rms_rel=8.616260e-3`, and Full-Attention block rows
  vs sequential decode `cosine=0.99996722`, `rms_rel=8.264119e-3`.
- The remaining Metal len>1 drift is therefore no longer pinned on an
  obviously wrong row-state kernel. It is cumulative `mul_mm`/`mul_mv`
  numerical divergence across many Q4_K layers, or it must be accepted as the
  same class of batched-prefill/decode non-bitexactness that llama.cpp uses.
- GPU-local argmax is required for greedy/triage decode. Metal now mirrors the
  CUDA path by running `kernel_argmax_f32` after the LM head and reading back
  only the token id for `temperature=0, top_k=1`.
- The 0.8B archive's approximate DeltaNet SIMD variants are not promoted here:
  `lanes4_sharedqk` improved speed but had hidden-state drift, and
  `gated_norm_simd32x4` regressed or drifted. Only exact paths are acceptable
  for the default summarizer.
- The 27B CUDA archive confirms the same CUDA rule: once the ggml Q4_K matvec
  is at the byte floor, performance comes from scheduling and batching
  changes, not polishing scalar glue kernels.

Notes:

- `PP 512` is prompt/prefill throughput, used as the input-token baseline.
- `TG 128` is decode throughput, used as the output-token baseline.
- Flash attention was not enabled explicitly. Mac reported `flash_attn=false`; CUDA reported `flash_attn=-1`.
- CUDA host had three RTX A4500 cards, but this baseline pins to `CUDA0` with `split_mode=none`.
- The native Rust CPU backend is currently scalar/reference-speed and is the
  mandatory fallback when no GPU backend is available. Its throughput is not
  yet acceptable for release and must be replaced by batched SIMD Q4_K
  execution.
  The CPU perf rows use the same no-logits prompt prefill split as the
  accelerated path, so prompt tokens do not run output norm or the vocab LM head.
- Native Rust Metal now has a tokenwise full-forward path and an experimental
  batched prefill path. The tokenwise path remains the default for correctness,
  `brief`, and greedy triage. The batched path proves that the SIMD/tensor
  `mul_mm` route can raise input throughput, but it remains gated until the
  row-state parity drift is fixed.
- The native Rust CUDA input-token rows before "no-logits prefill" are the old
  decode-only prompt path that also ran output norm, LM head, and full logits
  download for each prompt token.
- The "no-logits prefill" rows skip output norm, LM head, and logits download
  for non-final prompt tokens. This is a real warm inference measurement of the
  native prompt state-update path, but it is still token-by-token and not yet
  architecturally comparable to llama.cpp batched `PP 512`.
- The "cache-only final prefill layer" rows additionally skip Q projection,
  Q-normalization, Q-RoPE, attention scores/softmax/values, attention output,
  and FFN work in the final layer for non-final prompt tokens. The cache-only
  path is validated by comparing later logits after optimized prefill against
  the full-logits state path.
- The "Qwen add-RMSNorm fusion" rows add a dedicated f32 CUDA kernel for
  `RMSNorm(hidden + sublayer_out)` and use it after attention. This removes
  separate residual copies, add kernels, and post-attention RMSNorm launches
  from the Qwen layer flow while preserving the full-logits parity test.
- The GPU greedy argmax rows avoid copying the full vocabulary logits back to
  the CPU for `temperature=0, top_k=1`; this is the path used by semantic
  triage on CUDA and Metal. The non-greedy `brief` sampler still needs either
  CPU logits or a future GPU top-k/top-p sampler.
