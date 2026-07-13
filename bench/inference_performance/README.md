# Native inference performance contract

This harness is the release contract for native Greppy inference versus a
pinned llama.cpp checkout. It does not accept the human-readable output from
`llama-bench`, and it does not contain historical or synthetic measurements.
All run artifacts belong under the repository's ignored `dev/` directory.

## Current status

EmbeddingGemma and production Greppy prompt producers are implemented. The
strict verifier covers every workload and all four release platforms. Qwen's
typed diagnostic API now accepts exact supplied token IDs, target-prefills all
512 rows, and returns 128 committed EOS-disabled greedy tokens from the
production MTP path together with separate stage timings.

No complete four-platform result set has passed the verifier yet. Until paired
native and pinned llama.cpp artifacts are collected on quiet release hardware,
the historical table in `bench/qwen35_llama_cpp_baseline.md` is regression
context only. Missing baselines, PP511 records, target-only TG128, or mismatched
hardware continue to fail closed.

## Workloads

Every gated case has at least five warm-cache raw samples per engine. Model
load, tokenization, and warmup are outside each timed interval.

### `qwen_pp512`

- Exactly 512 supplied token IDs are processed in one target-model prefill.
- All 512 rows are timed. Holding the last token out for decode is PP511 and is
  invalid.
- `output_tokens` is zero and `generation_path` is `target_prefill` for both
  engines.
- Native and llama.cpp input token hashes must match.

### `qwen_tg128`

TG128 is separate from PP512 and means 128 committed output tokens, not 128
arbitrary target-only forward calls:

1. The identical supplied prompt IDs are prefixed outside the TG timer.
2. Greedy sampling is used and EOS is ignored until 128 outputs are committed.
3. The first output comes from the final prefill logits.
4. The timed interval includes greedy selection and the 127 subsequent decode
   evaluations needed to commit outputs 2 through 128.
5. Native must use `production_mtp`; llama.cpp uses
   `target_greedy_reference`.
6. Every committed output ID must match, sample by sample.

This differs intentionally from `llama-bench -n 128`, which times 128 random
single-token decode calls and does not expose the IDs. A native target-only
loop is not a production MTP result.

### `embedding_encoder`

- The native producer applies the production EmbeddingGemma document prompt
  and tokenizes once.
- Native and llama.cpp process those exact IDs with non-causal attention and
  materialize the pooled embedding.
- Each raw sample records both encoder latency and token throughput. The gate
  uses throughput; latency remains available for operational analysis.
- `output_tokens` and `output_limit` are zero. TG is not applicable to an
  encoder and any embedding TG record is invalid.

### `greppy_brief`

- The committed fixture contains realistic Rust, Python, TypeScript, and Go
  definition spans.
- The actual production chat prompt must tokenize to 100-500 IDs.
- The public production summarizer and MTP model are mandatory.
- Each generation cap is at most 64 tokens.

This workload is mandatory evidence that the optimized engine still exercises
the production Greppy path. It is native-only and is not mislabeled as a
llama.cpp MTP comparison.

## Gate

The verifier evaluates every platform/workload/case independently:

- platforms: `apple_cpu`, `x86_cpu`, `metal`, and `cuda`;
- median native throughput / median llama.cpp throughput must be at least
  `1.05x`, without rounded passes; and
- the slowest native sample / median llama.cpp throughput must be at least
  `1.00x`.

A missing binary, failed producer process, empty producer output, missing
llama.cpp case, metadata mismatch, or invalid sample fails closed.

## Provenance and parity

`collect.py` appends the following to every raw sample:

- SHA-256 of the exact native or llama.cpp binary;
- SHA-256 of the exact GGUF and tokenizer files;
- SHA-256 of tracked plus non-ignored source content, including dirty files;
- canonical-JSON SHA-256 of static hardware metadata; and
- SHA-256 of length-prefixed little-endian `u32` token IDs.

The verifier requires the same model, tokenizer, hardware, input IDs, thread
count, P-core set, selected device, and visible device set for each native and
llama.cpp pair. Source and binary hashes are stable per engine but are expected
to differ between engines.

Hardware JSON must contain static facts only, for example:

```json
{
  "system": "macOS arm64",
  "cpu": "Apple M5",
  "memory_bytes": 34359738368,
  "p_cores": ["p0", "p1", "p2", "p3"],
  "gpus": [{"id": "0", "name": "Apple M5", "cores": 10}]
}
```

Do not include temperatures, free memory, clocks, timestamps, paths, or other
values that can change between paired runs.

## CPU and GPU isolation

- Apple CPU: use exactly the physical performance-core count. Both native
  pools and the llama.cpp driver select user-interactive performance QoS; the
  driver fails if `--threads` differs from `hw.perflevel0.physicalcpu`.
- Linux x86 CPU: launch collection under `taskset` with one logical processor
  from each physical P-core. The llama.cpp driver fails unless inherited
  affinity exposes exactly `--threads` processors.
- Windows x86 CPU: set process affinity to one logical processor per physical
  P-core before collection. The driver checks the inherited affinity count.
- Metal: the llama.cpp driver requires exactly one enumerated GPU.
- CUDA: set `CUDA_VISIBLE_DEVICES` to one ID. The collector rejects a conflict,
  and the driver fails unless exactly one GPU is enumerated.

For GPU runs, use the same P-core list and host thread count for both engines.
The collector requires one P-core entry per thread.

## Build

Build native producers from the exact source under test:

```sh
cargo build --release -p greppy-qwen35-native \
  --example qwen_inference_contract --features metal
cargo build --release -p greppy-embed-native \
  --example embedding_inference_contract --features metal
```

Use `--features cuda` on CUDA hosts and no feature for CPU-only builds. Keep
copied binaries and all build experiments under `dev/inference-performance/`.

Build the exact-token llama.cpp driver against a pinned source checkout:

```sh
cmake -S bench/inference_performance \
  -B dev/inference-performance/llama-build \
  -DLLAMA_CPP_SOURCE="$LLAMA_CPP_SOURCE" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON
cmake --build dev/inference-performance/llama-build \
  --target greppy-llama-contract -j
```

Use `-DGGML_METAL=OFF` for CPU-only builds or `-DGGML_CUDA=ON` for CUDA. The
source checkout itself must be pinned; its complete source content is hashed
during collection.

## Collect EmbeddingGemma

The native run emits exact production prompt IDs. The llama.cpp run reads
those IDs from the resulting JSONL before the collector appends its samples.
Variables below are illustrative paths, not measurements:

```sh
RESULT=dev/inference-performance/apple-embedding.jsonl
PROMPTS=bench/inference_performance/greppy_prompts.jsonl

python3 bench/inference_performance/collect.py \
  --run-id "$RUN_ID" --platform apple_cpu --engine native \
  --model-family embeddinggemma \
  --binary "$NATIVE_EMBED_BIN" --model "$EMBED_MODEL" \
  --tokenizer "$EMBED_TOKENIZER" --source-root "$GREPPY_SOURCE" \
  --hardware "$HARDWARE_JSON" --threads 4 \
  --p-core-set p0,p1,p2,p3 \
  --device-kind cpu --device-id cpu --gpu-count 0 \
  --output "$RESULT" -- \
  "$EMBED_MODEL" "$EMBED_TOKENIZER" "$PROMPTS" cpu 5 1

python3 bench/inference_performance/collect.py \
  --run-id "$RUN_ID" --platform apple_cpu --engine llama.cpp \
  --model-family embeddinggemma \
  --binary "$LLAMA_CONTRACT_BIN" --model "$EMBED_MODEL" \
  --tokenizer "$EMBED_TOKENIZER" --source-root "$LLAMA_CPP_SOURCE" \
  --hardware "$HARDWARE_JSON" --threads 4 \
  --p-core-set p0,p1,p2,p3 \
  --device-kind cpu --device-id cpu --gpu-count 0 \
  --output "$RESULT" -- \
  --model "$EMBED_MODEL" --cases "$RESULT" \
  --model-family embeddinggemma --workload embedding_encoder \
  --device cpu --threads 4 --samples 5 --warmups 1
```

For CUDA, pass `--device-kind cuda --device-id 0 --gpu-count 1
--visible-gpu-id 0` and use `cuda` in the producer arguments. Metal uses the
same shape with `metal` and one visible GPU ID.

## Collect production Greppy prompts

```sh
python3 bench/inference_performance/collect.py \
  --run-id "$RUN_ID" --platform apple_cpu --engine native \
  --model-family qwen35_mtp \
  --binary "$NATIVE_QWEN_BIN" --model "$QWEN_MODEL" \
  --tokenizer "$QWEN_TOKENIZER" --source-root "$GREPPY_SOURCE" \
  --hardware "$HARDWARE_JSON" --threads 4 \
  --p-core-set p0,p1,p2,p3 \
  --device-kind cpu --device-id cpu --gpu-count 0 \
  --output dev/inference-performance/apple-greppy.jsonl -- \
  "$QWEN_MODEL" "$QWEN_TOKENIZER" \
  bench/inference_performance/greppy_prompts.jsonl cpu 5 1
```

The exact-token llama.cpp driver supports `qwen_pp512` and `qwen_tg128`. Feed
the collector-enriched native JSONL into `--cases`, exactly as in the embedding
example. The driver selects the requested workload from the combined native
Qwen rows and fails if that workload is absent. Never use `llama-bench` output
for this contract.

## Verify

Combine platform JSONL artifacts under ignored `dev/` and run:

```sh
python3 bench/inference_performance/verify.py \
  dev/inference-performance/all-platforms.jsonl
```

Development checks may limit required platforms and omit Greppy evidence:

```sh
python3 bench/inference_performance/verify.py results.jsonl \
  --platform apple_cpu --allow-missing-greppy
```

Release verification uses the defaults: all four platforms, at least five
samples per engine/case, all three gated workloads, and production Greppy
evidence.

Run harness tests with:

```sh
python3 -m unittest discover -s bench/inference_performance -p 'test_*.py' -v
```
