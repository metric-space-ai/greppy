# Hardware evidence

Reproducible proof that greppy's embedded inference (EmbeddingGemma vector
search + Qwen3.5 MTP summaries) works on real hardware beyond GitHub-hosted
runners. Every JSON file in this directory is one run of
[`bench/hardware_evidence.sh`](../hardware_evidence.sh) and conforms to
[`bench/hardware-evidence.schema.json`](../hardware-evidence.schema.json).

Artifacts are public release evidence. They record hardware make/model,
driver/OS versions, product and model digests — and nothing
host-identifying (no hostnames, usernames, serials, or absolute paths;
the harness has a scrub gate that refuses to emit otherwise).

## What each artifact proves

| Artifact | Backend | Proves |
| --- | --- | --- |
| `*-macos-aarch64-metal.json` | metal | The shipped Metal backend selects and runs end to end (index → semantic-search → brief → expand) on Apple Silicon. `platform.apple_silicon_generation` + the note in `measurements.notes` say whether the Metal 4 **tensor-ops** matmul path (M5-class GPU) or the simdgroup fallback was exercised. |
| `*-linux-x86_64-cpu.json` | cpu | The Linux binary's CPU backend runs end to end; `backend.cpu_capabilities` records which SIMD tiers the product's own probe engaged (avx2 / avx-vnni / avx512f). |
| `*-linux-x86_64-cuda.json` | cuda | **Pending hardware — see below.** The CUDA backend runs end to end on a real NVIDIA GPU *without any CUDA toolkit installed*: the harness strips `nvcc` from `PATH`, unsets `LD_LIBRARY_PATH`, and asserts the backend library the binary materializes needs no `cudart`/`cublas`/`nvrtc` — only the driver's `libcuda.so.1`. It also records the VRAM peak (nvidia-smi polling). |
| any artifact with `backend.baseline_x86_required: true` | cpu | **Pending hardware — see below.** The binary runs correctly on an x86-64 CPU that *lacks* AVX-VNNI/AVX-512, i.e. the SIMD fallback paths carry the full contract-check suite. |

Every artifact additionally locks in, on that hardware:

- backend selection: `greppy doctor --json` picked exactly the requested
  backend, and that backend is compiled into the binary;
- functional contracts: brief/semantic-search/expand JSON contracts,
  deterministic hit ordering across reruns, byte-exact grep passthrough;
- timings on the fixed 2-file fixture (`measurements.fixture.tree_sha256`
  pins the workload, identical across platforms): cold `index` wall time
  and warm p50/p95 over 20 sequential `semantic-search` and `brief` calls.
  Per-call latency includes full CLI process startup + daemon round trip —
  it is the latency an agent actually experiences, not bare model forward
  time (`bench/inference_performance/` measures that).

## Stale-evidence detection

`models.embedding.model_sha256` / `models.summary.model_sha256` are the
digests of the models embedded in the measured binary. Final release
numbers must be re-measured with the release model: an artifact whose
digests do not match `crates/cli/assets/MODEL_ASSETS.json` on the release
SHA is stale and must be regenerated. Likewise `product.source_sha` /
`product.binary_sha256` tie the artifact to the exact build measured.

## How to re-run

The harness is self-contained (fixture generated inline) and needs only
`bash`, `jq`, `perl`, plus `nvidia-smi`/`ldd` for the CUDA leg. Target
runtime is well under 10 minutes per platform.

### macOS / Metal (Apple Silicon)

```sh
tools/fetch_model_assets.sh
cargo build --locked --release --bin greppy --features metal
bench/hardware_evidence.sh target/release/greppy --backend metal \
  --source-sha "$(git rev-parse HEAD)" \
  --out bench/hardware-evidence/$(date -u +%F)-macos-aarch64-metal.json
```

### Linux / CPU

```sh
tools/fetch_model_assets.sh
cargo build --locked --release --bin greppy            # or --features cuda; the CPU leg forces --device cpu
bench/hardware_evidence.sh target/release/greppy --backend cpu \
  --source-sha "$(git rev-parse HEAD)" \
  --out bench/hardware-evidence/$(date -u +%F)-linux-x86_64-cpu.json
```

### Linux / CUDA — pending hardware (run after host reboot)

Building needs a CUDA toolkit (`nvcc`); **running does not** — the harness
proves that by hiding the toolkit from the product. Pin the run to one GPU
so multi-tenant boxes stay undisturbed:

```sh
tools/fetch_model_assets.sh
cargo build --locked --release --bin greppy --features cuda
CUDA_VISIBLE_DEVICES=0 GREPPY_EVIDENCE_GPU_INDEX=0 \
bench/hardware_evidence.sh target/release/greppy --backend cuda \
  --source-sha "$(git rev-parse HEAD)" \
  --out bench/hardware-evidence/$(date -u +%F)-linux-x86_64-cuda.json
```

Status 2026-07-13: the CUDA workstation (RTX A4500) currently has one
hardware-failed GPU that poisons driver initialization for every device
(`cuInit` fails, zero devices enumerate — recorded via the product's own
probe: `available: false`). The CUDA binary is already built and staged on
that box; execute the command above after the pending host reboot. Until
then the CUDA leg is **pending hardware**.

### Old x86 (no AVX-VNNI / AVX-512) — pending hardware

```sh
bench/hardware_evidence.sh target/release/greppy --backend cpu --require-baseline-x86 \
  --source-sha "$(git rev-parse HEAD)" \
  --out bench/hardware-evidence/$(date -u +%F)-linux-x86_64-cpu-baseline.json
```

`--require-baseline-x86` adds two checks: the machine is x86-64 **and**
the product's CPU probe (`doctor --json`, the same
`backend.cpu_capabilities` recorded in every artifact) reports neither
`avx-vnni` nor `avx512f`. A passing run therefore proves the quantized
kernels' SIMD fallback tiers, not just that the flag parses. No such
machine is currently available (the Linux box in this directory reports
`avx-vnni`); the leg is **pending hardware** until an older x86 host is
provisioned.

## Validating an artifact

```sh
python3 -c 'import json, jsonschema; s=json.load(open("bench/hardware-evidence.schema.json")); \
jsonschema.Draft202012Validator(s).validate(json.load(open("bench/hardware-evidence/ARTIFACT.json")))'
```

Only artifacts with `"status": "pass"` belong in this directory; the
harness exits non-zero (and records the failing check) otherwise.
