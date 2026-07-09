# Vendored GPU kernels — provenance & license

This directory vendors a small slice of GPU kernel sources so the CUDA and
Metal backends of `greppy-embed-native` can be built from source. The bulk
of these files are **verbatim (or lightly reduced) sources from
[`ggml-org/ggml`](https://github.com/ggml-org/ggml)** (the library behind
`llama.cpp`), which is **MIT-licensed**.

Their license is preserved alongside them in [`LICENSE-ggml`](./LICENSE-ggml):

> Copyright (c) 2023-2026 The ggml authors — MIT License.

The MIT notice is also reproduced at the top of several upstream headers
(e.g. `cuda/ggml-include/ggml-cann.h`, `cuda/ggml-include/ggml-sycl.h`).

## File provenance

| Path | Origin | License |
|------|--------|---------|
| `cuda/ggml-cuda/*.cu`, `*.cuh` | ggml `src/ggml-cuda/` (quantize + MMQ/MMVQ kernels, dot products, tensor-core MMA primitives) | MIT (ggml authors) |
| `cuda/ggml-cuda/vendors/cuda.h` | ggml `src/ggml-cuda/vendors/` | MIT (ggml authors) |
| `cuda/ggml-include/*.h` | ggml `include/` public headers | MIT (ggml authors) |
| `metal/shaders/ggml/ggml-metal.metal`, `ggml-metal-impl.h`, `ggml-common.h` | ggml `src/ggml-metal/` | MIT (ggml authors) |
| **`cuda/embed_native_cuda.cu`** | **greppy-authored** wrapper that packs f32 activations via ggml `quantize_mmq_q8_1`/`quantize_row_q8_1` and dispatches ggml `mul_mat_q`/`mul_mat_vec_q`; includes and links the vendored ggml kernels | MIT (greppy authors); derivative — see note below |
| **`metal/shaders/ggml/embed_native.metal`** | **greppy-authored** mean-pool + dispatch shader | MIT (greppy authors) |

The two `embed_native_*` files are original greppy code; they `#include`
and call into the vendored ggml kernels, so they are derivative works of ggml
and are distributed under the same MIT terms, with the ggml notice preserved
here.

## Why vendored (not a dependency)

`greppy-embed-native` is a from-scratch Rust EmbeddingGemma engine (CPU path
is 100% Rust via the `gemm` crate + hand-written Q4_K/Q6_K/Q8_0 SIMD dot
kernels). It does **not** depend on `ggml`/`llama.cpp` as a library. For the
opt-in GPU backends we reuse ggml's numerically-exact quantized matmul
kernels rather than reimplement them: MMQ for batched/prefill-style matmul,
MMVQ for batch-1 decode matvec, plus the matching quantization kernels.
This small subset is compiled directly:

- **CUDA** (`--features cuda`): `build.rs` invokes `nvcc` over
  `cuda/embed_native_cuda.cu` + `cuda/ggml-cuda/quantize.cu`.
- **Metal** (`--features metal`): the `.metal` shaders are compiled into a
  `.metallib` at build time.

Neither backend is built by default (`default = []`).

## Updating

When refreshing from upstream ggml, keep this file and `LICENSE-ggml` in sync
with the upstream copyright year range, and re-record the upstream commit here.
