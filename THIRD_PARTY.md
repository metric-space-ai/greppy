# Third-Party Notices

Greppy source is MIT-licensed. Embedded model weights, Rust dependencies, and
vendored accelerator kernels retain their own licenses and notices. Release
archives must include this file and the complete `licenses/` directory.

## Embedded model assets

### EmbeddingGemma-300M Q4_K

- Purpose: code-query and source-span embeddings.
- Upstream model: `google/embeddinggemma-300m`.
- Bundled GGUF source: pinned
  `cduk/embeddinggemma-300m-GGUF-with-dense-modules` snapshot, byte-identical.
- Bundled files: `embeddinggemma-300M-Q4_K.gguf`, `tokenizer.json`.
- Terms: Gemma Terms of Use plus the incorporated Gemma Prohibited Use Policy.
- Notice: `licenses/EMBEDDINGGEMMA-NOTICE.txt`.

The Gemma terms require a copy of the agreement and a specific Notice for
redistribution. They are not replaced by Greppy's MIT license. Public release
remains blocked until the redistribution review is signed off and the current
official terms are compared with the packaged copies. The bundled Q4_K_M bytes
have been independently reproduced from the pinned public F32 GGUF; the exact
source digest, llama.cpp revision, x86_64 command, and output digest are in the
provenance record.

See `licenses/EMBEDDINGGEMMA-MODIFICATIONS.txt` and
`licenses/EMBEDDINGGEMMA-PROVENANCE.json`. The release workflow verifies these
records and refuses to publish while their release lock remains unresolved.

### Qwen3.5-0.8B MTP Q4_K_M

- Purpose: short function-purpose navigation hints.
- Base model/tokenizer: pinned `Qwen/Qwen3.5-0.8B` revision.
- Finetune: Greppy full-parameter function-purpose SFT with an MTP draft layer.
- Quantization: merged BF16 checkpoint converted and quantized to Q4_K_M with
  pinned llama.cpp; the bundled GGUF contains target and MTP weights.
- Bundled files: `Qwen3.5-0.8B-MTP-Q4_K_M.gguf`, `tokenizer.json`.
- License: Apache License 2.0; see `licenses/QWEN3.5-APACHE-2.0.txt`.

This is a modified model, not an unchanged Unsloth artifact. See
`licenses/QWEN3.5-MODIFICATIONS.txt`, `licenses/QWEN3.5-PROVENANCE.json`, and
`licenses/QWEN3.5-TRAINING-DATA-MANIFEST.json`. The current checkpoint remains
blocked from release until the recorded quality and redistribution gates pass.

Model outputs are non-authoritative navigation hints. Source spans, signatures,
and graph evidence remain deterministic even when summary inference fails.

## ggml (vendored GPU kernels)

The embedding engine (`crates/embed-native`) is a native Rust
EmbeddingGemma implementation. Its CPU path is 100% Rust. For the **opt-in**
GPU backends (built only with `--features cuda` or `--features metal`) it
vendors a small slice of quantized-matmul and quantization kernels from
[`ggml-org/ggml`](https://github.com/ggml-org/ggml), compiled from source by
`nvcc` / the Metal toolchain.

The ggml project is **MIT-licensed**:

> Copyright (c) 2023-2026 The ggml authors
> Licensed under the MIT License.

The vendored sources live under `crates/embed-native/vendor/`. The MIT license
text is preserved there at `crates/embed-native/vendor/LICENSE-ggml`, and
per-file provenance (upstream-derived versus Greppy-authored wrappers) is
documented in `crates/embed-native/vendor/README.md`. Every build embeds both
models; the `metal` and `cuda` features control only accelerator kernels.

## Qwen3.5 native summarizer kernels

The Qwen3.5 summarizer crate (`crates/qwen35-native`) is original Rust source.
Its opt-in CUDA and Metal backends reuse the same MIT-compatible vendored ggml
kernel slice documented above under `crates/embed-native/vendor/`; no separate
Qwen3.5 runtime dependency on llama.cpp, libggml/libllama, Candle, ONNX,
Python, or an external server is introduced.

## Rust dependencies and release SBOM

Rust crate licenses are declared by their packages and resolved in
`Cargo.lock`. The release workflow must generate an SPDX SBOM from the exact
packaged source and attach it to the release. A tag is not publishable when the
SBOM, model notices, checksums, signatures, or build provenance are missing.
