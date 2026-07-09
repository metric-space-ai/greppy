# Third-Party Notices

`greppy` is original Rust source. It depends on third-party Rust crates
under their own licenses; those licenses are recorded in each crate's
`Cargo.toml` and resolved by `cargo metadata`. This file documents the one
non-crate notice obligation: vendored GPU kernels.

## ggml (vendored GPU kernels)

The embedding engine (`crates/embed-native`) is a from-scratch Rust
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
per-file provenance (which files are upstream ggml vs. greppy-authored
wrappers) is documented in `crates/embed-native/vendor/README.md`. The default
build (`default = []`) ships no ggml code.

## Qwen3.5 native summarizer kernels

The Qwen3.5 summarizer crate (`crates/qwen35-native`) is original Rust source.
Its opt-in CUDA and Metal backends reuse the same MIT-compatible vendored ggml
kernel slice documented above under `crates/embed-native/vendor/`; no separate
Qwen3.5 runtime dependency on llama.cpp, libggml/libllama, Candle, ONNX,
Python, or an external server is introduced.
