# Qwen3.5 Native Vendor Sources

The Qwen3.5-0.8B summarizer reuses the GPU kernel package owned by
`greppy-embed-native`. Vendored code must be source-compatible with this MIT
workspace.

- CUDA: production Qwen3.5 kernels live in `greppy-embed-native` alongside the
  vendored ggml CUDA MMVQ/MMQ sources so the decode pipeline can stay
  device-resident.
- Metal: production Qwen3.5 kernels live in `greppy-embed-native` alongside the
  vendored ggml Metal Q4_K matrix kernels and greppy-authored state/attention
  kernels. No AGPL code is vendored.
