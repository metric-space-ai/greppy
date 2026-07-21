# EmbeddingGemma-300M Q4_K Assets

Every Greppy build embeds:

- `embeddinggemma-300M-Q4_K.gguf`
- `tokenizer.json`

The model is the byte-identical Q4_K GGUF from the pinned
`cduk/embeddinggemma-300m-GGUF-with-dense-modules` snapshot. That repository
identifies `google/embeddinggemma-300m` as its base model and includes the dense
post-processing modules required for sentence-transformers-compatible
embeddings. The model is governed by the Gemma Terms of Use, not the Greppy
Apache-2.0 source-code license. Redistribution notices and a frozen review copy of the
terms are under the repository-level `licenses/` directory.

Source: https://huggingface.co/cduk/embeddinggemma-300m-GGUF-with-dense-modules/tree/16eaef07700282e488368e27b992b8fe5a40c423

Greppy did not convert or alter the GGUF bytes. The pinned third-party snapshot
also contains the F32 GGUF from which Greppy independently reproduced the
bundled Q4_K_M bytes. Two x86_64 runs of `llama-quantize` at llama.cpp revision
`56fc38b9655fbe1869d8bd6cfb269418196cea69` were bit-stable and byte-identical
to the bundled file. The exact source digest, command, architecture, and output
digest are recorded in `licenses/EMBEDDINGGEMMA-PROVENANCE.json`.

This reconstruction proves the packaged quantized artifact is reproducible; it
does not claim knowledge of the producer's unpublished build environment. The
redistribution review for this asset is recorded in
`licenses/EMBEDDINGGEMMA-PROVENANCE.json`; the binding use restrictions are
stated in `licenses/EMBEDDED-MODEL-TERMS.md`.

Verified asset digests:

- GGUF: `53f7d1c0d5c84a81e46f3bea8e0f17c94f459ffbaa8b06f7f52f1f09e58996f2`
- tokenizer: `6852f8d561078cc0cebe70ca03c5bfdd0d60a45f9d2e0e1e4cc05b68e9ec329e`
