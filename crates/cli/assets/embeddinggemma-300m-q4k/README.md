# EmbeddingGemma-300M Q4_K Assets

Every Greppy build embeds:

- `embeddinggemma-300M-Q4_K.gguf`
- `tokenizer.json`

The model is a Q4_K GGUF derivative of `google/embeddinggemma-300m` for local
code retrieval. Its source model is governed by the Gemma Terms of Use, not the
Greppy MIT source-code license. Redistribution notices and a frozen review copy
of the terms are under the repository-level `licenses/` directory.

Source: https://huggingface.co/google/embeddinggemma-300m

The bundled GGUF is a modified distribution: the upstream model was converted
to GGUF and quantized to Q4_K. Its exact historical conversion record has not
yet been recovered, so `licenses/EMBEDDINGGEMMA-PROVENANCE.json` and the global
redistribution lock keep this asset blocked from a production release. Greppy
will replace it with a reproducibly converted asset if that record cannot be
recovered.

Verified asset digests:

- GGUF: `53f7d1c0d5c84a81e46f3bea8e0f17c94f459ffbaa8b06f7f52f1f09e58996f2`
- tokenizer: `6852f8d561078cc0cebe70ca03c5bfdd0d60a45f9d2e0e1e4cc05b68e9ec329e`
