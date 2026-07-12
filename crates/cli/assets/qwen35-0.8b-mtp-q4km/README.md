# Qwen3.5-0.8B MTP Q4_K_M Assets

Every greppy build expects these Git-LFS assets in this directory:

- `Qwen3.5-0.8B-MTP-Q4_K_M.gguf`
- `Qwen3.5-0.8B-MTP-Q4_K_M.gguf.sha256`
- `tokenizer.json`
- `tokenizer.json.sha256`

The GGUF is Greppy's 2026-07-11 function-purpose finetune of the pinned
`Qwen/Qwen3.5-0.8B` base model. Greppy changed the model through full-parameter
supervised finetuning and trained an MTP draft layer. The merged BF16 checkpoint
was converted and quantized to Q4_K_M with llama.cpp; the checked-in GGUF
contains both target and MTP weights.

This checkpoint is an engineering candidate, not a released production model.
Its independent navigation-quality evaluation does not yet meet Greppy's
release threshold, so the release lock marks it `release_ready: false`.

The tokenizer JSON is from the pinned `Qwen/Qwen3.5-0.8B` revision and is
unchanged by Greppy.

Both sources identify Qwen3.5 as Apache-2.0. The complete license shipped with
Greppy is `licenses/QWEN3.5-APACHE-2.0.txt`. Exact base, data, training, export,
quantization, and modification records are in the repository-level
`licenses/QWEN3.5-*.json` and `licenses/QWEN3.5-MODIFICATIONS.txt` files.

Verified asset digests:

- GGUF: `d45e08ad7bb8787ae9b6f56b6915e8b44ac6e13c6b740fdc7bd591249209a72c`
- tokenizer: `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`
