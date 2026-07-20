# Changelog

All notable changes are documented here. Greppy follows Semantic Versioning.

## [Unreleased]

## [0.2.1] — 2026-07-20

First gate-qualified release: cut only after CI, CodeQL, the dependency
security audit, the task-bank reproducibility audit, the navigation-regime
agent benchmark, and the summary-quality gate passed on the release commit,
then signed, notarized, and attested (SBOM + provenance). Ships the complete
four-model MSCC evidence (MiniMax-M3, GLM-5.2, Qwen3.6-27B, Kimi-K3) and the
accompanying paper.

### Added

- Embedded Qwen3.5-0.8B Q4_K_M/MTP purpose summaries for `brief` and
  `semantic-search`, with CPU, Metal, and CUDA inference.
- Versioned JSON contracts with exact spans, source signatures, summaries, and
  durable expand handles.
- Shared inference backend registry, device probing, memory checks, daemon
  status, and model digests in `greppy doctor --json`.
- Managed cache inspection, garbage collection, and explicit clearing.
- Windows named-pipe transport for the embedded inference daemons.

### Changed

- Ordinary grep invocations are byte-exact real-`grep` passthrough and have no
  index, model, or cache side effects.
- Freshness is fail-closed: Greppy does not knowingly print stale source
  evidence.
- EmbeddingGemma and Qwen model assets are mandatory product assets in every
  binary; only the inference backend/device is selectable.
- Model idle TTL is 300 seconds and daemon process idle TTL is 1800 seconds.

### Fixed

- `greppy index` publishes the complete graph snapshot when embedding
  inference degrades (model load failure or failed batches) instead of
  discarding all indexing work with `EXIT_IO`; the vectors that did embed are
  kept and the next semantic query resumes the remainder in the background.

### Removed

- Synthetic grep-output augmentation, sidecars, and
  `NON_CANONICAL_CODE_HINT`.
- The `--vectors` switch and public model-disable/model-path controls.
- Installation or packaging under the binary name `grep`.
- The in-product self-updater.

### Licensing and hosting

- Source relicensed MIT → Apache-2.0 (embedded model terms unchanged:
  EmbeddingGemma under the Gemma Terms, the in-house Qwen3.5 fine-tune under
  Apache-2.0).
- Model weights hosted as public, ungated Hugging Face repos
  (`metricspace/embeddinggemma-300m-q4k`, `metricspace/greppy-qwen35-mtp-q4km`);
  the build fetches and SHA-256-verifies them before embedding.

## [0.1.2]

- Last pre-`v0.2.0` development release.
