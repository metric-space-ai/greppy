# Portable experimental head candidates

The candidate format connects GPU3 training exports to one portable Rust forward
pass. It is deliberately an uncalibrated experiment format. The loader rejects
nonempty validated-backend lists and calibration/release claims; loading a
candidate cannot authorize automatic activation.

`tools/embedding_heads/export_candidates.py` checks the completed run identity,
feature bundle, parameter names, shapes, finite values and checksums. It emits:

- `manifest.json`: `greppy.heads.candidate.v1`, head/objective, dimensions, exact
  input and representation hashes, source run ID, weight and golden checksums;
- `weights.f32le`: a 64-byte checked header followed by little-endian float32
  normalization parameters, row-major affine layers/biases and optional cutpoints;
- `golden.json`: same-vector reference outputs from PyTorch;
- a top-level inventory binding every candidate manifest and expected case count.

The header has magic `GRPYHD01`, format version 1, input/hidden/output dimensions,
objective and head-kind tags, float count and zero reserved bytes. Class order is
error/warning/progress/text. Supported architectures are linear and exact-GELU
MLPs with 128 or 256 hidden units. Ordinal ranking has three strictly increasing
cutpoints; pairwise ranking returns the learned scalar directly.

`crates/embed-native/src/head_candidate.rs` checks the entire manifest and binary
layout before inference, including an independently supplied expected input and
representation contract. It rejects corrupt/truncated files, incompatible heads,
wrong dimensions, non-finite values, non-positive scales and invalid cutpoints.
Arithmetic overflow also returns an error. The batch helper preserves caller IDs
and rejects duplicate IDs or any invalid row as a whole; it never generates text.

The portable implementation shares the existing R5 GELU/erf approximation. Its
error must be checked against reference outputs, not assumed identical to
PyTorch. `head_candidate_check` verifies candidate inventory coverage, checksums
and every reference value with an absolute tolerance of 5e-5. This is a numerical
head check, not native encoder/backend equivalence or task acceptance.

Run the focused Rust checks and candidate verifier through `greppy bash-smart`.
On macOS put CARGO_TARGET_DIR and TMPDIR under `/Volumes/tmp`. The complete
candidate export is diagnostic data derived from synthetic vectors; production
weights, calibration, a release manifest and runtime integration remain pending.

## Verified arithmetic

All six focused loader tests pass on macOS. The complete 45-candidate inventory
passed all 720 same-vector reference cases; maximum absolute difference was
1.430511474609375e-6, below the predeclared 5e-5 tolerance. The CPU-only build
reports an existing irrefutable-pattern warning in the untouched line-state
backend dispatch. No latency or native encoder parity claim follows from this
loaded-head arithmetic check.

The report and source hashes are retained at
`/Users/michaelwelsch/.local/state/greppy-heads/2026-09-05/portable-candidate-verification-v1.json`.
