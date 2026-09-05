# bash-smart R5 classifier asset

`classifier-v1.f32le` is the historical R5 **classifier only**, restored from
`wip/head-classifier-0.3.1` for reproducibility and native compatibility tests.
It is not enabled in the CLI. Both thresholds are historical calibration values;
neither is a validated threshold for the current encoder on every backend.
The previous corrected Gate-2 recall estimate omitted unaudited true negatives.
The old holdout also contained no progress examples. Neither establishes a
complete four-class release gate. The R5 ranker is not present in this asset.

The original asset SHA-256 is recorded in `MODEL_ASSET.json` and the adjacent
sidecar. The restored golden test verifies the classifier and golden hashes.
The historical parent manifest/build-script wiring has not been restored.

## Binary layout (`greppy.r5-classifier.f32le.1`)

All integers and IEEE-754 `f32` values are little-endian. Weights are row-major.

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 8 | magic `GRPYR5H1` |
| 8 | 4 | format version = 1 |
| 12 | 4 | header bytes = 128 |
| 16 | 4 | input width = 768 |
| 20 | 4 | hidden width = 256 |
| 24 | 4 | output width = 4 |
| 28 | 4 | labels = 4 |
| 32 | 4 | frozen error threshold |
| 36 | 4 | frozen warning threshold (provenance only) |
| 40 | 8 | payload `f32` count |
| 48 | 32 | source PyTorch checkpoint SHA-256 bytes |
| 80 | 32 | frozen-threshold JSON SHA-256 bytes |
| 112 | 16 | zero/reserved |
| 128 | 768×4 | StandardScaler mean |
| ... | 768×4 | StandardScaler scale |
| ... | 256×768×4 | linear 1 weight |
| ... | 256×4 | linear 1 bias |
| ... | 4×256×4 | linear 2 weight |
| ... | 4×4 | linear 2 bias |

Class order is `error, warning, progress, text`; activation is exact GELU
(`x * 0.5 * (1 + erf(x / sqrt(2)))`) followed by softmax.

`golden-v1.f32le` stores the first 64 frozen fresh-holdout block vectors plus
Python/PyTorch logits and probabilities. `golden-blocks-v1.json` stores their
fixed IDs/texts for end-to-end device checks. The enforced unit-test budget is
`max_abs_diff <= 5e-5`; the measured portable-head difference is recorded in
the engineering report.
