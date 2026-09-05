# Embedding heads: evidence and optimization work

Status: investigation and reproducibility, 2026-09-05. No newly trained model or
production-quality improvement is claimed. User priorities: logs and structured
web observations have equal weight; training uses GPU3.

Worktree: `/Users/michaelwelsch/greppy-worktrees/embedding-heads-optimization`.
Branch: `codex/embedding-heads-optimization`, base `49d889cc`.
Remote outputs: `/mnt/nvme1/greppy-heads-optimization-20260905/` on `ts-gpu3`.
Local evidence: `/Users/michaelwelsch/.local/state/greppy-heads/2026-09-05/`.
The Web task's worktree and existing model/data artifacts remain unchanged.

## Verified provenance

The head engineering work was deliberately parked in commit
`0f9efcc8f31f21368bca4eb00236e92c562d47d1`, branch
`wip/head-classifier-0.3.1`, on 2026-08-06. Its commit message explicitly records
that it belonged to neither the 0.3.0 release tree nor the 0.3.1 release tree.
It also contains unrelated benchmark files, so cherry-picking the whole commit
would mix scopes. The current base has no classifier CLI wiring.

GPU3 `/mnt/nvme1/greppy-head-eng` is based on
`03ec09d79f3204ad39ac9c2bd8377b41150640c7` plus dirty tracked changes and
untracked head sources/assets. The full tracked binary diff and selected
untracked files are archived under local `provenance/`. One dirty Qwen hash
sidecar is unrelated and is not imported.

The historical CLI path used `GREPPY_BASH_SMART_HEAD=1`, default off, and only
added implicit-error blocks. It never decoded warning scores or used HEAD2.
It required nonzero exit, eligible hidden groups, a ready CUDA daemon and a
<=2,000 ms cost estimate. CPU/Metal disabled themselves because raw-text
threshold parity had failed. The 2,000-block CUDA diagnostic skipped inference.
These are historical source/report facts, not current CLI behavior.

The restored library module, fixed-vector test, example and original asset in
this worktree enable compatibility measurements. The CLI featureflag, build
embedding and selection wiring have not been restored.

## Reproduced R5 baseline

`tools/embedding_heads/reproduce_r5.py` hashes the checkpoint, exported asset,
frozen thresholds, labels and native CUDA vector cache before scoring. It
writes into a new directory exclusively and never trains or tunes on the old
holdout. GPU3 execution reproduced exact stored counts:

| Frozen threshold | TP | FP | FN | TN | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Error | 910 | 1310 | 119 | 1661 | 40.99099% | 88.43537% |
| Warning | 48 | 137 | 154 | 3661 | 25.94595% | 23.76238% |

These references are the original M3 labels, not a claim of audited truth.
The 4,000 blocks contain error=1,029, warning=202, text=2,769, progress=0.
The report additionally stores the four-class argmax confusion matrix and
per-class metrics. Progress recall is undefined, not zero or successful.

PyTorch evaluation of exported float32 parameters vs original checkpoint on
the same cached vectors: maximum probability difference `1.3709068298339844e-6`,
zero error-threshold disagreements. This does not test the current encoder.
The portable Rust golden passes on CPU and in a Metal-feature build at
`max_abs_diff=0.000017166`, budget `0.00005`, N=64. The Metal-feature fixed-vector
test still computes the head in portable Rust; it is not a GPU-encoder test.

The current raw-text Metal diagnostic reproduces the old mismatch exactly:
`max_abs_diff=0.240339994`, three error-threshold disagreements out of 64.
This was a debug build, so its runtime is not a production latency measurement.
The separate CPU raw-text debug diagnostic was stopped by this task after
several minutes; it produced no final result. CPU raw-text parity remains open.
The teacher/data pipeline contract is recorded in
`tools/embedding_heads/TEACHER_CONTRACT.md`.

The previous Grok audit estimates precision=57.218%, conditional recall=91.434%.
TN was unjudged. `audit_metrics.py` reports full recall as null in this case.
Given the other point estimates, 23 additional true errors in TN would put
recall below 90%. This sensitivity is not a confidence interval. Six unit tests
cover missing strata, census arithmetic, empty populations and invalid counts.
The second Sol audit judged the same 619 blocks, not a fresh test set.

## Rubric adoption audit

`tools/embedding_heads/audit_training_labels.py` replays the original pooling
projection and compares the later warning rubric's 180 reversals with current
merged judgments. All 180 still have the old label; none were applied.
179 map to stored spans, one does not. Majority pooling changes 96 model-block
labels: 84 fitting examples and 12 calibration examples.

| R5 projected label -> rubric projection | Blocks |
| --- | ---: |
| error -> text | 61 |
| text -> error | 17 |
| text -> warning | 14 |
| error -> warning | 2 |
| text -> progress | 2 |

The replay maps 15,335 judgments to 11,918 blocks, matching the old report.
Both original and revised projections contain one block with conflicting votes.
No base calibration output ID occurs in the admitted extra/loghub/pkgforge/
implicit/terminal fitting sources. This is only ID disjointness, not a complete
content/template leakage audit. The generated ID-only overlay does not claim
to relabel the other training examples.

## Evaluation contract before training

- Keep the encoder frozen and use native Q4_K vectors, with hashes for model,
  tokenizer, binary, prompt, layer, normalization and pooling configuration.
- Freeze a new final cohort and labels before model selection. Already exposed
  R5/FARM/16 web pilot runs remain development diagnostics. Split by source or
  complete output/scenario, never individual neighboring blocks.
- Logs: independently measure all four classes, implicit errors/warnings,
  cause@3/5/8 on full long outputs, and hard negatives from the same output.
  Audit all original-label strata; report uncertainty and output clustering.
- Web: learn task-conditioned relevance separately from severity. Evaluate
  forms, dependent fields, ambiguous dialogs, menus, SPA transitions, delayed
  content, virtual lists, frames, reload, covered/replaced nodes and noise.
  Similar irrelevant events must occur in the same observations.
- Log and Web gates have equal weight. A gain in one cannot hide a regression
  in the other. No shared micro-average over differently sized datasets.
- Preserve exit codes and explicit errors deterministically. For Web preserve
  identities/revisions/refs, actionable state, validation and ambiguity/stale
  errors, focus/dialog/navigation changes, dispatch/effect/expectation status,
  partial chain progress, continuation and raw evidence. A head may only rank
  eligible optional material. It cannot manufacture fields, refs or success.
- Measure task success, causes/action hints retained, expansions, repeat calls,
  latency, result bytes and model calls. No paid/fallback model is assumed to be
  a valid browser interpreter without a dedicated evaluation.
- Training stays on GPU3 and never terminates foreign GPU jobs. Existing
  artifacts and old holdouts are not overwritten. Web content remains untrusted;
  secrets and private reasoning are excluded from training and shared caches.

Pending: current raw-text backend parity, complete new evaluation cohort,
TN/four-class independent audit, controlled label/data/ranking ablations,
Web relevance dataset and head, training, end-to-end tool measurement and a
verified opt-in integration. Historical metrics are not a release decision.
