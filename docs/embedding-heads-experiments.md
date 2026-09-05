# Reproducible head experiments on GPU3

`tools/embedding_heads/experiments.py` trains three separate experimental heads
on the host `gpu3-a4500`. Both the CLI and the training entry point refuse other
hosts and non-CUDA devices. GPU3 is the host; CUDA devices on it are indexed
0 through 2. The runner does not start an encoder, generate labels, grant data
admission, export a production model, or activate any backend.

## Experiment matrix

One invocation runs exactly 45 variants:

- Log classification: linear, exact-GELU MLP-128 and MLP-256, each with three seeds.
- Log ranking: the same architectures, ordinal and pairwise objectives, three seeds.
- Web ranking: independently the same architecture/objective/seed matrix.

Classification uses four-class cross entropy in error/warning/progress/text order.
Ordinal ranking uses a scalar score with three learned, monotonically ordered
cutpoints and cumulative binary cross entropy. Pairwise ranking uses logistic
preference loss on higher/lower grades within one output or observation and task.
Sampling is bounded per comparison scope, recorded in the configuration, and
reproducible per epoch. It never constructs a quadratic all-pairs matrix.

Normalization statistics are fitted on training data only, with streaming
variance accumulation. Feature arrays stay memory-mapped; normalized batches
are materialized as needed. Development metrics include all four confusion
strata per class, NLL/Brier/ECE, and per-scope nDCG/required-evidence recall at
several k values. These diagnostics do not choose a production compression rate
or establish agent task success. Calibration is explicitly unfitted.

## Feature bundle boundary

The bundle schema is `greppy.heads.feature-bundle.v2`. Version 1 remains readable
only for historical synthetic pipeline fixtures. Every head has exactly
train and development partitions, each with hash-bound JSONL rows and float32
NumPy arrays of shape [rows, 768]. Other partitions are rejected; final test data
must never be supplied. Native feature extraction remains a separate prerequisite.

Rows bind candidate/source/comparison/group identities, the exact input-contract
hash, input/annotation/evidence hashes, split, head, admission status and label.
Every row binds the captured source content hash; one source ID cannot change
content within a bundle. Ranking rows additionally bind task and full conditioning
hashes. Web rows bind observation identity and the actual action-context hash.
Comparison scopes cannot mix observations or action contexts even when the
episode and task are identical. The same candidate can have different labels
under different conditioning, but identical prepared inputs cannot carry
conflicting labels. Equal inputs, source contents and declared related groups
cannot cross splits.
Unadmitted rows, missing classes, corrupt arrays, wrong dimensions, non-finite
values and files escaping the bundle directory are refused.

Each head's input contract binds its preprocessor and prompt hashes, token limit,
layer, pooling, normalization and whether it is task-conditioned. A real
development candidate requires native frozen EmbeddingGemma-300M Q4_K provenance,
encoder/tokenizer/binary hashes and a specific backend. It also binds the source
registry, split manifest, annotations, admission review and teacher configurations
as files. Hash binding preserves provenance; it does not independently prove
the truth of those artifacts or replace the admission review. Bundles must be
immutable throughout an experiment. There is no automatic exporter from a
teacher agreement report into this boundary yet.

Synthetic pipeline tests have a separate role and representation kind and cannot
claim native encoder provenance. `synthetic_experiment_fixture.py` only creates
such test bundles. Its programmed labels and random vectors are never model
quality evidence.

## Persistence

Run identity binds the complete bundle hash, trainer and contract code hashes,
representation/input contract, label provenance, configuration and numerical
runtime. Output directories are keyed by that identity and protected by a process
lock. Completed results verify their asset hashes and return without rewriting.

An atomic checkpoint after each epoch contains model and optimizer state plus
loss history. Restart resumes the last committed epoch. Sampling is derived from
seed and epoch; these architectures contain no dropout. Interrupted partial
writes are not treated as checkpoints. Results contain NumPy weights, checkpoint
checksums, diagnostics and an explicit `release_gate: not_evaluated` with no
validated backend. The NumPy archive is an experiment artifact, not the versioned
portable production format.

Run on GPU3:

```sh
python3 experiments.py --bundle /data/immutable/manifest.json \
  --out /data/head-experiments --device cuda:0 --seeds 17 43 101
```

Route actual training and verification commands through `greppy bash-smart`,
as required by the repository. Do not run training on the Mac.

## Verified scope

On GPU3, 57 Python tests passed. A synthetic smoke run completed all 45 variants
for three epochs. All losses decreased; all 45 completed replays preserved result,
manifest, weight and checkpoint hashes and modification times. Injecting an
interruption immediately after epoch one for each of the five head/objective
combinations with MLP-256 reproduced bit-identical weights, losses and development
metrics after restart.

Evidence is retained at:
`/Users/michaelwelsch/.local/state/greppy-heads/2026-09-05/experiment-smoke-v1-verification.json`.
Full synthetic runs remain on GPU3 at
`/mnt/nvme1/greppy-heads-optimization-20260905/experiment-smoke-v1/`.

With feature-bundle v2, 68 Python tests passed on GPU3. A fresh independent
synthetic runner check again completed 45 variants, 45 immutable replays and
five bit-identical interruption/restart checks. Its receipt is
`/Users/michaelwelsch/.local/state/greppy-heads/2026-09-05/experiment-smoke-v2-verification.json`;
its GPU3 artifacts are under `experiment-smoke-v2/` beside the v1 directory.
These results validate the revised grouping contract, not native model quality.

Shared native target/context extraction now has CPU/CUDA raw-text probe evidence;
see `docs/embedding-heads-native-inputs.md`. The new representation has not been
admitted into real training bundles.

Still outstanding: admitted broad corpora, the admission-to-feature bundle bridge,
rubric/data ablations, genuine head selection, calibration per backend, portable
production loading, complete log/Web runtime use, and independent agent
workflow acceptance. This work does not establish production readiness.
