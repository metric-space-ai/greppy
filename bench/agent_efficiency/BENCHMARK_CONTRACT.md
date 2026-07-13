# Greppy Agent Benchmark Contract

## Decision under test

For a long-lived coding agent doing structural code navigation, Greppy may be
recommended as the primary navigator when it reduces exploration cost without
reducing answer correctness. Source, build, and test verification remain
required.

## Fixed comparison

- Candidate: one `greppy` binary with embedded EmbeddingGemma and Qwen summary
  inference active.
- Gate baseline: the same agent with ordinary shell grep/find and file reading,
  without Greppy-specific coaching (`explorer`).
- Diagnostic baseline: a coached efficient-grep prompt (`grep`), never used to
  determine product acceptance.
- Agent runtime: Pi Code.
- Contract model: `MiniMax-M3` through the checked-in provider extension.
- Task bank: `tasks_v2.json`, with classes in `task_classes_v2.json`.
- Repositories: commits in `realcorpus/MANIFEST.json`; synthetic controls are
  generated deterministically by `gen_corpus.py`.

Every run records SHA256 for the task bank, class document, repository
manifest, prompt texts, and Greppy binary plus the Greppy Git commit and model
ID. A result without that `RUN_MANIFEST.json` is not release evidence.

## Measurement

Each task runs with the same question, model, timeout, and repository snapshot.
Index/model setup occurs before measured agent work. Per-task rows retain:

Arm order is deterministically balanced per task by hashing the benchmark
order version, task ID, and arm name. This prevents one treatment from always
running earlier in a provider session while keeping the complete order
reproducible from the manifest.

- accepted mechanical quality/correctness evidence;
- explicit hard-negative graph terms are hard failures, so an answer cannot
  pass by naming the expected symbols alongside a known false edge;
- input and output tokens separately;
- variable input tokens after the common first-turn prompt;
- tool calls and source-open calls;
- returned search-context volume;
- wall-clock time and provider errors.

Provider failures, timeouts, missing grades, stale repositories, prompt/hash
mismatches, or incomplete pairs make a run non-decision-capable.

## Release gates

`v0.2.0` requires all gates against `explorer`:

1. Greppy must have at least as many observed paired correctness wins as losses.
2. A one-sided exact discordant-pair test must not detect a correctness
   regression at alpha 0.05. This is a regression alarm, not a claim of
   population equivalence or non-inferiority.
3. Candidate total tool calls on structural `locate` tasks are at most 80% of
   baseline.
4. Candidate source-open calls on structural tasks are at most 80% of baseline.
5. Candidate variable input tokens on structural tasks are at most 80% of
   baseline.
6. Every comparable row has accepted machine-readable quality evidence.

`release_gate.py` implements these thresholds. They are fixed before the run
and may not be relaxed after observing results.

## Publication

Publish `RUN_MANIFEST.json`, per-task results, mechanically graded results,
aggregate report, forensics report, release-gate report, and summary with the
release. Raw Pi JSONL traces may contain source snippets and are retained only
for local audit; they are never release artifacts.
