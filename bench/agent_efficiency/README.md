# Agent efficiency benchmark

A reproducible multi-repository measurement of correctness, input/output
tokens, source-open loops, tool calls, and wall time for the same coding agent
with and without Greppy. The pre-registered release rules are in
[`BENCHMARK_CONTRACT.md`](BENCHMARK_CONTRACT.md).

## Inputs

| File | Purpose |
|---|---|
| `tasks.json`, `task_classes.json` | 100 deterministic synthetic control tasks and router classes. |
| `tasks_v2.json`, `task_classes_v2.json` | 115 product tasks: six pinned real repositories plus synthetic controls. |
| `realcorpus/MANIFEST.json` | Repository URLs, commits, languages, and licenses. |
| `realcorpus/candidates.json` | Audited graph-oracle candidates used to generate real tasks. |
| `gen_corpus.py`, `gen_corpus.sh` | Deterministically materialize synthetic repositories. |
| `real_corpus.py` | Clone and verify pinned real repositories. |
| `gen_tasks.py`, `gen_real_tasks.py` | Deterministically regenerate v1 and v2 task banks. |
| `verify_tasks.py`, `verify_real_tasks.py` | API-free ground-truth and corpus verification. |
| `run_bench.py` | Low-level per-task Pi runner and aggregate reporter. |
| `parallel_acceptance_run.py` | Product runner with isolated stores, parallel tasks, grading, and gates. |
| `grade_answers.py`, `forensics.py` | Machine-readable quality grading and regression analysis. |
| `release_gate.py` | Fixed correctness, tool/source-open, and input-token thresholds. |
| `large_repo_stress.py` | Reproducible index, integrity, incremental-update, and RSS stress gate. |

Synthetic corpus repositories are generated locally and intentionally not
committed because each contains its own `.git` directory. Real repositories are
also cloned locally, never redistributed, and must match the commits in
`realcorpus/MANIFEST.json`.

## Task contract

Each task has a stable ID, repository/language, question type, question,
ground truth, and a machine-checkable descriptor. Example:

```json
{
  "id": "t001",
  "repo": "rust_medium",
  "lang": "rust",
  "type": "locate",
  "q": "Who calls compute_checksum?",
  "check": {
    "kind": "who_calls",
    "symbol": "compute_checksum",
    "expect_members": ["normalize_record"],
    "min_count": 70
  }
}
```

`tasks_v2.json` contains 115 tasks over Rust, Python, Java, JavaScript,
TypeScript, and Go. Real repository commits are pinned for Serde, Tokio, Flask,
Django, Gson, and Zod.

The runner deterministically balances arm order from the task ID and recorded
benchmark version, so Greppy is not systematically run before or after the
baseline while every ordering remains reproducible.

## Reproduce

```bash
# Product binary. Both models are embedded and active.
cargo build --release --bin greppy

# Synthetic controls.
bash bench/agent_efficiency/gen_corpus.sh
python3 bench/agent_efficiency/verify_task_classes.py
python3 bench/agent_efficiency/verify_tasks.py --index

# Pinned real repositories and v2 task contract.
python3 bench/agent_efficiency/real_corpus.py setup \
  --repos serde flask gson zod tokio django
python3 bench/agent_efficiency/verify_real_tasks.py

# Same model, same tasks, Greppy vs uncoached explorer baseline.
export MINIMAX_API_KEY=...
python3 bench/agent_efficiency/parallel_acceptance_run.py \
  --tasks tasks_v2.json \
  --agents grep,greppy,explorer \
  --parallel 5
```

Five workers is the default production setting. Higher concurrency can trigger
provider rate limits and does not make a partial run decision-capable. The
runner has bounded retry/backoff and a circuit breaker for sustained provider
failure.

## Artifacts

Each acceptance run is isolated under
`bench/agent_efficiency/acceptance_runs/<run-id>/` and produces:

- `RUN_MANIFEST.json`: Git/binary/task/class/repository/prompt/model hashes;
- `results.json`: per-task raw metrics and final answers;
- `results.mechanical.json`: accepted machine-readable quality grades;
- `aggregate.txt`: aggregate factors by class, size, type, and language;
- `FORENSICS_explorer_VS_greppy.md`: product-baseline analysis;
- `release-gate.json`: fixed release thresholds and pass/fail status;
- `SUMMARY.md`: executed steps and exit codes.

Raw Pi JSONL is retained locally for debugging and forensics. It may contain
source snippets and full trajectories and is excluded from Git and release
artifacts. The publishable files above contain per-task metrics and grading but
no raw traces.

## Release gates

The result is accepted only when every comparable row has quality evidence,
Greppy has at least as many observed paired correctness wins as losses, the
exact paired regression alarm does not fire, and on structural tasks Greppy
uses at most 80% of the baseline's tool calls, source-open calls, and variable
input tokens. The exact test is not presented as proof of population
equivalence. `release_gate.py` enforces those thresholds; `forensics.py
--enforce` additionally rejects missing evidence and router violations.

## Large-repository stress

```bash
python3 bench/agent_efficiency/large_repo_stress.py \
  --files 300 \
  --functions-per-file 3 \
  --fanout 2 \
  --timeout-s 180 \
  --incremental-timeout-s 90 \
  --max-initial-seconds 60 \
  --max-incremental-seconds 30 \
  --max-peak-rss-mib 768 \
  --max-db-mib 128 \
  --json
```

## Secrets

The provider key is read from `MINIMAX_API_KEY` or, on macOS, from the user's
launchd environment. It is never accepted on argv or written to results,
manifests, logs, or reports.
