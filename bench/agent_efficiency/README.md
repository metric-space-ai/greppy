# Agent efficiency benchmark (grepplus vs the uncoached grep agent)

A **real**, multi-repo, multi-language measurement of how many tokens, tool-call
loops, and seconds an LLM coding agent spends to answer code questions — once
using plain `grep`, once using `grepplus` — across a generated corpus of repos
of varied size and language.

## Layout

| File | Purpose |
|------|---------|
| `gen_corpus.sh` / `gen_corpus.py` | Deterministically generate the `corpus/` repos (git-init each). |
| `tasks.json` | ~100 benchmark tasks (locate + research) with machine-checkable ground truth. |
| `task_classes.json` | R7 router/regression classes for all 100 tasks: direct similarity, hybrid seed-graph, literal controls and graph controls. |
| `gen_tasks.py` | Regenerates `tasks.json`. |
| `verify_tasks.py` | Indexes every corpus repo, asserts cross-file edges, and checks every task's ground truth against the live grepplus graph. |
| `verify_task_classes.py` | Verifies that every task belongs to exactly one regression/router class and that known hard-negative sets are present. |
| `run_bench.py` | Runs the grep-agent and grepplus-agent per task; records **input and output tokens separately**, tool calls, wall-clock; aggregate reporter. |
| `grade_answers.py` | Conservative answer grader against `tasks.json`; `smoke` mode is triage, `mechanical --accept-mechanical` is the strict synthetic-bench gate. |
| `forensics.py` | Mandatory post-run forensics: winner/loser classes, quality-gate status, raw command paths, optimization backlog. |
| `acceptance_run.py` | Serial research orchestrator (product runs use `parallel_acceptance_run.py`); non-explorer baselines are stamped DIAGNOSTIC. |
| `large_repo_stress.py` | R3/R8 black-box index stress gate: synthetic git Rust repo, initial index, one-file incremental reindex, RSS sampling, graph.db integrity/size and symbol proofs. |
| `minimax-provider.js` | pi.dev provider registering MiniMax (API key from `$MINIMAX_API_KEY`). |

## Corpus

`gen_corpus.sh` produces six git-initialised repos under `corpus/`, spanning
five languages and three size classes (small ≈ 15 files, medium ≈ 80, large ≈
400+). Each repo is layered `core → service → app` so functions call across
files, modules import each other, and a few hub symbols are called from many
sites — giving graph/research questions real answers.

| repo | language | size | files | cross-file graph |
|------|----------|------|------:|------------------|
| `rust_medium`  | Rust       | medium | ~83  | full cross-file CALLS |
| `python_large` | Python     | large  | ~425 | full cross-file CALLS |
| `go_small`     | Go         | small  | ~16  | full cross-file CALLS |
| `java_medium`  | Java       | medium | ~81  | full cross-file CALLS |
| `js_small`     | JavaScript | small  | ~15  | same-file + resolved cross-file CALLS, IMPORTS |
| `ts_large`     | TypeScript | large  | ~412 | same-file CALLS, IMPORTS |

The corpus is **reproducible** (no randomness, no timestamps in content) and is
**not committed** (see `.gitignore`) — regenerate it with `gen_corpus.sh`.

## Tasks

`tasks.json` holds ~100 tasks, each:

```json
{"id":"t001","repo":"rust_medium","lang":"rust","type":"locate",
 "q":"Who calls compute_checksum? ...",
 "ground_truth":"compute_checksum is called by ...",
 "check":{"kind":"who_calls","symbol":"compute_checksum","expect_members":[...],"min_count":70}}
```

`type` is **`locate`** ("where is X / who calls X / find usages of X") or
**`research`** ("how does subsystem X work", "what breaks if Y changes", "trace
the data flow from A to B", "which module owns Z and what are its entry
points"). The `check` descriptor makes each ground truth mechanically
verifiable.

## Run it

```bash
# 0. build grepplus
cargo build --bin grepplus

# 1. generate the corpus (git-inits each repo)
bash bench/agent_efficiency/gen_corpus.sh

# 2. validate class coverage, corpus + ground truth WITHOUT any API key
python3 bench/agent_efficiency/verify_task_classes.py
python3 bench/agent_efficiency/verify_tasks.py --index

# 3. provide the model key VIA ENV (never commit it) and run the A/B benchmark
export MINIMAX_API_KEY=sk-...
# macOS alternative for Codex/Desktop processes:
# launchctl setenv MINIMAX_API_KEY sk-...
python3 bench/agent_efficiency/parallel_acceptance_run.py   # product gate: grepplus vs explorer
python3 bench/agent_efficiency/run_bench.py            # raw runner only
python3 bench/agent_efficiency/run_bench.py --repo go_small   # one repo
python3 bench/agent_efficiency/run_bench.py t001 t042         # a subset

# 4. aggregate report: MEDIAN and MEAN factors for INPUT and OUTPUT tokens
#    separately, by repo-size, task-type, and language
python3 bench/agent_efficiency/run_bench.py --report

# 5. mandatory post-benchmark forensics
#    --enforce exits non-zero until every comparable row has quality evidence
python3 bench/agent_efficiency/grade_answers.py \
  --mode mechanical \
  --accept-mechanical \
  --output bench/agent_efficiency/results.mechanical-graded.json
python3 bench/agent_efficiency/forensics.py \
  --results bench/agent_efficiency/results.mechanical-graded.json \
  --baseline explorer \
  --candidate grepplus \
  --output bench/agent_efficiency/FORENSICS_explorer_VS_grepplus.md \
  --enforce

# 6. R3/R8 large-repo index stress gate, no API key required
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

`run_bench.py` writes per-task rows to `results.json` incrementally (with input
and output token usage recorded separately for both agents).

No benchmark run is accepted from the aggregate report alone. A matching
forensics report must exist, and speed wins are only accepted when answer
quality is machine-readable, marked `accepted_for_speed_claim=true`, and not
worse than the baseline. Without that accepted quality evidence, wins are
optimization hints only. Use `--mode smoke` for triage. Use `--mode mechanical
--accept-mechanical` only for the synthetic 100-task bench where `tasks.json`
is the ground-truth contract. `forensics.py --enforce` also treats vector or
EmbeddingGemma usage on `literal_control` / `graph_control` tasks as a router
violation, even when the answer text happens to pass.

The preferred full product command is (R8: parallel, never serial):

```bash
python3 bench/agent_efficiency/parallel_acceptance_run.py \
  --agents grep,grepplus,explorer --parallel 20
```

It writes isolated artifacts under `bench/agent_efficiency/acceptance_runs/<run-id>/`
with a run-local `GREPPLUS_STORE_DIR`. The product GATE is fixed: `grepplus`
versus `explorer` (the uncoached "normales grep" agent, BENCHMARK_CONTRACT
§Baselines); only the explorer forensics leg decides the exit code (exit 2 if
not accepted). The coached `grep` agent is always co-reported as a marked
diagnostic row — it is never the gate and never product status.

Research-only ablation agents such as `plus` or `gemma` can still be requested
explicitly (`run_bench.py --diagnostic`) to diagnose which internal mechanism
helps or hurts. They are not separate grepplus products and must not be
reported as product status. The product claim is always the single `grepplus`
agent versus the uncoached explorer baseline.

The grepplus-agent's system prompt explicitly steers the model to grepplus's
**code-returning context queries** (`grepplus context` / `--code` style
retrieval, provided by `search-code`, which returns the matching source lines)
on top of the graph commands (`who-calls`, `callees`, `find-usages`, `path`,
`trace`), so it pulls real code, not just file:line pointers.

## Security

The API key is read from `$MINIMAX_API_KEY` at runtime and is **never** stored
in any file here. On macOS, `run_bench.py` and `acceptance_run.py` also fall
back to `launchctl getenv MINIMAX_API_KEY` when the current process did not
inherit the shell environment. `minimax-provider.js` only references the env var
(`apiKey: "$MINIMAX_API_KEY"`). The bench scripts never accept the key on argv,
so they are safe to drive from an orchestrator.
