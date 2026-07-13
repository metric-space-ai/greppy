# greppy benchmark suite

The `bench/` directory contains black-box compatibility, freshness, and agent
utility corpora.

| Script | Corpus | What it measures |
|--------|--------|------------------|
| `grep_compat.sh`     | grep-compatibility corpus | Byte-exact stdout/stderr/exit-code preservation against real `grep`, with no index or model side effects. |
| `agent_utility.sh`   | agent-style grep corpus   | Representative recursive agent searches remain byte-exact even with a fresh graph in scope. |
| `freshness_bench.sh` | freshness benchmark          | 9 mutation scenarios (cold start, fresh after index, edit, delete, add, rename, commit, branch, agent-temp-file) asserted against `greppy_freshness::check_files` via the `freshness-probe` example binary. |

A combined runner is provided as `run_all.sh`.

## Running

```bash
cargo build --workspace --examples
bash bench/run_all.sh
```

You can also run an individual script directly:

```bash
bash bench/grep_compat.sh
bash bench/agent_utility.sh
bash bench/freshness_bench.sh
```

## Runtime footprint evidence

`runtime_footprint.py` measures an exact Greppy binary against a real Git
repository without retaining repository content:

```bash
python3 bench/runtime_footprint.py \
  --greppy /absolute/path/to/greppy \
  --repo /absolute/path/to/repository \
  --semantic-query "find the retry scheduler" \
  --brief-symbol "run_retry_loop" \
  --device metal \
  --warm-repeats 5 \
  --output runtime-footprint.json
```

The harness uses and removes a private `GREPPY_STORE_DIR` outside the target
repository. It stops resident daemons between index, semantic-search, and brief
so each first measurement includes a real cold model load; subsequent samples
measure the warm path. Its atomic JSON artifact contains binary and redacted
command-template hashes, platform facts, timings, cache byte counts, sanitized
doctor/backend/model/daemon states, and daemon RSS when a reported PID is
readable. It never stores the query, symbol, source, command output, absolute
repository path, endpoint, secret, or raw trace. Run
`python3 -m unittest bench/test_runtime_footprint.py` for its redaction,
cleanup, failure, timing-schema, and atomic-write tests.

## Hardware evidence

`hardware_evidence.sh` runs one inference-evidence pass of a built greppy
binary on real hardware (backend selection, functional contracts, cold index
and warm p50/p95 latencies, VRAM peak on CUDA) and emits a scrubbed JSON
artifact conforming to `hardware-evidence.schema.json`. Committed artifacts
live in `hardware-evidence/` — see `hardware-evidence/README.md` for what
each platform leg proves and how to re-run it.

The fixture-based scripts are non-destructive: they reindex
`bench/fixtures/sample/` from scratch and clean up after themselves. The runtime
footprint harness reads the explicitly supplied Git repository and keeps its
temporary index outside that repository.
The fixture's git state is committed at the start of the run; each
mutation scenario resets the working tree via `git clean -fdx` +
`git checkout -- .` so successive runs are reproducible.

## How to read the output

Each script prints a `summary` block at the end:

```text
=== grep_compat.sh summary ===
pass: 35
fail: 0
```

A `pass` preserves stdout, stderr, and exit code byte-for-byte against real
grep. Semantic code navigation is exercised separately through explicit
commands such as `semantic-search` and `brief`.

A `fail` is an invocation that violated the contract. The script
prints the expected vs. actual output so a regression can be
diagnosed quickly.

`agent_utility.sh` reports exit code and output byte counts for each invocation.

`freshness_bench.sh` prints `expect / actual / elapsed_ms` per
scenario. Elapsed is the per-check wall time; the production
The Greppy passthrough gate is budgeted at 200 ms (per invocation,
search-path-scoped), while the bench probe uses 30 s because it
walks the whole repo.

## Fixture

`bench/fixtures/sample/` is a hand-crafted Rust project with the
symbols the corpora query. It is committed to a local git repo so
the freshness bench can also exercise the git-fingerprint path.

- `src/lib.rs` — `Greeter`, `ProcessOrder`, `UserService`,
  `InMemoryUserService`, `hello`
- `src/greeter.rs` — secondary module
- `src/orders.rs` — order-handling helpers
- `src/script.py` — Python file (exercises `Language::Unsupported`)

## Adding a new bench entry

1. Decide which script the entry belongs to:
   - Pipeline-sensitive grep invocation → `grep_compat.sh`.
   - Agent-style recursive grep invocation → `agent_utility.sh`.
   - Workspace mutation → `freshness_bench.sh`.
2. Append the entry to the `CORPUS` (or `probe` call) in the
   appropriate script.
3. Re-run `bench/run_all.sh` and verify the entry passes.
4. Commit the new entry + any fixture changes together.

## What this suite does NOT measure

- Statistical agent-utility numbers (real LLM traces, real Bash
  chains, real tool-call counts) — those belong to the
  agent-utility corpus but require running an actual coding agent.
  This corpus is a smoke-test, not a benchmark.
- Performance / wall-time regression detection — the freshness
  bench reports `elapsed_ms` but does not assert on it. A real
  performance suite would gate on a budget.
- Cross-platform behavior — the scripts assume `/usr/bin/grep` and
  macOS-style temp dirs (`$TMPDIR`). A Windows run would need the
  paths and the `find` calls adjusted; that is a future task.
