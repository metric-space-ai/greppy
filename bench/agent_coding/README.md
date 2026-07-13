# Paired agent coding benchmark

This harness measures coding outcomes for the same pinned mutation with
MiniMax-M3 through Pi Code:

- `explorer`: an uncoached coding agent using normal repository exploration;
- `greppy`: the same coding agent with a concise Greppy navigation treatment.

This is **coding-outcome evidence that complements, and does not replace, the
navigation benchmark** in `bench/agent_efficiency/`. The navigation benchmark
isolates code-finding behavior; this benchmark asks whether an agent can edit
the repository and pass an independent test.

## Experimental contract

Every pair uses the same pinned repository commit, ordered setup argv arrays,
mutation patch, user task, test argv, timeout, Pi version, MiniMax-M3 model,
built-in tools, shared system prompt, and user prompt. The only intended prompt
delta is the preregistered navigation treatment: the Greppy arm receives the
Greppy command guide, while the explorer arm receives no code-intelligence
workflow. Full prompt hashes, the shared user-prompt hash, and the setup-command
hash are recorded in `MANIFEST.json`.

The order of the two arms is deterministically counterbalanced from the task
ID. Each arm gets a separate temporary Git worktree, `GREPPY_STORE_DIR`, and
`PI_CODING_AGENT_DIR`. Before measurement, a third disposable worktree runs
setup, requires the independent test to pass on the clean pinned source, then
applies the mutation and requires that same test to fail without timing out.
Each arm independently reruns setup before applying the mutation. With
`--warm-greppy`, only the Greppy arm is indexed after mutation but before the
measured agent timer starts; setup and warmup durations are recorded separately.

Pi is launched with the existing explicit
`bench/agent_efficiency/minimax-provider.js` plus `--no-extensions`; the latter
disables ambient extension discovery. This combination was checked against Pi
`0.80.2`, where the explicit extension remains active. Every real harness run
also performs a local `--list-models` registration probe with the exact flags
before cloning tasks. The probe makes no model request, and the manifest records
the Pi version and that the probe passed.

Pi receives `MINIMAX_API_KEY` through its environment because the existing
`bench/agent_efficiency/minimax-provider.js` extension requires it. The key is
never placed in argv, prompts, status output, results, or manifests. Raw agent,
setup, test, warmup, and patch artifacts are exact-value redacted before local
write. Setup, test, and warmup subprocesses do not receive the provider key.

## Task format

Task files must match [`task.schema.json`](task.schema.json). The runtime also
performs strict standard-library validation, so no JSON Schema package is
required. `setup_commands` is required on every task and is an ordered array of
argv arrays. Each inner argv array must be nonempty and contain only nonempty
strings; shell command strings such as `"npm install && npm test"` are rejected.
Use `[]` when a repository needs no setup. Setup and test commands are executed
directly and are never interpreted by a shell.

```json
{
  "schema_version": "greppy.agent-coding-tasks.v1",
  "tasks": [
    {
      "id": "parser-null-guard",
      "repository": {
        "url": "https://github.com/example/project.git",
        "commit": "0123456789abcdef0123456789abcdef01234567"
      },
      "setup_commands": [
        ["python3", "-m", "pip", "install", "--disable-pip-version-check", "-e", "."]
      ],
      "mutation_patch": "diff --git a/src/parser.py b/src/parser.py\n--- a/src/parser.py\n+++ b/src/parser.py\n@@ -10 +10 @@\n-if value is None:\n+if False:\n",
      "user_task": "Restore the parser's null handling without changing public behavior.",
      "test_command": ["python3", "-m", "unittest", "tests.test_parser"],
      "timeout_seconds": 600
    }
  ]
}
```

The commit must be a full 40- or 64-hex object ID. Repository URLs containing
credentials, query strings, or fragments are rejected because the repository
identity is included in the publishable manifest.

## Validate and run

Validation does not clone repositories, invoke Pi, or use the network:

```bash
python3 bench/agent_coding/run_benchmark.py \
  --tasks /path/to/tasks.json \
  --validate-only
```

A real benchmark run requires built `greppy` and `pi` executables and the API
key in the environment:

```bash
export MINIMAX_API_KEY=...
python3 bench/agent_coding/run_benchmark.py \
  --tasks /path/to/tasks.json \
  --greppy-bin target/release/greppy \
  --warm-greppy
```

Use `--task ID` repeatedly to select tasks. Use `--output-dir DIR --run-id ID`
for stable orchestration paths, and add `--resume` to keep atomically completed
arms. Resume refuses changes to the task bank, prompts, binaries, model,
platform, setup contract, or gate contract, and rejects duplicate or foreign
result rows. Changing any setup argv changes both the task-file identity and
the per-task setup hash, so an old run cannot be resumed under new setup. No API
key option exists.

## Measurement and artifacts

For each arm the harness records:

- setup command status, duration, redacted-output hash, status hash, argv hash,
  and aggregate setup hash, separately from agent metrics;
- correctness from the independent post-agent test;
- model-reported total prompt input (including cache read/write accounting),
  uncached input, and output tokens;
- total tool calls and source opens, where a source open is a Pi `read` call or
  a shell call beginning a command segment with `cat`, `head`, `tail`, or
  `sed -n` (the same definition used by `run_bench.py`);
- measured agent wall time, excluding setup, Greppy warmup, and the independent
  test;
- test exit status and duration;
- pre-test and post-test `git diff --binary --full-index` SHA-256 and byte count;
- the final Git `HEAD` and cleanup status.

`results.json` and `MANIFEST.json` are replaced atomically after every arm.
`MANIFEST.json` is publication-safe: it contains the full Greppy source commit
and tracked-worktree state, task/prompt/binary identities, OS/architecture,
executable version strings and hashes, metrics, grading, and input hashes, but
no prompts, source snippets, diffs, command output, trace paths, or secrets.

Raw Pi JSONL, stderr, setup/test/warmup output, and binary patches stay under
`bench/agent_coding/raw_traces/<run-id>/`. That path is locally ignored by Git.
Default checkpoint directories under `bench/agent_coding/runs/` are also ignored;
their manifest can be uploaded directly as a benchmark artifact.

Pi and every command it starts run in their own process group. A timeout kills
the complete process group. Worktree removal runs in `finally`, followed by
forced directory removal and Git worktree pruning; task-level temporary roots
provide a second cleanup layer for exceptions, timeouts, Ctrl-C, SIGTERM, and
SIGHUP. As with any process, SIGKILL or machine power loss cannot execute
cleanup handlers; stale operating-system temporary directories can then be
removed normally.

## Preregistered gate

The gate is fixed in code and copied into every manifest:

1. Greppy must have at least as many observed paired correctness wins as losses.
2. A one-sided exact paired McNemar regression alarm must not fire at `p <
   0.05`. This alarm is not presented as proof of population equivalence.
3. A decision requires at least 30 complete pairs and at least 20 pairs where
   both independent tests pass; smaller runs cannot pass.
4. Only pairs where **both** independent tests pass enter efficiency grading.
5. Across those solved pairs, Greppy must use at most `0.80x` the explorer's
   summed tool calls, at most `0.80x` its summed source opens, **and** at most
   `0.80x` its summed input tokens. All three conditions are required.
6. Wall time is descriptive only and is computed only for solved pairs. A
   failed test can never receive or contribute a speed win.
7. Missing or invalid arms fail the gate. Pi timeouts, nonzero exits, reported
   model errors, and zero-turn sessions make an arm invalid even if a partial
   edit happens to pass the test.

## Unit tests

The tests use only Python's standard library and local temporary Git
repositories. They neither invoke Pi nor access the network:

```bash
python3 -m unittest discover -s bench/agent_coding -p 'test_*.py' -v
```

They cover patch validation/application (including binary diff capture),
worktree cleanup on exceptions, strict argv-only setup validation, setup failure
and secret handling, clean-pass/mutated-fail preflight behavior, setup exclusion
from measured agent wall time, metric parsing and secret redaction, exact paired
correctness grading, the 20% efficiency threshold across all three metrics,
30/20 minimum sample sizes, strict agent success validity, manifest
platform/version provenance, resume identity, and exclusion of failed tests
from speed credit.
