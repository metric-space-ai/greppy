# Bench runbook — how to actually run the benchmarks

The contract documents (`agent_efficiency/BENCHMARK_CONTRACT.md`,
`agent_coding/README.md`) say what is measured. This file says how to run it,
on which machine, with which prerequisites, and which traps have already cost
hours. Written 2026-07-31 after rediscovering four of them by failure in one
session. **Add to it the moment you learn something; do not rely on the
session context — it gets summarized and operational detail is what summaries
drop first.**

## Two benchmarks, two different claims

| Harness | Tasks | Answers |
|---|---|---|
| `agent_efficiency` | 115 (104 locate, 11 research) | navigation: fewer tool calls, source opens, input tokens for the SAME questions. Says nothing about editing. |
| `agent_coding` | 41 (+15 hard) | real work: pinned repo, injected mutation, INDEPENDENT test decides. Arms: explorer / greppy / greppy-edit. |

A 0.3.0 release claim about READ/EDIT needs `agent_coding`. The efficiency
bench alone is evidence for the navigation surface only.

## Machine: gpu3 (`ssh ts-gpu3`, Tailscale 100.71.114.101, 3x RTX A4500)

### Prerequisites, each of which failed once
```bash
# Rust: /usr/bin/cargo is too old — "lock file version 4 requires -Znext-lockfile-bump".
export PATH=$HOME/.cargo/bin:$PATH            # rustup toolchain

# Node: the system node is v12; pi 0.80.2 needs modern node (optional chaining).
# nvm install fails while ~/.npmrc carries a `prefix=` line — delete that line first.
export PATH=$HOME/.nvm/versions/node/v22.23.1/bin:$PATH
npm i -g @earendil-works/pi-coding-agent@0.80.2   # the agent runner: `pi`
npm i -g pnpm                                     # zod tasks (7 of 41)

# Go: hugo tasks (25 of 41) need it; user-level install, no sudo.
export PATH=$HOME/opt/go/bin:$PATH                # go1.23.4 in ~/opt/go

# Provider key (never printed, never committed):
set -a; . ~/.config/secrets/minimax.env; set +a   # MINIMAX_API_KEY
```

One PATH line that satisfies everything:
```bash
export PATH=$HOME/.nvm/versions/node/v22.23.1/bin:$HOME/opt/go/bin:$HOME/.cargo/bin:$PATH
```

### Store and embedding policy
```bash
export GREPPY_STORE_DIR=$HOME/greppy-bench-store   # never the shared user cache
```
`greppy index` exiting 0 proves that the graph generation was published; it does
not prove that semantic embeddings are ready. Keep the same sandbox, Store
namespace, device visibility, and daemon lifecycle alive while polling:

```bash
greppy index
greppy index status --json
```

Start an agent arm only after `index status --json` itself exits 0 and reports
both `healthy: true` and `embedding_complete: true`. Poll with a bounded
deadline and fail the arm as a harness-readiness error on timeout. Do not infer
readiness from elapsed time, `current_embedding_rows`, or the exit status of
`index` alone. A Linux GPU prewarm must expose only the selected NVIDIA device
and required driver nodes, but it must not hide `/dev/nvidia*` entirely or end a
`--die-with-parent` namespace while embedding is still active.

## Navigation bench
```bash
cd ~/greppy-030
python3 bench/agent_efficiency/verify_task_classes.py
python3 bench/agent_efficiency/verify_tasks.py --index        # must end RESULT: PASS
python3 bench/agent_efficiency/verify_real_tasks.py           # 115 tasks, 0 violations
python3 bench/agent_efficiency/parallel_acceptance_run.py \
    --tasks tasks_v2.json --agents grep,greppy,explorer --parallel 5
```
115 tasks x 3 arms ran in ~50 min at parallel 5. Artifacts land in
`bench/agent_efficiency/acceptance_runs/<stamp>/`: `release-gate.json`,
`FORENSICS_*_VS_greppy.md`, `aggregate.txt`, `results.mechanical.json`.

Single-task smoke with full traces (read them before any full run):
```bash
python3 bench/agent_efficiency/run_bench.py --tasks tasks_v2.json --agents greppy \
    --results /tmp/smoke.json --save-raw --raw-dir /tmp/smoke-raw r001 r092 r103
```

## Coding (edit) bench
```bash
python3 bench/agent_coding/run_benchmark.py --tasks bench/agent_coding/tasks_v2.json \
    --task <task-id> --arms greppy-edit --warm-greppy --output-dir /tmp/edit-smoke
```
Grading needs BOTH `explorer` and `greppy-edit` rows per task; running one arm
alone reports `invalid_or_missing_task_ids` — that is the grader missing a
pair, not the run failing. Raw Pi traces are NOT kept in `--output-dir`.

## Traps already paid for

- **The greppy arm's prompt must BE the shipped `AGENTS.md`.** Embedded copies
  drift: the efficiency bench advertised `semantic-search`, `find-usages`,
  `search-symbols`, `search-code`, and the coding bench's whole greppy-edit
  treatment was the retired M4 grammar — every one refused by 0.3.0 with exit
  64. Measured that way, the arm scores against a stale harness, not the
  product.
- **`greppy` must be ON PATH for the agent.** The manual writes `greppy
  who-calls S`; with only an absolute path the agent burns calls on `command
  not found`. `run_bench.py` now symlinks it into a run-local bin dir.
- **A source open is a source open, whoever returns it.** `source_open_calls`
  once counted `cat/head/tail/sed` and the pi read tool but not greppy's own
  `read`/`read-smart`/`read-file`/`--code` — a bias in the candidate's favour
  in the diagnostic metric being compared.
- **The ground-truth verifier speaks the shipped vocabulary too.** It once
  failed 17/100 tasks purely because it invoked retired verbs. `find_usages`
  maps to `who-calls` (0.3.0 walks CALLS+USAGE); multi-term literal checks
  need per-token, order-free verification, and English descriptions belong to
  `search`, not `search-pattern`.

## Known harness defects — the coding bench is NOT valid evidence until fixed

1. **Edit detection misses every 0.3.0 verb.** `re.search(r"greppy[^\n]*\bedit\b")`
   never matches `replace`/`patch`/`rename`/`write`/`undo`/`insert-lines`/
   `delete-lines`, so `edit_calls` stays 0 — and the post-edit re-read gate
   divides by it, passing VACUOUSLY without a single observed greppy edit. The
   file extraction looks for `--file X`, which 0.3.0 does not use (positional).
2. **Unequal tool palettes.** `ARM_TOOLS`: explorer gets `bash,read,edit,write`,
   greppy-edit gets `bash` only. The cost comparison then measures the palette
   cut as well as greppy.
3. **The solution is reachable from inside the worktree.** `git clone --mirror`
   brings the full upstream history including the real fix; the agent has
   `bash`, so `git log --all -p` can be read instead of solving the task.

Until 1-3 are repaired, coding-bench numbers are not a release proof.

## gpu3 environment for the agent benches (agent_efficiency)

A non-interactive `ssh gpu3 '...'` shell does NOT get the right toolchain and
the failure is silent in the results file: every arm records `return_code 1`,
zero tokens, zero tool calls, and `error: None`. Read as "8/8 budget
compliance" that is really "nothing ran".

```
export PATH=$HOME/.nvm/versions/node/v22.23.1/bin:$HOME/.local/bin:$HOME/.npm-global/bin:$PATH
set -a; . ~/.config/secrets/minimax.env; set +a
```

- **`pi` requires Node 22.** The system node is v12.22.9 and the nvm default is
  v20.20.1; both crash inside undici with
  `TypeError: webidl.util.markAsUncloneable is not a function`. Verify with
  `pi --version` (expect 0.80.2) before trusting any run.
- `pi` itself lives at `~/.local/bin/pi`, the provider key at
  `~/.config/secrets/minimax.env` (never echo it).
- Always check a finished run for `return_code` and non-zero token counts
  before computing any ratio from it.
