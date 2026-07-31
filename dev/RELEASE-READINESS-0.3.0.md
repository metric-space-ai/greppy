# 0.3.0 release readiness, corrected 2026-08-01

> **The 2026-07-31 version of this file was wrong on its central claim.** It
> reported "13 suites + lib: 271 tests green" and concluded the only blocker was
> benchmark evidence. Running the workspace suite the way CI runs it
> (`CI=true cargo test --workspace --features greppy/ci-test-assets --no-fail-fast`)
> shows **15 failing targets**, and CI has been **red on every 0.3.0 branch
> since f627fbb (2026-07-28)** made the platform GPU backend non-optional. The
> release was never one benchmark away from tagging. The corrected picture is
> below.

## The surface is complete

- All four families built and accepted against real output: SEARCH,
  NAVIGATE (where-am-i, who-calls, callees, brief, impact, path), READ
  (read, read-smart, read-file), EDIT (11 verbs).
- `dev/smoke_pass.py`: 33/33 — every line the prompt advertises, run against the
  built binary. This gate is genuine and remains green.

## What is actually red, measured

1. **CI could not build at all on Linux/Windows.** `release.yml` installs the
   pinned CUDA toolkit so `build.rs` can compile the vendored ggml-cuda kernels;
   `ci.yml` never did. Fixed 2026-08-01 (9a28043): Linux CI installs the same
   pinned toolkit, Windows builds `--features cpu-only`.
2. **`--features cpu-only` did not work.** It relaxed only the assertion in
   `lib.rs`, never the target dependency that enables `cuda` unconditionally on
   Linux/Windows, so `build.rs` panicked regardless. A missing nvcc is now a
   configuration rather than a build error. Fixed in the same commit.
3. **`bash_smart.rs` was gated on `unix` only**, so compiling bash-smart out left
   four tests calling a verb the binary no longer has. Fixed.
4. **Six suites tested the removed nested `edit <verb>` surface** that
   `DEAD_EDIT_VERBS` reserves as permanently dead. Removed; `edit_family`
   covers the refusal and stays green.
5. **15 targets still pin pre-0.3.0 output shapes** — twelve `graph_grid_*`
   suites plus `grep_passthrough_safety`, `stats_path_callees` and
   `greppy-parser --lib`. They expect `"(no callers)"` or empty stdout where the
   output laws now produce `"no callers"` / `"no callees"`. In work.
   `grep_passthrough_safety` is treated as a suspected real regression, not a
   stale expectation: greppy must never alter real grep's exit code.

None of items 1-5 is visible from a partial local run, which is how the earlier
claim survived: `bash_smart` sorts before `edit_*`, which sorts before
`graph_grid_*`, so the first failure masked every later one.

6. **Three of the four required workflows cannot run on a release branch at
   all.** `codeql.yml`, `security-audit.yml` and `task-bank-audit.yml` trigger
   only on `push: branches: [main]`, on pull requests, on schedule, or on
   `workflow_dispatch`. A push to `recovery/cli-umbau` triggers none of them,
   which is why the release SHA has no runs for any of the three — nothing was
   forgotten, the gate is simply unsatisfiable from a branch.

   Worse, two carry path filters, so merging is not sufficient either. Measured
   against `main..HEAD`:

   - `security-audit.yml` — watches `**/Cargo.toml`, `Cargo.lock`. 0.3.0 changes
     `Cargo.lock`, `crates/cli/Cargo.toml`, `crates/edit/Cargo.toml`, so a merge
     **does** trigger it.
   - `codeql.yml` — no path filter, a merge triggers it.
   - `task-bank-audit.yml` — watches the task-bank generators and JSON corpora.
     0.3.0 touches **none** of them, so a merge does **not** trigger it, and
     `release.yml` would refuse the tag with no obvious cause.

## The tag sequence, in the only order that works

1. Get `ci.yml` green on the release subject (items 1-5).
2. Land the work on `main` — this is what makes `ci`, `codeql` and
   `security-audit` run on the release SHA.
3. **Dispatch `task-bank-audit.yml` manually on the release SHA.** It cannot
   trigger itself for this change. A `workflow_dispatch` run records the branch
   head as its `headSha`, which is exactly what the gate queries. Skipping this
   step is the one failure mode that looks like a bug in `release.yml`.
4. Bind summary-quality, agent-benchmark and coding-benchmark evidence to that
   same SHA.
5. Tag.

## Other release facts, unchanged

- `AGENTS.md` frozen byte for byte by a twelfth guard (`APPROVED_SHA256`);
  verified that a one-byte append fails it.
- Version bumped to 0.3.0; `CHANGELOG.md` rewritten against the shipped
  binary (the draft described verbs that do not exist).
- `bash-smart` compiled out (`--features bash-smart` for development): its
  training-free layers are green, but the head that makes it smart is 0.4.0,
  and the release ships no name that promises more than it delivers.
- Model redistribution integrity: verified. Workflow action pins: verified
  across 10 workflows.

## What blocks the tag — and it is enforced, not agreed

`.github/workflows/release.yml` triggers on `v*` and refuses unless, bound to
the exact release SHA:

- `ci.yml`, `codeql.yml`, `security-audit.yml`, `task-bank-audit.yml` are
  green on that commit, and
- the publish job can bind summary-quality, agent-benchmark and
  coding-benchmark evidence to that same SHA.

So the remaining release work is not signing, notarizing, SBOM or publishing —
those are CI steps that run on the tag. `tools/release_artifacts.py` even
refuses to record a build environment outside CI ("runner OS mismatch"), which
is correct: provenance produced on a laptop is worthless.

**The release is mechanically blocked on benchmark evidence.** That is the v3
corpus work, and it is the honest reason 0.3.0 cannot be tagged today.

## What the current benchmark evidence actually supports

- The navigation bench (115 tasks, 3 arms, run 2026-07-31) passes its release
  gate against the uncoached baseline: source opens 0.42x, tool calls 0.71x,
  variable input 0.55x, correctness 11 wins / 4 losses / 100 ties. Against the
  coached grep baseline it does not pass (0.93x variable input, 1.15x tool
  calls). It measures navigation only — 104 locate, 11 research, no edit task.
- The coding bench measures editing but is not yet valid evidence: three
  structural defects were repaired on 2026-07-31 (blind edit detection with a
  vacuously passing gate, unequal tool palettes, a workspace from which the
  solution commit was readable), and the v3 redesign that gives it scale and
  an unbiased corpus is in progress.
- Prior coding-bench numbers must not be quoted: on `gate-v4-n30` all three
  arms solved 30/30 and greppy cost 1.11x the baseline (greppy-edit 1.99x,
  partly from the palette defect). There is no prior positive coding-bench
  cost evidence.
- The "0.46x" release criterion is not recorded anywhere in this repository
  and could not be traced to a harness. Until its provenance is established it
  must not be used as a pass/fail line.

## Therefore

0.3.0 is **feature-complete on the surface it advertises** — the smoke gate
proves every prompt line against the binary — and **not releasable** for two
independent reasons, either of which alone blocks the tag:

1. **The mechanical gates are not green on the release subject.** CI must pass
   on the exact tagged SHA, and codeql, security-audit and task-bank have never
   run on it at all. The build-level causes are fixed; the remaining 15 stale
   test targets are in work.
2. **There is no benchmark evidence.** That is the v3 corpus work, whose
   feasibility and validation-survival rate are now measured
   (`bench/agent_coding/v3/CORPUS_AMENDMENT.md`).

The honest summary is that the release needed more than a benchmark, and the
previous version of this file said otherwise. Both blockers are now named,
measured, and being worked rather than assumed absent.
