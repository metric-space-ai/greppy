# 0.3.0 release readiness, 2026-07-31

## The code is done and locally green

- All four families built and accepted against real output: SEARCH,
  NAVIGATE (where-am-i, who-calls, callees, brief, impact, path), READ
  (read, read-smart, read-file), EDIT (11 verbs).
- 13 suites + lib: 271 tests green. `dev/smoke_pass.py`: 33/33 — every line
  the prompt advertises, run against the built binary.
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

0.3.0 is **feature-complete, self-consistent and locally verified**, and
**not releasable** until the v3 coding corpus exists and the benchmark
workflows run green on the release commit. Tagging earlier would produce a
release whose own workflow refuses it.
