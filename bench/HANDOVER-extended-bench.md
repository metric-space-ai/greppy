# Handover: building the extended 0.3.0 coding bench

Written 2026-07-31 by the orchestrator who ran the 115-task navigation bench
and smoked the coding bench. Everything below cost real time today or was
found by a reviewer; none of it is theory. Read `bench/RUNBOOK.md` for the
machine setup (gpu3, toolchains, PATH, store policy) — this file is about
getting the MEASUREMENT right.

## The one rule that generated most of the defects

**Anything the harness says about greppy drifts away from what greppy is.**
Three separate copies of the tool's vocabulary had gone stale unnoticed:
the navigation prompt, the coding-bench prompts, and the ground-truth
verifier. All three advertised or invoked verbs that 0.3.0 refuses with exit
64. A bench built on a stale copy measures the copy, not the product.

Consequence for the new bench: the greppy arm's treatment must BE the shipped
`AGENTS.md`, read from the repo at run time, never an embedded paraphrase.
Record its sha256 in the manifest. If a check needs to invoke greppy itself
(ground-truth verification), it must use the shipped vocabulary too, and a
CI-cheap test should assert that no dead verb string appears anywhere in
`bench/`.

## Fairness: the things that silently tilt the result

1. **A source open is a source open, whoever returns it.** Counting
   `cat`/`head`/`sed` and the agent's `read` tool but not `greppy read` /
   `read-file` / `--code` puts the same act on the baseline's bill and off the
   candidate's. That was live in the navigation bench today, in the exact
   metric the release gate rests on.
2. **Identical tool palettes.** The coding bench gave explorer
   `bash,read,edit,write` and greppy-edit only `bash`. Then the comparison
   measures the palette cut too. Both arms get the same tools; the ONLY
   intended difference is the manual in the system prompt.
3. **A modern baseline has ripgrep.** The old baseline prompt explicitly
   denied `rg`. Beating an artificially crippled agent proves nothing; give
   Arm A `rg` + read + edit and let greppy earn the difference.
4. **Prompt tokens are a first-order cost, not a footnote.** `AGENTS.md` is
   ~1,500 tokens and is re-sent every turn. Measured today: greppy's first
   turn cost 3,704 prompt tokens vs 2,439 (explorer) and 2,472 (grep). Arm B
   must earn that back through fewer/cheaper turns. REPORT the prompt overhead
   as its own line, otherwise the headline factor is uninterpretable — it
   moved from 0.46x (short prompt, 0.2.1) to 0.93x vs grep today purely
   because the manual got longer and honest.
5. **Report median AND mean with n.** Today: input median 0.81x but mean
   1.35x — the median task and the hard tasks tell opposite stories. A single
   number hides that.

## Gates: make them fail loudly, never vacuously

The post-edit re-read gate divided by the number of observed greppy edits.
Detection missed every 0.3.0 verb (it matched the literal word `edit`), so the
count was 0, the rate was 0.0, and the gate PASSED — hardest exactly when
nothing had been observed. Any gate of the form "rate = X/Y" needs an explicit
`Y == 0 -> FAIL, reason stated` branch.

Same class of bug: the file attribution looked for `--file X`; 0.3.0 takes the
file as a positional. Derive detection from `AGENTS.md`'s EDIT section, not
from memory of an older grammar.

## Leakage: two independent doors, both open today

1. **History.** `git clone --mirror` + `git worktree` gives the agent the full
   upstream history including the real fix; the agent has `bash`, so
   `git log --all -p` reads the solution. The agent's workspace must contain
   exactly ONE commit (pinned tree + any mutation). Prove it in the harness:
   `git log --all --oneline | wc -l == 1`.
2. **The task id.** `expected_task_id()` is `{repo}-{type}-{fix_commit[:12]}`
   — the id literally carries the solution's sha prefix. Never let the id
   reach the agent's workspace, prompt, or environment; and consider deriving
   ids from a content hash instead.
3. Also verify the hidden tests are not present during the agent phase —
   apply the test patch only after the agent finishes, and keep PASS_TO_PASS
   as a regression guard so a fix that breaks neighbours is not a success.

## Method that is already right — do not reinvent

`swe_bench_adapter.py` already implements the honest shape: merged PR closing
a real issue, repo state at `M^1` (the commit BEFORE the fix), issue title +
body verbatim as the prompt (not authored, not leading), the PR's own tests as
hidden verification, and a task is kept only if it mechanically validates
(FAIL_TO_PASS genuinely flips red->green with the gold patch, PASS_TO_PASS
stays green). 18 repositories across python-pip / rust-cargo / go-test /
java-maven / ts-pnpm are already configured.

The gap is scale and spread, not method: the harvest ran on 6 of 18 repos at
`--per-repo 12`, which is why the bank is 41 tasks and 61% hugo. Run it wide
(`--repos` all, `--per-repo` 15-18) and cap any single repo at ~10% of the
final bank.

**Budget reality:** harvesting is minutes (gh API). Validation is the brick:
every candidate needs clone + setup + two test runs, so ~20-60 CPU-hours for
~250 candidates. gpu3 has 16 cores and 1.8T free. Maven is NOT installed there
yet — without it the whole Java family drops out.

## Process, from the day that produced this file

- **Read three full trajectories before any full run.** Three tasks exposed a
  56 KB single-command token bomb (`impact` ignoring the size law) and a
  `command not found` loop. Aggregates over 115 tasks hide both.
- **Measure the fix, don't assume it.** Same three tasks after the two fixes:
  context 15,150 -> 1,741 chars, one task 112s -> 9.2s.
- **Budget parity and censored runs.** Same turn/timeout cap for both arms, and
  record cap hits separately — a truncated run is a censored observation, not
  a capability failure.
- **Partial edits after a failed edit** (the owner's metric 5) is where
  greppy's transactional promise is actually tested: capture the workspace's
  dirty state BEFORE any cleanup, per arm.
- Diagnostic capability tags belong AFTER the fact, for the question "where
  does greppy help". They must never become targets, or the bench optimizes
  the tool instead of measuring it.
