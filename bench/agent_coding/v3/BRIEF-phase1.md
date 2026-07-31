# v3 Phase 1: a natural sample, not a designed one

`bench/agent_coding/v3/REDESIGN_PLAN.md` is normative — read it first, in
full. This brief implements its Phase 1 only. The goal of the whole redesign
is that no task enters the corpus because someone expected greppy to do well
on it; today the contract still selects by task class, patch size and
navigation pressure, and it gates the release on greppy adoption.

Work in YOUR CURRENT DIRECTORY — the launcher has put you in an isolated
worktree of the repo. Do NOT cd to any other checkout.

## Slice 1 — remove every selection signal that is not technical

From `corpus_contract.json` and the code that reads it (`pipeline.py` and its
tests), remove as ADMISSION or SELECTION criteria:

- `corpus.task_class_quotas` and the per-repository `task_class_slots`
  (`Quotas.task_class_quotas`, the "class slots do not sum to its quota"
  check, and every consumer).
- any minimum/target for changed production files or changed lines.
- selection by cross-file / migration / refactor / bugfix share.
- `validation.minimum_candidate_pool_per_repo_class_slot`.
- `candidate_admission.navigation_pressure_*` as an admission property.
- `release_gate.mechanism.greppy_task_adoption_minimum` — adoption moves to
  DIAGNOSTICS and can never gate a release.

What stays is exactly the technical/scientific exclusion list in the plan
("Auswahlregeln: Was bestehen bleibt") — merged PR with a linked real issue,
reconstructible parent/merge tree, not docs/format/generated/vendor/dep-bump,
observable code or config fix, derivable independent behaviour tests,
parent+hidden fails for the intended reason, gold+hidden and PASS_TO_PASS
reproducibly green, no internet/paid/privileged/mutable-data requirement, no
embargo/credential/solution leak, not already in SWE-bench/V2/denylist, and
executable under the registered budget (with the cause logged, and the cause
may never be a greppy-specific one).

The corpus shape stays: 144 tasks, 24 repositories, 8 languages, 6 tasks per
repository, plus at least 2 sealed reserves per repository.

## Slice 2 — deterministic natural sampling

Implement the plan's sampling exactly:

1. Registry, time window, model, agent, budgets and the selection algorithm
   are frozen BEFORE the harvest (a frozen spec with a hash).
2. Every merged PR of each repository in the window is a candidate — not a
   hand-picked count.
3. Technical exclusions are applied to all candidates.
4. The survivors of each repository are ordered by a deterministic keyed
   pseudo-random rank: `HMAC-SHA256(secret, repo_id + "\\0" + candidate_id)`,
   sorted ascending. The secret is sealed; publish only a commitment
   (e.g. `sha256(secret)`) before the harvest, and the secret itself with the
   corpus, so the ordering is verifiable afterwards but not steerable during.
5. Validation walks that order until 6 tasks plus >= 2 reserves per
   repository reproducibly pass.
6. A failing candidate may only be replaced by the NEXT candidate of the SAME
   repository. No backfill across repositories or toward a wanted class.

## Slice 3 — a candidate ledger that can be audited

Every candidate ever seen is one ledger row: repository, candidate id, HMAC
rank position, admitted or excluded, and on exclusion the exact reason from
the technical list (one enumerated reason, not free text). Validation outcome
and whether it became task / reserve / neither. The ledger must answer: "did
validation quietly leave only one kind of task standing?"

## Slice 4 — the suite runs without a magic environment

`uvx pytest bench/agent_coding/v3/` currently fails collection with
`ModuleNotFoundError: No module named 'bench'` and only passes with
`PYTHONPATH=.`. Fix it properly (conftest.py at the repo root or package
`__init__.py` files) so a plain invocation works — a suite that needs an
undocumented env var is a suite CI will skip.

## Acceptance — run these and paste REAL output
- `uvx pytest bench/agent_coding/v3/ -q` — green WITHOUT PYTHONPATH.
- `grep -rn "task_class_quota\|adoption_minimum\|navigation_pressure\|minimum_candidate_pool" bench/agent_coding/v3/` — no hits outside diagnostics/documentation contexts; show the remaining hits and say why each is not a gate.
- A test proving the HMAC order is stable for a fixed secret and CHANGES for a different secret, and that a repository's replacement comes from the same repository's next rank.
- A test proving a zero-denominator rate is `N/A` and cannot pass a gate.

## FILE WHITELIST
ONLY `bench/agent_coding/v3/**` and, for slice 4, a repo-root `conftest.py`.
FORBIDDEN: `bench/agent_coding/run_benchmark.py`, `bench/agent_efficiency/**`,
`AGENTS.md`, `crates/**`, the v2 task banks.

## Hard rules
- Do not commit unless acceptance is green. Commit message per slice:
  `feat(v3): <slice>`
- ESCAPE HATCH: if you believe you need scope beyond the whitelist, STOP and
  justify it in the report. Never widen on your own.
- If a test pins a removed quota, UPDATE it to pin the new rule — never
  weaken or delete an assertion.
- NO SUBAGENTS.

## REPORT TAIL (fixed form, at the very end)
CHANGED: <files>
OUTPUT: <the acceptance commands' real output, verbatim>
TESTS: <suite result lines>
OPEN: <what you could not do, and why>
COMMIT: <sha(s) or "not committed">
