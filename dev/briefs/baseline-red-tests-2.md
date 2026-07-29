# Baseline debt, round 2: four more suites pin pre-0.3.0 contracts

Found during the search --path acceptance (verified pre-existing: identical
pass/fail with the fix's sources reverted):

- `search_code_empty` — 0/2: tests a `search-code` subcommand that no longer
  exists (dead-listed vocabulary). Likely a retirement pin, like edit_m4:
  dead vocabulary is REFUSED (exit 64, "unrecognized subcommand"), never
  grepped, never answered.
- `context_and_code` — 25/26: one red, `who_calls_code_includes_callers_body`.
  Diagnose against the 0.3.0 contract (who-calls --code prints the byte-exact
  statement span per NAV-OUTPUT-SPEC); rewrite the pin to the new contract.
- `output_budget` — 1/3: two reds. Same treatment.
- `semantic_fallback` — 0/3, "missing-asset environment tests". Diagnose
  FIRST: if these tests exercise behaviour with model assets absent, run them
  in an environment that matches their assumption (or fix their setup); if
  they pin removed semantic-search behaviour (dead vocabulary), they become
  retirement pins.

For every red test: name the old contract, find the 0.3.0 contract (AGENTS.md
+ dev/*-OUTPUT-SPEC*.md normative), REWRITE at the same strength — never
weaken. If a test cannot go green without a source change, STOP for that test
and report it as a real 0.3.0 defect (that rule found and fixed a real
who-calls bug in round 1).

Work in YOUR CURRENT DIRECTORY — the launcher has put you in an isolated
worktree of the repo. Do NOT cd to any other checkout.

## Acceptance — run and paste REAL output
- cargo test -p greppy --test search_code_empty --test context_and_code
  --test output_budget --test semantic_fallback — ALL green (or the reported
  defects listed).
- cargo test -p greppy --test prompt_contract — 11/11, untouched.
- Per rewritten test: one line old contract -> new contract.

## FILE WHITELIST
ONLY crates/cli/tests/search_code_empty.rs, crates/cli/tests/context_and_code.rs,
crates/cli/tests/output_budget.rs, crates/cli/tests/semantic_fallback.rs.
Source files FORBIDDEN — a test needing a source change is a reported defect.

Commit message: test: baseline round 2 — four suites pin the 0.3.0 contracts
NO SUBAGENTS.

REPORT TAIL: CHANGED / OUTPUT (incl. old->new mapping) / TESTS / OPEN / COMMIT
