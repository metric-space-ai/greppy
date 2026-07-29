# The 14 pre-refactor tests: pin the 0.3.0 contracts

Three suites still assert pre-0.3.0 behaviour and are red on this branch
(documented in the brief-rework report): graph_grid_cpp (4: `(no callers)`
wording, retired `search-symbols` verb, `hop 1` prefixes), nav_postel (4:
old who-calls shapes), cli_suggestions (6: old search-code behaviour).

The bench gate requires an all-green baseline. For each red test: understand
what behaviour it pinned, find the 0.3.0 contract for that behaviour (AGENTS.md
+ dev/NAV-OUTPUT-SPEC*.md + dev/SEARCH-OUTPUT-SPEC.md are normative), and
REWRITE the test to pin the new contract at the same strength. Never weaken:
if the old test asserted an exact string, the new one asserts the new exact
string. If a behaviour genuinely no longer exists (a retired verb), the test
becomes the passthrough/refusal assertion for it.

Work in YOUR CURRENT DIRECTORY — the launcher has put you in an isolated
worktree of the repo. Do NOT cd to any other checkout.

## Acceptance — run and paste REAL output
- cargo test -p greppy --test graph_grid_cpp --test nav_postel --test cli_suggestions
  — ALL green.
- cargo test -p greppy --test prompt_contract — 11/11, untouched.
- Per rewritten test: one line in the report naming old contract -> new contract.

## FILE WHITELIST
ONLY crates/cli/tests/graph_grid_cpp.rs, crates/cli/tests/nav_postel.rs,
crates/cli/tests/cli_suggestions.rs. Source files are FORBIDDEN — if a test
cannot be made green without a source change, STOP and report it: that is a
real 0.3.0 defect, not a test problem.

Commit message: test: the pre-refactor suites pin the 0.3.0 contracts
NO SUBAGENTS.

REPORT TAIL: CHANGED / OUTPUT (incl. the old->new mapping) / TESTS / OPEN / COMMIT
