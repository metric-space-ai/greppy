# Baseline debt, round 3: cli_hardening still speaks the dead vocabulary

`cargo test -p greppy --test cli_hardening`: 10 passed, 29 failed. The reds
INVOKE removed verbs (`search-symbols`, `search-code`, `semantic-search`) and
assert their pre-0.3.0 outputs; since the vocabulary refusal landed they die
at invocation ("unrecognized subcommand", exit 64).

For each red test: the hardening property it guards (root resolution, scope
env, freshness, budget, …) is usually still REAL — only its vehicle verb is
dead. Rewrite the test to guard the SAME property through the living verb
(search / search-symbol / search-pattern, new row format, truthful
`"command"` field — the JSON `command` now names the invoked verb, e.g.
"search-pattern", never "search-code"). A test whose entire point was the
dead verb becomes a retirement pin (refusal, exit 64). Never weaken: same
strength, new contract. If a property cannot be guarded without a source
change, STOP for that test and report the defect.

Work in YOUR CURRENT DIRECTORY — the launcher has put you in an isolated
worktree of the repo. Do NOT cd to any other checkout.

## Acceptance — run and paste REAL output
- cargo test -p greppy --test cli_hardening — ALL green (or reported defects).
- cargo test -p greppy --test prompt_contract — 11/11, untouched.
- Per rewritten test: one line old vehicle -> new vehicle, property unchanged.

## FILE WHITELIST
ONLY crates/cli/tests/cli_hardening.rs. Source files FORBIDDEN — a test
needing a source change is a reported defect.

Commit message: test: cli_hardening guards its properties through the living verbs
NO SUBAGENTS.

REPORT TAIL: CHANGED / OUTPUT (incl. mapping) / TESTS / OPEN / COMMIT
