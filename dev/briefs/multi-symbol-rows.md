# who-calls/callees for several symbols: the new rows, grouped — no legacy

AGENTS.md promises: "who-calls and callees answer for several symbols at once:
`who-calls A B C`." The single-symbol path emits the 0.3.0 row format
(`file:line  name`). The multi-symbol path still emits the PRE-0.3.0 rows:

    $ greppy who-calls parse_path data_set        # today, WRONG
    edit-src/data.rs::Function::data_set edit-src/data.rs:81
    cli-src/lib.rs::Function::dispatch_edit_inner cli-src/lib.rs:8827

Same disease as the removed find-usages and brief-multi bundles: a legacy
emission surviving behind an argument shape. Remove the legacy function
entirely (grep that nothing else reaches it) and route multi-symbol through
the SAME emission as single-symbol, grouped:

    parse_path
    edit-src/data.rs:99  data_set
    edit-src/data.rs:234  data_delete

    data_set
    cli-src/lib.rs:8842  dispatch_edit_inner

One bare line naming the queried symbol opens each group (that line answers
"which question", it is not decoration); a blank line separates groups; within
a group the rows, truncation, distribution line, failure paths and flags
(--code, --all, --json) behave EXACTLY as in the single-symbol path — reuse
print_nav_rows and the shared helpers, do not fork them. A missing or
ambiguous symbol inside a multi query uses the same failure output as the
single-symbol path, inside its group; exit is 0 if at least one group answered,
1 if none did.

Work in YOUR CURRENT DIRECTORY — the launcher has put you in an isolated
worktree of the repo. Do NOT cd to any other checkout (the acceptance cd to
the test repo is the one exception).

## Acceptance — run and paste REAL output
- cd /Volumes/tmp/outputs-repo && export GREPPY_STORE_DIR=/Volumes/tmp/wc-store2
- greppy who-calls parse_path data_set — new rows, two groups, exit 0
- greppy callees parse_path data_set — same shape
- greppy who-calls parse_path xyzzy_frobnicate — group 1 answers, group 2 the
  missing-symbol output, exit 0
- greppy who-calls parse_path — unchanged single-symbol output (byte-identical
  to before your change)
- grep for the removed legacy emission: zero hits
- cargo test -p greppy --test prompt_contract — 11/11; --test graph_nav green
  (the documented pre-existing reds tolerated)

## FILE WHITELIST
ONLY crates/cli/src/nav.rs, crates/cli/src/lib.rs, crates/cli/src/cli_surface.rs,
crates/cli/tests/*.rs.

Commit message: fix(nav): multi-symbol rows are the single-symbol rows, grouped
ESCAPE HATCH: scope beyond the whitelist -> STOP and justify in the report.
NO SUBAGENTS.

REPORT TAIL: CHANGED / OUTPUT / TESTS / OPEN / COMMIT
