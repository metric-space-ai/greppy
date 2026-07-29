# The search family honours --path — ON EVERY COMMAND means every command

AGENTS.md's footer promises `--path P — only results under that file or
directory` for EVERY command. The search family rejects it:

    $ greppy search-pattern "parse_path" --path edit-src
    error: unexpected argument '--path' found

(who-calls and friends already honour it: `no callers under path filter: …`.)

Wire `--path P` (repeatable, like the nav commands) into search,
search-symbol and search-pattern: filter hits to files under any given path
BEFORE counting, so every printed count is the count of the filtered set —
a count from before the filter is a false number (that exact bug was already
fixed once for --kind; reuse that discipline and its helpers). Empty result
under the filter says so the way the nav commands do, with the search
family's grep exit codes (1 for no hit). --json carries the same filtered
numbers.

Work in YOUR CURRENT DIRECTORY — the launcher has put you in an isolated
worktree of the repo. Do NOT cd to any other checkout (the acceptance cd is
the one exception).

## Acceptance — run and paste REAL output
- cd /Volumes/tmp/outputs-repo && export GREPPY_STORE_DIR=/Volumes/tmp/wc-store2
- greppy search-symbol parse_path --path edit-src — hits only under edit-src, exit 0
- greppy search-symbol parse_path --path no-such-dir — says so, exit 1
- greppy search-pattern "fn parse_path" --path edit-src --json — counts equal the filtered rows
- greppy search "restrict a value to a range" --path edit-src — filtered, grep exit codes
- cargo test -p greppy --test prompt_contract — 11/11; the search suites green.

## FILE WHITELIST
ONLY crates/cli/src/search.rs, crates/cli/src/cli_surface.rs, crates/cli/tests/*.rs.

Commit message: fix(search): --path filters before every count
ESCAPE HATCH: scope beyond the whitelist -> STOP and justify. NO SUBAGENTS.

REPORT TAIL: CHANGED / OUTPUT / TESTS / OPEN / COMMIT
