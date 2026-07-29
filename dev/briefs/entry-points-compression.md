# where-am-i: the entry-points wall becomes fractal

`greppy where-am-i` passed acceptance except one line found afterwards: the
`entry points:` line names 60+ file paths in one breath (and `tests:` mixes
granularities — "inline #[test] modules, inline test definitions"). That
violates the fractal size law: no level of output drops thousands of tokens at
once; each level is at most a screen, the rest priced behind an expand id.

Target shape (adapt wording to the existing style):

    entry points: crates/cli/src/main.rs, bench/agent_efficiency/run_bench.py,
      tools/release_artifacts.py … 58 more — greppy expand <id>
    tests: crates/cli/tests/, crates/edit/tests/, training/qwen35/tests/,
      inline #[test] modules

Rules:
- The shown few are the most central ones: prefer roots the graph knows are
  reached from outside (main.rs, build.rs) and paths at the repo's top levels;
  cap at ~4-6 shown.
- ONE count for the remainder, true, with a real expand id whose pack lists
  every entry point (reuse the same pack machinery the inventory rows use).
- `tests:` lists each distinct root once, no duplicate phrasings.
- The --json shape carries the full list as today (JSON is for machines; the
  fractal law binds the text).

Work in YOUR CURRENT DIRECTORY — the launcher has put you in an isolated
worktree of the repo. Do NOT cd to any other checkout.

## Acceptance — run and paste REAL output
- cargo run -p greppy -- where-am-i in this repo — the entry-points line shows
  at most 6 paths + one true count + expand id; paste the line and run the
  expand: its list length equals shown+count.
- cargo test -p greppy --test prompt_contract — 11/11 green.

## FILE WHITELIST
ONLY crates/cli/src/nav.rs (and cli_surface.rs if a flag needs wiring),
crates/cli/tests/*.rs.

Commit message: fix(where-am-i): entry points obey the fractal size law
ESCAPE HATCH: scope beyond the whitelist -> STOP and justify. NO SUBAGENTS.

REPORT TAIL: CHANGED / OUTPUT / TESTS / OPEN / COMMIT
