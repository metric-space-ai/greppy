Implement the ORIENT dissolution per dev/ORIENT-DISSOLUTION-SPEC.md. Read that
file first — it is normative, including the amendment blocks. Work in YOUR
CURRENT DIRECTORY — the launcher has put you in an isolated worktree of the
repo. Do NOT cd to any other checkout.

TASK:
1. NEW `where-am-i` (NAVIGATE): one screen, facts only, empty categories
   omitted. Root line (path, languages, files, definitions), one line per
   top-level entry largest-first with size, most-used symbols (highest
   incoming CALLS+USAGE degree, ties by name) and an expand id; entry points
   from the graph; test roots from the indexer. Single-child directory chains
   collapse (src/main/java/… shows the first level with more than one child).
2. The expand packs are the fractal census: a level with ≤25 definitions
   delivers full rows `file:line  name  kind` (trailing `test` where it
   applies), grouped by file; a bigger level delivers one line per child in
   the same shape as the hub (name, count, most-used, expand id); a file
   above the budget paginates like read-file. Packs store content hashes and
   relocate-or-refuse on drift. Reuse the existing expand-pack machinery and
   nav_short_name/nav_kind_word/nav_is_test.
3. DELETE the commands `map`, `outline`, `changes`, `verify`: clap variants,
   dispatch arms, SUBCOMMANDS entries, their dispatchers (map.rs, changes.rs,
   verify.rs where they die entirely), and impact's undocumented
   --since/--base git scopes. Add all four names to the unknown_verb_refusal
   guard so none becomes a grep pattern. `edit --verify` is untouched.
4. Update tests that exercise the dead commands; add tests for where-am-i
   (hub shape, fractal pack, guard on dead names).
5. AGENTS.md is handled by the orchestrator — do not touch it. Keep
   `cargo test -p greppy --test prompt_contract` green.

ACCEPTANCE — run and paste real output:
  cd /Volumes/tmp/outputs-repo && export GREPPY_STORE_DIR=/Volumes/tmp/wc-store2
  greppy where-am-i                     # hub, one screen, module lines with expand ids
  greppy expand <id of edit-src/>       # ≤25 defs? full rows : child lines
  greppy map                            # unknown subcommand, not a grep, exit 64
  greppy outline x.rs                   # unknown subcommand, exit 64
  greppy changes                        # unknown subcommand, exit 64
  greppy verify -- true                 # unknown subcommand, exit 64
  cargo build --release && cargo test -p greppy --lib
  cargo test -p greppy --test prompt_contract
FILE WHITELIST: crates/cli/src/{lib.rs,cli_surface.rs,nav.rs,map.rs,
changes.rs,verify.rs,emit.rs}, crates/cli/tests/*.rs, lib_tests.rs.
FORBIDDEN: AGENTS.md, prompt_contract.rs, search.rs, read.rs, edit.rs,
every stable --json shape except deleting those of dead commands.
Do not commit unless green. Commit message:
  feat(nav): where-am-i is the hub; map, outline, changes and verify go
ESCAPE HATCH: need more scope? STOP and justify. NO SUBAGENTS.
REPORT TAIL: CHANGED / OUTPUT (verbatim) / TESTS / OPEN / COMMIT
