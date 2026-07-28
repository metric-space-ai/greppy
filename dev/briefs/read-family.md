Rebuild the read family per dev/READ-OUTPUT-SPEC.md. Read that file first — it
is normative and complete. Work in YOUR CURRENT DIRECTORY — the launcher has
put you in an isolated worktree of the repo. Do NOT cd to any other checkout.

THE CODE LIVES IN MODULES: read implementations in crates/cli/src/read.rs, the
clap surface in cli_surface.rs, nav helpers (nav_report_missing,
nav_refuse_ambiguous, nav_short_name) in nav.rs, the span summarizer in
summarize_daemon.rs via summarize_definition_span. Reuse them.

TASK:
1. `read S [S …]` — symbols only, whole, doc-extended span (/// and attributes
   above; the header names the range actually printed). --head M / --tail N,
   combinable. No guess heuristic: a path argument to `read` is not resolved
   as a file any more.
2. NEW `read-smart S [S …]` — structural folding below --depth N (default 1),
   gap lines `    … A-B ⟨sentence⟩ — greppy expand ID`, sentence generated per
   folded block via the summarizer, expands chained (a filled gap folds its own
   sub-blocks). Packs store address + content hash, relocate-or-refuse.
3. NEW `read-file PATH [PATH …]` — 400-line pages, continuation line
   `N more lines — greppy expand ID continues at LINE`, --lines A:B, --all.
4. Failures through the nav helpers: `no symbol`, ambiguity refusal, exit 1
   (never 10), no `read:` prefix. Partial delivery on multi-symbol.
5. Delete: --context, --symbol, --level remnants, the name-vs-path heuristic,
   `read PATH` (moved to read-file).
6. AGENTS.md READ section is ALREADY updated — do not touch AGENTS.md.

ACCEPTANCE — run and paste real output:
  cd /Volumes/tmp/outputs-repo && export GREPPY_STORE_DIR=/Volumes/tmp/wc-store2
  greppy read parse_path                 # doc line included, header 27-71 or true range
  greppy read data_set parse_path        # two blocks, blank line between
  greppy read dispatch_edit_inner --head 20 --tail 5   # two blocks, own headers
  greppy read-smart parse_path           # top level raw, loop folded, gap line with sentence + full expand command
  greppy expand <id from above>          # exactly the folded lines, verbatim
  greppy read zzz_nix                    # no symbol, exit 1
  greppy read run                        # 5 definitions, exit 1
  greppy read-file config.json           # whole file
  cargo build --release && cargo test -p greppy --lib
  cargo test -p greppy --test prompt_contract
  (graph_nav: symbol_queries_heal_single_file_edits... fails on the untouched
   baseline; every OTHER test must pass.)

FILE WHITELIST: crates/cli/src/read.rs, cli_surface.rs, lib.rs (dispatch +
SUBCOMMANDS + guard only), nav.rs (only if a helper needs a pub(crate)),
crates/cli/tests/*.rs, lib_tests.rs.
FORBIDDEN: AGENTS.md, prompt_contract.rs edits beyond keeping it green,
search.rs, edit.rs, every --json shape.

Do not commit unless green. Commit message:
  feat(read): read is bytes; read-smart folds by depth; read-file pages
ESCAPE HATCH: need more scope? STOP and justify in the report. NO SUBAGENTS.
REPORT TAIL: CHANGED / OUTPUT (verbatim) / TESTS / OPEN / COMMIT
