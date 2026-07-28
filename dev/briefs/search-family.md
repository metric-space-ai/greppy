Rebuild the search family per dev/SEARCH-OUTPUT-SPEC.md. Read that file first —
it is normative and complete, with real examples from the sample repo. Work in YOUR CURRENT DIRECTORY — the launcher has put you in an isolated worktree of the repo. Do NOT cd to any other checkout.

THE CODE MOVED. lib.rs was split into modules; the search implementations live
in crates/cli/src/search.rs, the clap surface in crates/cli/src/cli_surface.rs,
shared emitters in emit.rs, the resolver helpers in resolving.rs, and the nav
row helpers (nav_short_name, nav_kind_word, nav_is_test, nav_report_missing)
in crates/cli/src/nav.rs. Reuse them — a second version of any of them is a bug.

TASK:
1. Rename: `semantic-search` -> `search`, `search-symbols` -> `search-symbol`,
   `search-code` -> `search-pattern`. Clap variants, SUBCOMMANDS list, usage
   table, dispatch arms. No aliases. Add the three OLD names to the
   unknown_verb_refusal guard in lib.rs (next to find-usages/references) so
   they cannot fall through to the grep passthrough.
2. `search` joins all trailing positionals into one query string.
3. Output per spec: the three line formats, ranking rules, the 25-row
   distribution rule for search-pattern, at most 8 hits for search, the miss
   cascade on search-symbol, `no matches` + the case-insensitive fact line on
   search-pattern, grep exit codes (0 hit / 1 none / 64 malformed).
4. Delete from these commands: handles in output, `NNNNN |` line prefixes,
   whole-definition dumps, flattened signatures, spans in addresses, expand
   offers, all indentation.
5. AGENTS.md and prompt_contract.rs are ALREADY DONE — the SEARCH section is
   rewritten and guarded. Do not touch either file; your build must merely keep
   `cargo test -p greppy --test prompt_contract` green.

ACCEPTANCE — run and paste real output:
  cd /Volumes/tmp/outputs-repo && export GREPPY_STORE_DIR=/Volumes/tmp/wc-store2
  greppy search restrict a value to a range     # ranked, sentences, no spans
  greppy search-symbol data_set                 # exactly one row, kind at end
  greppy search-symbol dataSet                  # cascade: similar names, exit 1
  greppy search-pattern "data path must start"  # one row: edit-src/data.rs:32  parse_path
  greppy search-pattern zzz_no_such_text        # no matches, exit 1
  greppy semantic-search foo                    # unknown subcommand, not a grep
  cargo build --release
  cargo test -p greppy --lib
  cargo test -p greppy --test prompt_contract
  (graph_nav: symbol_queries_heal_single_file_edits... fails on the untouched
   baseline; every OTHER test must pass.)

FILE WHITELIST: crates/cli/src/search.rs, cli_surface.rs, lib.rs (dispatch +
guard + SUBCOMMANDS only), emit.rs (only if a shared emitter must change),
crates/cli/tests/*.rs (not prompt_contract.rs), lib_tests.rs.
FORBIDDEN: AGENTS.md, prompt_contract.rs, nav.rs, edit.rs, read.rs, every --json shape, the grep passthrough
behaviour itself.

Do not commit unless green. Commit message:
  feat(search): one family, one axis — search, search-symbol, search-pattern
ESCAPE HATCH: need more scope? STOP and justify in the report. NO SUBAGENTS.
REPORT TAIL: CHANGED / OUTPUT (verbatim) / TESTS / OPEN / COMMIT
