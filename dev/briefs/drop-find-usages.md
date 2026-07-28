Delete the `find-usages` command and let `who-calls` absorb what was true in
it. Work in YOUR CURRENT DIRECTORY — the launcher has put you in an isolated worktree of the repo. Do NOT cd to any other checkout.

WHY (measured, do not re-litigate): on `Snapshot` in the sample repo
find-usages reported 15 of 43 real references and zero imports, while the
system prompt promises "calls, uses, imports". A rename driven by it breaks the
build. See dev/NAV-OUTPUT-SPEC.md, the amendment at the top.

TASK, in this order:

1. `who-calls` walks `CALLS` **and** `USAGE` edges, not only `CALLS`.
   `dispatch_find_usages` already walks the reference edges — take the edge
   selection from there. Everything else about `who-calls` stays exactly as it
   is now: the row format, the ordering, the 25-row threshold, `--code`, the
   refusals. Do not touch `print_nav_rows`, `nav_statement_span`,
   `nav_short_name`, `nav_is_test` or `nav_kind_word`.

2. Because a struct now HAS answers, relax the incoming refusal:
   `NavDirection::Incoming::answerable` must accept every kind except those
   that cannot be referenced at all. Keep `Outgoing` exactly as it is.
   `who-calls Snapshot` must then print rows, not a refusal.

3. Delete the `find-usages` subcommand: the clap variant, `dispatch_find_usages`
   and its dispatch arm, its JSON branch, its tests. No alias, no deprecation
   message, no "was replaced by" line — a removed command is an unknown
   subcommand, and clap already says that. Remove it from AGENTS.md too (the
   NAVIGATE section), and nowhere else in that file.

4. Any helper that only `find-usages` used dies with it. `cargo check` must have
   no `never used` warning that your change introduced.

ACCEPTANCE — run these and paste the real output in your report:

  cd /Volumes/tmp/outputs-repo && export GREPPY_STORE_DIR=/Volumes/tmp/wc-store
  greppy who-calls Snapshot          # rows now, no refusal, exit 0
  greppy who-calls parse_path        # unchanged: two rows, exit 0
  greppy who-calls data_set          # unchanged: 7 rows, 6 marked test
  greppy find-usages Snapshot        # unknown subcommand, non-zero exit
  greppy who-calls run               # unchanged refusal, 5 addresses, exit 1

  cargo build --release
  cargo test -p greppy --test graph_nav
    -> `symbol_queries_heal_single_file_edits_and_wait_for_edit_refresh` fails
       on the untouched baseline too; it is an indexer failpoint test and not
       yours. Every OTHER test in that target must pass.

FILE WHITELIST: crates/cli/src/lib.rs, crates/cli/tests/*.rs, AGENTS.md.
FORBIDDEN: dispatch_brief, dispatch_impact, dispatch_path, search-code,
semantic-search, read, edit, and the `--json` shape of any command other than
find-usages (which goes away entirely).

Do not commit unless the acceptance block is green. Commit message:
  refactor(nav): who-calls answers every reference; find-usages goes

ESCAPE HATCH: if you need scope beyond the whitelist, STOP and justify it in
the report. Never widen on your own. NO SUBAGENTS.

REPORT TAIL:
  CHANGED: <files>
  OUTPUT: <the five real command outputs, verbatim>
  TESTS: <the graph_nav result line>
  OPEN: <what you could not do and why>
  COMMIT: <sha or "not committed">
