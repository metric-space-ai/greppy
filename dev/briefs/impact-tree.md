Rewrite `greppy impact` so its output is a tree, per
dev/NAV-OUTPUT-SPEC-IMPACT.md. Read that file first — it is normative and it
contains the exact reference output, produced from the real index of the sample
repo. Work in YOUR CURRENT DIRECTORY — the launcher has put you in an isolated worktree of the repo. Do NOT cd to any other checkout.

THE SUBSTANTIAL PART IS NOT PRINTING. The current walk records how far a node
is, not which edge it arrived on, so the parentage that the tree needs does not
exist yet. Change the collection to keep the parent. Do NOT re-derive it with a
second query per node.

REUSE, DO NOT REINVENT. These already exist and are already correct:
  nav_short_name, nav_is_test, nav_kind_word, nav_file_lines,
  nav_refuse_non_callable, nav_refuse_ambiguous, nav_report_missing,
  NavDirection
`impact` gets the same rows and the same four refusals as `who-calls`. If you
find yourself writing a second version of any of them, stop and use the one
that is there.

ACCEPTANCE — run these and paste the real output in your report:

  cd /Volumes/tmp/outputs-repo && export GREPPY_STORE_DIR=/Volumes/tmp/wc-store
  greppy impact parse_path
      -> must match the reference tree in the spec line for line
  greppy impact parse_path --depth 1
      -> the two direct callers only, no indentation below them
  greppy impact data_set --direction outgoing
      -> a tree of callees; `LimitedFrame::write` WILL appear, that is a known
         resolver defect and you must NOT filter it away
  greppy impact Snapshot          # `Snapshot` is a struct, not a function, exit 1
  greppy impact run               # 5 definitions, five addresses, exit 1
  greppy impact xyzzy_frobnicate  # no symbol, exit 1

  cargo build --release
  cargo test -p greppy --test graph_nav
    -> `symbol_queries_heal_single_file_edits_and_wait_for_edit_refresh` fails on
       the untouched baseline too; it is an indexer failpoint test and not yours.
       Every OTHER test in that target must pass.

FILE WHITELIST: crates/cli/src/lib.rs, crates/cli/tests/*.rs.
FORBIDDEN: dispatch_who_calls, dispatch_callees, dispatch_brief, dispatch_path,
search-code, semantic-search, read, edit, AGENTS.md, and the `--json` shape of
every command including impact's own.

Do not commit unless the acceptance block is green. Commit message:
  feat(nav): impact is a tree, not a flattened list

ESCAPE HATCH: if you need scope beyond the whitelist, STOP and justify it in the
report. Never widen on your own. NO SUBAGENTS.

REPORT TAIL:
  CHANGED: <files>
  OUTPUT: <the six real command outputs, verbatim>
  TESTS: <the graph_nav result line>
  OPEN: <what you could not do and why>
  COMMIT: <sha or "not committed">
