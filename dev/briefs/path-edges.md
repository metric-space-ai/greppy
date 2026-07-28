Rewrite `greppy path` per dev/NAV-OUTPUT-SPEC-PATH.md. THE CODE MOVED: lib.rs was split; dispatch_path and every nav_* helper live in crates/cli/src/nav.rs now. Read that file first —
it is normative and contains the exact reference output, produced from the real
index of the sample repo. Work in YOUR CURRENT DIRECTORY — the launcher has put you in an isolated worktree of the repo. Do NOT cd to any other checkout.

THE POINT: a path is made of EDGES, not of nodes. Today `path` prints one node
per line with the node's definition address, which is indistinguishable from
`callees` output. The new output prints the start symbol at its definition and
then one line per call, indented by the step, where the address is the CALL SITE
inside the parent and the name is what is called there. Those are the lines an
agent edits to break a connection, and no other command produces them.

Several paths become one tree with the common prefix written once. Every leaf of
the tree is the target — that is what separates `path` from `impact`.

REUSE, DO NOT REINVENT:
  nav_short_name, nav_refuse_non_callable, nav_refuse_ambiguous,
  nav_report_missing, NavDirection
`path` gets the same refusals as `who-calls`. If you write a second version of
any of them, stop and use the one that is there.

ACCEPTANCE — run these and paste the real output in your report:

  cd /Volumes/tmp/outputs-repo && export GREPPY_STORE_DIR=/Volumes/tmp/wc-store
  greppy path --from data_set --to parse_path
      -> exactly two lines, the second indented, address edit-src/data.rs:99
  greppy path --from data_set --to sha256_hex
      -> must match the reference tree in the spec line for line (14 lines)
  greppy path --from parse_path --to data_set     # no path from … to …, exit 0
  greppy path --from data_set --to data_set       # no path from … to …, exit 0
  greppy path --from run --to parse_path          # 5 definitions, exit 1
  greppy path --from data_set --to xyzzy_frobnicate   # no symbol, exit 1
  greppy path --from data_set data_delete --to parse_path
      -> clap's "unexpected argument" plus usage, exit 64, unchanged

  cargo build --release
  cargo test -p greppy --test graph_nav
    -> `symbol_queries_heal_single_file_edits_and_wait_for_edit_refresh` fails on
       the untouched baseline too; it is an indexer failpoint test and not yours.
       Every OTHER test in that target must pass.

ALSO in this change, both are documented lies in `path --help`:
  * `--edge USES` and `--edge TYPE_REF` are accepted but the store holds `USAGE`
    and `TYPE_ASSIGN`; they match nothing and return exit 0. Accept the stored
    names and reject the others as invalid values.
  * `--code` is described in its own help as "Accepted for agent ergonomics —
    no-op". Remove the flag from `path`.

FILE WHITELIST: crates/cli/src/nav.rs, crates/cli/src/cli_surface.rs, crates/cli/tests/*.rs.
FORBIDDEN: dispatch_who_calls, dispatch_callees, dispatch_brief, dispatch_impact,
search-code, semantic-search, read, edit, AGENTS.md, and the `--json` shape of
every command including path's own.

Do not commit unless the acceptance block is green. Commit message:
  feat(nav): path is a tree of call sites

ESCAPE HATCH: if you need scope beyond the whitelist, STOP and justify it in the
report. Never widen on your own. NO SUBAGENTS.

REPORT TAIL:
  CHANGED: <files>
  OUTPUT: <the seven real command outputs, verbatim>
  TESTS: <the graph_nav result line>
  OPEN: <what you could not do and why>
  COMMIT: <sha or "not committed">
