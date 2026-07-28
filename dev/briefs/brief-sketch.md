Rewrite `greppy brief` per dev/NAV-OUTPUT-SPEC-BRIEF.md. Read that file first —
it is normative and contains the reference output built from the real body of
`edit-src/data.rs:81-209`. Work in YOUR CURRENT DIRECTORY — the launcher has put you in an isolated worktree of the repo. Do NOT cd to any other checkout.

THE POINT: today `brief` prints the definition, then `-- CALLERS (n) --`,
`-- REFERENCES (n) --`, `-- CALLS (n) --`. That is who-calls and callees in one
call with ASCII bars around them, and it is byte-identical to v0.2.1 — every
output literal of dispatch_brief is unchanged between the tag and HEAD.

The new `brief` is the body of a function, SKETCHED: the sentence, the verbatim
signature head, then one line per step of the body naming the symbol used there
and what happens. Every symbol name stays real so the agent can carry it into
the next command.

REUSE, DO NOT REINVENT: nav_short_name, nav_is_test, nav_kind_word,
nav_file_lines, nav_refuse_non_callable, nav_refuse_ambiguous,
nav_report_missing, NavDirection.

SCOPE LIMIT — read this twice. The per-line descriptions for steps that contain
NO call would need a generation step greppy does not have. Do not invent one and
do not fake those lines: a sketch line is emitted only for a call site or a
branch the parser can see. Everything else is left out.

ACCEPTANCE — run these and paste the real output in your report:
  cd /Volumes/tmp/outputs-repo && export GREPPY_STORE_DIR=/Volumes/tmp/wc-store
  greppy brief parse_path      # sentence, signature, sketch, no ASCII bars
  greppy brief data_set        # the branch structure of the match statements
  greppy brief Snapshot        # a struct: whole definition, no sketch
  greppy brief run             # 5 definitions, exit 1 — NOT five function bodies
  greppy brief xyzzy_frobnicate    # no symbol, exit 1
  cargo build --release
  cargo test -p greppy --test graph_nav

FILE WHITELIST: crates/cli/src/lib.rs, crates/cli/tests/*.rs.
FORBIDDEN: dispatch_who_calls, dispatch_callees, dispatch_impact, dispatch_path,
search-code, semantic-search, read, edit, AGENTS.md, every --json shape.

Do not commit unless acceptance is green. Commit message:
  feat(nav): brief sketches the body instead of bundling three commands

ESCAPE HATCH: need scope beyond the whitelist? STOP and justify. NO SUBAGENTS.

REPORT TAIL:
  CHANGED / OUTPUT (verbatim) / TESTS / OPEN / COMMIT
