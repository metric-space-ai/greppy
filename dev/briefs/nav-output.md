Rewrite the TEXT output of `who-calls` and `callees` in
/Users/michaelwelsch/greppy-030 (crates/cli/src/lib.rs).

A previous attempt changed only the not-found messages and left the normal
output untouched. The normal output IS the task. Do the checklist below first;
the not-found messages come last and are the smallest part.

READ FIRST: dev/NAV-OUTPUT-SPEC.md — normative, complete, do not deviate.

MECHANICAL ACCEPTANCE. Every one of these must hold when you are done. Run them
yourself and paste the output in your report.

  A) These strings must no longer be reachable from who-calls or callees:
       grep -n "this sample usually answers" crates/cli/src/lib.rs
       grep -n "unresolved textual candidates" crates/cli/src/lib.rs
       grep -n "(no callers)" crates/cli/src/lib.rs
       grep -n "(no callees)" crates/cli/src/lib.rs
     If a hit belongs to another command, say which and leave it.

  B) Build and run these, paste the exact output:
       cd /Volumes/tmp/outputs-repo && export GREPPY_STORE_DIR=/Volumes/tmp/wc-store
       greppy who-calls parse_path
       greppy who-calls data_set
       greppy who-calls data_set --code
       greppy who-calls resolve_root
       greppy who-calls parse_path zzz_nix
       greppy callees data_set
       greppy callees parse_path
     Expected shapes, from the spec:
       who-calls parse_path  ->  exactly two lines
           edit-src/data.rs:99  data_set
           edit-src/data.rs:234  data_delete
       who-calls data_set    ->  7 lines, the 6 test callers carry a trailing "test"
       who-calls resolve_root -> a "51 callers: <file> <n>, …" line, blank line,
                                 then exactly 5 result lines, nothing else
       callees data_set      ->  12 lines, address is the callee definition start
       callees parse_path    ->  the single line: no callees
     No count line when nothing is hidden. No Expand offer. No path printed twice.
     No "Function::" prefix.

  C) cargo build --release must pass (CARGO_TARGET_DIR is already set for you).

FILE WHITELIST: crates/cli/src/lib.rs and crates/cli/tests/*.rs only.
FORBIDDEN: dispatch_brief, dispatch_find_usages, dispatch_impact, dispatch_path,
search-code, semantic-search, read, and every --json branch. Do not add a
suggestion cascade, git rename detection or semantic matching — the spec's
not-found rules are exactly what is wanted and nothing more.

Existing tests asserting the old strings must be updated to the new spec.
Do not commit unless B and C are green. Commit message:
  feat(nav): who-calls and callees answer instead of packaging

ESCAPE HATCH: need scope beyond the whitelist? STOP and justify in the report.
NO SUBAGENTS.

REPORT TAIL:
  CHANGED: <files>
  CHECK-A: <grep output>
  CHECK-B: <the seven real command outputs, verbatim>
  CHECK-C: <build result>
  OPEN: <what you could not do and why>
  COMMIT: <sha or "not committed">
