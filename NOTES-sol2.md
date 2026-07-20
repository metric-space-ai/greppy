# NOTES — solution blocked by crate-only scope

The requested contract gaps cannot be closed end to end while editing only
`crates/edit/`; all three workstreams require changes in the forbidden CLI
caller at `crates/cli/src/lib.rs`.

## Workstream 1 — verb-path CAS

`replace_body`, `insert_adjacent`, and `delete_span` receive only a path and a
byte range, then take their first `Snapshot` inside the verb. After a mutation
has already happened, that snapshot contains no information from which the
planned file SHA-256 or target SHA-256 can be recovered. The expected hashes
can be added to `VerbOptions`, but `resolve_edit_target` / `dispatch_edit` in
`crates/cli/src/lib.rs` must capture and pass those hashes from target
resolution; otherwise the real direct CLI verb path still has no CAS
precondition. The four ignored tests currently also call
`VerbOptions::default()` and provide no planned bytes or hashes, so a genuine
file-SHA plus target-SHA comparison cannot make them stale without changing
their setup or adding caller wiring. Inferring staleness from a changed parsed
node length would only fit these fixtures and would not satisfy compare-and-swap
for same-length mutations.

## Workstream 2 — compact stdout versus full report

`finish_edit` in `crates/cli/src/lib.rs:7064-7072` serializes `Certificate` once,
prints that exact string to stdout, and writes the same string to `--report`.
The edit crate is not told the report path or serialization destination, so it
cannot emit a compact stdout representation and a heavy report representation
selectively. The edit crate can expose separate compact/full serializers, but
`finish_edit` must call the compact serializer for stdout and the full serializer
for the report file. Changing `Serialize` globally would make both destinations
compact and violate the requirement that the report retain node text,
postcondition detail, and validator output.

## Workstream 3 — residual postcondition

The edit crate can add `VerbOptions::expect_residual`, scan same-language
workspace files, and report `residual_occurrences`. However, the forbidden CLI
`EditCommand::RenameSymbol` and `EditCommand::ChangeSignature` variants have no
`expect_residual` field, and their `VerbOptions` construction cannot receive a
`--expect-residual` value. Without edits to the CLI command definitions and
dispatch arms, the required option cannot exist on the user-facing paths and
the planted-leftover override cannot be exercised as `--expect-residual 1`.

Per the task's hard rule, implementation stopped rather than introducing
fixture-specific stale heuristics or serialization behavior that would leave
the actual CLI contract broken. No files under `crates/edit/` were modified.

## Verification at stop point

`cargo test -p greppy-edit` passes the unchanged baseline: 84 tests pass and the
four `grid_rust_*_stale` tests remain ignored, so the requested 88/88 state was
not reached. `cargo clippy -p greppy-edit -- -D warnings` passes with no
warnings.
