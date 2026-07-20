# NOTES — grid_rust.rs coverage and deviations

## Scope

`crates/edit/tests/grid_rust.rs` is a 4×4 cross-verb × cross-scenario
grid for greppy-edit's four byte-range single-op verbs on Rust sources:

|            | unique | ambiguous | stale            | syntax-breaking |
|------------|:------:|:---------:|:----------------:|:---------------:|
| replace-body | ✓      | ✓ (10)    | **#[ignore]** 12 | ✓                |
| insert-after | ✓      | ✓ (Err)   | **#[ignore]** 12 | ✓                |
| insert-before| ✓     | ✓ (Err)   | **#[ignore]** 12 | ✓                |
| delete       | ✓      | ✓ (Err)   | **#[ignore]** 12 | ✓                |

`✓` = passing, exit code and certificate status match the contract table.

## Ambiguous-cell deviations (exit 10 / `Err`, not 11)

The contract binds exit 11 (Ambiguous) to multi-match operations that
emit a candidate list. The four byte-range single-op verbs here take a
single caller-supplied `def_range`; they have no candidate-list
semantic.

- `replace-body` refuses body-resolution failure as `NotFound` (exit 10).
  The test `grid_rust_replace_body_ambiguous` covers this with a
  `def_range` entirely inside a leading comment line — `body_range_within`
  returns `None`, the verb emits a `not-found` certificate, and the file
  is unchanged.
- `insert_adjacent` and `delete_span` have no refusal-certificate path
  at all. Their only refusal is `Err` from `apply_in_memory` when the
  resolved splice lies past the file's end. The "ambiguous" cells assert
  `is_err()` and `file unchanged` — not a contract exit code. Mapping
  the verb-layer `Err` to a CLI exit code is the calling layer's
  responsibility (the binding is in `docs/contracts/EDIT_CONTRACT.md` exit
  20 for "invalid edit specification" — left to the CLI).

## Stale-cell defects (the four `#[ignore]`d cells)

`docs/contracts/EDIT_CONTRACT.md` row 12 binds exit 12 (Stale) to
operations that carry a CAS hash:

- `replace-span` / `patch-span` carry an `EditHandle` whose `verify`
  re-checks `file_sha256` and `target_sha256` against live content
  (`crates/edit/src/handle.rs:82`).
- `apply_plan` enforces `PlanPreconditions::file_sha256` and
  `expect_git_head` (`crates/edit/src/plan.rs:160`, `:184`, `:141`).

The four byte-range single-op verbs (`replace_body`, `insert_adjacent`,
`delete_span`, plus `rename_in_span`) do **not** thread a handle, a
file hash, or any plan precondition through their signatures. Their
implementations re-read the file on every call and re-resolve the
smallest containing node against the live content. A concurrent
mutation between plan and apply is therefore:

1. **Silently re-applied** when the mutation preserves the resolved
   definition's structure (e.g., body rewrite, sibling declaration) —
   the verb returns `Applied` with `exit_code() == 0`, the pre-call
   edit overwrites the user's mutation, and the certificate reports
   `published: true`. This is a transaction-safety defect: the store
   re-resolution and the live file have drifted, but neither side
   detects it.
2. **Mis-labeled `NotFound`** when the mutation removes the addressed
   definition entirely — the verb returns exit 10, not 12. The file is
   preserved, but the contract binds this case to 12 (file-hash stale),
   not 10 (target not found).

Each `grid_rust_*_stale` test is `#[ignore]`d with a comment pointing
back to this note. The assertion body is the failure fingerprint: when
CAS is added to these verbs (analogous to `EditHandle::verify`), running
`cargo test -p greppy-edit --test grid_rust -- --include-ignored`
should turn those four cells green without changing any other test.
Expected work: introduce a `def_range` hash bound to the verb API
(target span sha256 + file sha256 at read time) and short-circuit with
`Status::Stale` in the same way `replace_span` does.

## Verification assumptions

1. `body_range_within` returns the `block` node span including both
   `{` and `}`; the test's `replace_body_unique` /
   `replace_body_syntax_breaking` therefore pass the full `{\n  ... \n}`
   as the replacement rather than just the body lines.
2. `delete_syntax_breaking` relies on tree-sitter-rust emitting a
   MISSING `}` node for `fn foo() {\n    42\n` after the closing
   brace is removed. `delete_span` calls `run_pipeline(…,
   enforce_structure=false)`, so the structural-context check does
   not fire — only the MISSING-node count carries this case. If
   tree-sitter-rust silently auto-closes the block, this test should
   also `#[ignore]` and the note amended.
3. `insert_adjacent` swallows one trailing newline of the addressed
   range for `After` and prepends a blank line for `Before`, producing
   exactly `\n\n` between declarations; the unique-cell assertions
   depend on this layout.
4. No existing test in `crates/edit/tests/` was modified.
