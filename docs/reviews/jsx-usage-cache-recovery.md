# PR250 existing-cache recovery boundary

PR250 fixes new extraction; it does not yet recover an existing completed cache.
The reported Workjet store must not be considered repaired by this patch alone.
No indexer version or repair marker has been bumped, and no reporter store was
read or modified. This is an architectural scope blocker, not a failed runtime
experiment. The source evidence below is from base 8d5afe5a plus PR250.

## Why simply rerunning the existing JS/TS repair is unsafe

`crates/indexer/src/lib.rs::recover_visible_effect_fn_bindings_inner` returns
immediately for `greppy.effect_fn_repair_v8.<project>=complete`. Without that
marker, it fingerprint-validates visible JS/TS sources, compares complete raw
contributions, replaces differing contributions in `main.raw_edges`, records
`main.js_ts_reference_override_files`, and rebuilds visible overlay edges.
It preserves immutable Base bytes and uses a savepoint, but its storage model
is a complete file-level raw override, not a usage-only compatibility mask.

`crates/cli/src/store_cow.rs::validate_delta_visibility` permits off-manifest
JS/TS overrides only when both complete raw-edge multisets are exactly equal
between Delta and Base. A source-valid old TSX Base missing JSX facts necessarily
has a different multiset after extraction. Invalidating v8 or adding a v9 marker
and allowing it through that SQL would bypass the actual membership proof.
Pretending the unchanged file is dirty would require real Delta file-state and
content ownership and could trigger graph/embedding preparation; that is not a
usage-only migration preserving the requested identities and cached data.

## Existing pattern that can support a separate safe migration

`crates/store/src/raw_edge.rs::replace_validated_rust_edge_kind` stores Base-owned
replacement facts in bounded schema-meta rows with a per-file override mask,
while privately owned facts remain in `main.raw_edges`. The composed raw and
resolved views in `crates/store/src/store.rs` interpret those keys and filter
hidden paths. This avoids claiming Delta ownership of unchanged Base files.
Those keys and views are currently Rust-specific; reusing them for TSX would
mix masks/rows with Rust recovery and could overwrite the Rust contribution.
They must be extended with a distinct JS/TS usage namespace, not repurposed.

There is also no independent JSX query-open migration trigger:
`crates/cli/src/store_cow.rs::persisted_v7_delta_needs_repair` is gated by the
Rust repair marker. A completed Rust marker skips both single-store and overlay
query-open repair before the Effect.fn helper is reached. Single-store recovery
in `crates/indexer/src/lib.rs::rebuild_single_store_rust_edges` extracts and
replaces only `.rs` relations. Changing only the parser or Effect.fn marker
cannot reach already-completed private or linked-worktree stores reliably.

## Required migration and acceptance

A focused follow-up must extend all three contracts together:

1. A distinct usage-only storage mask/row namespace and composed-view handling,
   preserving existing Rust masks, non-USAGE facts and ordinary hidden-path
   filtering. Do not broaden the equal-multiset JS/TS visibility exemption.
2. Source-validated extraction of visible JS/TS usage facts, checked against
   persisted file fingerprints and graph definition completeness before any
   write. Unsafe paths, missing/deleted/changed files, degraded extraction and
   absent source identities must refuse or defer without certifying completion.
   Commit resolved facts and an independent completion marker atomically.
3. Query-open pending detection for completed private and linked/overlay stores,
   under the existing admission and writer-lock/recheck lifetime. Subsequent
   fresh opens must reuse the marker. Ordinary refresh must retire/filter
   compatibility rows as ownership/visibility changes.

Required tests: simulate completed v8 and Rust markers with missing old JSX
facts; verify exact imported-component callers recover in private and linked
stores without touching nodes, graph generation, file state, content, vectors
or immutable Base bytes. Test failed/deleted/changed/escaping source rollback,
ambiguous imports and shadows, mixed Rust+JS override preservation, visibility
hide/delete, no-op second open and sparse publication carry-forward. These
migration tests have not been implemented or executed in this source package.
The parser/indexer extraction regressions in PR250 also remain unrun pending
parent admission. No claim of original-store recovery or latency closure.
