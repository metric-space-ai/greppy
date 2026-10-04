# PR250 completed-cache recovery

The initial parser-only delivery and architectural analysis are preserved in
commits 657b747d3 and 844bf8cf2. This follow-up implements the end-to-end migration;
it has not been compiled or executed. Parent owns admission and runtime checks.

## Publication contract

`recover_persisted_js_ts_usages` independently checks
`greppy.js_ts_usage_repair_v1`. It reads stable visible JS/TS source, validates
persisted SHA-256 fingerprints, safe/contained paths, complete parser output,
and existing definition identity/line spans before writing. Missing, changed,
deleted, escaping or incomplete sources refuse recovery. Older Effect.fn
Variable identities that differ from current definitions also refuse; this
migration does not rewrite them. Existing overlay Effect.fn recovery runs first.

Usage replacement reuses the validated-reference Store pattern with a separate
`greppy.js_ts_usage_override_files.<project>` mask and
`greppy.js_ts_usage_override_rows.<project>` namespace. Rust keys and rows are
unchanged. Base-owned facts stay in schema-meta compatibility rows; private
facts stay in private raw edges. Composed raw/resolved views mask only Base
USAGE contributions for these files and filter hidden paths. No JS/TS raw-edge
multiset visibility exemption was loosened.

The source-validated USAGE facts resolve through existing import resolution.
Raw usage replacement, resolved usage publication and the completion marker are
inside one savepoint; any error rolls back all three. No nodes, vectors, file
state/content, generation, or immutable Base bytes are rewritten. Overlay
resolved usage rows carry the existing repair provenance so sparse publication
can preserve them, subject to existing source/target visibility filters.

Query-open pending detection includes the independent JS/TS marker even when
Rust and v8 repair markers are already complete. Both private and overlay
paths retain existing admission, lock, deadline and post-lock recheck behavior.
The existing admitted index-publication path completes this migration before
final graph publication. There is no full indexer-version invalidation.

## Source regressions and remaining limits

Added tests simulate old completed private/overlay caches missing JSX facts,
verify recovery and no-op second migration, exact import resolution, retained
nodes/file-state/workspace state and immutable Base bytes, separate concurrent
Rust override metadata, changed/deleted source refusal, rollback on completion
marker failure, Delta private-path validation, an independent pending trigger,
and hidden/deleted-path filtering on reopened overlays. Parser tests cover
opening/self-closing tags, callbacks and conservative shadow refusal.

All executable tests remain unrun. Formatting and diff checks passed. Original
Workjet store acceptance remains pending; no reporter store was opened/mutated.
The implementation makes one admitted source-validation pass over visible JS/TS
files and project definitions. Member/namespace tags are still not guessed;
shadow analysis follows active lexical scopes, function parameters and hoisted var bindings; later same-scope lexical declarations shadow through their temporal dead zone. The Store
completion marker follows the existing single-project query-open convention.
