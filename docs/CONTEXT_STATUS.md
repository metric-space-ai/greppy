# Contextual readiness notices

Greppy's built-in agent attaches useful readiness transitions to successful
human `rg`, `grep`, and `read-file` tool outcomes. Its `GreppyEnv` captures the
subprocess streams unchanged, then invokes the CLI status hook and appends a
notice to the existing `ToolOutcome.content` envelope. Saved session identity
is propagated to query subprocesses, including resumed interactive/headless
sessions. Machine output and explicit other-root invocations are excluded.
Direct grep/rg streams and exit codes remain byte-identical.

`context_status` is a bounded transition engine, not a progress poller. Actual
query preparation and stale-refusal boundaries record per-session restrictions;
literal calls do not infer restrictions from unrelated running jobs. Successful
queries acknowledge availability, avoiding a redundant notice afterwards.
Verified atomic publication records graph and semantic readiness separately.
Publication revision must advance beyond the restriction, so a new session's
pending semantic query cannot erase or consume an earlier shared-ready signal.
Generation and graph.db metadata must still match, failed/cancelled preparation
suppresses notices, and consumed state is persisted per session/capability.
Known source-freshness refusals invalidate readiness until another publication.

State reads/writes are capped at 16 KiB and 32 session/capability entries.
Serialization is nonblocking and replacement is private and atomic. Failure to
record status never fails the user's tool. No status operation opens SQLite,
loads models, starts preparation, or scans source. Publication availability is
an invitation to retry the capability; the query's normal freshness validation
remains authoritative. Arbitrary source edits not yet observed by a query
cannot be detected from the publication metadata alone.

A successful direct human `read-file` in a Codex thread can consume the same
engine's restriction on stderr. External Codex shell/tool execution has no
supported Greppy-owned result envelope for attaching rg metadata, so those rg
callers are not covered. No system prompt was changed.

Focused tests: `cargo test -p greppy --lib --features cpu-only,ci-test-assets
context_status::tests -- --test-threads=2` through the shared heavy-job gate.
Seven tests include the real injected-binary `GreppyEnv` consumer, session and
capability isolation, graph-ready/embedding-running, persisted dedupe,
cancellation and known-stale invalidation, machine/other-root exclusions,
publication revision requirements, stale/new-generation metadata and bounds.
These stub/metadata fixtures require neither a model nor an indexer. They do
not establish installed CLI acceptance.
