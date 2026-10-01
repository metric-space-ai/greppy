# Contextual readiness notices

Successful human `read-file` calls in a Codex thread may attach one readiness
notice on stderr after a previous read observed a live workspace preparation job.
Literal output and return codes remain unchanged. Calls outside Codex, machine
output, ordinary reads with no preparation evidence, and grep/rg remain quiet.

`context_status` is a small shared transition decision engine, not a progress
poller. Verified index publication records graph and semantic availability
separately. A later read can consume the corresponding thread/workspace/generation
restriction exactly once. Publication receipts bind to graph.db size and mtime;
a changed snapshot, failed/cancelled journal, different generation, or unfinished
semantic publication cannot authorize a notice. Readiness means the capability
can be retried; its normal source freshness validation still applies.

State reads are capped at 16 KiB. At most 32 scope/capability restrictions are
retained per workspace; oldest entries are evicted. Nonblocking serialization,
private atomic replacement, and best-effort failures keep literal reads usable.
No status operation opens SQLite, loads models, indexes, or scans source.

Grep/rg has a byte-identical stream contract and no existing contextual-result
envelope consumer was identified in this package. Attaching notices there
requires a supported caller envelope; this change leaves that integration open.

Focused checks: `cargo test -p greppy --lib --features cpu-only,ci-test-assets
context_status::tests -- --test-threads=2` through the shared heavy-job gate.
The five tests cover required prior restriction, consumption/durable dedupe,
semantic separation, failure/snapshot/generation suppression, bounded state and
oversized metadata refusal. They do not establish installed CLI acceptance.
