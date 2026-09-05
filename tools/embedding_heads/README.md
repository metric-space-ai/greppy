# Embedding head engineering

This is an in-progress implementation. No new production head has been trained
or enabled. `TEACHER_CONTRACT.md` defines the three supervised tasks and data
admission rules. `docs/embedding-heads-optimization.md` records baseline evidence.

## Teacher pipeline

`contracts.py` validates exact annotation coverage and source evidence IDs.
`queue_store.py` stores jobs, immutable source-group splits, results and events
in SQLite WAL with transactional worker leases. `teachers.py` calls MiniMax-M3
Responses and the signed-in Grok CLI (`grok-4.6`) with tools disabled. Only visible
structured annotations and numeric usage enter the result ledger.

Input is JSONL using `greppy.heads.example.v1`: id, source_id, group_key, family,
domain (`log` or `web`), split, task, records (id/text/protected), optional context
and last_action. Explicit `privacy_review` must be `public-redacted` or `synthetic`.
That admission tag is a caller obligation, not a claim that regex redaction alone
can identify every personal datum. Never import raw agent reasoning or arbitrary
trace metadata. The teacher allowlist excludes previous labels and predictions.

```sh
python3 tools/embedding_heads/teacher_queue.py --db /durable/heads/queue.sqlite enqueue /durable/heads/pilot.jsonl
python3 tools/embedding_heads/teacher_queue.py --db /durable/heads/queue.sqlite work --provider minimax --watch
python3 tools/embedding_heads/teacher_queue.py --db /durable/heads/queue.sqlite work --provider grok --watch
python3 tools/embedding_heads/teacher_queue.py --db /durable/heads/queue.sqlite status
```

Supply MINIMAX_API_KEY through the existing private environment file; never put
it in argv. Grok uses the existing signed-in CLI. `GROK_BIN` can select that
executable. On macOS scratch is required on `/Volumes/tmp`; set TMPDIR to a
subdirectory there for Python tests and other temporary files. Teacher ledgers
and irreplaceable results belong on durable storage.

Quota pauses the entire provider and resumes after Retry-After (30 minutes when
unavailable). Authentication pauses until `resume-provider PROVIDER` is explicitly
called after recovery. Transient/schema errors allow at most three retries.
Expired worker leases become `uncertain`, never automatically replayed: a request
may already have completed. A completed job is immutable and not reenqueued when
restarting. Changing input, rubric or provider/model creates a different cache key.

A worker exiting zero means it completed its processing loop, not that labels
passed semantic review. Inspect persisted statuses; queued, failed or uncertain
jobs are not admissible training data. The independent audit/admission stage and
large corpus import are still under implementation.

## Verification to date

- 21 Python tests cover coverage, evidence integrity, blind prompts, common
  secret redaction, restart idempotency, concurrent claims, bounded retries,
  quota/auth pauses, worker expiry and split leakage refusal.
- Live synthetic smoke: one M3 batch and one independently judged Grok batch,
  four records each, both structurally valid and matching on severity/relevance.
  Grok's actual JSON container is `structuredOutput`; the adapter accepts it.
- Historical R5 counts reproduced exactly on hash-pinned native CUDA vectors.
- Restored portable Rust classifier golden passes in CPU and Metal builds.
  Current raw-text Metal still differs on 3/64 old CUDA threshold decisions.

Run checks through greppy as required by the repository instructions:

```sh
greppy bash-smart -- env TMPDIR=/Volumes/tmp/dev-artifacts/greppy/embedding-heads-optimization/tmp python3 -W error::ResourceWarning -m unittest discover -s tools/embedding_heads -p 'test_*.py'
```
