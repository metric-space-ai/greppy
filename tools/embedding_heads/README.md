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
called after recovery. Transient provider errors allow at most three retries.
Invalid annotation/schema responses fail immediately for explicit review; repeating
an idempotent request is not a repair for an invalid result. Each new job stores its
exact prompt and output schema, and adapters verify the bound request before dispatch.
Expired worker leases become `uncertain`, never automatically replayed: a request
may already have completed. A completed job is immutable and not reenqueued when
restarting. Changing input, the full prompt/schema, rubric, provider/model or bound
call configuration creates a different cache key. Legacy jobs remain stored; jobs
without a bound configuration cannot dispatch through the new adapters.

A worker exiting zero means it completed its processing loop, not that labels
passed semantic review. Inspect persisted statuses; queued, failed or uncertain
jobs are not admissible training data. `admission.py` compares blind teacher jobs
and requires matching independent evidence receipts; conflicts and missing evidence
are held. A receipt must itself be verified against its referenced artifact. This
review report does not certify production readiness or automatically export labels.

`corpus.py` partitions log lines into byte-exact target spans, preserves full
context, and joins content/template/lineage relations before assigning splits.
Frozen split conflicts fail closed; existing source-group IDs survive extensions.
Oversized full contexts are held rather than truncated. `catalog_archive.py`
creates a source-offset/hash inventory without exporting private trace metadata.
Archive presence alone does not establish capture completeness or privacy admission.

`web_records.py` preserves the actual typed observation, including unknown fields
and absent state. Explicit goal/version and last action are required. Source record
IDs stay independent of goal versions; example IDs invalidate goal-dependent work.
`prepare_observations.py` checks source checksums and creates reviewable candidate
JSONL from an explicitly authored manifest. These are training preparation helpers;
the product session-goal API is prepared and tested separately. Daemon/action
wiring and runtime head integration are still outstanding.

## Verification to date

- 57 Python tests pass on GPU3's Python runtime and cover coverage, evidence integrity, blind prompts, common
  secret redaction, restart idempotency, concurrent claims, bounded retries,
  quota/auth pauses, worker expiry and split leakage refusal.
- Live synthetic smoke: one M3 batch and one independently judged Grok batch,
  four records each, both structurally valid and matching on severity/relevance.
  Grok's actual JSON container is `structuredOutput`; the adapter accepts it.
- Historical R5 counts reproduced exactly on hash-pinned native CUDA vectors.
- Restored portable Rust classifier golden passes in CPU and Metal builds.
  Current raw-text Metal still differs on 3/64 old CUDA threshold decisions.
- The GPU3-only experiment runner completed 45 synthetic variants, preserved all
  completed runs on replay, and reproduced bit-identical weights after injected
  interruptions for five head/objective combinations. See
  `docs/embedding-heads-experiments.md` for the contract and remaining work.

Run checks through greppy as required by the repository instructions:

```sh
greppy bash-smart -- env TMPDIR=/Volumes/tmp/dev-artifacts/greppy/embedding-heads-optimization/tmp python3 -W error::ResourceWarning -m unittest discover -s tools/embedding_heads -p 'test_*.py'
```
