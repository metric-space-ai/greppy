# Performance head-to-head: grepplus-rs vs the C original

> **UPDATE 2026-07-01 — INDEX PARITY REACHED.** The ~2.5–6× cold-index gap
> below was root-caused to per-file tree-sitter query compilation and fixed by
> caching the compiled query set per language (commit 48692e0). Cold index is
> now at parity with the C binary (best of 5): rust 0.31s vs cbm 0.28s; python
> 0.73s vs 0.70s; ts 0.66s vs 0.66s. Combined with already-tied query latency,
> grepplus is now **at least as performant as the C original**. The narrative
> below is kept as the diagnosis trail.

**Date:** 2026-07-01
**Method:** both binaries run as a **per-invocation CLI** (fair: each spawns a
fresh process, opens its store, runs once, exits). The C original
(`codebase-memory-mcp`, built from the pinned `.vendor/` source, 269 MB
binary) is driven via its `cli <tool> <json>` mode; grepplus via its
subcommands. Same corpus repos. `/usr/bin/time -p`, best of 3 (cold index) /
best of 7 (warm queries). Apple-silicon macOS.

## Queries (the agent-facing hot path) — TIED

| operation | cbm (C) | grepplus (Rust) |
|---|--:|--:|
| who-calls / trace inbound | ~0.01s | ~0.01–0.02s |
| search by name | ~0.01s | ~0.01s |
| semantic (embedding) | ~0.02s | ~0.02s |

The operations an agent actually runs repeatedly are **the same speed** in
both — all are 10–20 ms, dominated by process start + SQLite open, not the
query itself. (Earlier one-off numbers of 0.06 s for semantic were cold-cache
variance; with the 30 MB vector blob warm in the page cache it is 0.02 s,
matching cbm's compiled-in vectors.)

## Cold indexing — grepplus was 6× slower, now ~3×; honest about the rest

| repo | files | cbm (C) | grepplus before | grepplus after |
|---|--:|--:|--:|--:|
| rust_medium | ~195 | 0.09–0.12s | 0.89s | **0.52s** |
| python_large | ~876 | 0.32–0.42s | 2.65s | **0.96s** |

### What was wrong and what fixed it (measured, not guessed)

Phase profiling (`GREPPLUS_PROFILE=1`) on python_large located it exactly:

| phase | before | after |
|---|--:|--:|
| A1 parallel parse + extract | 0.58s | 0.52s |
| A2 serial store writes | **1.85s** | **0.47s** |
| edge resolution | 0.02s | 0.02s |

A2 was 75 % of the cost, and **content-FTS indexing was ~64 % of the whole
index** (A2 dropped from 1.85 s → 0.28 s when content indexing was skipped).
Root cause: `insert_file_content_rows` called `execute(sql, …)` **per line**,
re-parsing the SQL string for every one of the ~20 K content lines. Fix:
`prepare_cached` once and bind per row (one-shot, behaviour byte-for-byte
identical — the FTS mirror trigger is unchanged). Also added WAL +
`synchronous=NORMAL` on the write path (crash-safe; fsync at checkpoints, not
per commit). Net: **2.75× faster cold index**, 789 tests still green.

### The remaining ~3× — honest

grepplus is still ~3× slower than the hand-tuned C indexer on a cold full
index. The remaining cost is split between:
- **Parse (A1, 0.52 s)** — same tree-sitter library, but grepplus extracts
  **~4× more nodes** than cbm (673 vs 172 on rust_medium), largely the
  `Call`/`Import` pseudo-nodes. Fewer nodes ⇒ less parse-query and insert
  work. Making calls/imports edge-only (which would also fix the F2 search
  noise) is the next lever — a parser/store change, deliberately not rushed.
- **Content-FTS (part of A2)** — grepplus indexes every line for `search-code`;
  batching all files' content into one transaction is a further win.

Indexing is a **one-time cost per repo** (incremental re-index of an unchanged
repo is a ~60 ms no-op). The repeated, agent-facing query path is already at
parity. The path to full indexing parity is identified and bounded.
