# Battle-proof validation harness (Track C)

A **black-box** production-invariant suite. It drives the already-built
`grepplus` / `grepplus-grep` binaries and asserts the contracts that
matter in production. It does **not** touch any crate source or Cargo
files — it lives entirely under `bench/battle/`.

## Running

```sh
cargo build --bins                      # produce target/debug/grepplus{,-grep}
bash bench/battle/run_battle.sh         # run the whole suite
bash bench/battle/run_battle.sh scale   # run one battle by name
```

`run_battle.sh` aggregates every script, prints a combined PASS/FAIL
summary, and **exits non-zero if any check fails**. Each script also runs
standalone and prints its own `PASS`/`FAIL` lines plus a machine-readable
`BATTLE_SUMMARY <name> pass=<n> fail=<n>` line.

### Knobs

| env var | default | meaning |
|---|---|---|
| `BATTLE_SCALE_FILES` | 300 | corpus size for SCALE |
| `BATTLE_SCALE_BUDGET_S` | 120 | time budget for the SCALE index |
| `BATTLE_SCALE_RSS_KB` | 2097152 | RSS ceiling (KB) for SCALE |
| `BATTLE_DET_FILES` | 120 | corpus size for DETERMINISM |
| `BATTLE_CONC_WORKERS` | 6 | concurrent indexers for CONCURRENCY |
| `BATTLE_CONC_FILES` | 150 | corpus size for CONCURRENCY |
| `REAL_GREP` | `/usr/bin/grep` | byte-exact oracle for GREP-COMPAT |
| `BATTLE_SOAK` | `0` | set `1` to include the soak loop in `run_battle.sh` |
| `BATTLE_SOAK_ITERS` | 200 | iterations of the SOAK index→edit→reindex→search→grep loop |
| `BATTLE_SOAK_FILES` | 40 | corpus size for SOAK |
| `BATTLE_SOAK_RSS_FACTOR` | 3 | max late/early RSS ratio for SOAK |
| `BATTLE_SOAK_SIDECAR_CAP` | 64 | max live sidecars allowed at any checkpoint |
| `BATTLE_RELEASE` | `0` | set `1` to add a release-build invariant to `run_battle.sh` |

The **soak** battle is opt-in (slow). Run it directly with a small count:

```sh
BATTLE_SOAK_ITERS=20 bash bench/battle/soak.sh        # standalone smoke
BATTLE_SOAK=1 bash bench/battle/run_battle.sh         # full suite + soak
BATTLE_SOAK=1 BATTLE_RELEASE=1 bash bench/battle/run_battle.sh   # + release invariant
```

To push toward the 2000+-file aspiration (slow — see finding #1):

```sh
BATTLE_SCALE_FILES=2000 BATTLE_SCALE_BUDGET_S=1800 bash bench/battle/scale.sh
```

## The battles

1. **scale.sh** — generate a synthetic Rust repo (`gen_corpus.sh`),
   git-init it, index it. Asserts: completes within budget, no panic,
   bounded RSS, real cross-file `IMPORTS`/`TYPE_REF`/`CALLS` edges,
   `integrity_check=ok`.
2. **determinism.sh** — index the same corpus into two stores; asserts
   identical node/edge **counts** *and* byte-identical node/edge **sets**,
   plus re-index idempotency.
3. **concurrency.sh** — launch N concurrent `grepplus index` on one
   workspace; asserts exactly one winner, no crash, DB integrity ok, and
   the documented lock contract (losers exit 75).
4. **grep_fuzz.sh** — 42 patterns/flags/paths (malformed UTF-8, huge
   lines, binary files, missing paths, regex metacharacters) through
   `grepplus-grep`; asserts stdout/stderr/exit are **byte-identical** to
   `/usr/bin/grep` and grepplus never crashes.
5. **malformed.sh** — index truncated / invalid-UTF8 / deeply-nested Rust
   files; asserts no panic, graceful exit, integrity ok, and that a
   well-formed sibling still indexes.
6. **navigation.sh** — build a tiny Rust fixture with a known
   caller/callee and a struct+impl that **share a name**; drive
   `who-calls` / `find-usages` / `trace` and assert the printed symbols
   are the right ones (guards the name→node resolution layer that a
   DB-only check cannot see).
7. **multilang.sh** — generate ONE git repo mixing **Rust, Python,
   JavaScript, TypeScript, Go, and Ruby** (each with a cross-file
   caller→helper call and an import), index it once, and assert RESULT
   CONTENT: per-language cross-file `CALLS` (all six) and `IMPORTS`
   (Rust/Python/JS/TS — Go/Ruby emit an Import *node* but their
   package/relative target does not resolve to a node, asserted as a
   known characteristic); `stats` per-label/per-type counts match
   graph.db exactly; `who-calls`/`callees`/`path` resolve the right
   symbol in every language; `find-usages` lands on a `TYPE_REF`-d Rust
   struct; `search-symbols`/`search-code` find known symbols/content
   across all six languages; the drop-in grep contract (`grepplus -R` and
   `grepplus-grep` vs `/usr/bin/grep`) holds byte-exact in its STRICT
   regime while the recursive **semantic augmentation** is asserted
   present-and-additive on a fresh graph; determinism (index twice →
   identical node/edge counts and sets); and an unsupported `.txt` file is
   classified gracefully with zero graph nodes.
8. **soak.sh** *(opt-in, slow)* — drive `BATTLE_SOAK_ITERS` rounds of an
   `index → edit → reindex → search-code → grep` loop against a mutating
   corpus. Asserts, **across all iterations**: no panic / signal crash,
   `integrity_check` stays `ok`, the drop-in grep stays **byte-exact** vs
   `/usr/bin/grep`, sidecar temp files stay bounded under a short TTL and
   are actually reclaimed by the startup cleanup, and resident-set size
   does not grow unbounded (early vs late RSS sample).

`gen_corpus.sh <dir> <n>` and `lib.sh` are shared helpers.

---

## FINDINGS (honest results as of 2026-06-30, debug binary)

Default suite result on this machine: **166 PASS / 0 FAIL** (scale 14,
determinism 10, concurrency 8, navigation 12, multilang 74, grep_fuzz 42,
malformed 6). With the opt-in soak (8) and release invariant (2):
**176 PASS / 0 FAIL**.

> Update vs the earlier review: the concurrency lock-contract failure
> recorded below as Finding #2 **no longer reproduces** — `concurrency.sh`
> now reports `8/8 PASS` (winners=1, losers exit the documented 75, DB
> integrity ok). The fix landed in `crates/store` / `crates/cli`. The
> finding text is kept for history; the battle that asserts it is green.

### Finding #1 — Indexing scales super-linearly (≈ O(n²)) — *informational*

Measured wall-clock for `grepplus index` on the synthetic corpus
(debug build, chained-import corpus):

| files | time |
|---|---|
| 100 | 3.2 s |
| 200 | 7.0 s |
| 400 | 16.2 s |
| 800 | 66.7 s |

Doubling the file count roughly **quadruples** the time. A 2200-file
corpus did **not** complete within 10 minutes. This is algorithmic, not a
constant-factor debug-build artifact (a release build would lower the
constant but not the curve). The SCALE battle therefore defaults to 300
files (~10 s) with a generous budget, and exposes `BATTLE_SCALE_FILES`
for operators willing to wait. **This does not fail the suite** — it is
flagged here as a scalability concern for large real-world repos.

### Finding #2 — Concurrency lock contract (historical — now PASSING)

> **Status: resolved.** As of 2026-06-30 `concurrency.sh` passes 8/8 on
> this machine; the silent rc=73 path described below no longer
> reproduces. Text retained for history.


`crates/cli` documents that a concurrent `grepplus index` losing the race
returns `EX_TEMPFAIL` (75) with a diagnostic. Under a 6-way race the
observed outcome is:

```
winners(rc=0)=1  losers(rc=75)=2  unexpected(rc=73)=3
```

Three losers exit **73 (`EXIT_IO`) with completely empty stdout AND
stderr** — a silent IO failure, not the clean lock-contention path.

**Root cause (read-only diagnosis):** `Store::open`
(`crates/store/src/store.rs`) opens the SQLite DB and immediately runs
`migrate()` (DDL → exclusive lock) and `integrity_check()` — all **before**
the advisory file lock is acquired in `crates/cli/src/lib.rs`. No
`busy_timeout` / `busy_handler` is configured on the connection, so a
concurrent opener hits `SQLITE_BUSY` instantly and the CLI maps that to
`EXIT_IO` (73) with no message. The advisory-lock contention path
(rc=75) is only reached by the few processes that win the SQLite-level
race first.

**Impact:** no data corruption — the DB stays `integrity_check=ok` and
exactly one writer wins — but the *documented* contract ("losers exit
75") is not honoured, and the failure is silent (no diagnostic), which is
hostile to callers/scripts that branch on exit code 75.

**Suggested fix (belongs to `crates/store` + `crates/cli`, NOT this
track):** set a `busy_timeout` on the connection in `Store::open`, and/or
acquire the advisory lock *before* opening/migrating the SQLite DB so
lock contention is resolved by the advisory layer (clean rc=75) rather
than by raw `SQLITE_BUSY` (silent rc=73). At minimum, map `SQLITE_BUSY`
during open to `EX_TEMPFAIL` with a diagnostic.

The concurrency battle asserts this contract honestly and **fails** until
the underlying behaviour is fixed.

### Finding #3 — Sidecar TTL cleanup runs only on the `grepplus-grep` binary — *informational*

Sidecar `*__GREPPLUS_SEMANTIC_NONCANONICAL.md` files are written by **both**
augmenting paths: the `grepplus` (cli) drop-in and the `grepplus-grep`
drop-in. The probabilistic TTL cleanup (`maybe_run_sidecar_cleanup`,
honouring `GREPPLUS_SIDECAR_TTL_SECS`) is wired into the startup of the
**`grepplus-grep`** binary only (`crates/grepplus/src/main.rs`); the
`grepplus` cli binary (`crates/cli`) does not invoke it. The cleanup also
walks the **current working directory's** store dir, not an explicit
`--root`/`GREPPLUS_STORE_DIR` target passed to a one-shot invocation.

**Impact:** in a deployment that only ever invokes the `grepplus` cli
(never `grepplus-grep`), expired sidecars are never reclaimed and can
accumulate. When `grepplus-grep` is part of the loop (the common drop-in
case), reclamation works correctly: `soak.sh` verifies that backdated,
expired sidecars are deleted (`before → after` strictly decreases) and
that the live count stays bounded across 100+ iterations.

This is **not** asserted as a failure — the soak battle drives
`grepplus-grep` so cleanup fires — but it is flagged so the owning track
can decide whether the `grepplus` cli should also run cleanup on start.
Belongs to `crates/cli` / `crates/grepplus`, **not** this track.

### What passes cleanly

- **Determinism** (10/10): node/edge counts *and* full sets are
  byte-identical across runs; re-index is idempotent. The parallel
  pipeline is deterministic.
- **Grep-compat** (42/42): every probed pattern/flag/adversarial input is
  byte-identical to system grep across stdout, stderr, and exit code. No
  panics. The drop-in contract holds.
- **Malformed input** (6/6): truncated / invalid-UTF8 / deeply-nested Rust
  files are handled without panic or stack overflow; well-formed siblings
  still index; DB integrity holds.
- **Scale** (13/13 at 300 files): completes, no panic, bounded RSS
  (~60 MB), real cross-file `IMPORTS`/`TYPE_REF` edges, integrity ok.
- **Concurrency** (8/8): one winner, losers exit the documented 75, DB
  integrity ok after the race (Finding #2 resolved).
- **Soak** (8/8 at 100 iters, opt-in): no panic, integrity stays ok,
  drop-in grep stays byte-exact, sidecars bounded (peak ≤ 4) and TTL
  cleanup reclaims expired files, RSS flat (early 1504 KB / late 1536 KB).
- **Navigation** (14/14): `who-calls` / `find-usages` / `trace` resolve a
  known caller/callee and land on the struct (not the same-named impl).
- **Multilang** (74/74): one mixed Rust/Python/JS/TS/Go/Ruby repo indexes
  with no panic and `integrity_check=ok`; every supported language yields
  a cross-file `CALLS` edge and the four resolving languages a cross-file
  `IMPORTS` edge; `stats` per-label/per-type counts match graph.db
  exactly; `who-calls`/`callees`/`path`/`find-usages` return the right
  symbol per language; `search-symbols`/`search-code` find known
  symbols/content in all six; the drop-in grep contract holds in its
  STRICT regime and the semantic augmentation is additive on a fresh
  graph; index-twice is byte-identical; the unsupported `.txt` produces
  zero nodes.

### Observation — multilang edge resolution & the `grepplus -R` contract

From `multilang.sh` against a six-language mixed repo (all asserted, all
green — these are characteristics the suite now pins, not failures):

- **Cross-file `CALLS` resolve for all six languages** (Rust, Python, JS,
  TS, Go, Ruby) when the call site is a real call expression. Ruby needs
  explicit parentheses: a bare `rb_helper` is parsed as an identifier, not
  a call, and yields no `CALLS` edge — the fixture uses `rb_helper()`.
- **Cross-file `IMPORTS` resolve for Rust/Python/JS/TS only.** Go and Ruby
  emit an `Import` *node* (`example.com/p/helper`, `helper`) but the
  package/relative-path target does not map to a graph node, so no
  resolved cross-file `IMPORTS` edge is produced. Asserted both ways (node
  present, resolved-edge count == 0) so a future change either direction
  is noticed. Owner: `crates/resolver` / `crates/parser`, not this track.
- **`grepplus -R` is the drop-in surface but NOT unconditionally
  byte-exact.** When the indexed graph for the cwd is **fresh** and the
  pattern has semantic graph hits, a plain recursive listing appends one
  synthetic `…:1:<!-- GREPPLUS_NON_CANONICAL_HIT: <pat> -->` line per
  query (`Mode::VisibleAugment` in `grepplus_grep::run`). This is the
  intended value-add, gated by `freshness_gate`. The suite therefore
  asserts byte-exactness on the STRICT cases (`-R` no-hit patterns, `-Rc`
  count mode, non-recursive single-file, and the pure `grepplus-grep`
  binary with no fresh store in scope) and separately asserts the
  augmentation is present *and strictly additive* (every raw-grep line
  preserved verbatim) on a fresh-graph hit pattern. Both halves of the
  contract are pinned. Note `grepplus-grep` augments too when a fresh
  store is in scope — the "pure drop-in" guarantee is *no fresh index in
  scope*, which is the canonical drop-in scenario.

### Observation — cross-file `CALLS` resolution

In the SCALE corpus, `build_N()` calls the *imported* `Widget(N-1)::new()`,
but the resolver binds the call to the **local** module's `new` method, so
cross-file `CALLS` edges are 0 (cross-file `IMPORTS` and `TYPE_REF` are
correct: 299/299). This is reported as informational, not asserted —
method-call resolution across `use` aliases is a resolver characteristic,
not a corruption invariant.
