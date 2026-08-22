# greppy 0.3.2–0.3.3 — Parallel Agent CoW Release Plan

Status: 0.3.2 implemented and locally gate-qualified; exact-commit CI, signing
and publication pending · 2026-08-21. 0.3.3 remains planned.

Scope decision: **two releases, two complete features.** The milestones below
are implementation order, not open research branches.

- **0.3.2 ships Store-CoW:** one immutable greppy Base Store per compatible Git
  tree plus one private Delta Store per integrated agent.
- **0.3.3 ships Filesystem-CoW:** instant private CoW workspaces, using the
  simplest production-capable backend that wins the registered end-to-end
  gates. Rift-style native snapshots/reflinks are evaluated before a custom
  virtual Git filesystem is authorized.

0.3.3 consumes the 0.3.2 Base/Delta infrastructure. Neither release waits for a
later unspecified project to deliver its stated feature.

## 1. Product thesis

The integrated `greppy -p` agent currently reuses one stable worktree and warm
Store per repository. Concurrent runs cannot share it: one holds the stable
lock, while every additional agent receives a cold temporary worktree and a
cold isolated `GREPPY_STORE_DIR`.

The two releases remove both repeated costs in sequence:

```text
0.3.1
  Agent 1: warm native worktree + warm Store
  Agent N: cold temporary worktree + cold Store

0.3.2
  Agent 1: warm native worktree + shared Base Store + Delta 1
  Agent N: cold temporary worktree + shared Base Store + Delta N

0.3.3
  Agent 1: warm native worktree + shared Base Store + Delta 1
  Agent N: virtual CoW worktree + shared Base Store + Delta N
```

In `auto` mode, 0.3.3 deliberately retains the warm native stable worktree for
the serial/lock-holder case. The CoW backend replaces the expensive temporary
worktree fallback for concurrent agents. Users can force either backend.

## 2. Release contracts

### 2.1 greppy 0.3.2 — Parallel Agent Store CoW

Release claim:

> Parallel agents share one immutable greppy index for unchanged code and keep
> only their own index delta, providing warm navigation and semantic search
> without sharing writable agent state.

### 2.2 greppy 0.3.3 — Parallel Agent Filesystem CoW

Release claim:

> Concurrent agents can start from the same Git tree through a private CoW
> workspace instead of a full physical copy. Startup and private storage scale
> with changed/generated data or the selected native snapshot mechanism, and
> ordinary Git and toolchain behavior remains compatible.

### 2.3 Explicit exclusions through 0.3.3

- chunk/extent-level CoW;
- RAM-first file content storage;
- writable build-cache sharing between concurrent agents;
- submodule support beyond the current behavior;
- Git LFS or external checkout filters inside CoW workspaces;
- arbitrary device nodes and filesystem features irrelevant to source/build
  workspaces;
- claiming identical performance or support status on all three platforms.

Repositories using unsupported checkout semantics automatically fall back to a
native worktree with a concise diagnostic.

## 3. Registered release targets

### 3.1 0.3.2 gates

| Metric | Gate |
|---|---|
| Overlay vs full private reindex | **100% result parity** on registered tests |
| Duplicate Base summaries/embeddings after Base publication | **0 across 10 agents** |
| Agent writes to published Base | **impossible by API and permissions** |
| Median query-latency regression | **≤ 10%** |
| P95 query-latency regression | **≤ 20%** |
| Warm-Base temp-fallback pre-first-turn time | **≥ 50% faster** than 0.3.1 on medium/large fixtures |
| Warm serial-agent regression | **≤ 5%** |
| Supported concurrency | **10 gated; 50 stress-tested** |
| Corrupt/incompatible Base | **safe private-Store fallback** |

### 3.2 0.3.3 gates

| Metric | Gate |
|---|---|
| Warm CoW workspace creation | **no full-tree traversal; median ≤ 500 ms, P95 ≤ 1 s** on the large fixture |
| Zero-change workspace private files | **O(1) control files; no per-repository-file materialization** |
| Source edit storage | **proportional to changed/created files plus metadata** |
| Correctness | **registered filesystem and agent E2E suites pass** |
| `git status`, `rg`, greppy navigation P95 regression | **≤ 30% vs native warm checkout** |
| Representative incremental build regression | **≤ 25% after caches are warm inside that workspace** |
| Concurrent creation | **10 workspaces gated; 50 stress-tested** |
| Quota enforcement | **no overshoot beyond one in-flight write buffer** |
| Crash recovery | **no stale mount used as a valid workspace; private state recoverable or safely reclaimable** |
| Active graph-refresh wait | **no silent/`ETA unavailable` wait beyond 5 s; report owner/state/progress at least every 10 s and unblock within 2 s after writer failure or exit** |
| End-to-end agent wall clock | **materially faster than the best of temp worktree, warm worktree pool, and Rift on the registered concurrent workloads** |
| Custom virtual-FS authorization | **only if it beats or uniquely satisfies a gate that Rift/native CoW cannot** |

If a platform misses a performance gate, `auto` does not select CoW there, but
the explicitly forced backend may ship with preview status if all correctness
and safety gates pass.

## 4. Command surface

### 4.1 Store selection in 0.3.2

Shared Base plus Delta is automatic for `greppy -p`.

```text
greppy -p --private-store "TASK"   # diagnostic fallback to a full private Store
greppy index --agent-worktree      # builds/validates the shared Base Store
```

`doctor` and `diagnostics` report Store mode, Base identity, Delta identity,
dirty/deleted file count, completeness, cache hits, and fallback reason.

### 4.2 Workspace selection in 0.3.3

```text
greppy -p --workspace auto "TASK"       # default
greppy -p --workspace worktree "TASK"   # force current backend
greppy -p --workspace cow "TASK"        # force the selected CoW backend
```

`auto` policy is frozen for 0.3.3:

1. acquire and reuse the existing warm stable native worktree when available;
2. when the stable worktree is busy, use CoW if the repository and installed
   platform adapter pass preflight;
3. otherwise use the existing temporary native worktree fallback.

Diagnostics report chosen backend, adapter, mount/state paths, Base commit/tree,
Delta usage, quota, open handles, cache hits, and fallback reason.

### 4.3 CLI contract fix discovered during plan validation

greppy 0.3.1 documents multi-symbol `brief` in the repository agent contract,
but the release binary accepts only `brief SYMBOL`; a second symbol is rejected
with exit 64. 0.3.2 aligns implementation and contract by accepting multiple
symbols with the same resolution, ordering, `--code`, `--json`, output-budget,
and per-symbol empty/error semantics already documented for batched navigation.
Parser, human-output, JSON, ambiguity, mixed-hit/miss, and output-budget tests are
release-blocking. The fix is independent of Store-CoW but lands in S0 so agents
can inspect several related workspace symbols in one high-yield call as promised.

## 5. 0.3.2 architecture — Store-CoW

### 5.1 Store identity

Define a versioned `BaseStoreIdentity`:

```text
BaseStoreIdentity {
    canonical_repository_identity
    git_object_format
    base_tree_oid
    store_schema_version
    indexer_version
    parser_and_extractor_versions
    summary_model_and_prompt_version
    embedding_model_prompt_dimensions_and_encoding
}
```

Hash its canonical serialization for paths and store the complete identity in a
verified manifest. Prefer the tree OID over commit OID so commits with identical
content can safely share graph data.

Conceptual storage:

```text
<cache>/agent-base-stores/<repo-hash>/<identity-hash>/
├── manifest.json
├── graph.db
└── COMPLETE

<agent-data>/delta/
├── manifest.json
└── graph.db
```

The trusted parent publishes `COMPLETE` last after integrity and semantic-
completeness validation. Published Bases are opened read-only and never migrated
in place; version changes select a new identity.

### 5.2 StoreView

Introduce an explicit read abstraction rather than spreading SQL `ATTACH`
logic through callers:

```text
enum StoreView {
    Single(Store),
    Overlay {
        base: ReadOnlyStore,
        delta: Store,
        visibility: VisibilityIndex,
    }
}
```

Writes target only `Store`. Query APIs consume `StoreView` or a narrower read
trait implemented by both modes.

Visible rows are:

```text
base rows whose source file is not dirty/deleted
+ delta rows
```

### 5.3 Logical identity and edges

SQLite node IDs remain layer-local. Cross-layer resolution uses a stable key:

```text
SymbolKey {
    project
    qualified_name
    symbol_kind
    language-specific disambiguator when required
}
```

Edges are contributions owned by their source file:

```text
visible contributions
  = clean Base source-file contributions
  + Delta source-file contributions
```

Resolve endpoints against the visible logical symbol namespace after filtering.
This suppresses calls removed from dirty callers while preserving unchanged Base
callers of a symbol redefined in the Delta. The normalized store-owned
`raw_edges` table becomes the overlay boundary.

### 5.4 Delta freshness

Compute dirty state against the pinned base commit, never mutable agent `HEAD`:

- modified tracked files;
- deleted tracked files;
- rename as old-path deletion plus new-path addition;
- new untracked, non-ignored, indexable files;
- exact reverts, which remove the path from the Delta.

Each refresh transaction produces a complete Delta generation. Queries observe
the previous or next generation, never a partial mixture.

### 5.5 Query coverage

Every 0.3.1 index-backed command ships in Single and Overlay mode:

- symbols, definitions, usages, callers, callees, references;
- fan-in/fan-out, trace, `impact`, and `path`;
- indexed code and symbol search;
- semantic/vector and fused search;
- `brief`, `expand`, graph locate, stats, freshness, and diagnostics.

Vector search scores compatible Base and Delta candidates, suppresses dirty Base
paths, overfetches, and performs one deterministic top-k merge. FTS overfetches
per layer and uses one deterministic reranker; independent BM25 scores are not
assumed directly comparable.

### 5.6 Base lifecycle

On a missing Base:

1. acquire an identity-scoped builder lease;
2. build from a pristine view of the pinned Git tree in a temporary directory;
3. complete graph, content, summaries, and embeddings;
4. validate schema, integrity, identity, and completeness;
5. atomically publish and create `COMPLETE` last;
6. release the lease.

Concurrent followers wait boundedly, then use the published Base. A crashed or
timed-out build is reclaimed through managed-cache ownership rules. Persistent
failure triggers a full private Store for that run.

## 6. 0.3.2 correctness and release work

### 6.1 Differential oracle

For every query family:

```text
query(Base + Delta) == query(full private reindex of final worktree)
```

Fixtures cover body edits, added/removed calls, create/delete/rename, symbol
rename with unchanged callers, symbol move, imports, type references, ambiguous
names, comment/text edits, exact revert, multiple edits, and an agent changing
or committing `HEAD` mid-run.

Property tests generate file mutation sequences and compare normalized result
sets and declared ordering after each sequence.

### 6.2 Concurrency and recovery

Tests prove one Base builder for ten concurrent agents, no cross-agent Delta
visibility, read-only Base permissions, atomic Base publication, atomic Delta
generations, corrupt-Base quarantine, lease-safe eviction, sandbox exclusion of
the Base path, and end-to-end `--private-store` fallback.

### 6.3 Operational defects found during 0.3.2 validation

The following greppy defects were exposed by real use of the 0.3.2 feature and
are release requirements, not deferred cleanup:

- test-mode inference suppression must also prevent detached embedding workers,
  while graph-only background work remains functional;
- every short-lived test fixture must leave no orphan summary/embedding daemon
  and must release its temporary Store, preventing silent fixture recreation and
  unbounded disk growth; SQLite/cache handles are explicitly closed before
  directory removal so the same cleanup contract holds on Windows;
- automatic Delta reindexing must never mix progress text into machine-readable
  JSON output;
- `bash-smart -- CMD ...` must leave every child argument after the delimiter
  untouched, including names that resemble retired greppy edit flags;
- tests that modify process-global environment variables must serialize those
  mutations, restore the prior environment even on panic, and keep sandbox
  fixtures independent of host build-output redirects such as
  `CARGO_TARGET_DIR`;
- concurrent agent fixtures must use production-shaped unique run IDs and
  scratch paths; shared literal fixture paths must never let one test delete
  another test's live sandbox roots;
- steady-state Overlay queries must open the persisted Delta read-only, just as
  Single-Store queries do, and must load cached visibility from that same
  connection rather than opening the Delta twice; migrations, WAL setup, and
  evidence cleanup remain writer-only work rather than recurring query latency;
- the release performance harness must run explicitly in release mode and emit
  its machine-readable measurements and thresholds; its internal
  `store-cow-release-perf` build disables inference variance without enabling
  the test bypass in shipped release binaries;
- benchmark gates must materialize their externally maintained corpus from one
  public, immutable Git commit and verify the tracked SHA-256 lock before use;
  an absent or mismatched navigation, summary, or coding bank fails closed
  before any claimed evidence is published.
- benchmark repository producers, task mirrors, and consumers must share one
  explicit corpus root outside the product checkout; CI uses runner scratch
  space so baseline agents cannot discover the candidate binary, includes the
  corpus generator in store-cache keys, and fails on a missing repository root
  before costly indexing or agent execution begins.
- summary-quality judging must treat socket/read `TimeoutError` like the other
  retryable provider failures, preserve each completed batch checkpoint, and
  resume without discarding already accepted judgments; one transient read
  timeout must not void an otherwise reproducible 204-case release run.
- the Windows release matrix must request Cargo's real `cpu-only` feature, not
  an inert `cpu` workflow sentinel; Windows CPU is the certified 0.3.2 scope,
  and the hard no-GPU-backend assertion must remain active for accidental
  unqualified builds rather than breaking the intended release artifact.
- every executable release, daemon-stress, and hardware-evidence probe on
  Windows and Unix must exercise the shipping `search` command (with `--json`
  for machine-readable probes) rather than the retired `semantic-search`
  spelling; a release probe must fail on invalid product vocabulary before
  attempting to parse its CLI error text as JSON.
- release-smoke daemon draining must observe only the probe's owned runtime
  directory; a global process-name scan must never couple one validation run
  to unrelated greppy agents or daemons on the same machine.
- the integrated-agent manual must route exact definition, caller, impact,
  path, and semantic questions directly to one compact graph command,
  distinguish symbol reads from file/range reads, request `--code` only when
  body evidence is actually required, and stop when that result is sufficient;
  the registered agent-efficiency gate must be met by better agent behavior
  rather than weakened thresholds.
- agent-efficiency `variable_input` must remove each arm's own fixed first-turn
  prompt and user-query baseline from every later turn. Explorer is deliberately
  uncoached while Greppy ships this manual; charging that static manual once per
  tool round is a treatment-confounded metric, not variable search context. The
  release-gate schema is versioned when this definition changes, an invariant
  test covers unequal fixed prompt lengths with identical tool histories, and
  the 20-percent acceptance threshold remains unchanged.
- isolated provider failures must never become product-efficiency datapoints.
  The parallel runner resumes only invalid arms in bounded serial recovery
  rounds; any remaining invalid arm fails coverage, while all cost ratios use
  completed error-free pairs only. The release-gate schema is versioned for
  this fail-closed contract and the 20-percent thresholds remain unchanged.
- synthetic-only diagnostic or `--index-only` acceptance runs must not require
  the external real-corpus `MANIFEST.json`; the run manifest records real-corpus
  provenance only when a selected task uses it, while a selected real-corpus
  run continues to fail closed on a missing or mismatched manifest. Synthetic
  verification must use the registered `GREPPY_BENCH_BIN`, and every verifier
  failure must propagate as a non-zero orchestrator exit rather than producing
  a false-green index-only run.
- release-gate unit tests must construct their minimal synthetic task input
  locally and remain runnable when the externally maintained `tasks_v2.json`
  bank is intentionally absent from the product checkout.
- local release validation must use a run-owned Cargo target or an immutable
  downloaded artifact and verify the binary still exists before each evidence
  command. Shared `/Volumes/tmp` targets can be removed by unrelated cleanup;
  that invalidates the local run but must never be mistaken for a product or
  source failure, and remote exact-SHA evidence remains authoritative.
- `read-file` must accept an explicitly absolute diagnostic or artifact path
  outside the repository, while relative `../` traversal remains confined.
  This is required for agents to inspect downloaded release-gate artifacts
  through Greppy itself and is covered by CLI integration tests.

These defects are fixed in the 0.3.2 implementation and covered by the workspace,
differential Store-CoW, and release-performance test gates. They remain named
here so a later refactor cannot remove the protections as “test-only” behavior.

One additional operational defect was recorded while navigating this checkout:
a graph command can wait behind an active refresh for more than 90 seconds while
reporting only `ETA unavailable`; a refresh that later fails because the
workspace kept changing gives the blocked caller no owner, state, or progress.
This does not change Store-CoW correctness and is not allowed to reopen 0.3.2
after its exact-commit gates pass. It is a bounded 0.3.3 release requirement:
surface refresh PID/state/progress on stderr without polluting JSON stdout, wake
promptly when the writer exits or fails, and cover held-lock, killed-writer, and
failed-publication cases with deterministic tests. The gate is registered in
section 3.2 rather than left as open-ended cleanup.

### 6.4 0.3.2 build order

- **S0 — Contracts/baseline:** identities, manifests, StoreView API, SymbolKey,
  timing schema, CLI flag, multi-symbol `brief` contract fix, and frozen 0.3.1
  benchmark.
- **S1 — Base lifecycle:** builder lease, pristine-tree build, validation,
  atomic publication, read-only opening, cache leases and eviction.
- **S2 — Delta lifecycle:** dirty/deleted detection, transactional generations,
  exact revert, sandbox paths, cleanup and keep-worktree behavior.
- **S3 — Direct query primitives:** node/content lookup, normalized edge
  contributions, caller/callee/reference, content search, differential fixtures.
- **S4 — Complete queries:** traversals, vector/FTS/fused merge, hints,
  `brief`/`expand`, stats, diagnostics, all differential tests.
- **S5 — Agent integration:** automatic overlay mode, Base warming through
  `index --agent-worktree`, migration coexistence, fallback, E2E concurrency.
- **S6 — Release:** meet section 3.1, docs/SECURITY/CHANGELOG, publish 0.3.2.

## 7. 0.3.3 architecture — Filesystem-CoW

### 7.1 Rift make-or-buy gate

[Rift](https://github.com/anomalyco/rift) is a mandatory reference implementation
and benchmark competitor, not an assumed dependency. It is MIT-licensed and
already creates private CoW workspaces using writable Btrfs snapshots, Linux
per-file reflinks, and APFS `clonefile`. It also supplies lifecycle/registry,
hooks, Git working-state preservation, garbage collection, and a Rust core.

Rift does not replace Store-CoW in 0.3.2: it clones filesystem workspaces, not
greppy's immutable graph/summary/embedding Base with private logical Deltas.
0.3.2 therefore adds neither a Rift runtime dependency nor an unimplemented
Rift benchmark claim. Its Store-CoW results are baseline input; F0 in 0.3.3 owns
the Rift comparison harness and the backend-neutral phase instrumentation.

F0 performs one bounded decision spike using a pinned Rift revision:

1. benchmark `rift create --copy-all` and default filtered creation against the
   temp-worktree fallback and warm native worktree pools of size 2, 5, and 10;
2. run the greppy agent, Git-compatibility, toolchain, cleanup, and crash suites
   in Rift workspaces rather than accepting creation microbenchmarks alone;
3. inspect direct Rust-library reuse, subprocess integration, and a small
   greppy-owned native snapshot backend; record maintenance and packaging cost;
4. verify behavior for dirty index/worktree state, nested workspaces, ignored
   build output, hooks, symlinks, case behavior, quotas, sandbox boundaries, and
   proposal publication; explicitly test `rift init`, because on Linux/Btrfs it
   may convert and swap an ordinary source directory into a subvolume, which is
   not acceptable as an implicit `greppy -p` side effect;
5. publish an ADR selecting one path: integrate/pin Rift, implement the narrow
   native snapshot subset in greppy, or proceed with the custom virtual backend.

The current Rift repository is explicitly experimental, its interfaces may
change without notice, and workspace creation is not implemented on Windows.
Its filtered reflink/clone paths may still traverse directory metadata, whereas
the virtual design can serve unchanged names directly from Git. Therefore Rift
cannot be the sole cross-platform backend without greppy-owned compatibility,
version pinning, fallbacks, and lifecycle tests. Conversely, equivalent native
CoW performance and correctness block construction of a custom mount stack:
engineering novelty is not a release gate.

The ADR is binding for the rest of 0.3.3. If Rift wins, greppy builds the Rift
adapter and does **not** also build WorkspaceCore/FUSE/FSKit/WinFsp. If a narrow
native backend wins, only that subset is built. The custom virtual stack is
authorized only by a measured unmet requirement, and the unselected branches
leave the release scope.

### 7.2 Backend boundary

Refactor the current workspace into a stable interface before adding mounts:

```text
trait AgentWorkspaceBackend {
    create(repo, pinned_commit, run_id, limits) -> Workspace
    path(workspace) -> Path
    usage(workspace) -> WorkspaceUsage
    finish(workspace, message) -> RunOutcome
    cleanup(workspace)
}

NativeWorktreeBackend   # existing stable/temp behavior and warm pool baseline
NativeCowBackend        # Rift integration or greppy-owned snapshot/reflink path
VirtualCowBackend       # custom mount path, only when authorized by F0
```

`run_agent` depends only on this interface. Proposal outcome, sandbox setup,
Store selection, keep-worktree, error reporting, and cleanup are backend-neutral.

### 7.3 Conditional WorkspaceCore — virtual-backend path only

Only when F0 authorizes the custom virtual backend, add `crates/workspace-cow`
with no platform mount code:

Sections 7.4–7.8 describe only this conditionally authorized virtual path; a
Rift or narrow-native ADR implements equivalent public contracts through its
selected backend without inheriting this internal design.

```text
WorkspaceCore
├── immutable Git base reader
├── namespace overlay
├── disk-backed whole-file Delta
├── tombstones and directory redirects
├── metadata and handle table
├── quotas
├── shared Base caches
├── private Git-state manager
└── lifecycle journal and telemetry
```

Use a read-only `gix` integration for commit/tree/blob access and Git object
format support. Continue using the Git CLI for compatibility-sensitive private
Git setup and controlled proposal publication.

### 7.4 Lookup and copy-up

```text
lookup(path):
  1. apply directory redirects
  2. if Delta entry exists, return it
  3. if tombstoned, return ENOENT
  4. otherwise resolve Git tree entry from pinned Base

open-for-write(path):
  1. create private native Delta file
  2. copy Base blob and metadata if the file existed
  3. atomically publish Delta entry
  4. direct subsequent I/O to the native Delta file
```

File content is disk-backed. RAM is a bounded cache for Git tree entries,
directory merges, stats, and decompressed blob pages—not the primary Delta.

### 7.5 Namespace operations

Support before release:

- create/open/read/write/append/truncate/fsync;
- mkdir/rmdir/unlink;
- atomic file and directory rename;
- executable mode and normal permissions;
- symbolic links;
- hard links within the private Delta, materializing a Base file first;
- stable stat identity for the workspace lifetime;
- directory enumeration with Base/Delta merge and tombstone filtering;
- file locking and memory-mapped reads/writes as required by supported adapters;
- user xattrs needed by normal toolchains, with unsupported namespaces returning
  the platform-correct error.

A rename of a non-materialized Base directory uses a logical prefix redirect and
tombstone; it never copies the subtree. Redirect cycles and ancestry violations
are rejected transactionally.

Git-incompatible case collisions are detected during preflight on case-folding
hosts and cause native-worktree fallback.

### 7.6 Delta storage and limits

Conceptual state:

```text
<cache>/agent-cow/<workspace-id>/
├── manifest.json
├── lifecycle.journal
├── metadata.db
├── upper/
├── git/
└── spill/              # reserved; ordinary upper files are already disk-backed
```

Limits:

```text
max_delta_bytes
max_created_files
max_single_file_bytes
max_open_handles
max_dirty_bytes
```

Default limits are documented and configurable through agent CLI options or a
single versioned config block. Quota accounting reserves space before writes;
failure returns `ENOSPC`/`EDQUOT` or the corresponding Windows status.

Build outputs are ordinary private Delta files. No default 2-GiB cap is chosen
that would make representative `cargo check`/`npm test` fail; defaults are based
on the registered build fixtures and available disk.

### 7.7 Private Git state

Each CoW workspace receives a private real Git directory:

```text
git/
├── HEAD
├── index
├── refs/
├── logs/
└── objects/
```

The private object database borrows existing objects from the repository through
Git alternates. New commits, refs, reflogs, and objects remain agent-private.
The mounted root exposes a `.git` file pointing to this state, so ordinary
commands work:

```text
git status
git diff
git add -A
git commit
```

`finish()` stages the visible final tree in the private Git state, creates a
commit whose parent is the pinned base regardless of agent `HEAD`, and then uses
trusted host-side Git to transfer the exact reachable objects and atomically
publish `refs/greppy/agent/<run-id>` in the main repository. A failed transfer
leaves private state recoverable and does not publish a partial ref.

### 7.8 Shared caches

Cache across agents sharing a Base tree:

- commit → root tree;
- path → Git tree entry;
- directory → immutable Base listing;
- blob OID → verified decompressed pages;
- path → stable Base stat template.

Agent-private directory merges and Delta handles are never shared. Caches are
bounded, content-verified, and accounted separately from workspace quotas.

## 8. Platform support and adapters

One conformance suite drives every selected adapter. The 0.3.3 contract is
Filesystem-CoW on every platform where the F0-selected backend passes the gates,
with the existing native-worktree fallback everywhere else. Unsupported CoW is
reported explicitly and never emulated by a hidden full copy.

### Linux — supported

- Rift, a greppy-owned snapshot/reflink backend, or FUSE3 according to the ADR;
- used by `auto` only when filesystem capabilities and preflight pass;
- packaged with all selected runtime requirements declared;
- CI on current Ubuntu x86_64 and ARM64, including cleanup/crash tests.

### macOS — preview

- Rift/APFS `clonefile`, a greppy-owned clone backend, or an FSKit app-extension
  according to the ADR; an FSKit choice includes a small signed helper;
- notarized packaging and install/uninstall path documented when required;
- `auto` selects it only after backend health/preflight succeeds and the
  performance gates pass; otherwise native worktree fallback;
- CI on current supported macOS ARM64 plus Intel where the release supports it.

### Windows — preview

- native-worktree fallback is supported and explicit because Rift workspace
  creation is currently not implemented on Windows;
- CoW is preview only if F0 selects or authorizes a Windows-capable backend;
- a virtual-backend choice uses native WinFsp, not its POSIX FUSE layer, with
  prerequisite detection and case/rename/lock/mmap/ACL/unmount coverage.

Preview means shipped, documented, correctness-gated, and force-selectable; it
does not promise a CoW adapter on a platform where the selected backend cannot
meet the gates. The native worktree backend remains fully supported everywhere.

## 9. 0.3.3 repository preflight and fallback

Before allocating or mounting state, reject CoW and fall back to native when:

- tracked submodules/gitlinks are present;
- Git LFS or required checkout/smudge filters would change visible blob content;
- case collisions cannot be represented by the host;
- the adapter/helper/runtime is absent or unhealthy;
- the Delta/state root lacks required capacity or permissions;
- the pinned object database is incomplete and cannot provide Base blobs;
- a repository feature used by the current Git version is unsupported.

Preflight is cached only against an identity that includes relevant Git config,
attributes, platform, adapter, and repository tree. Fallback never leaves a
half-mounted path registered as a worktree.

## 10. 0.3.3 correctness suites

### 10.1 Filesystem model tests

Run randomized state-machine sequences against both WorkspaceCore and a native
reference directory, comparing visible trees, bytes, modes, links, errors, and
directory listings after every operation. Include create, overwrite, append,
truncate, delete, recreate, file/directory rename, nested redirects, hard links,
symlinks, concurrent handles, fsync, and quota boundaries.

### 10.2 Git compatibility tests

For every supported adapter:

- `git status` clean at creation;
- modify/create/delete/rename reflected exactly;
- `git diff`, `add`, `restore`, `reset`, `commit`, and ignored files;
- executable-bit and symlink changes;
- agent mid-run commits cannot alter shared refs/objects;
- `finish()` captures final filesystem state against pinned Base;
- proposal commit/tree equals the result from the native worktree backend.

### 10.3 Toolchain tests

Run representative real projects through:

- greppy index/navigation/edit operations;
- `rg` and metadata-heavy recursive traversal;
- `cargo check`/test;
- `npm` install/test and `tsc`;
- `go test`;
- CMake/Ninja fixture.

Compare outputs and test results with native worktrees; benchmark separately.

### 10.4 Lifecycle and fault injection

Kill the agent/helper/core during mount, copy-up, rename, metadata transaction,
Git commit, proposal publication, unmount, and cleanup. Recovery either restores
a consistent kept workspace or safely reclaims private state. It never reuses an
identity with an incomplete lifecycle journal.

## 11. 0.3.3 build order

- **F0 — Backend contract/make-or-buy:** `AgentWorkspaceBackend`, frozen CLI,
  native backend migration without behavior change, repository compatibility
  checks, Rift plus warm-pool E2E baselines, bounded integration spike, and the
  backend ADR required by section 7.1.
- **F1 — Selected backend:** implement exactly the ADR winner: pinned Rift
  integration, the narrow greppy-owned native snapshot/reflink subset, or (only
  when authorized) WorkspaceCore plus its platform adapters.
- **F2 — Git/proposal parity:** preserve normal Git behavior and produce the
  same atomic proposal ref and tree as the native-worktree backend; use private
  Git state/object transfer only where the selected backend requires it.
- **F3 — Lifecycle and platform hardening:** packaging, capability detection,
  quotas, cleanup, crash recovery, and the common filesystem/Git/toolchain
  conformance suite on every selected platform.
- **F4 — Agent integration:** frozen `auto` policy, Store-CoW composition,
  diagnostics, keep-worktree, cleanup, concurrency and 50-agent stress tests.
- **F5 — Release:** meet section 3.2, optimize caches without weakening
  semantics, README/SECURITY/CHANGELOG/install docs, publish 0.3.3.

All F0–F5 milestones are in 0.3.3 scope. The ADR closes the two unselected
implementation branches; it does not turn them into deferred work. Platform
support is declared as in section 8 and a native fallback is never mislabeled
as CoW.

## 12. Measurement plan across both releases

Instrument monotonic phases:

```text
workspace.preflight
workspace.create_or_mount
store.base_open_or_build
store.delta_refresh
store.summary
store.embedding
agent.first_model_request
agent.loop
workspace.finish
workspace.unmount_or_remove
```

Benchmark matrix:

| Dimension | Values |
|---|---|
| Repository | small fixture, greppy, large/300k-file fixture |
| Concurrency | 1, 2, 5, 10; 50 stress-only |
| Base Store | absent, warm |
| Workspace | stable native, temp native, warm pool k=2/5/10, Rift, selected CoW |
| Changes | none, 10 edits, create/delete/rename mix |
| Build | none, cached incremental, clean representative build |

Report median/P95 phase latency and full agent wall clock, query latency, CPU, peak RSS, Base cache hit
rate, summary/embedding reuse, Delta bytes/files, mount/helper CPU, build output
size, workspace create/cleanup time, and fallback counts/reasons.

Use a fixed/no-op model gateway for startup measurements so inference variance
does not hide Store or filesystem effects.

Every baseline and release benchmark invokes an absolute binary path and records
that path, `greppy --version`, source commit, feature flags, Store schema, and
indexer version in its result manifest. Results produced by another greppy found
incidentally through `PATH` are invalid for the release gates.

## 13. Security and isolation

- Shared Base Store and Git Base objects are read-only to agent sandboxes.
- Store Delta, filesystem Delta, private Git state, and scratch are distinct
  writable roots scoped to one agent.
- The mount adapter accepts requests only for its assigned workspace ID and
  never trusts an agent-supplied host path.
- Paths are normalized once; traversal, symlink escape, and alternate-data-path
  tricks are rejected before host filesystem access.
- Host-side proposal publication pins repository and private Git identities and
  repeats the current tamper checks.
- Quotas govern Delta bytes/files/handles; they are not a substitute for process
  sandboxing, CPU limits, or network policy.
- Kept workspaces and caches follow existing ownership, permissions, lease, and
  trash rules; cleanup never recursively deletes unresolved broad paths.
- SECURITY documents that Base indices, blob caches, and private Deltas contain
  source-derived data and may require encrypted/ephemeral storage.

## 14. Migration and compatibility

### 0.3.2

Existing private Stores are not destructively migrated. The new Base is built
beside them; old data remains eligible for normal managed eviction. Invalid Base
state falls back to a new full private Store.

### 0.3.3

The current native worktree backend remains supported and selectable. `auto`
falls back before contacting the model when CoW cannot start. A CoW failure after
agent changes preserves private state and prints a recovery path; it never
switches live workspaces underneath the agent.

Proposal refs and user-facing outcome formats remain compatible across both
workspace backends.

## 15. Risks and binding responses

1. **Store overlay is broader than a SQL union.** Use StoreView, logical symbol
   identity, source-owned edge contributions, and full-reindex differential
   tests.
2. **FTS/vector merging changes ranking.** Register ordering fixtures,
   overfetch, deterministic rerank, and enforce the latency gates.
3. **Base identity omits a semantic input.** Manifest every input and fail
   closed to a new identity.
4. **Userspace filesystem metadata latency is high.** Stable stat identities,
   shared immutable caches, native Delta handles, batch/prefetch where adapters
   permit, and binding performance gates.
5. **Build output dwarfs source edits.** Disk-backed upper, realistic quota
   defaults, separate build benchmarks, no RAM-first promise.
6. **Directory rename materializes a subtree.** Prefix redirects are mandatory
   before adapter integration.
7. **Private Git is not actually private.** New objects/refs/index live in the
   private Git dir; shared objects are alternates opened read-only; publication
   is an explicit trusted transfer.
8. **Cross-platform packaging delays 0.3.3.** F0 includes packaging cost in the
   backend decision; only selected adapters enter the release path, and every
   unavailable CoW platform has an explicit native fallback rather than a late
   custom-adapter expansion.
9. **First-run Base construction serializes agents.** One builder publishes;
   followers wait boundedly or use safe private fallback.
10. **CoW adapter crashes strand mounts.** Lifecycle journal, backend health,
    startup scavenger, explicit recovery, and adapter-specific forced-unmount
    tests ship before release.
11. **A custom filesystem duplicates a simpler native-CoW solution.** Rift and
    warm worktree pools are compulsory E2E baselines; F0 blocks custom adapter
    work until the ADR identifies a concrete correctness, portability, or
    performance gap worth its lifetime maintenance cost.

## 16. Documentation deliverables

### 0.3.2

- shared Base and private Delta model;
- Base identity, location, leases, and eviction;
- `index --agent-worktree` warming behavior;
- `--private-store` and fallback diagnostics;
- isolation and source-derived cache data.

### 0.3.3

- `--workspace auto|worktree|cow` and exact auto policy;
- platform support/prerequisites and install/uninstall;
- unsupported-repository fallback conditions;
- quotas, build-output behavior, kept workspace recovery;
- performance characteristics and how to diagnose adapter selection.

## 17. Definitions of done

### greppy 0.3.2 is done when

- S0–S6 are merged;
- every 0.3.1 index-backed command passes Single-vs-Overlay differential tests;
- concurrency, corruption, crash, sandbox, migration, and fallback tests pass;
- section 3.1 targets hold;
- docs, help, SECURITY, and CHANGELOG are complete;
- ten concurrent agents demonstrate one Base and isolated Deltas;
- release artifacts are published as 0.3.2.

### greppy 0.3.3 is done when

- F0–F5 are merged;
- Linux FUSE3 is supported and macOS FSKit/Windows WinFsp ship with the declared
  preview contracts;
- model, Git, toolchain, lifecycle, and fault suites pass per platform status;
- section 3.2 targets hold or `auto` remains disabled on the failing preview
  platform as specified;
- native and CoW proposal commits are tree-identical for the registered suite;
- docs, help, SECURITY, install packaging, and CHANGELOG are complete;
- ten concurrent agents demonstrate no full temporary checkout materialization;
- release artifacts are published as 0.3.3.
