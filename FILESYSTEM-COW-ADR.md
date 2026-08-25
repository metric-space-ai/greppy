# ADR: Filesystem-CoW for Greppy 0.3.3

Status: candidate implementation; not released

Date: 2026-08-25

## Decision

Filesystem-CoW is the sole feature candidate for Greppy 0.3.3. Greppy uses a
hard, purpose-specific fork of Rift's native snapshot mechanics instead of
shipping Rift as a workspace manager or building a custom virtual filesystem.

The dependency is `greppy-rift-core` from
<https://github.com/mkh-welsch/rift>, pinned at
`319bbd1b5a31a88d3cf1cb929e50139178d12a30`. Its audited upstream base is
`anomalyco/rift` commit
`757a22cb247f9b24a849c9d6bd56f49c0ec494f8`. The retained API is limited to
capability probing, exact snapshot creation, and snapshot removal. Greppy owns
template preparation, private Git state, locking, fallback, sandbox roots,
proposal publication, diagnostics, and lifecycle policy.

## Why a hard fork

The needed native primitives are small and platform-specific. Rift's CLI,
JavaScript/Bun/Node FFI, manager, registry, hooks, markers, filtering, Git
policy, and conversion behavior would enlarge Greppy's trusted surface without
serving the product contract. The fork removes those components and does not
promise API or merge compatibility with upstream.

The retained backends are:

- macOS: exact APFS directory cloning with `clonefile`; file data is CoW but
  directory metadata is traversed;
- Linux Btrfs: writable snapshots of an existing subvolume;
- Linux filesystems supporting reflinks: exact per-file `FICLONE` trees;
- all other cases: explicit unsupported result.

No backend silently degrades to an ordinary recursive copy.

## Product contract

The CLI surface is exactly:

```text
--workspace-backend auto|native|cow
GREPPY_WORKSPACE_BACKEND=auto|native|cow
```

`auto` attempts CoW before the model starts only when probing reports
constant-time metadata creation, then falls back to the unchanged 0.3.2 native
Git-worktree backend. `native` always selects that 0.3.2 behavior. `cow`
requires a supported, successful exact snapshot and may select a per-file
reflink tree; otherwise it returns a typed, visible error.

With the retained native primitives, Btrfs subvolume snapshots satisfy the
constant-metadata contract. APFS directory `clonefile` and Linux per-file
`FICLONE` are exact CoW implementations but walk the namespace; they are
explicit `cow` previews and are not selected by `auto`.

A CoW workspace receives a real private `.git` directory. The pinned base
repository's objects are readable through `.git/objects/info/alternates`; the
workspace's index, refs, new objects, and commits remain private. `finish`
creates and verifies one proposal commit, imports that commit into the main
object database, and atomically updates only
`refs/greppy/agent/<run-id>`. Main HEAD and index are invariant.

Git identity or containment tampering blocks both publication and automatic
deletion. Cleanup invokes only the selected native backend and never treats an
unverified path as a snapshot owned by Greppy.

## Release and rollback rule

This is a binary release decision, not an open-ended migration:

1. Release 0.3.3 only when every correctness, safety, packaging, platform, and
   registered performance gate passes for the exact candidate commit.
2. Do not ship a partial CoW implementation, a hidden recursive-copy backend,
   a legacy CLI alias, or a compatibility layer for removed Rift behavior.
3. If a mandatory gate fails and cannot be corrected within this candidate,
   abandon the CoW 0.3.3 candidate. Preserve the already released 0.3.2
   behavior and select a different feature for a new 0.3.3 candidate.

Agent benchmarks may discover and reproduce bugs, but they are diagnostic
inputs for subsequent fixes, not release gates by themselves. A report changes
the product only after deduplication, minimal reproduction, classification,
blast-radius analysis, and a regression test.

## Evidence and remaining gates

The pinned fork commit passed its macOS, Linux, Btrfs, XFS-reflink, and
ext4-unsupported CI matrix in GitHub Actions run `32814344750`. A controlled
300,000-file macOS release-gate run on the preceding revision showed APFS
directory `clonefile` still inside the recursive clone after six minutes. The
current dependency therefore reports APFS metadata traversal truthfully and
locks that classification in its macOS native test. Greppy unit coverage
currently proves private Git isolation, tamper preservation, exact native/CoW
proposal-tree parity, cleanup, and ten concurrent CoW workspaces.

Before release, the exact Greppy candidate must still pass the complete
workspace test matrix, packaged-artifact/license/SBOM checks, Windows native
fallback, Greppy-integrated Linux Btrfs and reflink tests, crash/fault recovery,
50-workspace stress, and the preregistered end-to-end latency/storage gates in
`PARALLEL-AGENT-COW-PLAN.md`.
