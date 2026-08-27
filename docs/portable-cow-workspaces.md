# Portable CoW workspaces

Greppy 0.3.4 replaces the filesystem-specific 0.3.3 workspace experiment with
one portable contract. Copy-on-write here means that unchanged bytes are
shared by Greppy's own content store; it does not mean that the host filesystem
must implement reflinks, snapshots, or block cloning.

## Setup and health

Install the platform package, then run:

```text
greppy workspace setup
greppy workspace doctor --json
greppy workspace status --json
```

`setup` installs or activates the packaged adapter. `doctor` verifies the live
control manifest against the marker read through the mount, checks recovery and
CAS integrity, and performs mounted create/read/partial-write/rename/delete
operations. `greppy -p` repeats the mechanical health preflight and makes no
model request if the provider is unavailable, stale, recovering, or has a
different identity.

Setup also records the provider for the next login using the platform's
per-user lifecycle mechanism. Linux installs and enables a restartable systemd
user unit with the exact selected data and mount roots. macOS installs a
RunAtLoad LaunchAgent that repeats the idempotent FSKit setup/health check after
login; FSKit activation itself remains controlled by macOS. Configuration files
are atomically replaced and an existing symlink is replaced rather than
followed. Windows service registration is intentionally absent from diagnostic
builds until the signed hardlink-capable transport is selected and packaged.

There is one persistent user mount with workspaces below `workspaces/<id>`.
Creating an agent workspace updates namespace metadata; it does not create a
second mount or traverse and copy the repository.

## Storage model

- File data is split into fixed 1 MiB chunks addressed by BLAKE3.
- Chunks are appended to segment files rather than stored as millions of small
  files.
- SQLite WAL records baselines, inodes, directory entries, links, tombstones,
  redirects, chunk references, proposals, and recovery journals.
- Writes preserve untouched chunk references. A one-byte overwrite creates at
  most one new data chunk plus metadata.
- Garbage collection removes only unreferenced chunks and fully recovered
  orphan workspaces. Live workspaces and proposal-pinned baselines are roots.

The initial namespace is two immutable layers: the tree at a pinned commit and
a dirty snapshot containing staged, unstaged, deleted, and untracked paths.
Ignored paths and build caches are not captured. The repository HEAD, index,
status, affected contents, and metadata are sampled twice; a changing source
is retried once and then rejected.

Merge, rebase, cherry-pick, submodule, Git LFS, and arbitrary checkout or
smudge filters are rejected before workspace creation because Greppy cannot
prove an exact portable baseline for those states.

## Filesystem contract

The mounted namespace implements ordinary toolchain operations: reads, writes,
append, truncate, memory mapping, byte-range locks, symbolic and hard links,
executable bits, atomic replacement renames, directory enumeration, and
deletion. Paths are normalized and confined before dispatch; traversal,
symlink escape, foreign mounts, and cross-workspace handles fail closed.

Each workspace contains private Git control state. Its index, refs, and new
objects are writable and isolated. Existing repository objects are shared
read-only. Concurrent workspaces cannot see one another's namespace, handles,
refs, indexes, or unpublished chunks.

## Proposals and apply

An agent result is published under `refs/greppy/agent/<id>`. Its commit parent
is the pinned source commit and its tree is the complete final tree. For a
dirty source, the review patch is computed from the immutable initial snapshot
to the final state, so the user's pre-existing dirty work is not presented as
the agent's change.

Apply with:

```text
greppy agent apply refs/greppy/agent/<id>
```

Apply verifies the exact baseline manifest and refuses any mismatch. It leaves
the current Git index unchanged, transfers only the agent delta, and uses a
backup plus recovery journal. A crash is recovered or reported before another
agent can start. Ordinary `git cherry-pick` is not a safe apply mechanism for
dirty-based proposals.

`--keep-worktree` retains the namespace, provider data, and proposal pins for
diagnosis. Normal cleanup removes the private namespace and delta; proposal
baselines remain pinned until their refs are removed.

## Platforms

- Linux x86_64 uses FUSE3. Package setup validates `/dev/fuse`, the kernel
  component, and user mount permissions. The package includes the systemd user
  unit and `workspace setup` binds it to the actual per-user roots. No
  particular backing filesystem is required.
- macOS ARM64 requires macOS 15 or newer and a signed, notarized FSKit app
  extension. The one-time System Settings approval is an OS security boundary;
  Greppy remains unavailable until activation. A minimal Swift host bridges
  FSKit to the Rust core; the LaunchAgent only replays idempotent setup after
  login and does not bypass that approval boundary.
- Windows x86_64 is a release blocker until a licensed, signed kernel transport
  forwards real hardlink operations to Greppy's Rust provider. Official
  unchanged WinFsp 2.1 rejects `FileLinkInformation` before userspace, so the
  current diagnostic package intentionally fails the common mounted contract.
  Greppy does not substitute copies or aliases, does not use ProjFS, and does
  not rely on NTFS CoW behavior.

Release packages include the adapter, driver/runtime dependency, checksums,
signatures, SBOM, provenance attestations, and complete third-party notices.
The release remains blocked until clean-machine install, activation,
upgrade/uninstall, cross-platform conformance, crash/security, isolation, and
performance gates pass on one exact commit. Agent benchmarks are diagnostic;
each suspected product defect is reproduced and classified independently.

## 0.3.3 compatibility

Greppy 0.3.3 is unchanged and reproducible from its tag. Its Rift-derived APFS,
Btrfs, and Linux reflink paths, `auto`/`native`/`cow` selector, `--fresh`, and
native Git-worktree fallback are historical behavior only. They are not
compiled, packaged, licensed as current dependencies, or used at runtime by
0.3.4.
