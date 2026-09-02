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
followed. The Windows MSI installs an uninstall-safe machine Run entry that
replays setup at login; Greppy validates that it points to the current package
before starting the adjacent private provider and driver.

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
- The first exclusive opener runs SQLite `quick_check` plus foreign-key checks
  over namespace and chunk metadata before recovery. A logically corrupt
  database remains unavailable; Greppy does not mark it healthy or start an
  agent merely because SQLite can parse its header.

The initial namespace is two immutable layers: the tree at a pinned commit and
a dirty snapshot containing staged, unstaged, deleted, and untracked paths.
Ignored paths and build caches are not captured. The repository HEAD, index,
status, affected contents, and metadata are sampled twice; a changing source
is retried once and then rejected. Hardlink identity among captured dirty
files is sampled in both observations and becomes part of the baseline hash;
promoting any alias binds every visible peer to the same private inode.

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
an agent edit. Because Git trees do not encode hardlink topology, Greppy stores
canonical hardlink groups beside the proposal and binds their SHA-256 digest
into the proposal commit message. Apply and crash recovery verify that binding,
include every peer in the rollback journal, and recreate real hardlinks rather
than independent copies.

Apply with:

```text
greppy agent apply refs/greppy/agent/<id>
```

Apply verifies the exact baseline manifest and refuses any mismatch. It leaves
the current Git index unchanged, transfers only the agent delta, and uses a
backup plus recovery journal. A crash is recovered or reported before another
agent can start. Ordinary `git cherry-pick` is not a safe apply mechanism for
dirty-based proposals.

Proposal publication, apply, and both recovery paths share one OS-backed
exclusive lease per canonical repository. Publication journals couple the
baseline ref, pinned Core proposal, and public proposal ref; apply journals
couple the exact initial snapshot to every affected path. Journals and their
directories are synced before visible changes. The kernel releases the lease
after a process crash. If the lease is still held, recovery treats the
operation as active and fails closed instead of rolling it back underneath the
owner. A journal that is a symlink, names another repository, or disagrees with
the Core metadata is rejected before Git or filesystem restoration begins.

`--keep-worktree` retains the namespace, provider data, and proposal pins for
diagnosis. Normal cleanup removes the private namespace and delta; proposal
baselines remain pinned until their refs are removed.

## Platforms

- Linux x86_64 uses FUSE3. Package setup validates `/dev/fuse`, the kernel
  component, and user mount permissions. The package includes the systemd user
  unit and `workspace setup` binds it to the actual per-user roots. No
  particular backing filesystem is required.
- macOS ARM64 requires macOS 15.4 or newer and a signed, notarized FSKit app
  extension. The one-time System Settings approval is an OS security boundary;
  Greppy remains unavailable until activation. A minimal Swift host bridges
  FSKit to the Rust core; the LaunchAgent only replays idempotent setup after
  login and does not bypass that approval boundary. Developer-ID builds also
  require separate embedded Developer-ID provisioning profiles for
  `ai.metricspace.greppy.workspacefs` and
  `ai.metricspace.greppy.workspacefs.extension`. Both must authorize
  `group.ai.metricspace.greppy`, contain the selected signing certificate, and
  bind the same team; the extension profile must additionally authorize the
  FSKit Module entitlement. The release build fails before compilation when
  either profile is absent and validates signature, expiry, distribution type,
  exact bundle ID, certificate and entitlement allowlist before signing.
  Notarization without these profiles is insufficient for FSKit activation and
  is never accepted as release evidence. At runtime, `workspace setup` checks
  both profiles as regular embedded CMS files and verifies the complete bundle
  with Developer ID code-signing and Gatekeeper before it opens System
  Settings. An older or incomplete installation therefore fails with a
  reinstall diagnostic instead of directing the user to a switch macOS cannot
  activate. Replacing or updating the application bundle can cause macOS to
  require approval again. Before mounting, setup now queries FSKit for the
  enabled extension whose bundle URL belongs to the exact installed app. If it
  is disabled, missing, or still bound to the replaced app, setup opens the
  File System Extensions pane directly and names the `Greppy Workspace FS`
  switch; enable it and rerun `greppy workspace setup`. This does not bypass
  the macOS security boundary. Private provider data remains in the signed application-group
  container, while the actual FSKit mount is
  `~/Library/Application Support/greppy/workspace-mount`; macOS rejects
  filesystem mounts inside managed Group Container directories.

For the macOS release-signing handoff, the Apple Developer account for team
`2HS27B8739` must contain these exact resources:

- App ID `ai.metricspace.greppy.workspacefs`, with App Groups enabled for
  `group.ai.metricspace.greppy`;
- App ID `ai.metricspace.greppy.workspacefs.extension`, with App Groups and
  the FSKit Module capability enabled;
- one Developer ID provisioning profile for each App ID, both bound to the
  same Developer ID Application certificate used by
  `MACOS_SIGNING_IDENTITY`;
- a Developer ID Installer certificate, including its private key, for signing
  the final PKG.

Download the two profiles without renaming their contents. Before adding any
secret, verify the exact files by building the real app locally; the build
decodes each CMS profile and rejects the wrong team, bundle ID, application
group, FSKit role, expiry, distribution type, or signing certificate:

```bash
CODE_SIGN_IDENTITY='Developer ID Application: Michael Welsch (2HS27B8739)' \
APP_PROVISIONING_PROFILE=/absolute/path/greppy-workspace-app.provisionprofile \
FSKIT_PROVISIONING_PROFILE=/absolute/path/greppy-workspace-extension.provisionprofile \
GREPPY_CLI_BINARY=/absolute/path/greppy \
  platform/macos/build-fskit-app.sh /absolute/new/output-directory
codesign --verify --deep --strict --verbose=2 \
  /absolute/new/output-directory/GreppyWorkspaceFS.app
```

Store the validated inputs in GitHub under the exact names consumed by the
release workflow:

- `MACOS_FSKIT_APP_PROVISIONING_PROFILE_BASE64`;
- `MACOS_FSKIT_EXTENSION_PROVISIONING_PROFILE_BASE64`;
- `MACOS_INSTALLER_CERTIFICATE_P12_BASE64`;
- `MACOS_INSTALLER_CERTIFICATE_PASSWORD`;
- `MACOS_INSTALLER_SIGNING_IDENTITY`.

The existing application-certificate and notary secrets are separate inputs;
an application certificate cannot sign an installer package. The signed
release workflow imports both identities into an ephemeral keychain, validates
both embedded profiles before writing the app, notarizes and staples the app,
signs the PKG with the installer identity, then notarizes and staples the PKG.
Neither an ad-hoc build nor a notarized app without both profiles is valid
release evidence.

- Windows x86_64 uses Greppy's minimal WinFsp transport fork because unchanged
  WinFsp 2.1 rejects `FileLinkInformation` before userspace. The fork forwards
  real hardlink operations to the same Rust provider and ships with its exact
  corresponding source. The release requires a Hardware Dev Center
  HLK/dashboard signature: the returned catalog must contain Windows Hardware
  Driver Verification EKU `1.3.6.1.4.1.311.10.3.5` and must not contain the
  attestation EKU `1.3.6.1.4.1.311.10.3.5.1`. The catalog, signer evidence,
  signed driver and canonical unsigned PE payload are hash-bound in the
  release contract. Attestation signing is accepted neither as production
  evidence nor as a release shortcut. The release remains blocked until that
  exact driver is signed and the installed MSI passes the common mounted
  contract, upgrade, uninstall, isolation and performance gates. Greppy does not
  substitute copies or aliases, does not use ProjFS, and does not rely on NTFS
  CoW behavior.

For the Windows release-signing handoff, build the pinned fork on Windows,
submit that exact driver through the Hardware Dev Center HLK flow, and download
the returned `.sys` and `.cat`. Then generate and bind the evidence without
editing either returned file:

```powershell
tools/verify_windows_driver_signatures.ps1 `
  -DriverPath greppyworkspacefsp-x64.sys `
  -CatalogPath greppyworkspacefsp-x64.cat `
  -OutputPath greppy-windows-driver-signature-evidence.json
python tools/windows_driver_contract.py create `
  --unsigned path/to/fork/build/Release/greppyworkspacefsp-x64.sys `
  --signed greppyworkspacefsp-x64.sys `
  --catalog greppyworkspacefsp-x64.cat `
  --fork-manifest third_party/winfsp-greppy/upstream.json `
  --signature-evidence greppy-windows-driver-signature-evidence.json `
  --output greppy-windows-driver-contract.json
```

The release workflow receives the signed driver, catalog, and contract through
the three `WINDOWS_SIGNED_WINFSP_*_BASE64` secrets. It recreates the signature
evidence from the supplied files and requires an exact semantic contract match;
the evidence itself is therefore not accepted as a trusted secret.

Release packages include the adapter, driver/runtime dependency, checksums,
signatures, SBOM, provenance attestations, and complete third-party notices.
The release remains blocked until clean-machine install, activation,
upgrade/uninstall, cross-platform conformance, crash/security, isolation, and
performance gates pass on one exact commit. Agent benchmarks are diagnostic;
each suspected product defect is reproduced and classified independently.

### Cross-platform performance evidence

The existing `Portable CoW` workflow becomes the authoritative
three-platform release gate when manually dispatched with
`full_platform_performance=true`. Because that workflow already exists on the
default branch, it can execute the feature-branch RC before merge. It builds
the provider and measurement harness from the selected commit, generates the
identical 300,000-file fixture, and runs through the real FUSE3, FSKit, or
WinFsp mount. Linux and Windows use clean hosted runners. macOS uses an
ephemeral Apple Silicon runner carrying the `greppy-fskit-performance` label,
a configured
`MACOS_SIGNING_IDENTITY` secret used by the release workflow, and an OS/MDM
approval for the Greppy FSKit bundle
identity. The job refuses an existing data or mount root; this prevents warm
state from a previous run from becoming release evidence.

Each platform uploads `greppy.portable-cow-performance.v1`. The final job
runs `tools/verify_portable_cow_performance.py` and accepts exactly Linux
x86_64, macOS ARM64, and Windows x86_64 from the same full Git commit. It also
rejects a dirty source tree, non-release builds, modified fixture size,
relaxed limits, P95 above 500 ms, more than 1 MiB for an untouched workspace,
anything other than one changed 1-MiB chunk for a 1-byte write, fewer than 50
parallel workspaces, or Rust/Python/Node overhead above 20%. The immutable
release workflow requires a successful run of this workflow for its exact
subject commit; a Linux-only result can never authorize `v0.3.4`.

## 0.3.3 compatibility

Greppy 0.3.3 is unchanged and reproducible from its tag. Its Rift-derived APFS,
Btrfs, and Linux reflink paths, `auto`/`native`/`cow` selector, `--fresh`, and
native Git-worktree fallback are historical behavior only. They are not
compiled, packaged, licensed as current dependencies, or used at runtime by
0.3.4.
