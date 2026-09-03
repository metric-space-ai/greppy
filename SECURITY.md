# Security Policy

## Supported versions

Security fixes are provided for the newest published Greppy release. The
unreleased `main` branch is tested continuously but is not a supported release
channel. Pin production installations to an immutable tag and verify the
published checksum and build provenance.

## Release scope

A release is cut only when, on the exact release commit, all of the following
pass: CI, CodeQL, the Rust dependency security audit, the task-bank
reproducibility audit, and the summary-quality gate — plus code signing and
notarization. Agent benchmarks run continuously as diagnostic inputs for a
subsequent release; they never gate publication of the release under test.
Their findings require reproduction and product/harness/prompt/usage triage
before any product change.

## Verifying a release

The expected provenance identity is the repository
`metric-space-ai/greppy`, the signer workflow
`.github/workflows/release.yml`, and the release tag being installed. Download
all assets into an empty directory so the exact release manifest can reject
missing or unexpected files:

```bash
version=v0.3.3
mkdir "greppy-$version" && cd "greppy-$version"
gh release download "$version" --repo metric-space-ai/greppy
python3 - <<'PY'
import json
from pathlib import Path
manifest = json.loads(Path("RELEASE-ASSETS.json").read_text())
expected = {asset["name"] for asset in manifest["assets"]}
actual = {path.name for path in Path(".").iterdir() if path.is_file()}
if actual != expected:
    raise SystemExit(f"release asset mismatch: missing={expected-actual}, extra={actual-expected}")
PY
sha256sum --check SHA256SUMS
gh attestation verify SHA256SUMS \
  --repo metric-space-ai/greppy \
  --signer-workflow metric-space-ai/greppy/.github/workflows/release.yml \
  --source-ref "refs/tags/$version" \
  --deny-self-hosted-runners
```

On macOS, use `shasum -a 256 -c SHA256SUMS` when GNU `sha256sum` is not
installed. Also verify the selected package itself; this binds its digest to
the same repository, workflow, and tag identity:

```bash
asset=greppy-macos-arm64.tar.gz  # select the package for the current platform
gh attestation verify "$asset" \
  --repo metric-space-ai/greppy \
  --signer-workflow metric-space-ai/greppy/.github/workflows/release.yml \
  --source-ref "refs/tags/$version" \
  --deny-self-hosted-runners
```

The macOS binary must have a valid hardened-runtime signature:

```bash
mkdir unpack && tar -C unpack -xzf greppy-macos-arm64.tar.gz
codesign --verify --strict --verbose=2 unpack/greppy
codesign --display --verbose=4 unpack/greppy 2>&1 | grep -E '^(Authority|TeamIdentifier)='
```

The binary is notarized, but a bare Mach-O executable cannot carry a stapled
ticket, so `xcrun stapler validate` and `spctl --assess --type execute` report
errors on it by construction — that is not a defect. Gatekeeper fetches the
notarization ticket online when the binary first runs.

For Windows, verify both the aggregate checksum and the Authenticode chain and
timestamp before running the binary:

```powershell
$version = 'v0.3.3'
gh release download $version --repo metric-space-ai/greppy
$line = (Select-String 'greppy-windows-x86_64.zip$' SHA256SUMS).Line
$want = ($line -split '\s+')[0]
$got = (Get-FileHash greppy-windows-x86_64.zip -Algorithm SHA256).Hash.ToLowerInvariant()
if ($got -ne $want) { throw 'release checksum mismatch' }
Expand-Archive greppy-windows-x86_64.zip -DestinationPath unpack
$signature = Get-AuthenticodeSignature unpack/greppy.exe
if ($signature.Status -ne 'Valid' -or -not $signature.TimeStamperCertificate) {
    throw "invalid or untimestamped Authenticode signature: $($signature.Status)"
}
$signature.SignerCertificate | Format-List Subject,Thumbprint,NotAfter
gh attestation verify greppy-windows-x86_64.zip `
  --repo metric-space-ai/greppy `
  --signer-workflow metric-space-ai/greppy/.github/workflows/release.yml `
  --source-ref "refs/tags/$version" `
  --deny-self-hosted-runners
```

The GitHub attestation establishes the expected repository and workflow
identity. The Apple and Microsoft checks independently establish platform
trust and timestamp validity; the displayed certificate subject is diagnostic
and is not a substitute for the repository-pinned attestation.

`RELEASE-ASSETS.json` is the machine-readable, exact filename contract.
`SHA256SUMS` covers every listed asset except itself, including the manifest,
SBOMs, build-environment records, benchmarks, and Qwen training evidence. Do
not install a release if any contract, checksum, signature, or attestation
check fails.

## Dependency audit policy

Every Cargo dependency change and the weekly scheduled audit check both the
primary and embedded-Web-Runtime lockfiles against RustSec. Vulnerability
advisories are not allowlisted unless the affected capability is made
unreachable in the shipped binary and that fail-closed boundary has a release
test.

`RUSTSEC-2024-0436` is the sole informational exception. It reports that the
`paste` proc-macro crate is no longer maintained; it does not describe a
vulnerability. Greppy does not depend on `paste` directly. The locked version
is used transitively by `gemm`, `pulp`, `tokenizers`, and
`macro_rules_attribute` while compiling the binary, and is not linked as
runtime code. The exception must be removed as soon as those upstream crates
offer a compatible maintained replacement. Any source, version, or dependency
path change remains visible in `Cargo.lock`, Dependency Review, SBOMs, and the
release provenance checks.

`RUSTSEC-2023-0071` is a temporary, capability-disabled exception for the
embedded Servo 0.5.0 dependency line. Servo pins `rsa 0.10.0-rc.18` and no
fixed upgrade is available. Greppy ships that engine with
`dom_crypto_subtle_enabled=false`, so page scripts cannot reach RSA-OAEP (or
another `SubtleCrypto` operation); `crypto.getRandomValues` and
`crypto.randomUUID` remain available. The engine-preferences regression test
fails if that gate is re-enabled. Remove the exception and restore
`SubtleCrypto` as soon as Servo provides a constant-time RSA backend.

## Reporting a vulnerability

Report vulnerabilities privately through GitHub's **Security > Report a
vulnerability** flow for this repository. Do not open a public issue for a
suspected vulnerability that exposes repository contents, cache paths, local
privilege boundaries, model/backend loading, or daemon transport.

Include the affected version and platform, reproduction steps, expected impact,
and whether the issue requires local access. Maintainers will acknowledge a
complete report within five business days and coordinate disclosure after a fix
is available.

## Security boundaries

- Greppy processes local source code and stores indexed source spans in a local
  SQLite cache. It does not send code or model prompts to a network service.
- Ordinary grep passthrough invokes the real system `grep` and must not open an
  index, load a model, or mutate a Greppy cache.
- Structured commands treat source and graph evidence as authoritative. Qwen
  summaries are untrusted navigation hints and may be omitted on any inference
  or validation failure.
- Model and embedded CUDA artifacts are extracted only into private,
  content-addressed cache paths and are verified before loading. External
  backend-library overrides are not supported by release builds.
- Greppy does not install drivers, toolkits, updates, or other software. Release
  upgrades are explicit and use signed/checksummed artifacts.
- Filesystem-CoW snapshots are local private workspaces, not a security boundary
  between the agent and repository contents. Their Git directory, index, refs,
  and newly written objects are private to the run; base objects are exposed
  read-only through a Git alternate. Only a verified final proposal is imported
  into the main repository.

## Sensitive repositories

Set `GREPPY_STORE_DIR` to an encrypted or ephemeral user-private location when
repository contents require additional at-rest protection. Use `greppy cache
status --json` to audit stored paths and `greppy cache clear --root DIR --yes`
or `greppy cache clear --all --yes` to remove managed data.

Portable agents reuse two different content-identified bases. The graph Base
Store contains source-derived graph rows, indexed spans, summaries, and
embeddings. The workspace RepositoryBase and DirtyBaseline contain the pinned
Git tree plus the captured staged, unstaged, deleted, and untracked state. Both
are published atomically, verified against complete identity manifests, opened
read-only, and excluded from the agent's writable sandbox roots. Ignored files
and build caches are never captured. Every run writes only to its private
workspace namespace, private Git state, graph Delta Store, and recovery journal.
These shared bases are local confidentiality boundaries, not sanitised
artifacts: place `GREPPY_STORE_DIR` and the workspace data root on storage with
protection equivalent to the repository.

Base corruption, incomplete publication, provider failure, and identity
incompatibility are fail-closed. Greppy quarantines invalid graph generations
and rebuilds them under an exclusive lease, but an agent does not automatically
substitute a private graph store after a Base-preparation error. The explicit
`--private-store` option changes graph-index isolation only; it does not bypass
the portable workspace provider or any provider-health gate. Cache reclamation
validates Greppy ownership markers and holds the same lifecycle locks used by
live readers, so it cannot evict an in-use base or traverse unmanaged
directories.

The portable 0.3.4 workspace has no Rift, reflink, native snapshot, or Git-
worktree fallback. Chunk data is stored in append-only segments; SQLite-WAL
manifests bind chunks, metadata, tombstones, redirects, namespaces, and recovery
state. Private Git directories, indexes, refs, locks, and new objects stay
outside the content mount. Snapshot creation double-captures HEAD, index,
status, affected content, and metadata and fails closed if they do not remain
stable. Identity, containment, symlink-escape, type, and baseline checks run
again before proposal publication, apply, garbage collection, and cleanup. A
suspected rewrite, traversal, mount substitution, or incomplete journal blocks
new agents and preserves recoverable evidence; it never selects another
workspace backend.
