# Security Policy

## Supported versions

Security fixes are provided for the newest published Greppy release. The
unreleased `main` branch is tested continuously but is not a supported release
channel. Pin production installations to an immutable tag and verify the
published checksum and build provenance.

## Release scope

A release is cut only when, on the exact release commit, all of the following
pass: CI, CodeQL, the Rust dependency security audit, the task-bank
reproducibility audit, the navigation-regime agent benchmark, and the
summary-quality gate — plus code signing and notarization. Greppy's claim is
structural code navigation, and the navigation benchmark is a hard, per-commit
release gate that must be green on the released binary. The remaining
benchmarks run continuously and their verdicts are published with each commit.

## Verifying a release

The expected provenance identity is the repository
`metric-space-ai/greppy`, the signer workflow
`.github/workflows/release.yml`, and the release tag being installed. Download
all assets into an empty directory so the exact release manifest can reject
missing or unexpected files:

```bash
version=v0.2.1
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
$version = 'v0.2.1'
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

Every Cargo dependency change and the weekly scheduled audit are checked
against RustSec. Vulnerability advisories are never allowlisted.

`RUSTSEC-2024-0436` is the sole informational exception. It reports that the
`paste` proc-macro crate is no longer maintained; it does not describe a
vulnerability. Greppy does not depend on `paste` directly. The locked version
is used transitively by `gemm`, `pulp`, `tokenizers`, and
`macro_rules_attribute` while compiling the binary, and is not linked as
runtime code. The exception must be removed as soon as those upstream crates
offer a compatible maintained replacement. Any source, version, or dependency
path change remains visible in `Cargo.lock`, Dependency Review, SBOMs, and the
release provenance checks.

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

## Sensitive repositories

Set `GREPPY_STORE_DIR` to an encrypted or ephemeral user-private location when
repository contents require additional at-rest protection. Use `greppy cache
status --json` to audit stored paths and `greppy cache clear --root DIR --yes`
or `greppy cache clear --all --yes` to remove managed data.
