# Store-CoW build staging ownership

This is an internal lifecycle contract, not an alternative CoW backend.

The `greppy-base-build-*` and `greppy-linked-base-checkout-*` prefixes identify
potential staging, not abandoned work. A directory's modification time does
not change when a descendant graph file is written. Age alone must therefore
never authorize recursive deletion.

## Ownership

- A creator acquires a shared `locks/base-build-staging.lease` immediately
  after creating its unique staging directory and before starting work.
- The Base index child receives the staging and temporary-checkout lease roots
  through `GREPPY_BASE_BUILD_STAGING_LEASES` (the platform's joined-path format).
- The child opens existing shared leases before cache maintenance or indexing.
  It never creates a missing lease. Parent and child retain their own guards;
  the child's protection does not depend on its parent remaining alive.
- Reclamation requires an existing regular lease file and a nonblocking
  exclusive lock. A held lease, missing ownership information, symlink or
  malformed metadata leaves the directory untouched.
- The six-hour threshold is an additional retention policy, not liveness proof.
  Legacy staging without this ownership record stays unmanaged until separately
  audited. It must not be silently classified as abandoned during upgrade.

## Reporting and verification

Explicit GC includes reclaimed staging paths and bytes in its normal report.
Dry-run reports eligible candidates without removing them. Locked staging is
reported as skipped, not as removed. Acquisition and deletion remain inside the
exclusive-lease lifetime.

Required regressions include: writes below an old root, active and unidentified
staging, a second process retaining a shared lease after the creator releases
its own, cleanup after the last holder exits, dry-run/actual reporting parity,
and fail-closed handling of invalid or absent lease paths.

The original TTL-only defect was reproduced against an isolated synthetic
writer, not against a user's running build. On macOS the compiled Core cache
suite passed 15 tests; its one ignored helper is explicitly executed by the
subprocess regression. Source hashes were unchanged throughout that run
(`core-staging-test.8V2k1z/result.json`, 2026-09-06). The CLI child handoff and
cross-platform gates remain pending; this document is not release acceptance
evidence.
