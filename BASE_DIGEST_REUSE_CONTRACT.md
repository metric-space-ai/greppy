# Bounded verification snapshots for published Bases

Published immutable Bases reuse a verified digest across CLI processes for less
than 30 seconds after the original full read began. This policy is confined to
`BaseStoreLayout::read_verified_manifest`; ordinary `file_sha256` callers retain
their existing process-local behavior. A snapshot hit does **not** cryptographically
revalidate unread bytes.

## Binding and invalidation

Each record binds SHA-256 of the exact manifest bytes, its expected file digest,
and the actual opened regular file's device, inode, length, modification/change
timestamps and filesystem classification. Both file and proof opens reject
symlinks. Identity is checked again on the same file descriptor after proof
lookup or a full read. Detectable mutation/replacement invalidates reuse;
a digest mismatch never creates a proof. A miss always fully hashes the opened
file, independently of the process-local digest cache.

Unknown whole-second metadata falls back to full hashing. Fractional metadata
must already be two seconds old at full-hash start and remain eligible at its
end. An opened-file `fstatfs` HFS classification permits known coarse HFS change
times only after four seconds. A long hash cannot promote an initially fresh
identity. Reuse never extends the original full-verification time.

Expiry checks both wall time and system monotonic time. Future/unknown times,
malformed records and expired records miss. Proof timestamps start before the
full read, so its duration consumes the reuse window. No environment override
can lengthen the window or disable integrity checks.

## Proof storage and concurrent publication

Small operational records use the current user's production data namespace:
`~/Library/Application Support/greppy/verified-base-digests-v1` on macOS and
`~/.local/share/greppy/verified-base-digests-v1` elsewhere on Unix. Missing
components are created with mode 0700. Directory traversal uses `openat` and
`O_NOFOLLOW`, validates ownership and denies group/other-write ancestors; the
final directory must be private and owned by the current UID. On macOS, native
extended ACLs are queried on the actual directory/file descriptors, including
root and all traversed ancestors, the cache directory, records, locks and staging
files. Every mutation ALLOW grant is rejected conservatively (including owner
and inherited grants); read/search ALLOWs and DENY-only ACLs remain acceptable.
Unknown tags, permission bits and ACL query failures fail closed to a full hash.
The SDK-documented EINVAL end marker is accepted only on a validated independent
ACL copy using fixed FIRST/NEXT selectors. macOS volumes
with `MNT_IGNORE_OWNERS` are rejected, so owners-disabled `/Volumes/tmp`
sidecars never establish proof. Unavailable/unsafe storage simply misses.

Records must be private, singly linked regular files owned by the current UID,
and at most 4096 bytes. Reads compare descriptor metadata before/after reading.
There are 64 fixed slots; hash collisions discard an optimization and cannot
bypass exact record binding. Each slot has a fixed private lock and staging
file. Independent descriptor `flock` locks serialize writers across processes
and threads; nonblocking lock contention misses. Publication uses synced
complete bytes and atomic `renameat`. OS exit releases the lock; the next
writer overwrites abandoned staging bytes. Published records, locks and crash
leftovers are bounded to at most 192 small files, with no unbounded cache log.

## Integrity limits

A trusted record authenticates a recently verified snapshot under normal
filesystem metadata semantics. It cannot detect silent media corruption that
changes bytes without changing metadata. Such corruption is detected at the
next full verification, once the original 30-second window expires. Privileged
attackers, compromised processes running as the same UID and filesystems that
lie about change metadata are outside the snapshot trust boundary.

On 2026-10-03, `diskutil info /Volumes/tmp` identified Journaled HFS+, Owners
Disabled, Secure Digital, and SMART Status Not Supported. This policy neither
trusts that volume's proof sidecars nor claims to eliminate its media risk.
Existing manifest/COMPLETE/owner/repository checks remain required on every
command. This optimization does not close the pre-existing later-store-open
path race after manifest verification; consumers must preserve their existing
immutable Base lifecycle guarantees.

## Verification package

Tests cover an actual aged Base cache hit with a full-read counter, restored-mtime
mutation, cross-process proof loading, expiry/future clocks, exact manifest and
file identity, fresh and unknown coarse metadata, digest mismatches, unsafe
proof permissions, symlinks/hardlinks, malformed/oversized proofs and concurrent
atomic publication. HFS classification and start/end age tests from PR #222 are
included. Native Mac ACL negatives cover writable ancestors, cache directories
and proof files whose POSIX modes remain 0700/0600; a deny-only/read-search
positive exercises normal production namespace traversal. A separate command
reopens that namespace and proves a cache hit with zero full digest reads.
An invalid-descriptor ACL query proves failures are not empty ACLs.
The worker ran rustfmt only; the release owner owns compilation,
local tests and operational performance acceptance under the shared resource
lease. No test result or release acceptance is claimed here.
