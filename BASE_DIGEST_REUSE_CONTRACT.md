# Cross-command Base digest reuse: contract decision

Assessment against revision `b0929552097b7ddfe134ab8fe598199074aeacbd`, with
HFS classification safeguards from PR #222,
`2ee1f87cfd3254dbb4c912857a2083aac54881bf`.

## Existing guarantee

`BaseStoreLayout::read_verified_manifest` checks COMPLETE, ownership and
repository identity, then compares SHA-256 digests of graph.db and
summary_cache.db with the exact manifest. `file_sha256` explicitly states:

> Every process performs its first full digest.

Its process-local memoization is a narrower optimization. An earlier command's
verification does not currently substitute for a new process's first read.
There is no repository Markdown specification authorizing persistent reuse.

## Why a trusted proof alone cannot preserve that guarantee

Consider a fully verified file F, digest D, and proof containing its manifest
binding, device, inode, size, modification/change timestamps and verification
time. A later command opens the same file and obtains identical metadata.
There are two indistinguishable states for a metadata-only lookup:

| State | Opened-file metadata | Actual bytes | Required full digest |
| --- | --- | --- | --- |
| Unchanged file | Identical | F | D |
| Silent media corruption | Identical | F with a changed byte | Different from D |

A private, owner-checked, nofollow proof file prevents a different user from
forging the proof; it cannot distinguish those states. Sampling likewise
cannot detect a changed byte outside the sample. A finite expiry bounds the
detection delay but does not preserve first-command full-content verification.
Silent corruption is an information limit, not an observed failure on this Mac.

On 2026-10-03, `diskutil info /Volumes/tmp` identified Journaled HFS+, Owners
Disabled, protocol Secure Digital and SMART Status Not Supported. Its proof
sidecars must not establish trust. Known HFS classification and aged start/end
checks in PR #222 address ordinary coarse-timestamp races, not the above
unchanged-metadata corruption case.

## Safe release choices

Keep first-process full hashing under the existing contract. A long-lived
reader can reuse the existing process-local verification without claiming a
new persistent integrity guarantee.

If bounded detection is explicitly accepted as a changed contract, implement
persistent reuse only as a separately specified policy, with these obligations:

- Proof storage in a small production namespace on an ownership-enforcing
  filesystem; validate every ancestor, current UID, permissions, regular-file
  type and nofollow opens. Never use an owners-disabled tmp sidecar as proof.
- Bind the exact validated manifest, digest algorithm/version, opened-file
  device/inode/size/change time and modification time, and generation identity.
  Include creation/generation metadata where available to avoid inode reuse.
- Require known filesystem semantics, aged metadata at both full-hash start and
  end, and identical opened-file identity before/after every reuse. Unknown,
  fresh, malformed, future-time or expired records fall back to a full hash.
- Use a strict maximum verification age and maximum number of reuses. Reuse
  must never refresh the original full-verification time. Record replacement
  must be atomic and fail closed on unsafe ownership or concurrent publication.
- Test actual cross-process hits, tampered and symlinked proof paths, unrelated
  manifests, expired/future records, replacement, restored mtime, concurrent
  writes and untrusted/unknown/coarse metadata. Explain that metadata-only
  media corruption is detected at revalidation, not at every command.

No runtime policy change is included in this assessment. The bounded policy
needs an explicit contract decision; implementing it under the current promise
would silently weaken verification.
