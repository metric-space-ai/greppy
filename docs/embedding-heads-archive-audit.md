# Historical build-log archive audit

The Cargo and Go June archives under GPU3's `corpus-extra/raw` contain 424
members, but 337 are JSON retrieval errors. They must not be counted as complete
build outputs. Their acquisition failures can only become separately admitted
network-diagnostic examples with correct provenance and task context.

| Archive | Members | Retrieval errors | Text captures awaiting review | Maximum lines |
|---|---:|---:|---:|---:|
| pkgforge-cargo/06.log.7z | 203 | 156 | 47 | 21,004 |
| pkgforge-go/06.log.7z | 221 | 181 | 40 | 56,080 |

Both compressed archives passed decompression/CRC validation. Every extracted
member matched its archive's uncompressed byte count and CRC, and every member
was valid UTF-8. The audit also recorded individual SHA256 hashes and line counts.
No member reached 100,000 lines. This establishes archive integrity, not public
origin, complete build capture, privacy admission, source independence or fresh
final-test eligibility. No output was admitted automatically.

The observed retrieval-error example was a complete GitHub API JSON response
with status 403 and a rate-limit message. The archived payload itself contained
the error; extraction did not cause it.

Archive SHA256 values:

- Cargo: `c9f773bf13ac1517913e7fe1561e1526528b378fb6b42bf86a164504d98c6311`
- Go: `11ce14389d12dd4148587066fac4962b48fe5a480551591bd45b35d84036261a`

`tools/embedding_heads/audit_log_archive.py` records this distinction without
emitting raw source text. It rejects missing, duplicate, symlinked or escaping
members and size/CRC mismatches. Small complete JSON transport failures, empty
captures and HTML responses are kept separate from candidate text logs. Its
classification never treats an incomplete prefix as a complete JSON response.

The audit used py7zr 1.1.3 on GPU3. Both reports, including per-member identities
and hashes, are retained under
`/Users/michaelwelsch/.local/state/greppy-heads/2026-09-05/` as
`pkgforge-cargo-archive-audit-v1.json` and `pkgforge-go-archive-audit-v1.json`.

The full Python suite now has 61 passing tests. Source acquisition and admission
must be completed before these archives can contribute to the planned pilot or
broad training stage.

## Independent archive-origin verification

Both archives were subsequently downloaded anonymously through Hugging Face Hub
0.35.3 using its HTTP transport. Their SHA256 hashes matched the previously
inventoried local archives exactly. The downloads resolve to immutable revisions:

- Cargo: `4ad00ebc1ec578362ddc34c082eb4bc708a911de`
- Go: `a07fdc87ecd3e86cbed28a4f8a3cbe9846bce60e`

`tools/embedding_heads/verify_archive_origin.py` records revision, anonymous
retrieval, transport, downloaded size and hash, cache location and immutable URL.
It refuses to overwrite an existing receipt. The initial Xet transport attempt
failed with a local file-exists cache error; HTTP retrieval then passed. No shared
cache was deleted. Browser navigation to the public API was not used as evidence
because it timed out.

The matching origin receipts are `pkgforge-cargo-origin-v1.json` and
`pkgforge-go-origin-v1.json` under the same durable evidence directory. They prove
origin of these archive bytes. They do not grant privacy admission, prove complete
build execution, establish independent source lineages or make old data a final test.
