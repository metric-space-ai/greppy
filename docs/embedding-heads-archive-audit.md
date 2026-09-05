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
