# Third-Party Notices

`grepplus-rs` is original Rust source. It depends on third-party Rust crates
under their own licenses; those licenses are recorded in each crate's
`Cargo.toml` and resolved by `cargo metadata`. This file documents
non-crate obligations: notice preservation for vendored concepts and
projects consulted while porting from `DeusData/codebase-memory-mcp`.

## Upstream reference

This project is a Rust port of portions of
[`DeusData/codebase-memory-mcp`](https://github.com/DeusData/codebase-memory-mcp),
pinned to release [`v0.8.1`](https://github.com/DeusData/codebase-memory-mcp/releases/tag/v0.8.1)
(commit `f0c9be19c5d74b84f418d807bfdce7b5d6a261ff`). See `PORT_LEDGER.md`
for the source freeze table and license implications.

The original work is:

> Copyright (c) 2025 DeusData
> Licensed under the MIT License.

A copy of the upstream MIT license text is preserved at
`.vendor/codebase-memory-mcp.git/LICENSE` for reference. We do not translate
the upstream C source verbatim; we read it for design reference only.

## Tree-sitter

The Rust port uses the [`tree-sitter`](https://crates.io/crates/tree-sitter)
crate and per-language `tree-sitter-<lang>` crates. The tree-sitter project
is MIT-licensed (Copyright (c) 2018 Max Brunsfeld). Per-language grammar
crates retain their original licenses (mostly MIT; the `clojure` grammar is
CC0-1.0).

## Vendored library replacements

The original codebase vendors several C libraries under `vendored/` and
`internal/cbm/vendored/`. The Rust port replaces them with crates.io
equivalents. The table below mirrors `PORT_LEDGER.md` "License Implications".

| Upstream library | Upstream license | Rust crate replacement | Replacement license |
|------------------|------------------|------------------------|--------------------|
| SQLite 3 | Public Domain | `rusqlite` (with bundled `sqlite3-src` via `libsqlite3-sys`) | Public Domain / MIT |
| mimalloc | MIT | `mimalloc` | MIT |
| yyjson | MIT | `serde_json` (Apache-2.0 + MIT dual) or `yyjson-rs` (MIT) | Apache-2.0 + MIT |
| xxHash | BSD-2-Clause | `xxhash-rust` / `twox-hash` | BSD-2-Clause / MIT |
| TRE (regex) | BSD-2-Clause | `regex` (Rust standard) | MIT / Apache-2.0 |
| LZ4 | BSD-2-Clause | `lz4_flex` | MIT |
| Zstandard | BSD-3-Clause (dual BSD/GPLv2 — BSD selected upstream) | `zstd` | MIT / BSD-2 / Zlib |
| simplecpp | 0BSD | (not needed in initial scope) | n/a |
| Verstable | MIT | `std::collections::HashMap` / `hashbrown` | MIT / Apache-2.0 |
| wyhash | Unlicense (public domain) | `wyhash` / `ahash` | Unlicense / MIT / Apache-2.0 |

## Intentionally not shipped

Per `PORT_LEDGER.md` decisions DD-1, DD-3, DD-4:

- **Hybrid LSP** (`internal/cbm/lsp/`, `internal/cbm/lsp_all.c`,
  `internal/cbm/lsp/generated/*.c`) — the original Rust port does not
  include LSP-based type resolution. Re-evaluation trigger: end of Phase 4.
- **nomic-embed-code vectors** (`vendored/nomic/`) — Apache-2.0; not
  shipped. If reintroduced in a later phase, ship under Apache-2.0 with
  the upstream NOTICE preserved in this file.
- **Python typeshed stdlib data**
  (`internal/cbm/lsp/generated/python_stdlib_data.c`) — Apache-2.0; not
  shipped while Hybrid LSP is dropped.

## LSP reference behaviour only

The original Hybrid LSP layer is original C source code "structurally
inspired by" (no source copied from) the following language servers and
language specifications. The Rust port's name resolution likewise does
not copy any of their source code. Listed for acknowledgment:

- `microsoft/TypeScript` (Apache-2.0)
- `microsoft/pyright` (MIT)
- `golang/tools` (BSD-3-Clause)
- PHP language reference + Composer PSR-4 specification
- `dotnet/roslyn` (MIT)
- `llvm/llvm-project` (Apache-2.0 WITH LLVM-exception)
- `eclipse-jdtls/eclipse.jdt.ls` (EPL-2.0) — reference only
- `fwcd/kotlin-language-server` (MIT)
- `rust-lang/rust-analyzer` (MIT OR Apache-2.0)
