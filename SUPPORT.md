# Support Policy

## Release targets

The `v0.4.1` release gate covers:

| Platform | Inference backend | Daemon transport | Web tool | Portable CoW |
|---|---|---|---|---|
| macOS Apple Silicon | Metal | Unix-domain socket | yes (beta) | optional FSKit |
| Linux x86_64 | NVIDIA CUDA | Unix-domain socket | yes (beta) | optional FUSE3 |
| Windows x86_64 (debug source build) | CPU (debug only) | named pipe with user ACL | not shipped | private WinFsp fork |

Windows CUDA, macOS Intel, and Linux ARM64 are not release-gated for `v0.4.1`.
Production inference requires Metal on macOS Apple Silicon or CUDA on Linux
x86_64. Automatic selection refuses inference when a suitable GPU is unavailable;
it never silently switches to CPU. CPU execution is reserved for debug and test
usage. An explicitly selected unavailable backend fails with its diagnostic.
FSKit is optional acceleration, not a prerequisite for ordinary Greppy use.

The Windows package is released only after Microsoft Hardware Dev Center has
returned the exact HLK/dashboard-signed WinFsp driver and catalog bound by the
release contract. A source build or test-signed driver is not release evidence.

## Language support

Greppy bundles tree-sitter parsers for **more than 60 languages**. Every one of
them indexes symbols and answers definition and text search, and most of them —
every procedural language, from the mainstream set through Ruby, C++, C#,
Kotlin, Swift, Elixir, Scala, and dozens more — also extract call, usage, and
import graph relations, so `who-calls`, `callees`, `path`, and `impact`
work out of the box (e.g. `greppy who-calls` resolves callers in an Elixir file
with no extra setup).

Eleven languages — **Rust, Python, Java, JavaScript, TypeScript, Go, C++,
C#, Kotlin, Swift, and Ruby** — are additionally **acceptance-certified for
graph completeness**: per-language certification grids (12 cells over
cross-file CALLS/USAGE/TYPE_REF/IMPORTS plus reindex stability and freshness
behavior, `crates/cli/tests/graph_grid_*.rs`), language fixtures, and
real-repository tests guarantee their caller/callee/usage/impact relations
are correct and complete. Every other language extracts the same relations
without that formal completeness guarantee — treat its graph as strong
evidence, still verified against source. Purely declarative formats (JSON, YAML,
TOML, Markdown, …) provide symbols and text search but no call graph, by nature.

Static analysis can miss reflection, runtime dependency injection, generated
code, macro expansion, monkeypatching, and dynamic dispatch. Greppy fails closed
when indexed source evidence is stale; verify proposed changes with the
language toolchain and test suite.

## Getting help

Open a GitHub issue with:

- `greppy --version` and the exact release checksum;
- operating system, CPU, GPU, and driver version;
- `greppy doctor --json` with private paths redacted;
- the command, exit code, and minimal reproducible repository when possible.

Use GitHub's private vulnerability-reporting flow for security issues. See
[`SECURITY.md`](SECURITY.md).

## Operational defaults

Embedding and summary models remain resident for 300 idle seconds. Their daemon
processes exit after 1800 idle seconds. Workspace cache entries use a 14-day
default TTL plus an independent size quota. These values can be inspected with
`greppy cache status --json`.
