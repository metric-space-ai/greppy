# Support Policy

## Release targets

The `v0.2.x` release gate covers:

| Platform | Inference backend | Daemon transport |
|---|---|---|
| macOS Apple Silicon | CPU and Metal | Unix-domain socket |
| Linux x86_64 | CPU and NVIDIA CUDA | Unix-domain socket |
| Windows x86_64 | CPU | named pipe with user ACL |

Windows CUDA, macOS Intel, and Linux ARM64 are not release-gated for `v0.2.x`.
An unsupported accelerator must not prevent CPU operation in automatic mode.
An explicitly selected unavailable backend fails with a diagnostic instead of
silently switching devices.

## Language support

Greppy bundles tree-sitter parsers for **more than 60 languages**. Every one of
them indexes symbols and answers definition and text search, and most of them —
every procedural language, from the mainstream set through Ruby, C++, C#,
Kotlin, Swift, Elixir, Scala, and dozens more — also extract call, usage, and
import graph relations, so `who-calls`, `callees`, `find-usages`, and `impact`
work out of the box (e.g. `greppy who-calls` resolves callers in an Elixir file
with no extra setup).

Six languages — **Rust, Python, Java, JavaScript, TypeScript, and Go** — are
additionally **acceptance-certified for graph completeness**: language fixtures
and real-repository tests guarantee their caller/callee/usage/impact relations
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
