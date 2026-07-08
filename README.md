<img src="assets/logo.svg" align="right" width="160" alt="greppy logo"/>

# greppy

**Standard `grep`, plus a few commands your coding agent can use to navigate code — `who-calls`, `impact`, `semantic-search`, `brief`. Agents finish code-navigation tasks ~2× faster, ~3–4× cheaper, and more accurately. One native Rust binary.**

[![Release](https://img.shields.io/github/v/release/metric-space-ai/greppy?display_name=tag&sort=semver&color=22c55e&label=release)](https://github.com/metric-space-ai/greppy/releases/latest)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`greppy` is a drop-in `grep` — every flag works exactly as before — that *also*
answers the questions an agent normally burns rounds on: *who calls this
function, what breaks if I change it, where is the code that does X.* One line in
your agent's config (below) tells it the extra commands exist, and it stops
looping through text matches.

```bash
# Standard grep — every command works, unchanged:
greppy -rn "TODO" src/
greppy -i "connection refused" server.log

# A few extra commands, on the same binary:
greppy who-calls parse_config                  # who calls this function
greppy impact User --direction incoming        # what breaks if I change User
greppy semantic-search "restrict a value to a range"   # find code by meaning
greppy brief _split_blueprint_path             # definition + callers + callees
```

<img src="docs/assets/greppy-demo.gif" width="100%" alt="Split screen: the same coding agent answers one who-calls question, left with plain grep, right with greppy."/>

<sub>The **same** coding agent (MiniMax-M3, driven by [Pi Code](https://pi.dev)) answers one *who-calls* question on a real repo — **left with plain `grep`, right with `greppy`**. greppy resolves the callers in a single `greppy who-calls` call instead of a grep-and-read spiral: **2.3× faster, 14 → 5 tool calls, ~9× fewer input tokens**. Counters are live from the recorded run.</sub>

---

## Setup — two steps

**1. Install the binary.**

```bash
cargo build --release --bin greppy --features embedded-model
sudo install -m 0755 target/release/greppy /usr/local/bin/greppy
```

Everything is automatic — the code graph and the semantic model are **built into
the binary** and build themselves on first use. Nothing to index, nothing to
download, no flags to configure. (Prebuilt binaries for macOS / Linux / Windows
are on the [Releases](../../releases) page.) Want it as a transparent `grep`
drop-in too? Install it a second time as `grep`.

**2. Tell your agent the extra commands exist.** Delegate it — in your agent's
chat, say **`install https://github.com/metric-space-ai/greppy/`** — or
paste the snippet below into the file your agent reads for project instructions
(`CLAUDE.md`, `AGENTS.md`, `.cursor/rules`, `.windsurfrules`, or the system
prompt).

```text
This project has `greppy` — standard grep plus a few code-navigation commands
over a prebuilt symbol graph and an on-device semantic index. Every normal grep
invocation (and flag) works exactly as usual.

CODE-NAVIGATION COMMANDS. SYMBOL is a function / method / class / type name.
They return resolved results as `qualified_name file:line`, not text matches:
  greppy who-calls SYMBOL        the callers of SYMBOL (incoming calls)
  greppy callees SYMBOL          the functions SYMBOL calls (outgoing calls)
  greppy find-usages SYMBOL      every reference to SYMBOL (calls, uses, imports)
  greppy brief SYMBOL            SYMBOL's definition plus its callers and callees, in one call
  greppy impact SYMBOL           the transitive set of code a change to SYMBOL reaches
  greppy search-symbols NAME     definitions whose name matches NAME (a name or fragment)
  greppy path --from A --to B    a call chain from symbol A to symbol B, if one exists

SEMANTIC SEARCH — use when you do NOT know the symbol's name:
  greppy semantic-search "PLAIN-ENGLISH DESCRIPTION"
      Describe the behaviour or code you are looking for in plain English
      (e.g. "restrict a value to a range", "retry a failed HTTP request").
      Returns the signature and file:line of the closest-matching definitions
      to OPEN AND READ — a locator, not a written answer.

FLAGS (append to any command above):
  --code            include each result's source lines (so no separate read is needed)
  --all             return every result (turn off the default truncation)
  --json            machine-readable output with exact counts
  --root DIR        run against a repo other than the current directory
  --kind KIND       (search-symbols) restrict to function|method|class|struct|enum|trait
  --direction incoming|outgoing, --depth N   (impact) which way and how far to walk
  --from A --to B   (path) the two endpoint symbols

Prefer these over grepping a symbol name and reading every hit: who-calls /
callees / impact answer relationship questions directly, and semantic-search
finds code you cannot name.
```

---

## What it saves

What an agent actually pays for is **billed tokens** and **wall-clock time.**

<img src="docs/assets/nav-wins.svg" width="100%" alt="Per-task: on navigation questions the agent answers better and cheaper with greppy"/>

The benchmark: a real coding agent (MiniMax-M3, driven by [Pi Code](https://pi.dev)) answers **94 code questions** — *who-calls*, impact/blast-radius, call-chain traces, and vocabulary-gap "find the code that does X" — across **four real repositories** (Rust `serde`, Python `flask`, Java `gson`, TypeScript `zod`). The same agent runs each task twice: once with plain `grep`, once with `greppy`. The fixed system prompt is warmup and is excluded from every ratio. The harness is in [`bench/agent_efficiency/`](bench/agent_efficiency/). Medians:

| What you actually pay | Median | |
|---|---:|---|
| **Search-context tokens** | **~7.8×** | cheaper |
| **Loop prompt tokens** | **~2.3–2.9×** | cheaper |
| **Output tokens** | **~2.0×** | fewer |
| **Wall-clock time** | **~2.0×** | faster |
| **Tool-call rounds** | **~2.9×** | fewer |

And it is not a speed-for-accuracy trade: on the same graded 94 questions the `greppy` agent answered **85 correctly (1 wrong)** vs plain grep's **77 (2 wrong)** — cheaper *and* more correct.

**The gain depends on the model.** The extra commands only help a model that routes to them well — across a 14-model sweep the per-model efficiency gain varied widely and did **not** track a model's general agentic-benchmark score. Benchmark your own model; the median model still comes out ahead.

<img src="docs/assets/gain-vs-agentic.svg" width="100%" alt="Real dollar saving per model vs Artificial Analysis Agentic Index — correlation near zero"/>

_Preliminary (being re-measured on the diverse multi-language set)._

The win comes from one thing: **fewer model round-trips.** It is largest on structural questions (`who-calls`/`impact`) and vocabulary-gap searches (`semantic-search`), and ~1× on a plain literal search, where `grep` is already the right tool.

---

## How it works

- **Standard grep.** Any invocation that isn't one of the extra commands runs real `grep` and returns its output and exit code unchanged.
- **A precomputed code graph.** An indexed, typed symbol graph (`CALLS`/`USES`/`TYPE_REF`/`IMPORTS`) answers `who-calls`/`callees`/`find-usages`/`impact`/`path` directly — resolved relationships with `file:line`, not text matches — collapsing several grep+read rounds into one call.
- **Native semantic search.** For a natural-language query that shares no words with the code, `semantic-search` embeds the query with Google's **EmbeddingGemma** on greppy's own native Rust inference (CPU / Apple Metal / NVIDIA CUDA, auto-detected — no llama.cpp, no Python, no HTTP) and returns the nearest code spans by meaning. A small warm daemon keeps the model resident between calls and drops it after idle, so it never holds GPU memory while you're not searching.
- **One native Rust binary.** The EmbeddingGemma model is baked into the binary; tree-sitter parsers and SQLite are compiled in statically.

---

## Status

Early and evolving — the drop-in `grep` core is solid; the intelligence layers around it are beta.

- **Solid:** the `grep` drop-in and the code-graph commands (`who-calls` / `callees` / `find-usages` / `impact` / `path` / `brief`) on supported languages.
- **Supported languages (107):** `python`, `csharp`, `go`, `cpp`, `php`, `rust`, `swift`, `scala`, `c`, `java`, `javascript`, `typescript`, `ruby`, `bash`, `kotlin`, `fsharp`, `julia`, `ocaml`, `d`, `gdscript`, `zig`, `elm`, `erlang`, `crystal`, `gleam`, `objc`, `solidity`, `prisma`, `protobuf`, `css`, `dockerfile`, `json`, `groovy`, `lua`, `sql`, `make`, `nix`, `cmake`, `dart`, `fortran`, `elixir`, `scheme`, `vue`, `astro`, `svelte`, `verilog`, `glsl`, `hcl`, `matlab`, `r`, `purescript`, `racket`, `clojure`, `haskell`, `cuda`, `tcl`, `graphql`, `pascal`, `powershell`, `html`, `yaml`, `hlsl`, `cobol`, `fish`, `ini`, `vhdl`, `json5`, `awk`, `cairo`, `ada`, `hare`, `kdl`, `jsonnet`, `llvm`, `janet`, `jinja2`, `bicep`, `gotemplate`, `just`, `devicetree`, `liquid`, `assembly`, `hyprlang`, `gn`, `blade`, `cfml`, `cfscript`, `csv`, `bibtex`, `beancount`, `gitattributes`, `markdown`, `toml`, `xml`, `scss`, `perl`, `fennel`, `starlark`, `ron`, `dotenv`, `properties`, `po`, `diff`, `rst`, `mermaid`, `regex`, `linkerscript`. More land in each release.
- **Beta:** `semantic-search` — the on-device EmbeddingGemma inference is solid and the model ships inside the binary. Newer than the graph commands, so still labelled beta.

Not yet production-ready — use it as a fast code-navigation aid, not a system of record.

## License

MIT — see [LICENSE](LICENSE). Third-party notices: [THIRD_PARTY.md](THIRD_PARTY.md).
