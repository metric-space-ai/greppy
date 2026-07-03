# grepplus

**A drop-in replacement for `grep` that also answers code questions — so LLM agents stop looping.**
**~2× faster and cheaper agentic code discovery, in one native Rust binary.**

`grepplus` **is** `grep`. Run it with any grep arguments and you get grep — same flags, same output, same exit code. That is the default. It just *also* understands your codebase.

```bash
# Default behaviour: it's grep. Every grep call works, unchanged.
grepplus -rn "TODO" src/
grepplus -i "connection refused" server.log
alias grep=grepplus          # install it as grep — existing tools & scripts don't notice

# The "plus": extra commands on the same binary — for an agent, or for you.
grepplus who-calls parse_config                 # who calls this function
grepplus impact User --direction incoming       # what breaks if I change User
grepplus context "restrict a value to a range"  # find code by meaning, not keyword
grepplus brief _split_blueprint_path            # definition + callers + callees, one call
```

An LLM coding agent armed only with `grep` finds things by *looping* — grep → read → grep → read, guessing vocabulary and stitching call graphs together by hand. Each loop is another model round-trip, and billed tokens grow super-linearly with the number of rounds. grepplus answers those same questions from a **precomputed code graph** and **native semantic search**, so the agent reaches the answer in far fewer rounds.

![A grep agent vs a grepplus agent on one real benchmark task](assets/grep-vs-grepplus.svg)

> **Status:** early and in active development — an in-progress Rust port of a larger C code-intelligence engine (~40% of the upstream engine, ~18 languages wired for cross-file `CALLS`/`IMPORTS`; Rust additionally resolves `TYPE_REF`/`USES`). Not production-ready. The numbers below are real, reproducible medians from an agent benchmark (MiniMax-M3, 94 tasks, 4 real repositories) — not marketing figures. See [Status & scope](#status--scope).

---

## What it is

- **One binary, native Rust inference.** The embedding engine is [candle](https://github.com/huggingface/candle) (pure Rust) — no llama.cpp, no Python, no HTTP, no external inference runtime. At runtime the binary links only system libraries (`otool -L` → `libSystem`, `libiconv`). Parsers (tree-sitter) and the store (SQLite) are standard C libraries **compiled in statically** — nothing to install.
- **Grep-compatible by default.** Any invocation that isn't a grepplus subcommand is forwarded to the real `grep`, returning its stdout/stderr and exit code **verbatim** — even for non-UTF-8 patterns. Scripts and tools that call `grep` don't notice the swap. The extra commands live on the same binary; you opt into them by name.
- **The "plus":** a cross-file code graph, token-lean structural answers, and local semantic search that finds code by *meaning* when the query and the code share no words.

---

## How it works — the "plus"

grepplus removes agent round-trips with a handful of composable techniques. Every claim below is backed by the code, not aspiration.

**1. Drop-in `grep` passthrough (byte-exact).** Any invocation whose first token isn't a known subcommand is forwarded to the real `grep` verbatim; grepplus returns grep's exit code and, on a *miss*, returns immediately with grep's output untouched. Zero adoption cost, zero risk — the agent never spends a round learning a new tool for the common literal-search path.

**2. A precomputed code graph for structural questions.** An indexed, typed symbol graph (`CALLS`, `USES`, `TYPE_REF`, `IMPORTS`) answers `who-calls` / `callees` / `find-usages` / `impact` / `path` / `trace` / `fan-in` / `fan-out` directly. `grep` finds *textual name occurrences*; the graph returns *resolved call relationships* (with `file:line`) in a single call, collapsing several read+grep rounds — and it doesn't confuse definitions, comments, and same-named symbols.

**3. Token-lean locators with exact-count stop signals.** Structural results are compact `qualified_name file:line` pointer rows, capped to a sane default, followed by an honest completeness footer stating the true total (`— 3 callers`, or `N+` when the count is a genuine floor, or `… and N more` with an `--all` escape hatch). The exact count is a definitive *stop signal*, so the agent doesn't loop with `--all`/grep to re-confirm what it already has.

**4. Native semantic vector search for the vocabulary gap.** When a natural-language query shares no literal words with the target (`"restrict a value to a range"` won't match a function named `clamp`), grepplus embeds the query with Google's **EmbeddingGemma** — pure-Rust candle GGUF inference, mean-pooled and L2-normalized — and does exact cosine nearest-neighbour search over code-span embeddings computed at index time. Bare identifiers never touch the model; only genuine natural-language queries do.

**5. One-shot aggregate briefings.** `brief SYM` returns a symbol's definition (with source), its direct callers, and its direct callees in **one** call; `impact SYM` returns the full *transitive* blast-radius set with hop distances in one call (optionally scoped to a git diff). A grep-only agent would run `context` + `who-calls` + `callees` separately and iterate `who-calls` to build the transitive set — grepplus folds that into a single round.

**6. Freshness-gated incremental index.** An on-demand freshness check (a workspace fingerprint compared on every query) decides whether cached answers are trustworthy; when stale it refuses or clearly labels the answer rather than handing the agent a confidently-wrong stale result. Re-indexing reparses only changed files and produces a graph byte-identical to a full reindex.

**7. `plus` — hybrid ranked search.** One grep-shaped command fuses literal/full-text, symbol, an algorithmic semantic ranker, graph-neighbour proximity, and (optionally) native vectors, ranking the genuinely relevant hits first — while every row stays a plain `file:line score snippet`, search output, not a generated answer.

The through-line: **the dominant cost of an agent is the number of model rounds, not search latency.** Each technique above trades one precomputed structural answer for several speculative grep-and-read rounds.

---

## Install (drop-in)

There is no system installer — "install" means build the binary and put it on your `PATH`.

```bash
# 1. Build (default binary; loads the embedding model from the HF cache on demand)
cargo build --release --bin grepplus

# 1b. OR a true single file with the quantized weights baked in (no external model file)
cargo build --release --features embed-weights --bin grepplus

# 2. Put it on your PATH
install -m 0755 target/release/grepplus /usr/local/bin/grepplus

# 3. (optional) use it as a grep replacement — every plain grep call still works
echo 'alias grep=grepplus' >> ~/.zshrc

# 4. Index a repository once (a setup cost, not part of the agent loop)
grepplus index /path/to/repo
#   for the semantic vector path (default binary only):
#   grepplus index /path/to/repo --embeddings \
#     --embedding-gguf <embeddinggemma-300M-Q4_K.gguf> \
#     --embedding-tokenizer <tokenizer.json>
```

## Uninstall

```bash
rm -f /usr/local/bin/grepplus                                   # remove the binary
sed -i '' '/alias grep=grepplus/d' ~/.zshrc                     # remove the alias, if set
rm -rf "${GREPPLUS_STORE_DIR:-$HOME/Library/Caches/grepplus}"   # remove the index stores
```

The binary is self-contained — apart from the index store (path above), it writes nothing to your system.

---

## Use it in your coding agent

To make a coding agent (Claude Code, Cursor, Codex, …) use grepplus as its `grep` for **one project** — without touching your system `grep` — paste this to the agent:

```text
Use grepplus as your grep for THIS repository only. Do not change my global/system
grep, my shell config, or anything outside this project.

One-time setup:
1. Build it:        cargo build --release --bin grepplus
2. Index the repo:  ./target/release/grepplus index .
3. For this session only, put it first on PATH so you invoke it as `grepplus`:
       export PATH="$PWD/target/release:$PATH"

From now on, use `grepplus` instead of `grep` for every search — it is a byte-exact
drop-in for grep. For code questions, prefer these over grep+read loops:
- grepplus who-calls SYM / callees SYM / find-usages SYM   # callers / callees / uses
- grepplus impact SYM --direction incoming                 # what breaks if SYM changes
- grepplus path --from A --to B                             # the call chain from A to B
- grepplus context "plain-English description of the code"  # find code by meaning
- grepplus brief SYM                                        # def + callers + callees, one call
```

This is **project-scoped**: the `export PATH` only affects the agent's own session, so your system `grep` and other tools are untouched. For a persistent setup, drop the same instructions into the agent's rules file (`CLAUDE.md`, `AGENTS.md`, or `.cursor/rules`) — it then applies every session, still only inside this project.

---

## What it saves

Real medians from the agent benchmark (MiniMax-M3; discovery aggregate of 82 tasks — the 9 literal-control tasks are grep's home turf and excluded). grepplus agent vs. an uncoached agent baseline. **The fixed system prompt is warmup and is factored out of every ratio.**

| Axis | Factor (median) | What it measures |
|---|---:|---|
| Tool-call rounds | **4.0×** | how many times the agent calls the model |
| Billed input tokens (prompt-neutral) | **3.7×** | loop input, minus the fixed system prompt |
| Output tokens | **2.9×** | agent output |
| Wall-clock time (session, no setup) | **2.0×** | end-to-end agent loop time |
| Search-context tokens | **11.9×** | bytes the agent must read to *find* the answer |

The gain is **not uniform** — it's largest where `grep` structurally fails, and ~1× where `grep` is already optimal:

| Task class | Search-context median | Why |
|---|---:|---|
| structural graph queries (who-calls / callees / impact) | **~19×** | one resolved graph call instead of grep+read per caller |
| research / multi-hop trace | **13.5×** | one `impact`/`path` call replaces a manual graph walk |
| vocabulary-gap discovery (semantic) | **2.4×** | vector search finds what grep keywords miss; capped by the small Q4 model |
| literal definition search | **~1×** | grep's home turf — grepplus passes straight through |

The median search-context saving clears **10×**; the conservative headline of **~2× faster** (wall-clock) and **~3–4× cheaper** (billed tokens) is what you can count on across the mix.

---

## Status & scope

grepplus is an honest work-in-progress, not a finished product.

- **~40%** of the upstream C code-intelligence engine is ported.
- **~18 languages** are wired for cross-file `CALLS` and `IMPORTS`; Rust additionally resolves `TYPE_REF` and `USES`. Richer edge classes (`WRITES`/`READS`/`IMPLEMENTS`/`OVERRIDE`) and full non-Rust edge parity are **not** ported yet.
- The drop-in is **byte-exact** on a miss and for any scripted/piped grep; on a *match over a fresh index* it may append exactly one clearly-labelled non-canonical pointer line. For scripts that require byte-identical grep output in every case, don't alias `grep`.
- Not production-ready. Use it as a fast code-navigation aid for agents and humans, not as a system of record.

---

## Reproducing the numbers

The numbers above come from the harness in [`bench/agent_efficiency/`](bench/agent_efficiency/) — the exact one used to produce them, included so the claims are auditable.

```bash
# deterministic, no API key: byte-level search-context measurement
python3 bench/agent_efficiency/context_cost.py

# full agent benchmark (grepplus agent vs uncoached grep agent, MiniMax-M3 via Pi Code)
export MINIMAX_API_KEY=sk-...
python3 bench/agent_efficiency/real_corpus.py            # fetch the pinned real-repo corpus
python3 bench/agent_efficiency/run_bench.py --tasks bench/agent_efficiency/tasks_v2.json
```

See [`bench/agent_efficiency/README.md`](bench/agent_efficiency/README.md) for the methodology (baseline definitions, why the fixed system prompt is excluded as warmup, and the corpus manifest).

---

## License

MIT — see [LICENSE](LICENSE). Vendored assets and replaced dependencies: [THIRD_PARTY.md](THIRD_PARTY.md).
