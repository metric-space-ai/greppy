<img src="assets/logo.svg" align="right" width="160" alt="greppy logo"/>

# greppy

**Local code navigation for coding agents: deterministic symbol-graph evidence, native semantic search, compact function briefings, and byte-exact real-`grep` passthrough. One native Rust binary.**

[![CI](https://github.com/metric-space-ai/greppy/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/metric-space-ai/greppy/actions/workflows/ci.yml?query=branch%3Amain)
[![CodeQL](https://github.com/metric-space-ai/greppy/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/metric-space-ai/greppy/actions/workflows/codeql.yml?query=branch%3Amain)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`greppy` is a code-navigation tool that also accepts ordinary `grep`
invocations. Those invocations execute the real system `grep` and forward its
stdout, stderr, and exit code byte-for-byte; they do not open an index, load a
model, or mutate a Greppy cache. Greppy is installed only as `greppy`, never as
a global `grep` replacement.

Its structured commands answer questions an agent otherwise spends several
search-and-read rounds on: *who calls this function, what breaks if I change it,
where is the code that does X.* Deterministic source and graph evidence is the
authority. Locally generated summaries are short navigation hints attached to
the exact source signature, not a replacement for reading the returned code.

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

<sub>The **same** coding agent (MiniMax-M3, driven by [Pi Code](https://pi.dev)) answers one *who-calls* question on a real repo — **left with plain `grep`, right with `greppy`**. This recording illustrates the workflow only; the pre-registered, publishable release evidence is described below.</sub>

---

## Setup — two steps

**1. Install the binary.**

```bash
# Portable CPU build (both models are always embedded)
git lfs install
git lfs pull
cargo build --release --bin greppy
sudo install -m 0755 target/release/greppy /usr/local/bin/greppy
```

Every build embeds EmbeddingGemma and Qwen3.5 plus their tokenizers. No model is
downloaded at runtime and neither model can be disabled. Git LFS must materialize
the tracked model objects before Cargo runs; the build rejects missing or
incorrect assets. CPU inference is always available; build with `--features
metal` on Apple Silicon or `--features cuda` on Linux/NVIDIA to include the
accelerated backend. Runtime selection is automatic and can be made explicit
with `--device cpu|metal|cuda[:INDEX]` or `GREPPY_DEVICE`.

The first structured query builds its local workspace index. There is no
current prebuilt production package while `v0.2.0` is completing the release
gates listed below. Older archives remain available only as explicitly marked
legacy previews and are not the current production distribution. Build the
current `main` revision from source for evaluation. Do not rename or install the
binary as `grep`.

The deterministic graph is published first. Code-span embeddings are a one-time
local computation for each source generation and are reused by later agent
sessions. On a large repository they continue in a generation-bound background
job, so graph navigation remains available while Greppy trades local compute
once for lower repeated cloud-model search and context cost. `semantic-search`
never exposes partial vectors: until that generation is complete it returns
`status: "indexing"` in JSON (exit 75), the selected CPU/Metal/CUDA backend,
exact span progress, and an estimated completion time. Automatic selection
prefers a compatible Metal or CUDA device with sufficient memory and otherwise
uses the CPU fallback.

**2. Tell your agent the extra commands exist.** Delegate it — in your agent's
chat, say **`install https://github.com/metric-space-ai/greppy/`** — or
paste the snippet below into the file your agent reads for project instructions
(`CLAUDE.md`, `AGENTS.md`, `.cursor/rules`, `.windsurfrules`, or the system
prompt).

```text
This project has `greppy`, a local code-navigation tool over a symbol graph and
an on-device semantic index. Ordinary grep invocations are delegated byte-for-
byte to the real system grep, but Greppy must not be installed or invoked as a
global grep alias.

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
      Returns the closest-matching definitions by meaning (signature + file:line).
      While first-use embeddings are still building, returns a retryable status
      with the active backend, progress, and ETA instead of partial/empty hits.

EXPAND — get the full source in one call instead of opening files by hand:
  greppy expand ID
      who-calls / callees / impact / semantic-search may end their output with a
      line `Expand: greppy expand <id>`. Run it to print the prepared evidence
      pack — the full source of the top matches, bundled — in a single call,
      instead of reading each file:line yourself.

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

Treat returned source paths, exact spans, signatures, and graph relations as
evidence. The indented English sentence below a function signature is a local
Qwen navigation hint. Read the source and verify changes with builds and tests.
```

### Try it without committing to it

`greppy trial` runs one mechanically graded, own-project A/B observation. The
v1 protocol supports a `who-calls` check through Pi:

```bash
greppy trial \
  --root . \
  --question "Who calls parse_config, and from where?" \
  --check who-calls \
  --symbol parse_config \
  --expect load_application \
  --forbid legacy_loader \
  --runner pi \
  --provider minimax \
  --model MiniMax-M3
```

`--expect` and `--forbid` are repeatable, case-sensitive final-answer literal
checks. The command requires `--root` to be the exact top level of a clean Git
repository with a committed `HEAD`. Commit or remove staged, unstaged, and
untracked files first.

The harness creates two private detached worktrees outside the target
repository, at the same commit and tree. Each arm gets its own
`GREPPY_STORE_DIR`, Pi config directory, and session directory. Pi context
files, skills, prompt templates, extensions, themes, and session persistence
are disabled. Complete versioned system prompts are supplied with
`--system-prompt`; the Greppy prompt contains the exact requested symbol.
Only the Greppy worktree is indexed, before either measured arm. Arms then run
in deterministic `baseline`, `greppy` order and both worktrees must remain
clean at the pinned commit. Disposable worktrees and stores are removed after
the run.

Stdout is one `greppy.project-trial.v1` JSON object identified by
`schema_version`. It records commit and tree IDs; Pi and Greppy executable
paths, versions, and SHA-256 digests; exact normalized tool and source-open
calls; tool-result character counts; first-turn, later-turn, and aggregate Pi
token counters when reported; turns; wall time; answers; mechanical grades;
and trace SHA-256 digests. Raw Pi traces remain private and are deleted with
the disposable trial directory.

The only statuses are:

- `valid_observation` (exit 0): both valid arms passed the mechanical check.
- `quality_regression` (exit 1): the valid baseline passed and the valid Greppy arm failed.
- `inconclusive` (exit 2): setup, execution, contamination, cleanliness, or baseline-quality failure prevented that comparison.

A baseline trace that invokes Greppy is rejected. The `comparison` object gives
the observed quality relationship and Greppy-minus-baseline deltas and ratios
for this single pair. These are descriptive measurements, never a generalized
efficiency result or release claim.

The trial's disposable stores are removed automatically. To remove any Greppy
cache created during normal use and uninstall the binary:

```bash
greppy cache clear --root . --yes
sudo rm /usr/local/bin/greppy
```

---

## What it saves

Greppy is designed to replace exploratory search-and-open loops with one
structured query plus directly attached evidence. That benefit is a release
gate, not a marketing assumption.

Two complementary, pre-registered suites are checked in:

- [`bench/agent_efficiency/`](bench/agent_efficiency/) contains 115 pinned
  navigation tasks across six real repositories plus deterministic controls.
  It measures whether Greppy preserves answer correctness while reducing search,
  source-reading, and context cost.
- [`bench/agent_coding/`](bench/agent_coding/) contains 30 paired edit-and-test
  tasks across Flask, Hugo, Gson, Zod, Serde, and Tokio. Each task starts from an
  exact commit, proves that its independent test passes before mutation and
  fails after mutation, then runs isolated Greppy and ordinary-exploration arms.
  Setup is outside measured agent time; the post-agent test determines
  correctness.

Both suites record per-task correctness, tool calls, source opens, input/output
tokens, context or prompt volume, and wall time for the same agent and model.
Task banks, prompts, binaries, runtime versions, setup commands, and repository
commits are hashed into their manifests. Arm order is deterministically
balanced per task and its ordering scheme is versioned in the manifest.

`v0.2.0` may claim an efficiency win only when both published, mechanically
graded runs for the exact release commit prove all of the following:

- at least as many observed paired correctness wins as losses, plus no exact
  paired regression alarm at `p < 0.05` (the alarm is not presented as proof
  of population equivalence);
- at least 20% fewer tool calls and source-open calls on structural tasks;
- at least 20% fewer variable input tokens on structural tasks;
- exact repository commits, task-bank hash, prompt hash, model ID, Greppy
  binary hash, per-task rows, grading, aggregate, and forensics are published;
- raw agent traces remain private and are not release artifacts.

Historical charts and illustrative recordings are not treated as `v0.2.0`
evidence until current navigation and coding-outcome runs pass those gates.
Index construction is outside measured agent sessions because it is reusable;
release evidence reports its CPU/GPU wall time and resource cost separately and
includes the amortized break-even instead of presenting precomputation as free.

---

## How it works

- **Standard grep.** Any invocation that isn't one of the extra commands runs real `grep` and returns its output and exit code unchanged.
- **A precomputed code graph.** An indexed, typed symbol graph (`CALLS`/`USES`/`TYPE_REF`/`IMPORTS`) answers `who-calls`/`callees`/`find-usages`/`impact`/`path` directly — resolved relationships with `file:line`, not text matches — collapsing several grep+read rounds into one call.
- **Native semantic navigation.** `semantic-search` uses Google's embedded **EmbeddingGemma** to find code by meaning. Embedded **Qwen3.5-0.8B Q4_K_M with MTP** adds a short purpose hint under each returned function signature and to each definition printed by `brief`. Inference is local Rust plus vendored Metal/CUDA kernels: no llama.cpp runtime, Python, HTTP, or model server.
- **Bounded warm daemons.** The embedding and summary engines use separate local daemons. A used model remains resident for five idle minutes; the process exits after 30 idle minutes. Failed inference never removes deterministic source or graph output.
- **One native Rust binary.** Both model files and tokenizers are baked into every binary; tree-sitter parsers and SQLite are compiled in. CPU is universal, while release artifacts add the native GPU backend for their target platform.

## Local data and cleanup

Greppy stores workspace paths, source spans, graph edges, embeddings, and query
cache entries in a local SQLite-backed cache outside the repository. Directories
are private to the current user (`0700` on Unix), and cache objects are managed
only after ownership, type, and path validation. Set `GREPPY_STORE_DIR` to place
the data on an encrypted or ephemeral volume.

Full source bodies are not duplicated into SQLite. Exact code search reads the
current worktree through real `grep` where available, with an in-binary literal
fallback on clean Windows hosts. Freshness checks guard indexed graph spans and
embeddings.

```bash
greppy cache status --json       # inspect paths, sizes, locks, TTL and quota
greppy cache gc --dry-run        # preview TTL/LRU reclamation
greppy cache gc                  # reclaim eligible entries
greppy cache clear --root . --yes
greppy cache clear --all --yes   # explicit destructive operation
```

The default workspace-cache TTL is 14 days. `GREPPY_STORE_TTL_DAYS=0` disables
age eviction but not the independent size quota.

---

## Status

The current `main` branch is qualifying for the gated `v0.2.0` release. No
official release is cut until the packaged artifacts, native inference,
daemon-fault tests, summary-quality corpus, agent benchmark, hardware matrix,
signing, and notarization gates pass.

- **Production-certified parser set:** Rust, Python, Java, JavaScript,
  TypeScript, and Go.
- **Additional parser coverage:** many more languages can be indexed and
  searched, but are not claimed to have the same graph completeness until they
  receive language-specific fixtures and real-repository acceptance tests.
- **Supported release targets:** macOS Apple Silicon with Metal, Linux x86_64
  with CPU and NVIDIA CUDA, and Windows x86_64 CPU with named-pipe daemons.
- **Known boundaries:** reflection, runtime dependency injection, generated
  code, macros, and dynamic dispatch can hide relationships from any static
  graph. Freshness checks fail closed rather than knowingly returning stale
  source evidence.

Published releases are immutable and checksummed. Greppy has no self-updater;
pin a release or commit and upgrade through verified release artifacts. See
[SUPPORT.md](SUPPORT.md), [SECURITY.md](SECURITY.md), and
[CHANGELOG.md](CHANGELOG.md).

Contributions follow [CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). Research and benchmark users can cite
the exact software artifact through [CITATION.cff](CITATION.cff).

## License

Greppy source code is MIT-licensed; embedded model weights have separate terms.
See [LICENSE](LICENSE), [THIRD_PARTY.md](THIRD_PARTY.md), and the model notices
under [`licenses/`](licenses/).
