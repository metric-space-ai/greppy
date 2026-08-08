# Changelog

All notable changes are documented here. Greppy follows Semantic Versioning.

## [0.3.1] — 2026-08-08

### Answers that say what to do next

An answer about a stale index used to open with a freshness digest comparison
and never say what would fix it. The remediation now leads, phrased the way
`doctor` phrases it, and the digests follow as evidence. Index contention no
longer just reports that another indexer is "running" — it names the wait and
`greppy index status --json`, so the two messages stop forming a loop with no
exit.

`greppy search-pattern … --fixed` now names the flag when the flag is what
caused the miss: a pattern written as a regular expression cannot match under
`--fixed`, and the previous suggestions (look it up as a definition name,
reindex) led away from the cause.

### `index [PATH] [--agent-worktree]`

`--agent-worktree` prepares and indexes the worktree `greppy -p` runs in.
That worktree has its own workspace identity, so indexing the checkout left
the agent cold and it paid for the first index — graph and embeddings — inside
its own run.

### Corrections to the grep contract

`greppy PATTERN [FILE]` stays grep even when PATTERN happens to sit within two
edits of a command name: `greppy greppy FILE` was refused where real grep
printed the matching line. A mistyped subcommand carries no file operand, and
that is now what tells a typo from a pattern.

A mistyped flag is corrected rather than silently dropped — dropping `--jsoon`
handed text to a caller who asked for JSON.


`--deadline-secs N` / `GREPPY_DEADLINE_SECS` stops the agent loop cleanly between turns at a wall-clock budget and still delivers the proposal (or clean) outcome.

### `greppy -p` — a built-in coding agent

`greppy -p "TASK" [--model M]` runs a one-shot coding agent over the current
repository. The harness is greppy itself: a ~60-line static system prompt,
one tool — `greppy` (search/navigate/read/edit, plus commands via
`bash-smart -- CMD` so output arrives compacted) — and an agent loop ported
from pi v0.80.2 (MIT; see THIRD_PARTY.md and licenses/PI-LICENSE.txt).

Every run works in a per-repository agent worktree under the platform cache.
Tracked content is reset to HEAD before every run; ignored build caches are
kept deliberately so repeat runs stay fast (`--fresh` drops them too). Nested
repositories are removed; repositories with submodules are refused (the agent
worktree cannot reset them safely). The worktree's own greppy index is built on
first use and kept warm afterwards. The run ends as exactly one commit on
`refs/greppy/agent/<run_id>`, built from the base tree regardless of what
happened to HEAD in the worktree — nothing is edited in place. `git show <ref>`
reviews it, `git cherry-pick -n <ref>` applies it; `--apply` does so directly
but refuses a checkout with uncommitted changes. Concurrent `-p` runs fall
back to a disposable temp worktree when the stable tree is locked. Host-side
git against the worktree pins the linked git directory recorded at creation; a
rewritten worktree `.git` aborts the run (`Tampered`) rather than redirecting
into the user checkout.

Inference is localhost-only (plain HTTP, no TLS stack in the client): an
Anthropic-Messages-compatible gateway (standard: CLIProxyAPI on
127.0.0.1:8317; `GREPPY_ENDPOINT`, `GREPPY_MODEL`, `GREPPY_API_KEY`) with
strict SSE validation, transient-failure retry, and stream caps. No provider
SDKs, no stored credentials. Tool subprocesses are write-confined to the run
worktree, a per-run scratch dir (`TMPDIR`), an isolated greppy data root for
that worktree (store + lock namespace; not the operator's global greppy data),
and `~/.cargo/{registry,git}` only — never the platform cache wholesale, the
global temp root, or `~/.cargo/bin` / credentials (Seatbelt on macOS, Landlock
on Linux; `--no-sandbox` / `GREPPY_NO_SANDBOX=1` disables). Store GC is skipped
under `GREPPY_AGENT_RUN`. Reads and network stay open — network confinement
remains future work. Credential env vars are stripped from tool children and
agent runs refuse to nest (`GREPPY_AGENT_RUN`).

`AGENTS.md` gains an `AGENT:` section documenting `-p` as a delegation
primitive for host agents; the frozen-prompt hash moved with that approved
change.

## [0.3.0] — never released on its own; its changes ship in 0.3.1

A breaking release: greppy stops being a navigator that hands you file paths
and becomes the whole loop — find the symbol, read exactly it, change exactly
what you read, and get a receipt instead of re-reading. No aliases, no legacy
paths, no deprecation period: vocabulary that is gone is refused, not quietly
translated.

### The surface is the contract

`AGENTS.md` IS the system prompt, and twelve guard tests hold it: eleven pin
the concepts (which verbs exist per section, the result-line shape, that a
footer flag holds for every command in its section, that retired vocabulary
never returns), and one freezes the file byte for byte — changing it requires
updating an approved hash, which is a deliberate signature rather than an
accident. A separate gate (`dev/smoke_pass.py`, 33 cases) runs every
advertised line against the built binary, so the manual cannot promise what
the code does not do.

### Search — one family on one axis

`search "WHAT IT DOES"` (meaning), `search-symbol NAME` (name),
`search-pattern REGEX [--fixed]` (text). A miss cascades through case and
underscore variants, edit distance, word overlap and finally meaning, showing
only the best tier and labelling its confidence. Counts always describe the
set that was printed.

### Navigate

`where-am-i` opens a session: languages, file and definition census with
documentation and configuration counted separately, each module with its most
referenced symbols, entry points and test roots — each level a screen with a
true count and an expand id for the rest. `who-calls` and `callees` answer for
several symbols at once, grouped, in the single-symbol row shape
(`file:line  name`, tests marked). `brief` sketches a body — signature plus
one line per step naming the symbol used there — instead of bundling three
commands. `impact` is a tree of what a change reaches, each row with what it
does, obeying the same size law as everything else. `path --from A --to B`
prints the call chains as a tree of the call sites they hang on.
`find-usages` is gone; who-calls walks CALLS and USAGE, which is what it
promised.

### Read

`read S` returns the definition byte-exactly, `read-smart S` folds nested
blocks below `--depth` into one-line semantic descriptions, `read-file PATH`
paginates at 400 lines. `--handle` names exactly the span that was printed —
including when the output was cut — and `replace-span` takes it.

### Edit — eleven verbs, receipts that never lie

`replace`, `replace-text`, `replace-lines`, `replace-span`, `insert-lines`,
`delete`, `delete-lines`, `patch`, `write`, `rename`, `undo`. `NEW`/`DIFF`
come from stdin when absent. A receipt names every span that was touched
(`applied f.txt:1,3  4e689e`); `--dry-run` says "would apply" and changes
nothing; a refusal states its reason with evidence (`OLD occurs 2 times —
nothing written`) instead of guessing, and `undo` refuses when a later edit
touched the same span. `patch` lands whole or not at all.

### Everywhere

`--path`, `--json`, `--limit`/`--offset`, `--root` and `--help` hold for every
command. Empty answers exit 0 and say so; unanswerable questions exit non-zero
and say why; the search family keeps grep's codes and the grep-compatible form
stays byte-identical. The byte budget (`--max-bytes`) works again after the
module split had silently disabled it, and it never shrinks an answer to zero
rows with a retry that would loop.

### Build

GPU acceleration is the default: Metal on macOS, CUDA on Linux and Windows,
enforced at compile time; `--features cpu-only` is the single escape. `lib.rs`
went from 26k lines to 10.5k across thirteen modules.

### Not in this release

`bash-smart` is compiled out (`--features bash-smart` for development): the
training-free layers exist and are tested, but the classifier head that makes
them smart is 0.4.0 work, and the release does not ship a name that promises
more than it delivers. Known and deferred: ~13% of CALLS edges resolve on
ambiguous names, group imports extract only their first name, and the retired
git-scope search machinery is still present as dead code.


## [0.2.1] — 2026-07-20

First gate-qualified release: cut only after CI, CodeQL, the dependency
security audit, the task-bank reproducibility audit, the navigation-regime
agent benchmark, and the summary-quality gate passed on the release commit,
then signed, notarized, and attested (SBOM + provenance). Ships the complete
four-model MSCC evidence (MiniMax-M3, GLM-5.2, Qwen3.6-27B, Kimi-K3) and the
accompanying paper.

### Added

- Embedded Qwen3.5-0.8B Q4_K_M/MTP purpose summaries for `brief` and
  `semantic-search`, with CPU, Metal, and CUDA inference.
- Versioned JSON contracts with exact spans, source signatures, summaries, and
  durable expand handles.
- Shared inference backend registry, device probing, memory checks, daemon
  status, and model digests in `greppy doctor --json`.
- Managed cache inspection, garbage collection, and explicit clearing.
- Windows named-pipe transport for the embedded inference daemons.

### Changed

- Ordinary grep invocations are byte-exact real-`grep` passthrough and have no
  index, model, or cache side effects.
- Freshness is fail-closed: Greppy does not knowingly print stale source
  evidence.
- EmbeddingGemma and Qwen model assets are mandatory product assets in every
  binary; only the inference backend/device is selectable.
- Model idle TTL is 300 seconds and daemon process idle TTL is 1800 seconds.

### Fixed

- `greppy index` publishes the complete graph snapshot when embedding
  inference degrades (model load failure or failed batches) instead of
  discarding all indexing work with `EXIT_IO`; the vectors that did embed are
  kept and the next semantic query resumes the remainder in the background.

### Removed

- Synthetic grep-output augmentation, sidecars, and
  `NON_CANONICAL_CODE_HINT`.
- The `--vectors` switch and public model-disable/model-path controls.
- Installation or packaging under the binary name `grep`.
- The in-product self-updater.

### Licensing and hosting

- Source relicensed MIT → Apache-2.0 (embedded model terms unchanged:
  EmbeddingGemma under the Gemma Terms, the in-house Qwen3.5 fine-tune under
  Apache-2.0).
- Model weights hosted as public, ungated Hugging Face repos
  (`metricspace/embeddinggemma-300m-q4k`, `metricspace/greppy-qwen35-mtp-q4km`);
  the build fetches and SHA-256-verifies them before embedding.

## [0.1.2]

- Last pre-`v0.2.0` development release.
