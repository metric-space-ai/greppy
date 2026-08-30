# Changelog

All notable changes are documented here. Greppy follows Semantic Versioning.

## [0.3.4] — 2026-08-30

### Portable Chunk-CoW agent workspaces (0.3.4)

`greppy -p` now has one fail-closed workspace path on macOS ARM64, Linux
x86_64, and Windows x86_64. The 0.3.3 Rift/native snapshot backends, Git
worktree fallback, `--workspace-backend`, and `--fresh` are removed from the
current build. The portable Rust core stores fixed 1 MiB BLAKE3-addressed
chunks in append-only segments and keeps namespace manifests, tombstones,
redirects, references, and recovery journals in SQLite WAL.

The initial view combines a pinned Git commit with a double-validated immutable
snapshot of staged, unstaged, deleted, and untracked paths. Ignored files and
build caches are excluded. Merge, rebase, cherry-pick, submodule, Git LFS, and
arbitrary checkout/smudge-filter states fail before model startup. Partial
writes replace only touched chunks; private Git indexes, refs, and new objects
remain isolated while existing objects are read-only.
Dirty-baseline hardlink identity is now double-captured and hashed. Proposal
metadata stores canonical hardlink groups, binds them into the proposal commit,
and restores real shared inodes through crash-safe apply instead of degrading
them into byte-identical independent files.
Proposal staging hashes each hardlink group once and explicitly writes the
same blob to every corresponding Git index entry. This avoids platform-specific
Git stat-cache shortcuts from producing a commit whose tree disagrees with the
bound hardlink topology.

New `workspace setup`, `workspace doctor --json`, `workspace status --json`,
and `workspace gc` commands manage one persistent per-user provider mount.
Linux uses FUSE3 and installs a restartable systemd user service; macOS 15+
uses a bundled FSKit app extension with a minimal Swift system boundary and a
RunAtLoad LaunchAgent that revalidates setup after login. These per-user
registrations are published atomically without following existing symlinks.
The Windows package remains diagnostic until the selected signed transport
can forward real hardlink operations to Greppy's Rust provider; unchanged
WinFsp 2.1 does not satisfy that contract. Adapter failure has no hidden
native fallback. Concurrent provider and CLI opens wait for short SQLite-WAL writers
instead of failing nondeterministically with `database is locked`.
Control and mounted provider manifests now bind the same immutable identity and
capability set while validating their independently refreshed heartbeats
separately; a healthy adapter is no longer rejected during the unavoidable
cross-file heartbeat publication window.
Committed Git trees are exposed with traversable virtual directory modes;
their type-only Git mode is never mistaken for filesystem permissions. A Base
builder compares source and mounted inventories before indexing and verifies
the final `file_state` row count before publishing `COMPLETE`, preventing a
root-files-only index from becoming a reusable false-success Base.

Release eligibility additionally requires one exact-SHA performance set from
the real Linux FUSE3, macOS FSKit, and Windows provider mounts. A shared
fail-closed verifier rejects missing platforms, mixed commits, dirty builds,
relaxed thresholds, or any failed latency, storage, parallelism, or toolchain
gate.

Dirty-based proposals record the pinned commit as parent but expose only the
initial-snapshot-to-final patch. `greppy agent apply REF` verifies the exact
baseline hash, preserves the existing Git index, applies only the agent delta,
and journals backup/recovery state. Ordinary cherry-pick is not advertised as
safe for dirty-based proposals.

Proposal publication and apply now use a shared, OS-backed exclusive lease per
canonical repository. Fsync-bound publication journals recover an interruption
between baseline ref, pinned proposal metadata, and the public proposal ref;
apply journals recover a process death after visible path materialization while
leaving the Git index byte-identical. Active operations are never mistaken for
crashes, and symlinked, foreign-repository, or metadata-mismatched journals fail
closed. The exclusive recovery opener also runs SQLite quick and foreign-key
checks over namespace and chunk metadata before restoring a healthy marker.

Agent Base prewarming is now fail-closed as well: a graph, embedding, summary,
manifest, or publication failure aborts `index --agent-worktree` and `greppy
-p` before the first model call instead of silently switching to a private
Store and returning success. Summary inference deterministically bounds very
large individual source spans while retaining the complete span for cache
identity and Base-completeness checks. The regression fixture is the exact
30,716-byte EODAG JSON input that exposed the CUDA prewarm failure. Explicit
`--device` and `--no-gpu` selections are preserved by the internal immutable
Base build instead of being silently replaced by automatic device selection.

`bash-smart` now applies leading `VAR=value` tokens to direct argv commands,
rejects unquoted shell syntax with an actionable quoted-command example, and
uses Bash pipefail for quoted Unix pipelines so a failed producer cannot be
masked by a successful `tail` or `grep`. Windows pipeline scripts are rejected
until an equivalent fail-closed status contract exists.

The Windows conformance gate remains fail-closed: official unchanged WinFsp
2.1 does not implement hard links, while the 0.3.4 workspace contract requires
them. No Windows release is eligible until the provider substrate satisfies
the same hard-link semantics as Linux and macOS; the test is neither skipped
nor emulated with a full file copy.

`index status` now distinguishes an active foreground or first-use index build
from a missing index before the first atomic snapshot is published. It reports
`status=indexing`, `writer_active=true`, actionable wait guidance, and temporary
exit 75 instead of the contradictory `no_index` response that told agents to
start a second indexer.

## [0.3.3] — 2026-08-25

### Filesystem-CoW agent workspaces

`greppy -p` can create exact private Filesystem-CoW workspaces through the
purpose-specific `greppy-rift-core` hard fork. `--workspace-backend auto` is
the default: it attempts only O(1)-metadata CoW before model startup and falls
back to the unchanged 0.3.2 native Git-worktree path when the platform,
filesystem, or snapshot setup is unavailable. `native` forces the 0.3.2 path;
`cow` requires exact CoW, including per-file reflink trees, and fails explicitly
instead of silently copying.

Every CoW run replaces the linked-worktree control file with a real private Git
directory. Existing repository objects are exposed read-only through Git
alternates; new objects, indexes, refs, and model-created commits stay private
until `finish` exports one verified proposal commit. Main-checkout HEAD and
index remain unchanged. Identity and containment checks preserve a workspace
on suspected tampering instead of deleting or publishing it.

The retained snapshot core is a hard, purpose-specific MIT-licensed fork of
`anomalyco/rift`, pinned to an immutable revision. It contains only APFS,
Btrfs, and Linux reflink mechanics plus capability probing and cleanup. The
former Rift CLI, registry, hooks, FFI, workspace policy, and compatibility
surface are not part of Greppy.

Capability reporting distinguishes Btrfs subvolume snapshots from namespace-
walking CoW trees. APFS directory `clonefile` and Linux per-file `FICLONE`
remain exact explicit-preview backends, but `auto` does not select them because
both traverse repository metadata; Windows continues to use the native
fallback.

Btrfs templates are created and sealed read-only through the retained native
API. A warm snapshot may bypass the repeated full-tree cleanliness traversal
only when capability probing confirms both constant-time metadata and the
filesystem-enforced immutable source; other backends retain validation.
The registered 300,000-file Btrfs CI fixture measures ten warm creations and
fails above 500 ms median or 1,000 ms P95; the current candidate measured
345.735 ms median and 385.708 ms P95.

Windows agent workspaces and sandbox Cargo paths now resolve from native
`LOCALAPPDATA`/`USERPROFILE` roots instead of falling through a Unix-only
`HOME` assumption to a mixed `/.cache\\...` path. The Windows CI executes the
agent workspace suite so this native-fallback contract cannot regress silently.
Canonical extended-length paths are converted to ordinary drive/UNC syntax
only at the Git subprocess boundary because Git for Windows rejects `\\?\...`
values passed through `--git-dir`.

## [0.3.2] — 2026-08-24

### Parallel Agent Store CoW

Concurrent `greppy -p` runs now share one content-identified, immutable Base
Store for unchanged code while keeping a writable Delta Store private to each
agent. The Base identity covers the Git tree, store schema, indexer/extractor
contract, and summary and embedding contracts. One leased builder publishes a
complete graph, text index, summaries, and embeddings atomically; followers
reuse it read-only. Corrupt or incompatible generations are quarantined and
rebuilt, with an explicit full-private fallback if preparation cannot complete.

Overlay queries cover graph navigation, callers/callees, traversals, path and
impact, symbol and content search, FTS and vector search, `brief`, `read`,
`expand`, stats, freshness, and diagnostics. Dirty and deleted Base paths are
suppressed against the pinned Base commit, cross-layer edges resolve through
logical symbol identity, rankings merge deterministically, and an exact revert
removes the Delta contribution. Atomic Delta snapshots preserve the previous
generation across a failed publication.

`greppy index --agent-worktree` now builds or validates the same shared Base
the integrated agent will consume. `greppy -p --private-store` opts one run out
of Base reuse for diagnosis. `doctor`, `diagnostics`, and index-health output
report Store mode, identities, completeness, changed paths, cache hits, and
fallback reason. Shared Agent Bases participate in ownership-checked TTL/LRU
cache management and are protected from eviction by live reader leases.

The sandbox exposes the Base only as readable data and gives each agent an
isolated writable Delta. Differential full-reindex tests cover edits, deletes,
renames, untracked files, exact revert, graph and text query families, and
pre-publication crashes; concurrency tests gate ten agents and stress fifty
isolated Deltas. Base summaries and embeddings are reused across ten agents
without duplicate private rows.

### Batched brief

`brief` now accepts multiple symbols, including newline-delimited selectors on
stdin. Human and JSON output preserve requested order, apply the global output
budget only at whole-result boundaries, and return a usable retry when the next
complete result does not fit.

### Release hardening

The agent-efficiency diagnostic now removes each arm's fixed prompt from later-turn
input and treats provider errors as invalid coverage rather than product-cost
measurements. Isolated invalid arms receive bounded serial recovery after the
parallel run; remaining invalid sessions fail closed without weakening any
efficiency or quality threshold.

The integrated-agent navigation contract now treats one semantic `search`
result as a ranked answer-candidate set and forbids paraphrased semantic retry
loops. Exact-candidate evidence showed the requested definition at rank two on
the first call while an unconstrained agent repeated 50-plus equivalent
queries. The diagnostic corpus and its 20-percent efficiency thresholds remain
unchanged.

Explicit and scheduled CI soaks now build Linux debug binaries with the real
`cpu-only` feature, while macOS keeps its default Metal build. The previous
unqualified Linux soak correctly tripped Greppy's hard no-GPU-backend guard
before the battle suite could start.

The coding-outcome diagnostic now partitions its registered 41 tasks and
three arms into six deterministic runner shards, with at most three shards
active concurrently. A serial candidate completed 47 of 123 arms before the
six-hour timeout; three threads on one runner reached only 75 because local
build and toolchain contention erased much of the gain. Shards retain atomic
checkpoints and deterministic within-task arm order, and a separate fail-closed
merge requires six disjoint, complete task partitions before grading all 123
results once. Tasks, models, prompts, grading, and release thresholds are
unchanged.

Coding shards now treat each registered Pi timeout as a symmetric compute
cutoff rather than an infrastructure failure. After at least one completed turn,
the exact worktree snapshot at the cutoff is independently tested: a passing
snapshot is correct, a failing snapshot is an ordinary measured loss, all cost
remains charged, and no timeout replay occurs. Zero-turn cutoffs and
provider-reported errors remain invalid. This replaces the harness-v3
fresh-session recovery after exact-candidate evidence showed that complete
replays can hit the same task budget and merely double cost without resolving
the measurement. The manifest now also freezes
the Greppy-edit prompt hashes and describes the actual provider-dollar and
post-edit re-read thresholds instead of stale token-ratio wording. The binding
edit contract now matches the tested harness: all three arms receive the same
explicit `bash,read,edit,write` palette, and treatment rather than a hidden tool
restriction defines the Greppy-edit contrast.

That edit treatment now fails closed when a changed worktree has no observed
Greppy edit or Pi's built-in edit/write controls were used. Unified diffs
streamed to `greppy patch` on stdin contribute their touched files to the
post-edit re-read metric, and the prompt names the accepted unified-diff form
instead of leaving agents to retry incompatible patch marker formats.

Coding mutation preflight now accepts Pytest's completed quiet-mode progress
line as framework-proven failure evidence when it contains an `F` or `E` marker.
Pytest `-q` can omit the collection banner even though tests executed; this
previously invalidated a real two-test failure as infrastructure. Bare failure
text and all-passing progress remain fail-closed.

The packaged-daemon SIGKILL recovery smoke now creates a cold replacement
owner and proves an active model load before killing it. Polling for the
microsecond-scale active window of three warm embeddings produced a false
failure on fast macOS runners even though stale-socket recovery, concurrency,
and inference had already passed.

`read-file` accepts explicitly absolute diagnostic and artifact paths outside
the repository, while relative `../` traversal remains confined to the repo.
This lets integrated agents inspect downloaded gate artifacts without leaving
Greppy's read/navigation workflow.

Automatic CLI option recovery now treats POSIX `--` as a hard boundary. A
malformed invocation with option-looking positional arguments after the
terminator is refused once with bounded usage output instead of recursively
rewriting the same arguments until stack overflow.

Release-package acceptance now seeds unmanaged 0.3.1-shaped model and workspace
cache namespaces before indexing, proves `index` followed by `where-am-i`, and
then verifies that cache clearing removes managed data while preserving those
unmanaged legacy files. This covers upgrades from an existing cache as well as
the already-gated fresh-store package path.

## [0.3.0] — 2026-08-12

The first release of the 0.3 line, and it ships the whole line: everything
below, developed as 0.3.0 and 0.3.1, lands here together. (An interim
internal tag named v0.3.1 existed briefly during stabilisation and was
retired.)

### Fixes landed during bench stabilisation (2026-08-08 … 2026-08-12)

`index --agent-worktree` warms the store the agent actually reads: `greppy
-p` runs against an isolated data root beside the worktree, and the warm-up
now writes there instead of the operator's root — before this, prewarmed
agents still paid the full first index inside the measured run (9× slower,
confirmed end-to-end by the benchmark).

One invalid provider record degrades the record, not the file. Template-heavy
C++ headers (ClickHouse) lost hundreds of definitions to a single anonymous
node produced by grammar error-recovery; `.h` files whose C parse shows C++
syntax are re-extracted with the C++ grammar, and the skip detail now names
the violated contract rule.

A dead embedding-daemon endpoint degrades to in-process embedding instead of
failing every semantic query. `Failed` stays reserved for a live daemon that
actually holds the model; the error texts name a next action.

Edit receipts echo the landed span (±3 context lines, read back from disk,
changed lines marked). `who-calls` on a module names the module's own
definitions and the two commands that answer the question as meant.

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

### The 0.3.0 development line

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
accident. A separate 33-case release smoke gate ran every advertised line
against the built binary, so the manual could not promise what the code did
not do.

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
