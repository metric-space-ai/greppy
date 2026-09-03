# Changelog

All notable changes are documented here. Greppy follows Semantic Versioning.

## [Unreleased]

### Agent session events

Interactive session JSONL now records additive `tool` and `turn` event lines
(start/finish/done/error) plus a `source` field on `meta` (`interactive` /
`headless`). Unknown or new line types still load without marking the session
recovered.

`greppy -p` persists a session in the same store, prints `session: <id>` as
the first stderr line, and accepts `--continue` / `--resume SESSION_ID`.
`greppy -p --json` streams newline-delimited JSON events on stdout; `greppy
agent --json` is rejected. The first SIGINT/SIGTERM cancels at a safe
boundary and still emits `result` with status `cancelled` and exit 130; a
second signal exits immediately.

### Agent session readers

`greppy agent sessions list|show|tail|path` reads persisted JSONL session logs
without starting the TUI or writing to the store. `list` is newest-first;
`show` renders the transcript; `tail --follow` polls every 200 ms until SIGINT;
`path` prints the JSONL location. Unknown JSONL types are ignored. Session ids
accept a unique prefix; unknown or ambiguous ids exit 2. Human `show`/`tail`
and client event rendering strip terminal control sequences from remote text;
`--json` stays byte-faithful.

### Agent control clients

`greppy agent status|send|attach|interrupt|quit` drive a live `greppy agent serve`
session over its control socket. Ids resolve like `sessions`; a session
without a live socket exits 3. Sockets use short hashed paths in a per-user
runtime directory so they fit the macOS Unix-socket limit; `sessions list
--json` reports the path. `send --wait` streams events until that turn
completes; without `--wait` it prints the queued prompt id and returns.
`attach` streams live events until Ctrl+C (exit 130).

## [0.4.0] — 2026-09-02

### Web tool for the agent, and worktrees that reuse the shared inference cache

0.4.0 is where the web-runtime line and the 0.3.x stability line converge.
Two things had to be true for this release: the built-in agent can drive a
browser through its own `greppy web` verbs, and a linked Git worktree never
re-embeds a repository the primary checkout has already embedded.

**Web tool (beta).** `greppy -p` and `greppy agent` receive the browser prompt
block from `assets/prompts/web-beta.md` via `include_str!`; the headless path
installs the attach token like the interactive one, and the runtime is started
by the unsandboxed parent before the tool children are sandboxed (macOS forbids
a nested sandbox). External agents still get only the pointer in AGENTS.md; the
verb surface is beta and may change.

Measured on the fourteen-task set `greppy-web-tasks-v1` (five repetitions each,
one long-lived runtime): engine tier 70/70, CLI-chain tier 70/70, both up from
78 % and 64 % at the start of the campaign. The findings behind that climb:

- Synthetic input has a delivery receipt for every move, press, release, wheel
  and touch leg. A fresh WebView whose display list is not committed is
  repainted and the event resent (up to eight times); exhausted retries fail
  the verb instead of reporting success (findings 019, 034).
- `check` / `uncheck` are real confirmed clicks verified against the resulting
  state, so framework apps that bind checkbox state to native clicks see the
  change; the property-and-events path stays as fallback (finding 033).
- Enter in an eligible form field performs the HTML implicit submit (038).
- Action and read verbs accept the same selector forms: unquoted descendant
  CSS (`css=body a`) and the quoted wrapper (`css="body a"`) both work in
  `click`, `hover`, `wait`, `extract` and `find` (032).
- `:visible` / `:hidden` are stripped before `querySelectorAll` and applied
  as a box-size filter instead of silently returning an empty list (031).
- The session CPU budget is a delta from the session's own baseline instead
  of the worker lifetime, checked before and after each operation; a
  respawned worker resets the baseline (039, and the release-suite limit
  tests that only the after-check can satisfy).
- The controller waits for the next script instead of a 120 s idle clock
  that killed a live controller during session create/close cycles; the
  1000-cycle gate finishes with a `test result:` line again (027).
- The runtime removes its socket and attach token on shutdown and heals a
  stale endpoint on bind only after a connect proves nobody listens; a live
  runtime keeps its socket and the caller gets a real address-in-use error
  (040).
- A content-worker crash answers the in-flight call with the typed error
  `worker_restarted` (`retryable: true`, `next_action` set) instead of prose;
  the CLI forgets the poisoned session and reopens (030). A plain timeout no
  longer poisons every session.
- Same-URL navigation returns the fresh document, not the one it replaced.
- `greppy web screenshot --render-complete` waits for a complete render;
  the default returns the framebuffer immediately.
- Document-bound `@N` refs from `observe` resolve locators and expire with
  `STALE_REF` after navigation or DOM loss.

**Worktrees and the shared inference cache.** The immutable Base build for a
linked worktree runs `greppy index` as a child with `GREPPY_STORE_DIR` pointed
at a private staging directory. Models and the inference cache derived from
that root, so the child embedded every span again with an empty cache and
copied the embedding model into staging: 41 012 spans, 1 520 s on an M-series
Mac for a tree the primary checkout had fully embedded. The parent now passes
its shared root through `GREPPY_SHARED_INFERENCE_ROOT`; the same worktree
build takes 161 s and touches no model file. Abandoned staging directories
(`greppy-base-build-*`, `greppy-linked-base-checkout-*`, 44 of them holding
23 GB on one machine, reported by `cache status` as unmanaged) are reaped
after six hours at build start and by `greppy cache gc`.

**Store-CoW freshness.** Hidden and filtered Delta files carry a content
identity, so a metadata-only change no longer loops `graph refresh`; the
background refresh no longer reports an old snapshot as newly published;
overlay navigation uses indexed Base/Delta queries (`path` from >120 s to
under 6 s); a detached HEAD reports only the paths that actually drift.

**Known limits.** A JavaScript search component that intercepts Enter (for
example Wikipedia's) still needs the button; the agent finds it, at roughly
five times the turns. Windows: the Web tool is not included in 0.4.0; Windows
web-runtime localization follows in 0.4.1. The Windows CLI reports that scope
explicitly instead of attempting runtime discovery. Live agent serve/control
is Unix-only in 0.4.0; Windows retains local one-shot and interactive agents.
The HTTPS extra-header tests trust the fixture certificate only in debug builds. On macOS, replacing
or updating the signed app can require one renewed approval of `Greppy
Workspace FS`. `workspace setup` detects approval for the exact current bundle
before mounting, opens the File System Extensions pane directly when needed,
and remains fail-closed until the user enables the named switch and reruns it.

## [0.3.4] — 2026-08-30

### Worktree indexing and agent feedback fixes

- Embedding and summary results are now cached user-wide by the complete
  model/prompt/task/input contract. Linked worktrees and portable CoW agents
  reuse exact content instead of re-embedding it in each private Store; only
  changed Delta documents consume inference.
- Warm embedding reuse now reads and touches the user-global cache in bounded
  transactions and imports each vector batch into a new Base with one Store
  transaction. Token-length-only rows are evicted before expensive vectors,
  and concurrent cache-writer contention cannot turn a valid hit into a new
  inference request. Live index status reports local reuse, global hits and
  genuine inference misses separately.
- Store-CoW embedding counting and indexing now page the writable Delta table
  directly. They no longer sort and deserialize every immutable Base node only
  to discard it afterward, eliminating long zero-progress
  `counting_embeddings` phases for small worktree deltas.
- Exact token sizing uses the lightweight tokenizer in the indexing client
  while all heavyweight model inference remains in the shared supervisor.
  First-use indexing no longer serializes one daemon request per candidate
  span before its first embedding micro-batch, and status reports
  `counting_embeddings` instead of freezing at a false `loading_model` 0%.
- Indexing and `bash-smart` no longer load EmbeddingGemma inside each client.
  All heavyweight embedding work goes through one user-scoped daemon endpoint
  per device/model contract, with document micro-batches, round-robin client
  scheduling and no queue-capacity rejection. There is no local heavyweight
  fallback during daemon startup or contention.
- The inference protocol is versioned into the daemon endpoint identity, so an
  on-the-fly binary replacement starts a compatible daemon while an older idle
  daemon drains and exits instead of receiving a request shape it cannot
  understand.

- Ordinary linked Git worktrees now share one immutable, Git-tree-bound Base
  index and publish only their committed/dirty private Delta. The binding is
  persisted per worktree, stays pinned when the primary checkout advances,
  and later worktrees reuse a verified Base without another checkout or full
  repository walk.
- Tracked Delta paths intentionally excluded by discovery policy (for example
  `.gitattributes` or `.github/workflows/ci.yml`) now persist both metadata and
  content identities without entering the code graph. Metadata-only changes
  therefore fall back to the content hash instead of starting a false
  Store-CoW freshness loop. Freshness validation also excludes structural
  Folder nodes from its file-owned private-path invariant.
- A stale query joining a background refresh now waits through the launcher's
  pre-lock window. It no longer mistakes the still-existing old graph for the
  new publication, prints `graph refresh published`, and immediately retries
  against the same stale generation.
- Store-CoW visibility diagnostics now report the exact symmetric difference
  between the indexed and live Delta manifests. A detached-HEAD move no longer
  labels every Base-relative path as changed when only one path differs.
- Typed caller/path traversal now constrains each Base/Delta edge layer by its
  indexed logical endpoint before composing visible node ids. It no longer
  scans the complete overlay edge view once per BFS node, and call-site lookup
  reads only files reached by candidate paths instead of sorting every raw
  edge in the project. `path --code` now explains that path is call-site-only
  and gives the exact `greppy read SYMBOL` recovery; the agent contract no
  longer advertises that unsupported combination.
- `index status` is lock-free while an index writer is active and reports the
  foreground phase, completed/total spans and ETA when available. Semantic
  search against an incomplete embedding generation now returns temporary
  exit 75 with an explicit `index status --json` retry condition.
- Cold graph builds publish real discovery, extraction, graph-write and
  structural phases with phase-local file/folder/edge counts. Structural
  edges are committed in one transaction instead of one transaction per
  edge, removing the multi-minute structural tail on large repositories.
  Active work is no longer mislabeled as stalled at zero percent. Cold model
  loading uses its own bounded stall threshold rather than the ordinary
  two-minute graph threshold. A linked worktree waiting on
  another immutable-Base publisher now stops after a bounded five seconds
  with exit 75, the exact builder-lock path and retry guidance instead of
  blocking silently in `flock` under the stale `preparing_base_checkout`
  phase.
- Immutable-Base publication no longer synchronously generates thousands of
  derived text summaries. The fully validated graph and embeddings publish
  immediately with a model-bound empty summary cache; requested summaries are
  generated lazily in the private workspace cache. A busy summary daemon can
  therefore no longer discard an otherwise healthy Base, and genuine Base
  failures persist their exact terminal error in `index status --json` instead
  of the generic `background index exited before successful publication`.
- First-use graph navigation starts indexing as one observable background job
  and waits at most two seconds for publication. Large repositories therefore
  return retryable exit 75 instead of hiding an unbounded index walk. Job state
  is published before process launch, immutable-Base child progress is relayed
  under the outer owner PID, and a status record with no update for two minutes
  is marked potentially stalled with bounded recovery guidance.
- Stale graph navigation no longer hides model loading or a large full-store
  rebuild inside `read`, `who-calls`, or another query. Vector-backed and large
  stores start one observable background refresh and return bounded retry
  guidance; small graph-only edits retain fast inline healing.
- `bash-smart` now emits a liveness heartbeat after 15 seconds of silent child
  execution and once per minute thereafter, including the direct child PID and
  elapsed time, so active `cargo`, `rustfmt`, and test runs are not mistaken for
  dead wrappers while their output is being compactly captured.
- Background first-use indexers run in an independent OS process group (and a
  detached no-window class on Windows), so an agent runner can return the
  bounded retry response without its process cleanup killing the cache build.
- A foreground `greppy index` now prints an immediate `preparing_base` line
  with its PID and the exact nonblocking status command, so a cold build on a
  large repository is never indistinguishable from a silent orphaned process.
- `bash-smart` no longer opens writable graph-pack storage while an indexer
  owns the workspace lock; the requested process starts immediately and only
  optional expansion storage is skipped. Implicit grep pipelines now allow a
  five-second producer-start window, preventing loaded but valid producers
  such as `ps` from being misreported as missing stdin.
- Exact `read-file --all` and `read-file --lines` operations no longer open or
  migrate the graph store, so they remain available during concurrent first
  indexing and cannot collide with the indexer's schema publication.
- Paginated `read-file` creates only its small continuation-pack store on a
  cold repository; it no longer starts a graph build or embedding job merely
  to return the next-page handle. Exact reads resolve the repository once and
  derive graph identity lazily, avoiding repeated filesystem walks under load.
- Linked-worktree root validation is structural on Windows and accepts the
  equivalent long-path/8.3 spellings emitted by the OS without weakening the
  rejection of real partial-subdirectory indexing.
- Literal-search text rows emit lossless qualified locators accepted by
  `greppy read`; enclosing provider nodes no longer produce a short name that
  cannot be read back.
- Edit verification selects a local verifier for the touched language instead
  of an unrelated root manifest, streams start/still-running/final state,
  never downloads a checker, and terminates the verifier process tree at a
  bounded timeout with an actionable direct command.
- Grep/rg passthrough tolerates realistically delayed pipeline producers. An
  explicit `-` remains the unbounded, byte-exact contract for slow producers,
  while an idle implicit stdin still fails with bounded recovery guidance.

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

### Web runtime and next feature line (0.4.0)

The 0.3.x line after 0.3.4 is reserved for bounded stability fixes. Interactive
web-runtime work and subsequent feature development converge on **0.4.0**;
there is no separate 0.3.5 web feature release. The current source remains an
experimental web-runtime spike until its own gates below are green; this is
not a Playwright-compatibility claim.

- The terminal UI now uses Ratatui 0.30.2 and one aligned Crossterm 0.29
  runtime. This removes the transitive vulnerable `lru` 0.12 implementation
  (GHSA-rhfx-m35p-ff5j) in favor of 0.18.3 without changing Greppy's TUI API;
  the complete CLI test suite and Clippy remain green.
- Synthetic pointer input now has an engine receipt for every move, press, and
  release. A fresh WebView whose display list is not yet available is repainted
  and retried instead of silently dropping the event; exhausted retries fail
  the command explicitly. This separately fixes the long-lived-runtime input
  loss tracked as Fund 034.
- Pressing Enter in an eligible form field now performs the HTML implicit-submit
  action through the native input path. This is the distinct Fund 038 fix and
  is not attributed to the pointer-delivery retry above.
- `web.check` and `web.uncheck` now use the same acknowledged native activation
  path as `web.click`, verify the resulting checkbox state, and fail explicitly
  if activation did not take effect. This fixes Fund 033: DOM state and page
  application state can no longer silently diverge because checkbox events were
  skipped; an already-matching state remains an intentional event-free no-op.
- Web-runtime release evidence now binds the exact source commit, records a
  dirty-tree bit and rejects production signing from a dirty checkout. The
  built runtime itself is inspected for the separate Fund 034 and Fund 038
  contracts, while a dedicated exact-SHA gate requires three serial passes of
  the complete session-daemon suite instead of treating one load-sensitive
  failure snapshot as a stable defect list.
- Cancelling the interactive agent during its visible startup phase now returns
  exit 130 deterministically. The terminal UI's cancellation result is no
  longer discarded in a race with a worker that can finish cleanly first.
- An ordinary web action deadline now returns the public timeout error (exit
  35) with a partial artifact without poisoning the session. A follow-up run
  on that same session succeeds instead of reporting the internal
  `Failed -> Busy` transition previously tracked as Fund 030.
- Release artifacts now carry the stamped web-runtime distribution beside the
  CLI on macOS, Linux, and Windows, including its executable, checksums, SBOM,
  provenance, coverage manifest, and signature-state marker. Package smoke
  tests resolve that exact sibling distribution through `web doctor`; Windows
  uses the native `web-runtime.exe` member instead of a Unix-only filename.

Local gates landed in this stream (exact tests, not inventory shrink):

- Layout actionability pumps the Servo event loop between samples. A moving
  locator target times out with `failed_check=stable` and is not clicked;
  a later-stable target is clicked only after two matching samples. Pump
  tokens are worker-owned monotonic nonces. Leftovers are reclaimed on the
  next engine call and after a successful wait. Repeated timeout cycles then
  recovery must leave `window.__greppyPump*` empty (`locator_click_waits_for_actionable_target`).
- The requested action timeout is preserved. Controller RPC adds explicit
  watchdog headroom so the engine can return the named actionability error.
- Closed Page, Context, and Browser throw `object_disposed`. `context.close`
  disposes descendant pages; sibling contexts stay live. Server-side
  `params.generation` is compared against stored Live/Disposed generation
  (`closed_page_and_browser_throw_object_disposed`).

Keyboard `type` / `insertText` / `press` follow a measured Chromium oracle
(isolated from fill/click setup): `insertText` is
`beforeinput` → `textInput` → `input` (no key events); `beforeinput.preventDefault`
blocks mutation; `keydown.preventDefault` still emits keyup on `type`. Remaining
gaps (IME, `isTrusted`, full layout, repeat, mac editing commands) stay
unsupported rather than claimed (`keyboard_down_and_up_are_separate_events`).
`Page.setViewportSize` stays fail-closed, including the current 800×600 size,
because the Servo renderer does not resize (`set_viewport_size_is_unsupported`).
Nested `frameLocator` / `childFrames` stay fail-closed. Child `Frame.tap` stays
on the touch path, not click.
`GREPPY_WEB_FIXTURE_URL` is no longer a silent CLI production path.
`Page.waitForFunction` / `Frame.waitForFunction` / `Locator.waitForFunction`
wait inside the content worker while pumping Servo (no 20ms controller poll).
`hydrated_spa_wait_for_function_sees_async_dom_update` requires the wait to
outlast an 80ms page timer and to run script-interval ticks.
Non-macOS workers refuse to start unsandboxed instead of `Ok(())` with no profile.
Engine RPC and content-worker action timeouts parse JSON integers and V8
`f64` numbers (`timeout_ms_from_json`); invalid values use the default and
the result is clamped. `page.waitForFunction({ timeout: 250 })` therefore
returns `timeout: waitForFunction` instead of racing the controller watchdog
(`timed out after 250ms`). Default `polling: 'raf'` follows pure JS heap
state (no DOM mutation required); the waiter is a chained rAF plus one
completion nonce, one slot-read, and cleanup on every path.
Relative ESM modules are a supervisor-mediated capability: the canonical
entry file's parent is a bounded script root, staged into an isolated
per-request temp path, with symlink and parent-directory escapes skipped
and deleted after the run. Host filesystem and Node builtins stay denied.
CommonJS `require("playwright")` is granted; `require("fs")` stays denied
(`relative_esm_inside_script_root_is_granted`).
JSON modules under the staged script root are granted only with
`with { type: "json" }` (`json_module_inside_script_root_is_granted`);
absolute `file://` JSON imports stay denied.
`Page.waitForEvent` for `pageerror` / frame / `requestfailed` /
`requestfinished` / `request` is controller-side `once` plus a one-shot
timeout, not a 20ms poll. After `goto`, network settlement waits on the
content-worker Condvar (`page.waitForRequest`) instead of raster-polling.
Sensitive artifact manifests omit model-facing `text`/`html`/`bytes` and
keep digest/path/label (`sensitive_model_facing_ref_omits_raw_bytes_and_keeps_digest_path_label`).
Claimed-entry source receipts live under `contracts/web-runtime/receipts/`
and must not set inventory `behavior:passing`.
Linux Landlock live deny and same-image `fexecve` stay OPEN until an
ubuntu-latest job log shows the deny/fexecve path. The reproducible gate is
`.github/workflows/web-runtime-linux-sandbox.yml`. Preliminary Linux
container evidence uses `rust:1.93-bookworm`; the pinned 1.95/Ubuntu-CI
receipt remains OPEN.

Debug `cargo test` / `cargo build` copies digest-verified GGUF/tokenizer
files from compile-time repo paths instead of `include_bytes!` into
`__TEXT`. That is a **dev/test profile optimization only**. It does not
close a production one-binary or signed/notarized-release cold-start gate.
Release still bakes the product models. `ci-test-assets` stays debug+CI
only. Signed/notarized release first-exec remains OPEN until measured.
Workspace `[profile.dev.package.sha2] opt-level = 3` speeds debug SHA-256
of the GGUF in `build.rs` (the `sha2` crate only). It is not a blanket
`build-override` (that would rebuild every proc-macro/build script at
opt-level 3). Other debug crates and release profiles are unchanged.
Digest verification is not skipped.

`setExtraHTTPHeaders` extra-header matrix passed `extra_http_headers_are_sent_on_goto`
(1/1, 53.66s) with extras on the wire for HTTP and HTTPS document, `/sub.js`
subresources, 302 `/jump` → `/landed`, and isolation (`tagged=false` on a
sibling context). Engine path is vendored `WebResourceLoad::continue_with_headers`
in `main_fetch` above TLS; PolicyProxy is pin/deny only. `spin_until` pumps the
Servo event loop so Continue loads can leave `LoadStatus::Started`. Goto no
longer sleeps 20ms after `Complete`; scripts are waited by `waitForFunction`.
`Deno.core.print` is not a controller logging channel (it is the engine IPC
stdout). Product label stays experimental web-runtime spike.

`Route.continue({ headers })` passed `route_continue_sends_extra_headers_on_http_and_https`
(1/1, 24.00s) with TLS `/continued` and `/sub.js` tagged. `url` / `method` /
`postData` stay fail-closed. The JS handler still runs at `page.route()`
registration and stores a static `RouteRule` (not per-request Playwright
callbacks).

`one_thousand_session_create_close_cycles` passed (1/1, 28.03s). Session ids
include a monotonic sequence so two creates in the same nanosecond cannot
collide. `web.session.close` publishes the session snapshot after remove.

Local runtime gates re-run after those fixes (exact, `--test-threads=1`):
`idle_supervisor_workers_are_not_cpu_hot` 20.89s, `idle_sessions_are_reaped`
19.29s, `worker_sandbox_denies_host_secret_paths` 23.64s,
`content_worker_crash_is_recovered_without_hanging` 17.50s,
`compatibility-inventory` 3/3 in 0.16s, `process-boundary` 7/7 in 29.79s,
`observe_read_search_research_screenshot_and_policy` 32.30s,
`twenty_independent_playwright_scripts` 186.39s (20 **repo fixtures**, not
the DoD's 20 independent third-party Playwright scripts).
`one_thousand_session_create_run_close_cycles` is active and passes in the
single-threaded session-daemon suite: 1000 create+run+close cycles complete
without the former harness abort or mozalloc SIGSEGV. The product 10s
`web.run` deadline is unchanged; the leak harness uses a 45-minute watchdog
so the full lifecycle gate can complete.

`web_doctor_json_does_not_spawn_runtime` 1/1 in 9.66s (bound 30s). The test
now isolates `GREPPY_STORE_DIR` so a skip-GC miss cannot scan the real
store. Doctor remains facts-only and does not spawn workers.

External blockers (codesign/notary, Windows-Linux CI, 8h soak, SBOM signatures,
human security review, 20 independent third-party Playwright scripts, size/RSS
vs Playwright+Chromium, macOS fexecve, `web-engine-servo` crate split) stay
unchecked.

### Interactive agent TUI (0.3.4)

`greppy agent` is a production interactive TUI on the same isolated
CoW/native worktree, sandbox, and `refs/greppy/agent/<run-id>` proposal as
`greppy -p`. The terminal event loop stays on the main thread; model
streaming and tools run on a worker. Token and thinking deltas are coalesced
through a capped buffer so a slow renderer cannot grow memory without bound.
Conversation history is preserved across prompts without cloning it on every
streamed token.

The surface is a one-line header (agent, repository, branch/worktree, model,
sandbox), a Markdown transcript with tool rows and collapsed thinking, a
grapheme-aware multiline composer, and a status footer with activity, tokens,
turns, and queued follow-ups. Submissions while the agent is busy become
visible queued follow-ups. Enter submits; Shift+Enter / Alt+Enter insert a
newline when the terminal reports those modifiers. PageUp/PageDown scroll by
the viewport. Follow-tail is automatic only at the bottom; End or a new
prompt restores it.

Slash commands: `/help`, `/clear` (confirms before discarding visible
context), `/model`, `/usage`, `/tools`, `/copy` (OSC 52 with a fallback
message), `/sessions`, `/name TITLE`, `/compact`, `/exit` `/quit` `/q`.
Ctrl+C cancels at a safe tool boundary and never interrupts an in-flight
edit; a second Ctrl+C exits after RAII terminal restoration (raw mode,
alternate screen, mouse, bracketed paste, cursor, title). Non-TTY stdin or
stdout refuse full-screen mode without emitting control sequences. `NO_COLOR`
and an ASCII fallback are honoured. Below 60×18 the UI shows a stable
"terminal too small" view and recovers on resize.

Sessions persist as versioned append-only JSONL under the Greppy data root
(`agent-sessions/<project>/`). `--continue` restores the latest project
session; `--resume SESSION_ID` restores a named one. A truncated tail
recovers the valid prefix. Persistence failure warns and keeps the in-memory
conversation. `/compact` retains recent messages plus an extractive summary.
Secrets and authorization headers are redacted from display and disk.

`--apply`, `--diff`, `--keep-worktree`, `--fresh`, `--workspace-backend`,
`--no-sandbox`, `--skip-selfcheck`, deadline, and token/turn limits remain
valid in interactive mode. `greppy -p` stdout/stderr/exit-code, sandbox,
workspace, proposal-ref, and apply behaviour are unchanged; `greppy -e -p`
still reaches grep passthrough.

The agent loop honours a cooperative cancel flag at the same safe boundaries
as the wall-clock deadline (between turns, after a stream, after a tool
returns) and exposes `LoopStop::Cancelled`.

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
