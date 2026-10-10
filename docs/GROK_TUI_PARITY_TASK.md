# Greppy 0.3.4: production-grade interactive agent TUI

Implement and verify a production-grade, independently designed interactive
TUI for Greppy's integrated coding agent. Work only from this specification and
the existing Greppy repository. Do not inspect, copy, translate, or derive from
third-party agent source code or restricted reference repositories.

## Repository rules

- The checkout is dirty and contains unrelated concurrent Web Runtime work.
  Preserve every pre-existing change. Never reset, restore, stash, clean, or
  rewrite unrelated files.
- Use `greppy` for every code-navigation step as required by `AGENTS.md`.
- Run every build, test, lint, and formatting check through
  `greppy bash-smart -- CMD`.
- Do not commit, push, or create a PR. Leave the verified implementation in the
  current checkout and report exactly which files you changed.
- Work from the existing TUI baseline. Relevant starting points are:
  - `crates/cli/src/agent_tui.rs`
  - `crates/cli/src/agent.rs`
  - `crates/agent/src/agent_loop.rs`
  - `crates/agent/src/protocol.rs`
  - `crates/cli/src/cli_surface.rs`
  - `crates/cli/Cargo.toml`
  - `README.md` and `CHANGELOG.md`
- `greppy -p "TASK"` is a stable headless interface. Preserve its stdout,
  stderr, exit-code, sandbox, workspace, proposal-ref, and apply behavior.
- `greppy agent ["INITIAL TASK"]` is the interactive entry point.

## Product objective

The result must feel like a mature terminal coding agent rather than a thin
chat box. It must remain recognizably Greppy: graph-first tool activity,
isolated worktree/proposal semantics, restrained visual hierarchy, fast native
rendering, and no marketing-style decoration.

## Required architecture

1. Separate terminal lifecycle, application state/update logic, rendering,
   input editing, transcript rendering, command handling, and agent-worker
   bridging into focused modules. Do not grow another monolithic file.
2. Keep the terminal event/render loop responsive while model streaming and
   tools execute on a worker thread.
3. Use bounded channels or explicit event coalescing so token/thinking deltas
   cannot grow memory without limit when rendering is slower than the model.
4. Preserve full conversation history across prompts. Avoid cloning the full
   history for every streamed delta.
5. Add cooperative cancellation. Ctrl+C during a run requests cancellation at
   a safe boundary and must never interrupt an edit halfway through. A second
   Ctrl+C exits after terminal restoration.
6. Use RAII terminal restoration for normal exit, errors, disconnects, and
   panics: raw mode, alternate screen, mouse capture, bracketed paste, cursor,
   and terminal title must all be restored.
7. Do not introduce an async runtime solely for the UI unless the measured
   design requires it. Prefer the repository's existing synchronous agent
   architecture and small, explicit concurrency primitives.

## Required visual surface

### Header

- Compact one-line identity: Greppy agent, repository, branch/worktree state,
  selected model, and sandbox state.
- No large banners or decorative boxes.
- Degrade cleanly on narrow terminals by dropping secondary fields before
  truncating the repository/model identity.

### Transcript viewport

- Distinct but restrained user, assistant, thinking, tool, warning, and error
  treatments that work in dark and light terminal themes.
- Stream assistant text incrementally without layout jumps or duplicated text.
- Render Markdown structure: paragraphs, headings, lists, inline code, fenced
  code blocks, and links. Syntax-highlight code when practical; fall back to a
  readable code style without failing.
- Tool rows show running/success/failure state, the Greppy command summary,
  elapsed time, and a bounded output preview. Tool details can be
  expanded/collapsed from the keyboard.
- Thinking is collapsed by default but visibly active while streaming and can
  be toggled for the current response.
- Auto-follow the stream only while the viewport is at the bottom. Manual
  scrolling disables follow; End or a new submitted prompt restores it.
- Support keyboard and mouse-wheel scrolling. PageUp/PageDown must scroll by a
  viewport-relative amount, not a hard-coded line count.
- Selection/copy must remain usable in terminals where mouse capture is
  disabled or bypassed with a modifier.

### Composer

- Real multiline, Unicode-safe editor with grapheme-aware cursor movement,
  Home/End, word movement, Backspace/Delete, line navigation, history, and
  bracketed paste.
- Enter submits. Shift+Enter and Alt+Enter insert a newline where the terminal
  reports those distinctions.
- The composer grows to a bounded height and then scrolls internally.
- While the agent is busy, new submissions become visible queued follow-ups;
  they are not silently rejected or lost.
- Show a compact completion menu for slash commands after `/` and input
  history navigation when the composer is otherwise empty.

### Footer/status

- Current activity, model, cumulative input/output/cache tokens, turn count,
  queued-message count, and stop/error state.
- Spinner animation must not repaint at an excessive rate when nothing else
  changes. Target at most 20 frames per second; idle UI should not busy-loop.
- Never expose API keys, authorization headers, full environment variables, or
  unredacted sensitive tool arguments.

## Required interactions

- `/help`: modal/overlay listing commands and active key bindings.
- `/clear`: clear only the visible transcript after confirmation when it would
  discard unsaved conversational context.
- `/model`: searchable model selector populated from the configured gateway or
  current known model when discovery is unavailable.
- `/usage`: detailed cumulative usage for the session.
- `/tools`: show tool executions and allow expanding a selected row.
- `/copy`: copy the last assistant response using OSC 52 when supported, with
  a clear fallback message.
- `/exit`, `/quit`, `/q`: finish safely, restore the terminal, and publish the
  normal Greppy proposal outcome.
- Ctrl+C: safe cancel while running; exit while idle; double Ctrl+C requests
  exit after the current non-interruptible tool boundary.
- Esc: close the top overlay, then clear completion state, then return focus to
  the composer.
- Tab/Shift+Tab: navigate completion/overlay choices without inserting literal
  tabs unexpectedly.
- Resize events must reflow content without panics, overlap, or losing the
  user's scroll position.

## Session behavior

- Persist interactive sessions in a versioned, append-safe format beneath the
  existing Greppy data root. Never write secrets or raw authorization headers.
- Restore the most recent project session with `greppy agent --continue` and
  allow `greppy agent --resume SESSION_ID`.
- Persist model, messages, usage, stop state, proposal/worktree identity, and a
  user-visible title. A truncated/corrupt tail must recover the valid prefix
  and report the recovery.
- Add `/sessions` as a searchable session picker and `/name TITLE` to rename the
  current session.
- Session persistence failure must not lose the active in-memory conversation;
  display a warning and continue.
- Context growth must be bounded. Add an explicit `/compact` operation or a
  documented safe compaction policy that retains recent messages and a summary.

## Workspace and proposal semantics

- One interactive session uses one isolated Greppy agent workspace.
- Every tool call remains confined to that workspace and existing sandbox
  roots. The TUI must not add broader filesystem or network permissions.
- On clean exit, publish the same `refs/greppy/agent/<run-id>` proposal used by
  headless mode. Show the ref, stat, inspect command, and apply command after
  leaving alternate-screen mode so the information remains in scrollback.
- Preserve `--apply`, `--diff`, `--keep-worktree`, `--fresh`,
  `--workspace-backend`, `--no-sandbox`, `--skip-selfcheck`, deadline, and
  token/turn limits in interactive mode.
- A failed or cancelled session keeps the worktree when the existing headless
  contract would keep it.

## Accessibility and terminal compatibility

- Respect `NO_COLOR` and non-TTY output. Refuse full-screen mode with a concise
  diagnostic when stdin/stdout are not suitable TTYs; never emit raw control
  sequences into redirected output.
- Use terminal capabilities rather than assuming true color, Unicode symbols,
  mouse support, clipboard support, or distinguishable Shift+Enter.
- Provide an ASCII-safe rendering fallback.
- All visible strings must fit at 120x36, 80x24, and 60x18 without overlap or
  underflow. Below the supported minimum, show a stable "terminal too small"
  view and recover automatically after resize.

## Tests and evidence

1. Pure state/update tests for every command, key path, queue transition,
   cancellation state, follow-tail transition, and worker disconnect.
2. Ratatui `TestBackend` snapshot/golden tests at 120x36, 80x24, and 60x18 for:
   idle, streaming text, thinking, running tool, failed tool, scrolled
   transcript, command overlay, model/session picker, queued follow-up, error,
   and terminal-too-small states.
3. Unicode/grapheme tests covering combining marks, emoji sequences, CJK width,
   multiline paste, and deletion at boundaries.
4. PTY integration tests with a deterministic local mock gateway covering
   startup, streaming, tool start/finish, follow-up prompt, resize, cancellation,
   clean exit, terminal restoration, and proposal publication.
5. Regression tests proving `greppy -p` output and routing remain unchanged and
   `greppy -e -p` still reaches grep passthrough.
6. Failure tests for worker panic/disconnect, event-channel saturation, session
   write failure, corrupt session tail, unsupported terminal, and tiny resize.
7. Provide deterministic PNG previews of the actual renderer at 120x36 and
   80x24 under `docs/assets/tui/`, plus the command that regenerates them. The
   previews must be generated from the same state/view code used in production,
   not from a separate mock layout.

Run at minimum:

```text
greppy bash-smart -- env CI=true cargo check -p greppy --features ci-test-assets,cpu-only
greppy bash-smart -- cargo test -p greppy-agent
greppy bash-smart -- env CI=true cargo test -p greppy --lib --features ci-test-assets,cpu-only
greppy bash-smart -- cargo fmt --all -- --check
greppy bash-smart -- env CI=true cargo clippy -p greppy-agent -p greppy --features ci-test-assets,cpu-only -- -D warnings
greppy bash-smart -- git diff --check
```

If unrelated concurrent changes make a whole-workspace command fail, isolate
the relevant package/test and report the unrelated blocker precisely. Do not
modify the unrelated files to make the check pass.

## Definition of done

- The required behaviors above are implemented, tested, and documented.
- The two deterministic screenshots exist and match the production renderer.
- No terminal state leaks after success, cancellation, panic, or error.
- No `greppy -p` regression.
- No unrelated dirty-worktree changes are reverted or reformatted.
- `README.md`, `CHANGELOG.md`, and `greppy agent --help` document the final
  behavior accurately, without claiming features that are incomplete.
- Final report starts with remaining gaps or failed checks, then lists changed
  files, tests run, screenshot paths, and exact commands for manual review.

