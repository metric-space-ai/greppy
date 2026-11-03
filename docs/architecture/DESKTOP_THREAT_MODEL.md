# Greppy desktop threat model

Status: binding for the `greppy desktop` MVP. This is not a claim that driving
the host GUI is safe. It records the boundaries the adapter actually enforces.

## Scope

In scope:

- `greppy desktop` and the `greppy-desktop` crate;
- the pinned `agent-desktop` crates (tag `v1.1.0`), called only from
  `crates/desktop/src/upstream.rs`;
- the built-in agent's decision to include the desktop manual and to allow the
  command;
- session logs, event previews, and the write sandbox.

Out of scope for this MVP, and therefore not implemented:

- clipboard, drag, raw mouse moves, and notifications;
- a claim that AT-SPI, AX, or UIA isolate one application from another;
- sandboxing the accessibility bus itself. The write sandbox still only
  confines filesystem writes.

## Assets

| Asset | Handling |
|---|---|
| UI names, tree text, window titles | Untrusted, same as page text. Wrapped as `UNTRUSTED_DESKTOP_CONTENT` before the model sees a non-image result. |
| Screenshots | PNG bytes are an image part for the model. The session log stores `[image/png omitted from session log]`, not the pixels. |
| Session transcript and `--json` event previews | `redact_text` drops the interior of the desktop fence before persistence. |
| Secrets typed into a control | `--from-env` and `--value-stdin` keep the secret off the argv the session log stores. `--value-stdin` is omitted from the built-in manual because a tool call has no stdin. |
| Desktop state (`AGENT_DESKTOP_HOME`) | See the sandbox section. |

Desktop output is not written to the tracing log by greppy. The desktop child
is started with `GREPPY_LOG=error` and `RUST_LOG=error` so an upstream debug
trace cannot copy a tree onto stderr and into the tool result. An operator who
later raises that filter can still see upstream diagnostics; do not do that on
a machine whose UI is sensitive.

## Activation

The default system prompt names `greppy capabilities enable web` and
`enable desktop`, and says their output is greppy's own manual and counts as
part of the system prompt. Enable prints that manual as the tool result. The
system prompt is not rewritten, so its cache stays valid. Calling enable again
in the same session prints the same manual.

The enable result is not desktop or page content. It is not wrapped in
`UNTRUSTED_DESKTOP_CONTENT`. Snapshots, window names, and screenshots from
later `greppy desktop` commands stay fenced and are omitted from the session log.

`--enable desktop` (and `web`, including `web,desktop`) on `greppy -p`,
`greppy agent`, `greppy agent serve`, and `greppy agent stdio` includes the
manual in the prompt from the first turn and unlocks desktop immediately.

Inside a built-in agent (`GREPPY_AGENT_RUN`), `greppy desktop` exits 40 with
`desktop_disabled` until that session's `enable desktop` has succeeded
(`GREPPY_CAPABILITY_DESKTOP`). The unlock lasts for the agent process. A
desktop command typed at a normal shell is not gated: the operator already
has the GUI.

A build compiles one platform adapter. Linux links `agent-desktop-linux`
(AT-SPI, and therefore zbus and x11rb). macOS links `agent-desktop-macos`.
Windows links `agent-desktop-windows`. The other adapters are target-cfg
dependencies and are not compiled into that binary. `agent-desktop-core` is
shared and does not open a platform bus.

Web commands are unchanged aside from this prompt gate. They still run when
the manual is absent.

## Write sandbox and `AGENT_DESKTOP_HOME`

Sandboxed tool children may write only under the prepared roots: the worktree,
its private git dir, the per-run scratch directory, the isolated greppy data
root, Cargo's registry and git caches, and the run's browser socket directory.
Reads and the network stay open.

`agent-desktop` stores session files in `$AGENT_DESKTOP_HOME` when that
variable is an absolute, non-symlink directory owned by the current user, and
otherwise in `$HOME/.agent-desktop`.

Under `SandboxMode::Enforce`, greppy redirects `HOME` to `<scratch>/home` and
sets `AGENT_DESKTOP_HOME` to `<scratch>/agent-desktop`. That directory is
created by the parent and is inside the scratch root, which Landlock
(PathBeneath) and macOS Seatbelt (subpath) already allow. An inherited
`AGENT_DESKTOP_HOME` that points outside the grant is replaced for that child.
Widening the sandbox to follow an arbitrary path would let a tool write the
operator's real home; the override is the containment.

With `--no-sandbox`, the variable is left as the operator set it, and the
upstream default `$HOME/.agent-desktop` applies. That process can write
wherever the user can write.

The state directory must not be a symlink. Upstream rejects a symlink
`AGENT_DESKTOP_HOME`. The parent creates a real directory.

## Platforms

The Linux, macOS, and Windows adapters are compiled only for their OS. Any
other target returns `platform_not_supported` and exit 40 without calling an
adapter. A missing desktop session (no AT-SPI bus, no permission) is an error
document in `greppy.desktop.v1`, not a panic.

## Residual risk

The accessibility bus can read and act on other applications of the same user.
Greppy does not add an allow-list of bundle ids. A model that has been
prompt-injected by a window title can still click if desktop is enabled. The
fence tells the model the text is data; it does not prove the model will obey
it. Enabling desktop is a deliberate grant of the user's GUI.
