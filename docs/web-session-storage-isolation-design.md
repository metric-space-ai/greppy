# Web session storage isolation design checkpoint

## Contract and regression

An ordinary `greppy web session create --profile project` must create an
ephemeral browser storage boundary. Tabs in one session share cookies and web
storage. Concurrent ordinary sessions do not share cookies, localStorage,
IndexedDB, or HTTP cache, and closing one session cannot be the mechanism that
protects another already-open session.

`ordinary_sessions_isolate_cookie_state` in
`crates/web-runtime/runtime/tests/session-daemon.rs` is the smallest native
regression. It creates A and B before A sets a cookie, verifies a new tab in A
receives it, verifies B does not, closes A, and verifies new C does not. The
exact selector is:

```text
cargo test -p web-runtime --test session-daemon ordinary_sessions_isolate_cookie_state -- --exact
```

The installed-binary equivalent is retained outside this disposable worktree
as `installed-cookie-isolation-baseline.py` in the task evidence directory. It
uses only normal `session create`, `open`, `tab new`, `goto`, `js`, and
`session close` commands against a loopback HTTP fixture.

## Demonstrated source gap

`ContentEngine` currently owns one `Servo`, one `SharedProfile`, one
`PolicyProxy`, and one `UserContentManager`. `browser.newContext` allocates an
ID and `ObjectLife`, while `context.newPage` always constructs its webview from
that single engine-wide Servo. `session.ensurePage` reaches `context.newPage`
without a storage-context parameter. A newly created CLI session is therefore
an ownership record, not a network/storage partition.

Servo constructs one public and one private resource/storage thread pair per
`Servo` instance. `WebViewBuilder` accepts a Servo but exposes no arbitrary
cookie/storage partition key. Private-browsing mode supplies only one alternate
global store and cannot isolate an arbitrary number of concurrent sessions.
Header interception is insufficient because script-visible cookies,
localStorage, IndexedDB, and cache would remain shared.

## Proposed implementation boundary

Make each ordinary browser context own a complete Servo bundle: its Servo,
profile, policy proxy, and user-content manager. Bind every daemon Session to
one engine context when the session is created or first materialized. Every tab
in that Session uses the same bundle. A different Session uses a different
bundle, including while both are live. Context disposal closes all of its pages
and drops the bundle.

The content worker then needs to route every page operation and event-loop pump
to the Servo that owns that page. A mechanical `spin_all_event_loops` fallback
would preserve progress but expands the scheduling surface; page/context-aware
routing is preferable where the operation already names a page. Engine-global
operations must iterate live context bundles deliberately.

Persistent profiles require a separate explicit contract. They may select a
durable storage root, but they must not cause ordinary unnamed sessions to
share a live context. This checkpoint does not claim that existing persistent
profile filesystem bookkeeping already provides Servo storage persistence.

## Implementation risks to resolve

- Confirm multiple Servo instances can coexist on the content-worker thread;
  the vendored builder initializes substantial process-global machinery even
  though storage/resource threads are instance-owned.
- Move `session.setProfile` from engine-global state to the bound context.
- Add a context owner field to each `PageSlot` or an equivalent reliable map so
  waits, evaluation, navigation, screenshots, and cleanup pump the right Servo.
- Preserve the current same-session tab behavior and generation/disposal
  checks.
- Exercise cookie plus at least one script storage primitive after the cookie
  regression passes; do not accept a cookie-only transport workaround.

## Verification state

This checkpoint was authored by `gpt-6-astra`. Only source review and
`cargo fmt --all -- --check` have run. No native compile, native test, installed
browser probe, build, model load, or index was started in this task while the
shared heavy lease was occupied. The draft PR is intentionally not ready to
merge until the production isolation boundary and admitted native/installed
verification are completed.
