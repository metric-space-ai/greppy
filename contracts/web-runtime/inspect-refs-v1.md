# Native inspection of observed refs

`greppy web inspect @1 --session SID [--tab TAB] [--attrs] [--html] --json`
uses the same session-bound snapshot resolver as native actions. It does not
turn the ref into a guessed CSS selector or simulate a click. Disabled nodes
are readable. `web.inspect` is also advertised by the runtime handshake.

The runtime request has `selector: {"type":"ref","value":1}`, `session_id`,
optional `tab_id`, and optional boolean `attrs`/`html`. The daemon binds the
selector to that session's observed page/snapshot before engine execution.
An explicit tab must belong to the session; it does not change the active tab.
Without one, the session's active page is used. CLI query-based inspection
also accepts the same explicit tab option.

The result keeps the query-inspection shape: `value.count`, `value.node`,
optional `value.html`, plus `serialized`, `session_id` and the existing
untrusted-content boundary. The DOM descriptor is shared by the CLI query
and native-ref paths so fields do not drift independently. Inspection returns
page-derived data, not proof that an application operation succeeded.

`@0` or malformed refs fail with `QUERY_SYNTAX` before starting a runtime.
A missing snapshot, wrong page/session, navigation, or replaced node fails
with `STALE_REF` and observe-again guidance. An unknown/cross-session explicit
tab fails with `TAB_NOT_FOUND`, without falling back to the active page.

Observation retains at most the current snapshot's 200 actual node identities.
A DOM clone retaining `data-greppy-ref` is not the observed node and cannot
inherit its ref. This identity check also protects native actions. It is a
stale-target safeguard, not a security boundary against a page manipulating
its own JavaScript environment. Ref numbers still refer to the latest
observation in the selected session; they are not durable global IDs.

This change removes the unsupported-ref/JS-query detour for inspect. It does
not fix the separate active Playwright context problem or establish a token,
latency, or task-success improvement; those require independent trials on a
matched Release CLI/runtime pair.
