# Native post-action page state (E1)

Implementation status: in progress; not a release acceptance claim.

Successful native navigation and user-input actions retain their existing
result fields and add `result.page_state`:

```json
{
  "schema": "greppy.web.page-state.v1",
  "status": "available",
  "snapshot": {
    "url": "https://example.test/",
    "title": "Example",
    "untrusted_content_boundary": "UNTRUSTED_PAGE_CONTENT",
    "actionable_schema": "greppy.web.actionable.v2",
    "actionables": [],
    "ref_count": 0,
    "refs_truncated": false
  }
}
```

The snapshot uses the same bounded, credential-redacted observation contract as
`web.observe`, including text, headings, links and actionable state. It belongs
to the session and actual target page of the completed action, including an
explicit non-active tab. It is captured before that operation becomes idle;
it must not acquire a second session operation or reset the execution budget.
`available` means an observation was obtained, not that an application-level
expectation was satisfied. An empty actionable list can be a valid observation.

If the action succeeded but the observation failed, retain the successful
action receipt and return this instead:

```json
{
  "schema": "greppy.web.page-state.v1",
  "status": "unavailable",
  "error": {
    "code": "OBSERVATION_UNAVAILABLE",
    "message": "The post-action page could not be observed."
  }
}
```

Do not replay the action, imply rollback, or claim business success. Errors
must be redacted. An action that itself failed retains its original nonzero
error contract; this envelope must not turn it into a successful action.
Read-only inspection and explicitly requested script results keep their own
contracts rather than silently performing another observation.

## Reference identity

- A reference names an actual DOM node, not an array index in the latest view.
- Value, checked state and focus changes on that same node preserve its ID
  across automatic and explicit observations.
- Replacing a node, including cloning all attributes, does not transfer its
  reference. A new observation must never recycle that old ID for the clone.
- Navigation and document/frame replacement invalidate old references even
  after a new observation has been returned. Session/page ownership is checked
  independently; another tab must not capture a stale reference implicitly.
- Keep only bounded current observation targets strongly reachable. Persistent
  node-to-ID bookkeeping must not retain detached DOM nodes indefinitely.

## Required regression evidence

1. Same node: change value, checked state and focus, observe again, and use the
   original handle successfully against that node.
2. Replacement: retain a handle, clone/replace its node, observe again, and
   reject the old handle with `STALE_REF`; the newly returned handle works.
3. Navigation: observe the next document, then reject all old document handles.
4. Native `open`/`goto` and input actions return usable state without a separate
   caller-initiated observe. A click updates application state exactly once.
5. A deliberate post-action observation failure preserves the dispatch receipt,
   reports `unavailable`, and does not dispatch the side effect again.
6. Bound-tab, sensitive-value redaction, existing CSS/inspect/type/ref behavior,
   lifecycle budgets and unchanged error exits remain covered.

Functional green tests are not an efficiency or release gate. The independently
pinned agent comparison must measure the resulting calls/tokens separately;
no runtime refresh workaround or repeated unchanged comparison hides failures.
