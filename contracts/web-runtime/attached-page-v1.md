# Explicit existing-page bindings (implementation in progress)

The planned opt-in `web pw --tab TAB` must continue on a named live page of the
selected session. The existing free-script path continues creating its own
page. Neither path may silently choose another page, reopen the URL, replay
previous actions, or claim to roll back already delivered actions.

Current implementation: `runtime/js/attached-page-bindings.js` validates a
versioned native descriptor and constructs the existing Browser, BrowserContext
and Page facade classes around its identities. Five component tests pass,
including native-call interception of the actual facade source: reconstruction
makes zero engine calls; title and keyboard operations then target the selected
existing page, with its generation on the locator operation. Sibling pages,
context relationships and generations are retained. Invalid graphs are rejected
before any constructor runs. No active CLI flag or daemon operation is wired yet.

Descriptor shape:

```json
{
  "schema": "greppy.web.attached-page.v1",
  "selected_page": "page-selected",
  "browser": {"id": "browser-owned", "generation": 1},
  "contexts": [{
    "id": "context-owned", "generation": 1,
    "pages": [{"id": "page-selected", "generation": 1, "url": "https://example.invalid/work"}]
  }]
}
```

IDs remain opaque and globally unique within the descriptor; the selected page
must appear exactly once. Missing IDs, wrong selection, unsupported schema,
invalid generations or incomplete URL fields are errors. Strings originating
in page content do not decide ownership or error recovery. This descriptor is
internal tool state, not text to be echoed into every successful response.

The native implementation still must:

1. Resolve and authorize the explicit session/tab before script execution.
   The daemon must supply the allowed page scope; the script must not be trusted
   to authorize itself. A foreign, closed or changed binding is an error.
2. Obtain browser/context/page identities from live engine state. CLI-created
   pages currently have optional parent IDs, so merely instantiating a Page
   wrapper around a guessed ID cannot establish a coherent context. Supplying
   facade metadata must not create another WebView or navigate the document.
3. Preserve the full relevant identity graph and lifecycle. Tab creation and
   closure during a bound script must reconcile with the same session tab list;
   explicit close calls remain real actions. Automatically closing borrowed
   browser/context/page objects at snippet completion is prohibited.
4. Pass returned values through a typed completion channel. The existing PW
   wrapper searches exception text for `PWRESULT`; that mechanism is under
   separate negative-test investigation and must not establish success for the
   new path.
5. Exercise CLI fill/click → bound script → CLI observe on the same unsaved
   state, including inactive tabs, another session, stale handles, explicit
   close, navigation, script errors and subsequent usability. Check that no
   new page or navigation was introduced merely by crossing the interface.

Component tests do not prove native ownership, Servo behavior, preservation of
real application state, or lower provider token usage. The descriptor helper
remains disconnected until the native lifecycle and result contracts are
implemented and verified. Existing prepared-script performance and the paired
Luna input/output/time gates remain mandatory.
