# Servo 0.5.0 embedding gaps

Contract revision: 2026-08-26
Phase: 0 (inventory only)
Crate: servo 0.5.0
Git revision: 77fccacc1f1fdce10498d50173aafaa09d02879e
Archive SHA-256: 331e15df72165ca15b3945970c6870c4b7367be116ded058fda4f41190b265b8
License: MPL-2.0 (verified)
Rust-version fact: 1.88.0

This document is the Phase 0 embedding-gap inventory required by `docs/PLAYWRIGHT_INTERACTIVE_WEB_RUNTIME_GUIDE.md` section 12. It maps required engine behavior onto hooks the Greppy `WebEngine` trait needs.

## 1. Inspection record

Local inspection of servo 0.5.0 **succeeded** after the crate was fetched as a `crates/web-runtime` dependency. Extracted path:

`/Users/michaelwelsch/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/servo-0.5.0`

Phase 1 uses these embedding APIs from that crate:

- `ServoBuilder::default().build()` — `servo.rs`
- `SoftwareRenderingContext::new(PhysicalSize)`
- `WebViewBuilder::new(&Servo, Rc<dyn RenderingContext>).delegate(...).build()` — `webview.rs`
- `WebView::show`, `focus`, `load`, `paint`, `evaluate_javascript`, `notify_input_event`, `load_status`, `url`
- `Servo::spin_event_loop`
- `EventLoopWaker` — `servo-embedder-traits-0.5.0`

Accessibility tree updates exist (`WebViewDelegate::notify_accessibility_tree_update`) but the Phase 1 spike resolves role/label via page-realm JavaScript plus real `getBoundingClientRect()` layout, not AccessKit.

`present` below means the crate exposes a general-purpose API that the spike used. It does not mean Playwright behavior compatibility. Remaining table rows that the spike did not exercise stay as previously recorded until they are implemented.

## 2. Servo change process (guide §12.2)

Binding posture, unchanged by the lack of local source:

1. Prefer the released Servo embedding API.
2. If a required capability hook is missing: document the missing hook and the affected Playwright methods; implement the narrowest general-purpose Servo API; add Servo-level tests; upstream the change when practical; pin the accepted source revision; avoid a large untracked fork.
3. Greppy-specific policy, evidence, and Playwright semantics MUST remain outside Servo.
4. `playwright-compat` depends on engine traits, not Servo concrete types. The supervisor selects the concrete adapter.
5. Fabricated geometry is forbidden. Visibility, bounding boxes, hit testing, intersection state, `offset*`, and `getBoundingClientRect()` require real layout semantics.

## 3. Hard layout constraint

Layout MAY avoid continuous rasterization. APIs that depend on geometry still need a real layout tree:

- visibility
- bounding boxes
- hit testing
- intersection state
- `offset*`
- `getBoundingClientRect()`

Returning fabricated, guessed, or hardcoded geometry is forbidden. A stub that always returns a 0×0 box, a full-viewport box, or "visible: true" is a failed gate, not a compatibility implementation. This constraint blocks honest Locator actionability (guide §13) and therefore blocks the Phase 1 vertical spike if click/fill cannot wait for a real visible, stable, unobscured target.

## 4. Required `WebEngine` trait (guide §12)

The adapter MUST expose this engine-neutral trait. Playwright compatibility code MUST NOT call Servo internals directly. The signatures below are Greppy-side requirements from the guide, not Servo crate members.

```rust
pub trait WebEngine {
    fn create_context(&self, options: ContextOptions) -> EngineResult<ContextId>;
    fn close_context(&self, context: ContextId) -> EngineResult<()>;
    fn create_page(&self, context: ContextId) -> EngineResult<PageId>;
    fn navigate(&self, page: PageId, request: NavigateRequest) -> Operation<ResponseInfo>;
    fn evaluate(&self, frame: FrameId, request: EvaluateRequest) -> Operation<SerializedValue>;
    fn query(&self, frame: FrameId, query: QueryRequest) -> Operation<Vec<NodeRef>>;
    fn perform_action(&self, target: NodeRef, action: ActionRequest) -> Operation<ActionResult>;
    fn screenshot(&self, page: PageId, request: ScreenshotRequest) -> Operation<ArtifactRef>;
    fn subscribe(&self, sink: EngineEventSink) -> Subscription;
}
```

Until crate source is inspected, whether servo 0.5.0 can implement each method is `unverified`.

## 5. Gap inventory

Each row is a required hook. `servo_crate_status` is `unverified`. `proposed_narrowest_general_purpose_api` is a gap description for a later Servo-side change, not an implemented API and not a promised type name.

### 5.1 Context and page lifecycle

| ID | Required hook | Affected Playwright symbols | Blocks Phase 1 spike | Proposed narrowest general-purpose Servo API | Upstream vs local-patch posture | servo_crate_status |
|---|---|---|---|---|---|---|
| E-CTX-CREATE | Create an isolated browsing context with cookie/storage partition | `chromium.launch`, `BrowserType.launch`, `Browser.newContext`, `Browser.newPage`, `BrowserType.launchPersistentContext` | Yes | Embedding host can create and destroy a profile-scoped browsing context with isolated storage, without Playwright types leaking into Servo | Prefer released API; if missing, smallest context-create hook plus tests; upstream | unverified |
| E-CTX-CLOSE | Deterministic context close and descendant invalidation | `BrowserContext.close`, `Browser.close` | Yes | Close a context, abort its network and page tasks, and emit a crash/close reason the host can journal | Prefer released API; pin revision if a close-reason hook is added | unverified |
| E-PAGE-CREATE | Create a page/WebView inside a context | `BrowserContext.newPage`, `Browser.newPage` | Yes | Host-owned page/view handle bound to a context, with crash propagation to the host | Prefer released API | unverified |
| E-PAGE-CLOSE | Deterministic page close | `Page.close`, parent dispose invalidating descendants | Yes | Page close that cannot recycle handles; host sees a generation counter | Prefer released API | unverified |
| E-CRASH | Crash propagation that does not take down the host | `Page.event.crash`, `Browser.event.disconnected`, `controller_terminated` | Yes for isolation; the spike MUST keep Greppy parent alive | Engine crash is delivered as an event to the embedder; the embedding process is not the Greppy parent | Policy stays outside Servo; crash channel is general-purpose | unverified |

### 5.2 Navigation

| ID | Required hook | Affected Playwright symbols | Blocks Phase 1 spike | Proposed narrowest general-purpose Servo API | Upstream vs local-patch posture | servo_crate_status |
|---|---|---|---|---|---|---|
| E-NAV | Top-level navigation to a URL with result metadata | `Page.goto`, `Frame.goto` | Yes | Navigate a view to a URL and report start, commit, HTTP status when available, and failure class | Prefer released API | unverified |
| E-NAV-REDIRECT | Redirect hops visible to the host | `Page.goto`, `Request`/`Response` lifecycle, research evidence `redirect_chain` | Partial: spike can proceed if final URL is honest; evidence/SSRF later require hops | Report each redirect URL to the embedder before following it, so the host can apply DNS/redirect policy | General-purpose redirect observer; Greppy SSRF policy stays in the supervisor | unverified |
| E-NAV-SAME-DOC | Same-document navigation | `Page.goto` fragments, `framenavigated` | No | Notify the embedder of same-document navigations | Prefer released API | unverified |
| E-NAV-HISTORY | Reload, back, forward | `Page.reload`, `Page.goBack`, `Page.goForward` | No | History traversal hooks with the same result metadata as E-NAV | Prefer released API | unverified |
| E-FRAME-LIFECYCLE | Frame attach/detach/navigate | `Page.event.frameattached`, `framedetached`, `framenavigated`, `Frame` | Partial: `goto` on the top frame is enough for the spike script; frame-aware locators need it | Tree of frame identities with lifecycle events | Prefer released API | unverified |

### 5.3 Evaluation and serialization

| ID | Required hook | Affected Playwright symbols | Blocks Phase 1 spike | Proposed narrowest general-purpose Servo API | Upstream vs local-patch posture | servo_crate_status |
|---|---|---|---|---|---|---|
| E-EVAL | JavaScript evaluation in a page/frame realm | `Page.evaluate`, `Frame.evaluate`, `Locator.evaluate` | Yes | Evaluate a source string in a chosen frame realm and return a host-serialized value. V8 MUST NOT replace Servo page JS | Prefer released API; serialization format is Greppy-side | unverified |
| E-EVAL-HANDLE | Persistent JS handles | `Page.evaluateHandle`, `JSHandle`, `ElementHandle` | No | Optional handle table in the engine or in the adapter; do not invent page-JS object IDs without engine support | Keep handle tables in the adapter if Servo only returns values | unverified |

### 5.4 Query, selectors, accessibility

| ID | Required hook | Affected Playwright symbols | Blocks Phase 1 spike | Proposed narrowest general-purpose Servo API | Upstream vs local-patch posture | servo_crate_status |
|---|---|---|---|---|---|---|
| E-QUERY-CSS | CSS selector query against current DOM | `Page.locator`, `Frame.locator`, `Page.querySelector` | Yes | Query nodes by CSS in a frame, returning engine node refs that die with the document | Prefer released API | unverified |
| E-QUERY-TEXT | Text selector | locators using text | Partial | General-purpose text search over rendered/plain text, not a Playwright parser inside Servo | Selector engines stay in Greppy; Servo provides text content | unverified |
| E-QUERY-ROLE | Accessibility role/name | `Page.getByRole` | Yes for the spike script (`getByRole("button", { name: "Load" })`) | Expose an accessibility tree snapshot (role, name, disabled, value) the host can match | Prefer released a11y tree API; do not encode Playwright role engines in Servo | unverified |
| E-QUERY-LABEL | Label, placeholder, alt, title, test-id | `getByLabel`, `getByPlaceholder`, `getByAltText`, `getByTitle`, `getByTestId` | Yes for `getByLabel("Query")` in the spike | Host reads labels/attributes from DOM plus a11y; Servo exposes attributes and label association | Prefer released DOM attribute/label hooks | unverified |

### 5.5 Actions, input, actionability, layout

| ID | Required hook | Affected Playwright symbols | Blocks Phase 1 spike | Proposed narrowest general-purpose Servo API | Upstream vs local-patch posture | servo_crate_status |
|---|---|---|---|---|---|---|
| E-LAYOUT | Real layout for geometry | `Locator.click` actionability, `boundingBox`, `ElementHandle.boundingBox`, `offset*`, `getBoundingClientRect`, visibility, hit testing, intersection | Yes | Layout results for a node: border box, viewport intersection, hit target at a point. Fabricated geometry is forbidden | Prefer released layout/hit-test API; this is the most likely upstream gap | unverified |
| E-ACTION-MOUSE | Mouse input at coordinates | `Page.click`, `Locator.click`, `Mouse`, `Page.hover`, `Page.dblclick`, `Page.dragAndDrop` | Yes (`click` in the spike) | Dispatch mouse events through the engine input path at layout coordinates | Prefer released input API | unverified |
| E-ACTION-KEYBOARD | Keyboard input | `Page.fill`, `Locator.fill`, `Keyboard`, `Page.type`, `Page.press` | Yes (`fill` in the spike) | Focus a node and insert text / key events through the engine | Prefer released input API | unverified |
| E-ACTION-TOUCH | Touch input | `Touchscreen`, tap | No | Touch event dispatch | Prefer released API | unverified |
| E-ACTION-FORM | Form, select, file chooser | `selectOption`, `setInputFiles`, `FileChooser`, `Page.check` | No | Set form control values and open/fulfill file choosers through engine APIs, not DOM hacks that skip validation | Prefer released API | unverified |
| E-ACTIONABILITY | Attached/visible/stable/enabled/editable/hit-target checks | Locator actions, strict mode | Yes | Host implements Playwright checks using E-LAYOUT, E-QUERY, and mutation/layout events. Servo MUST NOT grow Playwright-specific actionability | Actionability stays in Greppy; Servo provides layout + events | unverified |
| E-WAIT-EVENTS | DOM/layout/navigation events for auto-wait | Locator auto-wait | Yes if polling is forbidden as the primary wait | Mutation, layout, and navigation notifications so the host can wait without busy polling | General-purpose observer; polling only as bounded fallback | unverified |

### 5.6 Network, storage, dialogs, artifacts

| ID | Required hook | Affected Playwright symbols | Blocks Phase 1 spike | Proposed narrowest general-purpose Servo API | Upstream vs local-patch posture | servo_crate_status |
|---|---|---|---|---|---|---|
| E-NET-OBSERVE | Request, response, failure observation | `Page.event.request`, `response`, `requestfailed`, `BrowserContext` network events | No | Embedder-visible network events without secret headers | Prefer released observer | unverified |
| E-NET-ROUTE | Interception, fulfill, abort | `Page.route`, `Route.fulfill`, `Route.abort` | No | Pause a request, fulfill from the host, or abort, still subject to host DNS policy | General-purpose interception; Greppy policy outside Servo | unverified |
| E-STORAGE | Cookies, localStorage, sessionStorage, IndexedDB where Servo supports them | `BrowserContext.cookies`, `addCookies`, `storageState` | No | Read/write cookies and web storage per context. If IndexedDB is absent in the crate, record `missing` after inspection rather than faking it | Prefer released API; do not stub storage as success | unverified |
| E-DIALOG | JS dialogs | `Page.event.dialog`, `Dialog` | No | Dialog open/handle/dismiss callbacks to the embedder | Prefer released API | unverified |
| E-DOWNLOAD | Downloads | `Page.event.download`, `Download` | No | Download start/path/failure events into a host-chosen directory | Prefer released API | unverified |
| E-POPUP | Pop-ups | `Page.event.popup` | No | New-page requests from script or target=_blank, attributed to an opener | Prefer released API | unverified |
| E-SCREENSHOT | Raster only when requested | `Page.screenshot`, `Locator.screenshot`, `web.screenshot` | No | Optional raster of a viewport or node box using real layout. No continuous compositor requirement | Prefer released API | unverified |
| E-EMULATION | Viewport, device scale, media, locale, timezone, geolocation, permissions, color scheme, reduced motion | `Browser.newContext` options, `Page.setViewportSize`, `Page.emulateMedia` | Partial: default viewport may suffice for the spike | Per-context emulation knobs that Servo already has; missing knobs stay `unsupported` until a general-purpose hook exists | Do not claim options Servo cannot honor | unverified |
| E-ISOLATION | Browser-context isolation | `BrowserContext` | Yes | Two contexts MUST NOT share cookies or storage unless explicitly configured | Prefer released API | unverified |

## 6. Phase 1 vertical spike mapping

The Phase 1 unchanged script requires all of the following. Each is `unverified` against servo 0.5.0 until crate inspection.

| Spike step | Playwright symbols | Required hooks | Blocks spike |
|---|---|---|---|
| `chromium.launch()` | `chromium.launch`, `BrowserType.launch` | E-CTX-CREATE | Yes |
| `browser.newContext()` | `Browser.newContext` | E-CTX-CREATE, E-ISOLATION | Yes |
| `context.newPage()` | `BrowserContext.newPage` | E-PAGE-CREATE | Yes |
| `page.goto(fixtureUrl)` | `Page.goto` | E-NAV | Yes |
| `page.getByRole("button", { name: "Load" }).click()` | `Page.getByRole`, `Locator.click` | E-QUERY-ROLE, E-LAYOUT, E-ACTION-MOUSE, E-ACTIONABILITY, E-WAIT-EVENTS | Yes |
| `page.getByLabel("Query").fill("greppy")` | `Page.getByLabel`, `Locator.fill` | E-QUERY-LABEL, E-LAYOUT, E-ACTION-KEYBOARD, E-ACTIONABILITY | Yes |
| `page.locator("main").innerText()` | `Page.locator`, `Locator.innerText` | E-QUERY-CSS, real text from DOM/layout | Yes |
| `page.evaluate(() => document.title)` | `Page.evaluate` | E-EVAL | Yes |
| `browser.close()` | `Browser.close` | E-CTX-CLOSE, E-PAGE-CLOSE | Yes |
| Parent survives worker death | process model | E-CRASH, workers out of Greppy parent | Yes |

Screenshot, routing, downloads, dialogs, and CDP are not required to enter the Phase 1 spike. They remain in this inventory so later phases have a checklist.

## 7. What later phases MUST do on first crate inspection

1. Extract servo 0.5.0 at git `77fccacc1f1fdce10498d50173aafaa09d02879e` and record the local path.
2. For each ID above, cite the actual embedding type and method, or record `missing`.
3. Do not mark a hook `unsupported` merely because it was unverified here.
4. If hooks are missing, follow section 2. Narrow general-purpose APIs, Servo-level tests, upstream when practical, pin the accepted revision.
5. Stop if the required delta becomes an unbounded Greppy-specific fork.

No Servo embedding implementation, crate, or local patch is included in Phase 0.
