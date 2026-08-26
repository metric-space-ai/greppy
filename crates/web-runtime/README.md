# web-runtime

This is only a dependency/lifecycle checkpoint. It is **not** Phase 1 completion and it is **not** Playwright compatibility.

## Exact pinned engine versions

- `deno_core = "=0.410.0"`
- `servo = "=0.5.0"`

## Crate hashes and source revisions

- servo crate SHA-256 `331e15df72165ca15b3945970c6870c4b7367be116ded058fda4f41190b265b8`; source revision `77fccacc1f1fdce10498d50173aafaa09d02879e`
- deno_core crate SHA-256 `04d1a43a2716c6818a845f2449ae659da547687c0c4b27c34d242469dd5912bb`; source revision `5dfadcebf35d86f27e916a25c4109b5aa233ded7`

## Combined-engine crash (proven)

`phase1-probe` can compile and link `servo` and `deno_core` into one macOS executable. That is not a working in-process runtime.

Modes that construct `deno_core::JsRuntime` print `phase1_probe.*.ready` and then SIGSEGV during normal `JsRuntime` drop. `servo-only` is not this crash. A standalone `deno_core`-only binary (no Servo in the image) drops cleanly.

Verified still-failing run after a two-symbol localization attempt:

```
greppy bash-smart -e phase1_probe... -- cargo run --manifest-path crates/web-runtime/Cargo.toml -p phase1-probe -- deno-only
```

printed `phase1_probe.deno_core=ready` and `phase1_probe.process=ready`, then signal 11 / exit 139.

Newest matching crash report: `/Users/michaelwelsch/Library/Logs/DiagnosticReports/phase1-probe-2026-08-26-115308.ips` (`KERN_INVALID_ADDRESS at 0x10`).

Call target in the linked `target/debug/phase1-probe` (`__TEXT` vmaddr `0x100000000`):

1. `v8::internal::Isolate::Delete` at `0x10779c4a0` does `bl 0x10779c4c4` (`Deinitialize`).
2. `Deinitialize` at `0x10779c58c` does `bl 0x104b1184c` (`v8::internal::Isolate::~Isolate`).
3. That destructor is SpiderMonkey irregexp, not V8 `isolate.o`. Its body does `bl 0x104b1ad70` (`mozilla::SegmentedVector<mozilla::UniquePtr<void, JS::FreePolicy>, 256, mozilla::MallocAllocPolicy>::~SegmentedVector`).

SpiderMonkey irregexp (`Unified_cpp_js_src_irregexp0.o` in `libjs_static.a`) is a V8 fork. It defines `v8::internal::Isolate::~Isolate` under the same mangled names as V8:

- `__ZN2v88internal7IsolateD1Ev`
- `__ZN2v88internal7IsolateD2Ev`

ld64 coalesces those private-extern symbols. V8 `Isolate::Delete` then runs the SpiderMonkey destructor against a V8 isolate.

### What does *not* have to be renamed

Of 830 `v8::` mangled symbols defined in SpiderMonkey `*irregexp*` objects, **only 4 names** also exist in `librusty_v8.a`:

| category | mangled names | role in this crash |
| --- | --- | --- |
| irregexp `Isolate` destructor | `IsolateD1Ev`, `IsolateD2Ev` | the `bl` target; must be disambiguated |
| `v8::internal::PrintF` | two overloads | same-name overlap; not the faulting `bl` |
| other irregexp `v8::internal` (ActionNode, RegExpNode, …) | 826 names | no current match in V8; forked signatures |

A second, larger overlap is **ICU 77** (`icu_77` / `_77`): about 10k shared global names between `libjs_static.a` and `librusty_v8.a`. That is not this destructor crash. Do not apply a 10k-symbol hide/rename without mapping that category and re-validating both engines.

### Failed experiment (do not repeat)

Localizing only the two SpiderMonkey Isolate destructor symbols in `target/.../libjs_static.a` via `phase1-probe/build.rs` is **not** a fix. After that rebuild, `Deinitialize` still `bl`s `0x104b1184c` (the SpiderMonkey destructor). The README must not claim that localization made V8 keep the correct destructor. Mutating sibling `mozjs_sys` archives under `target/` is also non-hermetic and cache-corrupting; that `build.rs` was removed.

### Blocker

`mozjs_sys 140.14.0-0` links a **prebuilt** `libjs_static.a` (GitHub release tarball) unless `MOZJS_FROM_SOURCE` is set. There is no hermetic in-tree way to namespace irregexp without vendoring `mozjs_sys` and rebuilding SpiderMonkey from source. A scoped source fix is to wrap the irregexp V8 fork (`namespace v8 { namespace internal`) so at least `Isolate` and `PrintF` no longer share mangled names with Deno’s V8. That is a SpiderMonkey rebuild, not a probe-only patch.

### Implemented process-boundary checkpoint

The `runtime` workspace member now proves the process-boundary lifecycle with three separately linked executables, verified as Mach-O images on macOS:

- `web-runtime-supervisor` links neither engine;
- `web-controller-worker` links `deno_core`/V8, constructs a `JsRuntime`, executes a JavaScript startup probe, and holds it until shutdown;
- `web-content-worker` links Servo/mozjs, constructs Servo, and holds it until shutdown.

The supervisor starts both workers through explicit executable paths, validates their role and protocol version, requests shutdown, and requires a normal zero exit after each runtime has been dropped. Supervisor-side message waits and graceful process reaping have 30-second deadlines. Drop guards kill and reap the current single-PID workers on failure; there is no `process::exit`, `mem::forget`, intentional leak, or crash masking.

Verified checkpoint evidence:

- 15 protocol, worker, and supervisor unit tests pass;
- the real process-boundary integration smoke passes with both engines constructed and normally dropped;
- isolated symbol inspection confirms the supervisor contains neither engine, the controller contains V8 but no Servo/mozjs, and the content worker contains Servo/mozjs but no `deno_core`/`rusty_v8`.

This proves only dependency, framing, process isolation, and lifecycle viability. It is not a Playwright compatibility claim by itself.

### Phase 1 vertical spike (verified locally)

The same three processes now execute the guide's unchanged Playwright script:

```javascript
import { chromium } from "playwright";
const browser = await chromium.launch();
const context = await browser.newContext();
const page = await context.newPage();
await page.goto(fixtureUrl);
await page.getByRole("button", { name: "Load" }).click();
await page.getByLabel("Query").fill("greppy");
const result = await page.locator("main").innerText();
const value = await page.evaluate(() => document.title);
await browser.close();
```

The controller worker loads a virtual `playwright` module in `deno_core`, `firefox`/`webkit` launch fail with `browser_engine_not_available`, and other missing methods fail with `unsupported_playwright_operation`. The supervisor forwards engine calls. The content worker hosts a Servo `WebView` (`SoftwareRenderingContext`), navigates, waits for real layout via `getBoundingClientRect()`, clicks with Servo input events, fills through the page realm, and evaluates `document.title` in SpiderMonkey.

Verified:

```
greppy bash-smart -- cargo test --manifest-path crates/web-runtime/Cargo.toml -p web-runtime --features controller-runtime,content-runtime --test phase1-spike -- --nocapture
```

`unchanged_playwright_script_controls_servo_across_process_boundary` passed in 9.35s (exit 0) on macOS. The fixture hydrates the button/label/main tree with `queueMicrotask` after first paint. This is an **experimental web-runtime spike**, not product Playwright compatibility, not a signed distributable, and not a Linux/Windows CI receipt.

### Alternative one-process experiment

The alternative one-process experiment is: vendor `mozjs_sys`, rebuild with `MOZJS_FROM_SOURCE`, namespace irregexp away from `v8::internal`, then re-run `servo-only`, `deno-only`, `deno-then-servo`, and `servo-then-deno` with normal return (no `process::exit`, no skipped `Drop`). Only if those pass repeatedly is a one-binary claim allowed. ICU overlap remains a follow-on risk after irregexp.

## IPC

Worker IPC v1 is length-delimited (32-bit unsigned big-endian payload length, compact UTF-8 JSON, 1 MiB maximum), never JSON Lines. Every message carries schema `greppy.web-runtime.worker.v1` and version `1`. The closed message set is `Hello`, `Ready`, `Shutdown`, `ShutdownAck`, `RunScript`, `ScriptComplete`, `EngineCall`, and `EngineResult`.

This worker protocol is still smaller than the Greppy client/supervisor protocol in `contracts/web-runtime/protocol.v1.schema.json`.
