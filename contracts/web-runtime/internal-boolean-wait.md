# Experimental internal Boolean wait bridge

Status: targeted native and JavaScript regressions verified on a source-bound
development candidate; signed-package and complete stability gates remain open.
This is not a native page index or a new free-form JavaScript agent command.
CLI opt-in wiring retains its typed-condition validation and explicit opt-in.
Typed conditions over a shared native page index remain the intended endpoint.

Internal request operation: `web.wait`.

```json
{
  "session_id": "existing-session",
  "tab_id": "optional-existing-tab",
  "source": "!document.querySelector('#absent') && !document.querySelector('#ready').disabled",
  "timeout_ms": 2000
}
```

The CLI supplies its validated condition expression, explicitly returning a
Boolean. The existing CLI `{holds, detail}` object MUST NOT be passed directly.
Both `{holds:false}` and `{holds:true}` are invalid predicate results; strings,
numbers, null, undefined, Promises and DOM nodes are likewise rejected. The
legacy Playwright `page.waitForFunction` keeps its truthy-value contract.

The implementation reuses the existing nonce-bound MutationObserver/rAF wait
engine. It does not repeatedly cross the CLI/RPC boundary to evaluate a query.
Property-only state changes remain observable on rAF. At most one rAF callback
is pending per waiter. This is not a claim that page predicates never traverse
the DOM or that all operation costs are constant-time.

An existing live page is required. The operation does not create a blank page,
restart a dead content worker, or silently switch an explicitly selected tab.
The request deadline, explicit timeout and session remainder bound setup,
transport, waiting and the final observation together. A zero timeout means
no time remains, not infinity. The strict path does not borrow legacy 20ms/1ms
minimum waits or unrelated pump-cleanup budgets. Browser/OS work still needs
real integration evidence; these are budget rules, not a proof of preemption.

Success returns `held:true`, `waited_ms`, `session_id`, `tab_id`, `document_id`,
`page_state` and `untrusted_content_boundary`. `page_state` uses the bounded,
credential-redacted `greppy.web.page-state.v1` envelope; one observation is
made while the original session operation is Busy, capped by the remaining
budget and two seconds. `document_id` names its observation scope, not a URL.

If that observation fails or has no budget, the confirmed condition remains
successful, but page_state is explicitly unavailable with a reason, and
document_id is null. No old snapshot is relabeled fresh. An asynchronous page
may change between predicate confirmation and observation; `held` is not a
promise that the condition remains true forever.

Ref predicates must retain the existing observed-node/document resolution
contract. A propagated STALE_REF stays an error, not a request to retry against
a replacement node. CLI ref-predicate wiring and actual replaced-node/native
regressions exercise this binding; a synthetic STALE_REF exception alone does
not prove real ref binding.

Malformed source/RegExp returns INVALID_WAIT_SOURCE with correction guidance;
non-Boolean output returns INVALID_WAIT_PREDICATE. Do not expose internal
JavaScriptErrorInfo/stack structures or suggest Doctor/blind retries for syntax.
An unconfirmed condition returns TIMEOUT, not successful held:false.

Required evidence for integration acceptance:

- Native strict object/Boolean/absent/AND behavior and fresh state after a
  property-only update; sensitive fields redacted.
- Existing tab/session ownership; real stale refs after node/document replacement.
- Observation failure preserves held:true without fabricated current refs.
- Zero, exhausted session and near-deadline budgets; no retained waiter resources.
- Existing Playwright value/error/nonce and scheduler regressions remain green.
- Full native build and source-bound test receipts; isolated JS/Rust unit tests
  alone do not satisfy this gate.
## Observed reference conditions

CLI conditions using `@N` carry an internal `condition_ref` selector separately
from the Boolean/source expression. The daemon binds it to the selected
session and tab's observed snapshot using the native locator binding path.
Every predicate sample validates the actual node against that snapshot and
the existing WeakMap identity registry before evaluating the condition.

An absent, detached, replaced, foreign-document or foreign-tab reference is
`STALE_REF`, never Boolean false and never success through `--absent`. No ref
is translated directly into page-controlled CSS attributes without identity
validation. Both one-shot/legacy condition evaluation and native waits use the
same check. An older runtime lacking the lexical condition binding fails
rather than certifying absence; CLI and runtime must be upgraded as a set.

The JavaScript component test covers the guard and actual identity predicate;
the native regression additionally covers server binding, replacement during
an in-flight wait, navigation, and retained session usability. Native and
packaged acceptance remain required; a component test is not that evidence.
