# Selection evidence and compact Inspect: experimental integration

Greppy S07 repeatedly required additional DOM/JavaScript requests to discover
select option values. The shared descriptor now includes `select_choices`
using the same bounded helper as native Select diagnostics. Valid values,
labels, disabled options/groups, empty values, and explicit truncation remain
available. The helper reads at most eight options; sensitive controls are not
enumerated. This is an information-quality change, not semantic inference.

Human-readable successful `web.inspect` output now prints its decoded element
once. The serialized transport copy is omitted only for a recognized successful
Inspect result. Other receipt fields, unknown extensions, errors, malformed
shapes, and untrusted-content boundaries remain. Machine JSON is unchanged.

Validation on 2026-09-05:

- `cargo test -p greppy --features ci-test-assets --lib web::view::tests`:
  15 passed, exit 0; Exec 18483. Compilation 1m33s, test execution 0.07s.
  This compiles the composed descriptor constant, but does not execute Servo.
- Shared helper and descriptor Node tests: 10 passed, exit 0. These use
  DOM-shaped test objects, including disabled groups and sensitive controls.
- Native owner/descriptor probe 01: exit 1, retained under
  `/Users/michaelwelsch/.local/state/greppy-web-study/isolation-descriptor-20260905-01`.
  Distinct owners and distinct sessions were created; both foreign-session
  accesses returned `session_not_found`/32. The following implicit observation
  returned `NO_SESSION`/30. The probe stopped before descriptor evaluation.
  Both owned runtimes and the HTTP fixture were stopped in finally.
  This is not a passed isolation or descriptor integration test. The unexpected
  context loss is reported to the existing Greppy bug-fixing task.

The old immutable CLI/runtime remain unchanged. No new native Inspect binary
has been built or accepted, no model-token savings have been measured for this
change, and no comparison gate passes on the basis of these tests. Native
Select refusal/postcondition fixes are owned by the fix worker. Full Observe
integration is a separate pending change: Inspect alone does not remove the
first lookup when the initial page observation lacks option values.

Follow-up native probe 02 (Exec 6183, exit 1) completed the descriptor checks:
actual Servo DOM produced the exact empty/ascending/descending value-to-label
mapping and marked the disabled optgroup option unavailable. Candidate hashes
remained unchanged. This still used explicit `web js`, not rebuilt Inspect.

The context regression is now isolated: both implicit own observations worked
before the foreign access, both failed afterward, and both explicit own-session
observations still read the original pages. Engine state survives; the implicit
CLI association is lost. Probe 02 therefore remains failed even though its
13 preceding checks passed. Evidence is in sibling directory
`isolation-descriptor-20260905-02`; cleanup stopped both runtimes and HTTP thread.
The fix worker received the exact reproduction and verified recovery path.

A prospective Observe integration is retained as
`bench/web_study/basic_fixture/observe_select_choices.patch`. It applies cleanly
in dry-run against the fix worker's current content_worker.rs and was applied
only to a separate scratch copy. It injects the single shared helper and adds
`select_choices` only where applicable. No worker source was edited by Root.

`observe_choices_test.cjs` executes that patched OBSERVE_JS with DOM-shaped
objects. Three tests passed: initial actionable values and disabled groups,
unchanged non-select state, and sensitive-option non-enumeration. Use explicit
`GREPPY_OBSERVE_PROBE_SOURCE` pointing at patched content_worker.rs. This harness
probe is deliberately separate from the crate's ordinary unit tests because
Root's runtime branch does not yet contain the worker's E1 Observe generator.

Existing selected_options still scans node.options; no reduction in full
traversals is claimed. Native Observe integration and subsequent agent-token
measurements remain pending.

Native integration follow-up on 2026-09-06: probe 04 passed 18 checks against
preserved CLI 1d53d0a71c443f0f7fdff0ae569b89e08cd50af1f5a85308e308325124110281
and runtime d8f437c493d7780f39bb8868da9227ce8600dadaa7a7e393936954cf1877277b.
With `--check-native-choices`, both isolated owners returned exactly the shared
choice projection through actual `web observe` and native `web inspect` calls.
The context-preservation checks also passed. Both runtimes stopped and the HTTP
thread ended; executable hashes were unchanged. Evidence is retained at
`/Users/michaelwelsch/.local/state/greppy-web-study/isolation-descriptor-20260906-04`.
This establishes the compiled projection for this fixture. It does not yet prove
the Root human renderer in a newly built CLI, the opt-in native Wait adapter,
or reduced model input/output tokens in paired agent trials.
