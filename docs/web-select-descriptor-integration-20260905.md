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
