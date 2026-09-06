# Failed-action feedback fix and verification

S08 exposed two concrete costs after a safely rejected stale reference: the CLI
printed current page_state as raw partial_result JSON, and recovery guidance
requested another observation despite already supplying fresh refs. Public
S08-C1 response67 and the later C5 control preserve the exact evidence.

The fix worker's scoped view.rs change now reuses the same compact state renderer
for valid failed-action snapshots. The typed failure, exit code, partial receipt,
unknown fields, page fences, truncation and no-rollback statement remain. Invalid,
unavailable or unknown-version snapshots retain their explicit raw diagnostics.
The human view does not rewrite machine-readable response fields.

Root reviewed and committed this as c639020e. All 90 compiled CLI web tests pass,
including the normalized real S08 stale response with both choice projections,
six controls and current refs, and unavailable/malformed/future state coverage.
Receipt: `/Volumes/tmp/dev-artifacts/greppy/web-efficiency/cli-guarded-rinck5qk/receipt.json`.
No source changed during the gate and no guard fired.

The separate native guidance commit1c0b6e86 was integrated as0585f611. Its pure
policy tests pass 9/9 in Root. Only STALE_REF with an available observation changes
guidance; no action is retried and no timeout/worker budget changes. The added
native no-toggle/current-ref/guidance test has not yet run with a newly built
runtime. The frozen S08 runtime therefore still has the old guidance.

`error_feedback_probe.py` independently exercises real reference invalidation,
verifies rejected clicks change no fixture state, reads the compact current ref,
and completes one correct reservation after recovery and reload. Functional truth
and the output-format gate are separate. Its optional current-guidance gate is
reserved for a matching future native runtime; it is not silently assumed.

First attempt, `error-feedback-20260906-01`, reached the harness's 60-second open
deadline without a native reply. No Greppy exit code or formatter verdict was
observed. Own runtime and HTTP cleanup passed. This coincided with a Root CLI
build and other host load; it is retained as a startup/deadline event, not asserted
to be a formatter regression. The exact command/context and a cautious product
versus environment classification were sent to the designated fix worker. One
repeat without a concurrent Root build is in progress; deadlines are unchanged.

The isolated repeat `error-feedback-20260906-02` starts successfully and reproduces
exactly the old human-output defect: 4780 bytes, raw snapshot duplication and
redundant false choice flags, no readable compact ref. Rejected clicks change
nothing, recovery via the native JSON ref completes the independent 5/5 oracle,
and binary/source hashes and cleanup pass. Outer exit1 is the expected feedback
gate failure. A fresh CLI build now precedes the same after-fix native preflight.
The original startup timeout is not reproduced by this repeat.

The fix worker separately observed a different CLI process stalled in dyld on
the temporary artifacts volume, while the same executable hash ran successfully
from the internal volume. That is not proof of this probe's startup cause.
Candidate-volume placement and loader state need explicit controls before future
cold-start acceptance. S08 paths and all failed probe receipts stay unchanged;
the output-format comparison makes no startup-latency claim.
The fresh CLI build completed successfully from c639020e with unchanged listed
sources and no guard event. Executable SHA256:
`c18870dac98156fce835ae41fb7e26d7e07e6fd634def728e935570de211eac6`.
Receipt: `/Volumes/tmp/dev-artifacts/greppy/web-efficiency/cli-guarded-a6ikv94y/receipt.json`.
After-fix probe03 nevertheless timed out at open after 60.056 seconds, with no
stdout/stderr or Greppy exit. Cleanup passed in 3.782 seconds. It never reached
the formatter and does not establish either a formatter pass or regression.
No process stack was captured; the separate worker dyld sample is not its cause
by inference. The failure remains retained and reported.

For an executable-placement control, both frozen CLI versions and runtime4a7070
were copied byte-for-byte to the internal volume; all three expected SHA256s
matched. Original candidates remain untouched. Attempt04 was a Root harness
setup error: prepare_context explicitly requires scratch under /Volumes/tmp.
It started no browser; setup-failure.json records exit1 and the actionable error.
Attempts05/06 retain the required temporary scratch and change only executable
placement, with fresh contexts and unchanged 60-second action deadlines.

Internal-volume before probe05 completes the independent 5/5 oracle and safe
stale-reference refusal checks but fails the expected old-format gate: 4784 bytes,
no readable current ref, raw snapshot and redundant false flags. Candidate/fixture
integrity and runtime/HTTP cleanup pass. Internal placement allowed this run to
complete; a single run does not establish why previous starts timed out.

Internal-volume after probe06 is terminal PASS, exit0. The same native runtime
with CLI c18870 now returns 2383 bytes, one fenced content block, a readable
current ref and preserved choices, without duplicate snapshot JSON or false
choice flags. The independent oracle passes 5/5, rejected clicks change nothing,
and candidate/fixture integrity plus both cleanup checks pass. This is a 50.19%
reduction in this human error response's bytes, not a provider-token measurement.
The two internal-volume runs establish the formatter before/after result without
claiming loader causality, general cold-start reliability or Luna efficiency.

The native recovery guidance gate remains false in both runs, as expected for
unchanged runtime4a7070. Guidance source/policy tests are integrated, but the
worker's guarded native build was refused before spawn for insufficient temporary
disk space. No guidance integration pass is claimed. Next empirical work must
address modal scope and intermediate post-action state, documented separately in
`web-modal-feedback-20260906.md`, then measure actual Luna input and output tokens.

## Native guidance integration now verified

The worker subsequently completed the guarded build and both actual native tests
at 972ebdb7. Ref identity/no-toggle/current-guidance and strict Boolean wait each
pass 1/1; raw logs confirm harness reaped=true. Runtime SHA256:
`a7366cfe7d5f61f01e5c9605b6296361dc33a681f4e1447f2eb30940031c9441`.
Acceptance and source-bound receipts are under
`/Volumes/tmp/dev-artifacts/greppy/native-guidance-candidate.08TDRS/ACCEPTANCE.md`,
with capture directories 4vxy5l8_, u_8wj29r, nvfsd8cs named there. GL texture diagnostics
remain recorded; these are not screenshot/rendering acceptance tests.

Root verified the create-only candidate copy and ran probe07 with c18870 CLI,
this runtime and `--require-current-guidance`. Exit0, all feedback gates PASS,
independent oracle 5/5, rejected actions unchanged, candidate/fixture integrity and
runtime/HTTP cleanup PASS. Response 2527 bytes includes the new current-state
recovery instruction; this is not a new Luna token measurement. Probe06 remains
the prior compact-formatter-only checkpoint; no historical receipt is rewritten.

Of 36 captured source files, 35 match Root byte-for-byte; the only difference is
CLI common.rs. Captured native sources match, but this is still a worker-built
debug runtime and selected-source comparison, not a clean Root full-suite build.
Evidence: `/Users/michaelwelsch/.local/state/greppy-web-study/error-feedback-20260906-07`.
The earlier native-guidance integration gate is now passed for this candidate;
modal structure, action expectations, broader Luna trials and release gates remain.
