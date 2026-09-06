# Complete-agent development study

The experiment tests an optional record command against unchanged Greppy Web and the standard in-app Browser. The business task, frozen order and prompts are in the durable `records-20260906/series1/plan.json`. Fifteen fresh Luna/medium participants run serially across five seeds. This is one synthetic development case, not the original multi-case acceptance and not the actual Greppy-agent B arm.

## Implemented intervention

The record wrapper returns native refs, names, explicit inspected visibility and requested data attributes in one record shape, with native local filtering. It preserves errors and rejects incomplete source evidence. All other commands forward unchanged. The executable/runtime pair is frozen. It is an opt-in research prototype, not a native production command.

In the successful prepared pilot the filtered query returned eight eligible rows using 21 internal native commands in 14.55 seconds. The complete booking passed all six independent oracle checks. A standard-browser pilot passed the same checks. These prepared pilots are correctness evidence, not agent-token measurements or speed comparisons.

The wrapper uses sequential reads and is limited to the current document; it does not create an atomic snapshot or add frame support. A native bulk implementation would need a separate consistency and performance evaluation. The optional intervention may be ignored if the original page state already contains sufficient information. Changes in E cannot be attributed to records without verifying actual uptake.

## Evaluation

The owner's clarified criterion is aggregate efficiency over comparable correct runs. A modest isolated regression does not cancel nine savings. We report paired changes, ratio of aggregate totals and an exploratory paired-bootstrap interval after all five pairs; each outlier remains included. Correctness and functional/usability failures remain separate from stochastic token variation.

Input includes cached input; cached input is not added again. Output and uncached input are separate dimensions. Actual response-level telemetry is exported losslessly from the host, reconciled with turn totals and pinned by hash. No bytes-to-token conversion or guessed prices are used. Agent-turn time includes onboarding and browser startup, but post-hoc oracle collection does not establish end-to-end verified latency. Other host load is uncontrolled.

## Failure classification and recovery

Prepared pilot failures remain under the durable `records-20260906` directory. One default-session startup hit the harness watchdog before the record query; the subsequent expired-session response does not establish the original cause. Explicit project-profile sessions are a prospectively shared C/E condition, not an extension of wall time: the checked limits implementation leaves the default 120-second wall budget unchanged for both project and research profiles, consistent with the observed failures. Two later pilot failures were test-driver parsing assumptions: the chain emits multiple JSON documents, including action receipts with `ok: true`. The driver now validates each document and records calls incrementally. These parsing failures do not establish a Greppy matcher or record product bug.

In participant r1-c, early calls forwarded only stdout and omitted running-session metadata despite explicit onboarding. A subsequent unquoted multiword fill was rejected before executing the chain. The participant recovered through the UI; all six outcomes are correct and every response remains in its measured cost. Excess tokens alone are not evidence of a Greppy product bug.

## Evidence status

The frozen prototype is pushed on `codex/web-composition-study`, commit `6bf9dca647bffef106cc6fc4234a1f0d48891669`, in draft PR13: https://github.com/metric-space-ai/greppy/pull/13 . The study is still running; aggregate results will be added after collection. Full raw host traces stay in the owner's durable local evidence directory rather than being published in the repository.
