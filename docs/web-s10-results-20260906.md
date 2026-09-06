# S10: compact receipts do not yet remove interaction regressions

All ten Luna/medium trials pass the independent server oracle and reload check. Greppy C uses fewer input and output tokens in four of five pairs. Pair 1 still consumes more input: **474102 versus 432023, approximately +9.74%**. This is an unresolved efficiency regression, despite lower output in that pair and favorable aggregate medians.

Median paired changes are **−47.3710% input and −39.0221% output**. Provider response counters reconcile exactly with recorded and cumulative totals for all ten trials. Cached input remains part of input. No bytes-to-tokens conversion is used.

| Pair | A input | C input | A output | C output | A/C calls | A/C oracle |
|---|---:|---:|---:|---:|---:|---|
| 1 | 432023 | 474102 | 1397 | 1226 | 8/11 | pass/pass |
| 2 | 1058766 | 277500 | 2410 | 579 | 17/6 | pass/pass |
| 3 | 648642 | 314451 | 1409 | 747 | 12/7 | pass/pass |
| 4 | 538039 | 361225 | 1189 | 820 | 10/8 | pass/pass |
| 5 | 1197581 | 630275 | 3129 | 1908 | 20/14 | pass/pass |

The generic v6 onboarding is unchanged. CLI SHA `f3d06e96…` adds the independently verified compact workflow formatter; runtime remains `caed0528…`. S09 is historical context, not a randomized formatter ablation. Neither the difference from S09 nor this one-case block establishes the formatter's causal token effect, a 50% context reduction, or general browser superiority.

## Public trace episodes that change the next action

**C1: native workflow plus transport and parser recovery.** It waits for `4 matching items`, while the filtered task has three. It prints only `r.output`, dropping the host's pending handle when the command outlasts 10 seconds. Later `web fill @601 3 :: gw-… web click @602` is invalid chaining; the CLI strips operands and falls into grep errors. The corrected native fill/confirm workflow returns the old dialog. A duplicate Confirm then two observations and a status check follow, with two more empty forwarded chunks. These extra rounds remain counted. Original chaining misuse and dropped host handles are distinguished from the misleading parser recovery and old-dialog feedback.

**C2: effective use of expectations after one stale reference.** It retains complete exec envelopes, batches filtering, receives STALE_REF after a weak matching-count expectation, uses the supplied new reference, then confirms with `text=Reserved` and reloads. Six calls and no duplicate submit.

**C3: guessed outcome text, correct partial-failure interpretation.** It makes `observe body` and another observation after filtering, then expects `Reservation confirmed`. The application says `Reserved 3 × Cedar`. Greppy returns an expectation timeout after three seconds with `action status=ok`, phase=expectation and the saved state. Luna reloads successfully without submitting again. The condition mismatch is real; a false business-success claim is not inferred.

**C4: explicit regex guidance enables immediate correction.** Missing the closing regex delimiter yields `WORKFLOW_CONDITION_SYNTAX: expected /pattern/flags`, with zero attempted actions. Luna corrects the delimiter in the next call. It later receives a stale ref, uses the supplied current one and finishes. This syntax episode has actionable guidance and is classified as usage error, unlike C5's opaque CSS diagnosis.

**C5: opaque expectation diagnosis causes loss of batching.** The bare expectation `3 matching items` is interpreted as CSS and fails preflight. The response says only `SyntaxError: The string did not match the expected pattern`, step 3, and `see greppy web --help`. Luna reads general help, changes quotes around the select values, repeats the same invalid expectation, then reads workflow help and falls back to individual actions. It later recovers from a stale ref and presses Enter after Confirm returned the old dialog. Fourteen calls result. The concrete worker request is to name the failing expectation argument, query dialect and correction example while preserving zero-mutation preflight.

**Standard A also has recoveries.** A2 and A5 repeat Confirm after the first action still shows the dialog; the second selector call times out because the dialog has closed. A5 first guesses the wrong browser-client module path and performs multiple documentation reads before opening its page. These costs remain in A. The block measures actual cold tool discovery plus task interaction, not isolated browser-engine work. Future interpretation must keep startup/recovery separate from steady-state interactions without deleting either from task totals.

No private reasoning, motive or internal trust state is inferred. These episodes are based on public request/result pairs and subsequent visible actions.

## Actual provider cost associated with selected error rounds

The generalized `feedback_round_costs.py` verifies each manual call ID against its public argument text and response timeline. Selected generations remain whole; they can contain useful actions and cannot be treated as removable savings.

- C1's malformed chain and duplicate confirmation generations: **79201 input / 225 output**.
- C5's two invalid expectation attempts, stale click and subsequent Enter generation: **165354 input / 594 output**.
- A2's repeated confirmation generation: **66195 input / 140 output**.
- A5's wrong module import and repeated confirmation generations: **106123 input / 370 output**.

The incomplete annotation set is explicitly not a full failure classifier. Unannotated calls and normal verification stay in trial totals.

## Reporting and measurement integrity

The previous summarizer labeled a median improvement as `passes this development block only` even with an individual regression. Its original S10 output remains archived as `summary-final.json`. The corrected report, **`summary-regression-audited.json`**, retains the median result separately and states `median_improved_but_pair_regressions_or_gaps_remain`. Four focused tests cover a hidden individual regression, missing/equal telemetry, failed/empty blocks and strictly lower successful pairs. This strengthens reporting; it does not rewrite the frozen task plan or any trial.

All ten live observers produced valid timing receipts. A separate Cargo process, PID 44128, was confirmed alive with elapsed 10:37 near the end and was not touched. The worker's own pause therefore did not establish full load isolation. No controlled latency or p95 acceptance is claimed. C1 still takes about74.41 seconds versus A1's39.57; that slow case remains visible.

Evidence root: `/Users/michaelwelsch/.local/state/greppy-web-study/table-series-20260906-10`:

- Frozen `plan.json`, fixture files and `prepared-dispatches/` messages.
- Ten `trials/` public exports, provider counters, state snapshots and independent oracles.
- Ten `live/` bindings and terminal observer receipts.
- `provider-audit.json`: all10 reconciled; candidate hashes and session-ID checks pass.
- `usage-timelines/`, `error-round-annotations-v2.json`, `error-round-costs.json`.
- `runtime-cleanup.json` and `cleanup.json`: five own runtimes stopped, fixture PID38449 terminal exit143, port19270 closed, five temporary PATH aliases removed. The blocked root Office-tab close remains a separate limitation.

The worker received the malformed-chain regression and the expectation-diagnostic fallback with exact calls and requested checks. Scope, applied-sort state and confirmation feedback remain separate product work. Arm B is still blocked by the workspace provider. Real Word/Excel tasks, the twelve-task suite, held-out acceptance and prepared-script performance remain unfinished. No new default or release is authorized by this result.
