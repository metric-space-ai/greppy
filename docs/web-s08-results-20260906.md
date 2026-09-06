# S08 development results — no acceptance

All ten frozen Luna/medium trials are recorded. Greppy C passes 5/5 independent
oracles; standard Codex A passes 4/5. A4 stops before submission for an unnecessary
confirmation despite the shared disposable-test authorization. It is retained as
a failure, not a fast success. This is one development case, not the requested
12-case acceptance or the native Greppy-agent B comparison.

Actual provider tokens, including all retries and cached input:

| Pair | A input | C input | A output | C output | A/C oracle |
|---|---:|---:|---:|---:|---|
| 1 | 485146 | 598097 | 1552 | 1400 | pass/pass |
| 2 | 691027 | 487609 | 1953 | 1058 | pass/pass |
| 3 | 872209 | 489083 | 2074 | 989 | pass/pass |
| 4 | 589008 | 607280 | 1693 | 1653 | fail/pass |
| 5 | 534826 | 444861 | 1248 | 900 | pass/pass |

The median of paired percentage changes is **−16.8214% input** and **−27.8846%
output**. Input is still higher for Greppy in pairs 1 and 4. The shared summary
correctly reports `failed_or_unproven`: an incomplete baseline run, the remaining
50% context goal, broader correctness, B, latency and p95 gates are not passed.
No failed run or recovery cost was dropped to obtain the percentages.

The independent public-telemetry audit reconciles every cumulative delta with
provider counters and all ten recorded totals. All five C executable integrity
checks pass and no observed session ID is reused across C trials. This is not a
proof of complete storage/profile isolation. The response counts are A10/14/15/
12/11 and C14/12/12/14/11; smaller context per response coexists with more model
rounds in several pairs. Both sides' tool mistakes remain counted.

## What the traces support

All five C traces and their actual public calls/results are retained. C1 shows a
lost shell handle, duplicate open, failed chain, then separate actions. C2/C3
repeat Confirm and wait through timeouts after an already executed submission.
C4 additionally reads the unrelated standard Browser skill and starts a second
open; these are participant deviations, not permission to remove their cost.
The current error output's full page_state JSON and unnecessary observe guidance
are reported to the designated fix worker. Safe stale-ref refusals stay intact.
These observations motivate fixes; aggregate percentages do not establish their
individual causal contributions. No private reasoning was inspected.

C5 follows the full shell-envelope/poll instructions and uses action chains.
It still receives a stale row ref after sorting and repeats Confirm into a
timeout after a successful fill/Confirm chain. Thus transport misuse alone does
not explain the remaining loop. All five C runs contain post-sort stale refusals;
four contain a subsequent confirmation timeout. Exact calls/verdicts and source
positions are retained in `public-call-failures.json`; safe refusals are preserved.

All five C runs use the exposed exact select values EU and ascending without an
OPTION_NOT_FOUND recovery or a DOM/inspect/help fallback to discover the options.
That earlier selection-discovery failure is absent in this block. The remaining
loops occur at asynchronous table replacement and submission feedback. This is
consistent with the option-view fix helping discovery; isolating its causal token
contribution still requires an ablation rather than comparing whole old series.

## Timing and dispatch audit limits

A1's late observer is not backfilled. The original timing code also compared
`verified` UTC with a later monotonic sample after integrity hashing. A controlled
400ms integrity delay reproduces false clock-drift rejection. After all ten S08
observers completed, adjacent endpoint samples fixed this measurement error;
17 observer tests pass, including real clock-step rejection. Existing S08 timing
receipts remain unchanged and invalidated receipts remain invalid. Independent
host load and cold-versus-warm runtime conditions also preclude time acceptance.

Frozen messages are in prepared-dispatches, and this conversation contains the
actual spawn requests and returned task paths. The local rollout stores those
assignment messages opaquely and does not expose them as child user text. The
first two automated readback audits therefore produced false flags/failed
assertions that mean unavailable readback, not demonstrated prompt alteration.
No encrypted assignment was decoded. Exact on-disk dispatch-text equality is
unverified and must not be claimed from these audits. Model/effort, public task
paths, executed URLs, oracles and actual provider counters are independently
available. Prepared records alone remain insufficient proof of delivery.

Evidence directory:
`/Users/michaelwelsch/.local/state/greppy-web-study/table-series-20260906-08`.
Canonical summary: `summary-final.json`; per-trial records and usage-timelines
retain failures and telemetry. The isolated fixture server and each C runtime
were stopped after recording. No release activation follows from this series.
