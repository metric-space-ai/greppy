# S08 ongoing empirical investigation

Completed results and audit corrections are in `web-s08-results-20260906.md`.
The following records the investigation sequence, including retained mistakes.

Scope: five development A/C pairs on the table case, frozen Luna/medium, no
history fork, alternating order, independent fixture oracle. B is still blocked
by the unhealthy WorkspaceFS provider; no C trial is relabelled as B. Full
12-case acceptance, controlled latency/p95 and prepared scripts remain open.

Evidence: `/Users/michaelwelsch/.local/state/greppy-web-study/table-series-20260906-08`.
The plan and ten prompts were frozen before dispatch. CLI f6ef88 and runtime
4a7070 passed the native compact-feedback and 12 wait/ref preflights. The runtime
candidate is a preserved dirty-worker build, not a clean build from Root HEAD.

## First observed mechanisms

C1: actual input 598097 versus A1 485146, output 1400 versus 1552. Both oracles
pass. This pair fails the input-saving requirement. No conclusion is inferred
from its smaller output alone.

The public call/response trace shows:

1. Despite explicit instructions to forward the complete shell envelope, C1
   prints only r.output. The first cold open yields no output; the hidden running
   handle is lost. The agent opens again. Subsequent use of the second response's
   refs selects the other implicit context and is safely refused. This is agent
   integration misuse; no stale-ref safety check should be weakened to mask it.
2. After recovering, the agent switches from the attempted three-action chain to
   separate select, check and select calls. This is an observable batching loss,
   not a claim about private model reasoning or a quantified causal token share.
3. The sort response has value ascending but revision 2 rows in order 15/25/18.
   Its immediately used row ref is stale after asynchronous replacement. The
   refusal returns revision 3 and a usable new ref; no wrong node is clicked.
4. The error formatter dumps page_state as full partial_result JSON, bypassing
   compact state/choice formatting. The raw CLI error is 4780 bytes; its host
   result JSON is 6749 bytes. Neither measurement is reported as provider tokens.
   Recovery guidance asks for observe even though a fresh page state is present.
5. Confirm is sent once in C1, followed by observe and reload. C2 instead repeats
   Confirm after a successful fill/Confirm chain and incurs a timeout; the
   independently verified final state still has exactly one correct reservation.

The error-formatting and redundant-observe guidance defects were reported with
exact commands, cwd, version/hashes, trace locations and expected behavior to
`01a02118-0d61-7e10-a9d4-be496fa34879`. Fixes must preserve the typed failure,
correct ref invalidation, page fences and earlier side effects. Frozen S08
binaries/prompts are not changed during the series.

## Harness limitations retained

Preparation/onboarding was initially sequenced incorrectly before plan.json
existed; it was corrected before any participant started. A1's prospective
observer was attached after completion and refused to backfill timing. A1 keeps
its actual tokens and post-hoc oracle, with no live timing claim. C2's observer
started before completion but detected 1.0205 seconds of wall/monotonic clock
mismatch, so its time is invalidated even though the oracle passes. No correction
or latency subtraction is applied. Other host load prevents controlled timing
acceptance throughout this series.

A source cherry-pick of worker 6a22a0aa into Root 91c57523 exposed missing native
prerequisites. It was aborted without changing the frozen executables; the worker
is providing an exact prerequisite slice. No successful source integration is
claimed from that aborted attempt.
