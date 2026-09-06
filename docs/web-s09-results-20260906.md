# S09: five paired Luna trials completed

All ten preregistered Luna/medium trials pass the independent table oracle,
including exactly one correct reservation and persistence after reload. Greppy C
uses fewer provider input AND output tokens in every pair. Median paired changes
are **−37.3407% input and −39.1129% output**. This passes the development block's
token comparison; it does not meet the 50% context target or establish full
browser-tool superiority.

| Pair | A input | C input | A output | C output | A/C oracle |
|---|---:|---:|---:|---:|---|
| 1 | 593550 | 521926 | 1775 | 1359 | pass/pass |
| 2 | 580220 | 363562 | 1577 | 730 | pass/pass |
| 3 | 634094 | 276415 | 1540 | 650 | pass/pass |
| 4 | 846557 | 404222 | 1598 | 1152 | pass/pass |
| 5 | 429356 | 320013 | 992 | 604 | pass/pass |

The independent provider audit sums individual response counters and reconciles
input, output and cached input with both recorded trial totals and cumulative
provider totals for all ten trials. No byte-to-token estimates are used. All C
candidate integrity and observed session-ID isolation checks pass. These are
trace identity checks, not a proof of complete storage/profile isolation.

## What changed in actual interactions

All five C participants use native workflows. Their tool-request counts are
12, 8, 6, 9 and 7; standard A uses 11, 11, 12, 14 and 8. The remaining variation
matters: the six-call run does not erase the twelve-call run or its recoveries.
S09 tests the whole candidate plus generic capability documentation, not an
isolated causal contribution of workflow batching, modal output or onboarding.

- C2–C5 receive a post-sort stale-reference refusal and use the supplied current
  state. C1 has no recorded STALE_REF. An earlier coordinator message incorrectly
  generalized the refusal to C1; the complete audit corrects it to 4/5.
- C1–C3 choose matching-item text conditions already compatible with the prior
  table state. C2 returns revision 2 with the new select value but unsorted rows;
  its next click correctly refuses the old row ref and returns revision 3. A
  true DOM predicate does not establish that the requested sort has completed.
- C1 waits for `Ember` after saving; that text already appears in the dialog.
  The result says held=true after 8ms while the dialog remains open. It then
  retries confirmation, confuses wait's timeout flag, and makes extra reads.
  C3 also chooses an existing item-name condition but reloads and verifies
  successfully. These different public actions do not prove different private
  reasoning or an individually causal optimization.
- C4 expects the placeholder text `Reserve item`, but the opened dialog is
  `Reserve Ember`. The workflow correctly reports an expectation timeout with
  the successful action receipt and current dialog; subsequent filling works.
- C1, C3 and initially C4 print only the shell result's output field despite
  explicit complete-envelope guidance. C1 loses three running-result handles,
  produces NO_SESSION and opens again; C4 eventually preserves and polls a
  handle. C3's fast commands finish without exposing the same failure. Keep
  these costs; do not mistake an empty forwarded chunk for command completion.
- The human workflow output repeats identity/protocol fields per step, and
  non-modal observations repeat page-scope null/default fields and control refs.
  Both are reported efficiency defects. The worker's receipt formatter arrived
  after S09 and is being verified separately, never substituted into this series.
- C4's `observe role=dialog` returns a whole-page view. Worker source/help checks
  trace this to removal of the unsupported positional argument before reparsing,
  while repository guidance promises a query. This is a parser-recovery and
  documentation-contract defect; native scoping is still missing.

## Limits and next experiment

Prospective completion observers produced valid timing receipts, but uncontrolled
concurrent build load prevents time/p95 acceptance. C1 takes about 91.24 seconds
to verified completion versus A1's 52.03 seconds; that slow run remains visible.
The independent Greppy-agent B arm and deployed Office editors remain unavailable.
The twelve-task population, held-out cases and prepared-script gate remain open.

Priorities are compact lossless receipts, reliable transmission of pending shell
handles, and feedback that exposes an unfinished application update without
requiring the agent to invent a weak text predicate. Event-correlated completion
is a research hypothesis, not an implemented native index or a proven fix.
Each must be tested separately and then with repeated same-model tasks.

Evidence: `/Users/michaelwelsch/.local/state/greppy-web-study/table-series-20260906-09`:
`summary-final.json`, `provider-audit.json`, per-trial public metadata/oracles and
prospective `live/*/terminal.json`. All five C runtimes stopped; the owned fixture
server is terminal and its port is closed. Temporary PATH aliases were removed;
frozen aliases, sources, candidates and traces remain available for reproduction.
