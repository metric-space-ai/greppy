# S09: workflow candidate development study

Preregistered before the first participant: five A/C pairs, Luna/medium,
identical table-task facts per pair, alternating A/C order, isolated state and
Greppy runtime owners. Fixture and messages are frozen under
`/Users/michaelwelsch/.local/state/greppy-web-study/table-series-20260906-09`.
The new `browser_plugin_workflow_v6` condition leaves the standard-browser
message unchanged and documents general Greppy workflow/expectation semantics.
It supplies no page selectors, option values or task solution. Its extra input
counts toward provider usage. Fifteen onboarding tests pass.

Candidate CLI: `0a668f4cfad89b576a5875d6690929cd8061dc33022a5df04498465a24e9f1d9`.
Runtime: `caed05280ccc134f713d7db1df1fa08abdcd8073b52c3e3b708b9f5740391d0a`.
Both are frozen internal copies, separate from mutable build outputs. The
workflow receipt repetition defect remains in this candidate. Worker fixes
must not be substituted midway through the series.

This is a prospective development measurement of the whole candidate and its
onboarding, not a causal ablation of an individual feature. Both tools may batch
known actions and choose their own interaction strategy. All failures,
retries, help reads, tool results and provider input/output usage remain counted.
The standard browser's documentation/setup costs remain in its trial, and
Greppy's own documentation/setup costs remain in its trial. Bytes are never
converted into reported tokens.

Public call/result traces will be reviewed for:

- whether Luna discovers and actually uses native workflow/expectation support;
- whether returned state removes repeated observe, wait or confirmation rounds;
- whether stable references and modal scope avoid guessed targets or recovery;
- confusing syntax, unexpected output shapes and unfinished shell invocations;
- receipt fields repeated without informing the next decision;
- independent correctness, including persistence after reload and duplicate saves.

A pattern count alone is not a product-bug diagnosis. Each report must include
an exact public command/result and distinguish documented usage errors,
fixture/harness problems, expected refusals and actionable product defects.
No private model reasoning is inspected or inferred from public call text.

Completion observers record independent oracle timing prospectively. External
native compilation can run concurrently, so S09 timings are diagnostic and
cannot satisfy the total-time/p95 acceptance gate. The independent Greppy-agent
B arm remains unavailable: workspace status reported ready=false with a stale
provider heartbeat immediately before preparation. C is not a substitute for B.
The twelve-task acceptance, prepared scripts and Office editor readiness remain
open regardless of S09 outcomes.
