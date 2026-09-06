# Aggregate evaluation and record-command development study

The existing Basic evaluator now uses improving paired medians and aggregate input/output costs, retaining correctness, candidate integrity and session isolation checks. An isolated modest token regression remains visible without vetoing the block. Re-evaluating the recorded S10 trials gives **46.90% less total input and 44.62% less total output**, with all ten tasks correct; the single +9.74% input outlier no longer rejects this development block. This is a new evaluation of existing measurements, not a new browser run or full release acceptance.

The separate optional record-command prototype is **not supported for promotion by this study**. Fifteen new Luna/medium attempts are complete, with all costs retained. Its only actual invocation occurred after a session had already expired and correctly propagated that upstream failure. Four experimental participants did not invoke it at all. Group differences therefore cannot be attributed to the record implementation.

## New development results

One synthetic delivery case, five repeated seeds, three arms in a frozen order. A uses the standard in-app Browser; C uses an unchanged scope-development CLI/runtime pair; E adds an optional Python record wrapper to that same pair. C/E use an older scope candidate, not the S10 native-workflow candidate or a newly validated main build. These series must not be pooled or treated as an isolated before/after product comparison.

| Arm | All six state checks passed | Total input tokens | Total output tokens | Uncached input tokens |
| --- | ---: | ---: | ---: | ---: |
| A: standard Browser | 3/5 | 2,299,915 | 6,971 | 149,771 |
| C: unchanged Greppy | 5/5 | 2,938,401 | 9,024 | 178,209 |
| E: optional records | 4/5 | 5,626,572 | 15,637 | 268,236 |

These are observed attempt costs, including failures and recovery, not the cost of equivalent successfully completed tasks. C used 27.76% more input and 29.45% more output than A in aggregate while completing more state oracles. E used 91.48% more input and 73.28% more output than C. No new efficiency acceptance follows. Every paired change and the exploratory bootstrap results are in `bench/web_study/evidence/records-20260906/summary.json`; failed attempts and the operator-stopped run make those intervals unsuitable as successful-task performance claims.

Input includes cached input; cache is not added a second time. Output and uncached input remain separate dimensions. Actual response-level usage reconciles with cumulative turn totals for all fifteen attempts, with no missing tool responses. No byte-to-token conversion or estimated prices were used. Turn time includes onboarding and startup; post-hoc state checks do not establish end-to-end verified latency, and host load was uncontrolled.

## Outcome and failure review

- **r2-a:** stopped before confirmation to request additional approval despite explicit authorization for a synthetic test. The oracle records an incomplete booking. A later administrative cleanup turn was not a task retry and is excluded from its original turn cost; the tab was already unavailable.
- **r4-a:** saved a EUR 7 offer while an enabled EUR 6 offer was visible in the returned snapshot. Storage succeeded, selection correctness failed.
- **r3-e:** repeated invalid-command recovery and 120-second session limits without saving. The coordinator requested a stop after multiple restarts; all 35 responses and 2,024,317 input tokens remain. No numeric per-trial deadline had been frozen: this is an explicitly documented, unplanned administrative stop, not an unassisted completed trial. Cleanup of r2-a also overlapped this attempt.
- **r5-e:** all six stored-state checks passed. Its final sentence nevertheless said confirmation had not completed. That statement conflicts with the independent saved state; it is recorded separately rather than changing the preregistered state oracle after the fact.

In several Greppy attempts, multiword fill values were not shell-quoted. The chain rejected invalid argv before execution; participants initially treated the earlier selection as completed and needed extra UI steps. Early stdout-only forwarding also discarded running-session metadata despite onboarding. These observed mistakes and recoveries stay in measured costs. Excess tokens alone do not establish a product bug.

The checked limits implementation leaves the default wall budget at 120 seconds for both project and research profiles. Selecting project is not an extension of that budget. The observed expiry is consistent with that policy; the prototype did not raise limits or replace an upstream error with an empty successful result.

## Implemented prototype and pilot

`record_tool.py` returns native refs, names, explicit inspected visibility and requested data attributes in one record shape, with filtering through native `web match`. Other commands forward unchanged. It rejects incomplete observations and upstream errors. It is an opt-in research wrapper, not a deployed native command.

The successful prepared pilot returned eight eligible records through **21 internal native commands in 14.55 seconds** and passed all six outcome checks. The standard-browser pilot passed the same checks. These prepared pilots prove the tested workflow, not agent-token savings. Reads are sequential and restricted to the current document; atomic snapshots and frame support are not implemented.

One earlier startup hit the pilot watchdog. Two later pilot failures were test-driver parsing assumptions: a chain emits several JSON documents, including action receipts with `ok: true`. The driver now validates each document and persists every call incrementally. Original failures are preserved, not counted as participant trials or silently discarded.

## Reproducibility and limits

All fifteen requested model/effort pairs and fresh-context flags match the plan. Dispatch order, the frozen prototype/fixture/runner source hashes, and CLI/runtime executable hashes match. The local coordinator export stores message arguments as opaque values, so plaintext prompt equality cannot be automatically verified there. The corrected dispatch audit reports this as unavailable and keeps overall `ok: false`; its non-message checks pass. The initial ciphertext-hash comparison is retained as superseded local audit evidence. Frozen plaintext prompts remain in the durable plan, without claiming an independent host byte-match.

Frozen CLI: `a3d6d1f5e77e44b8187ad7611735c06427d11701e5e263a270cf0865d4ffc31e`; runtime: `6e925e498338e5e9d5237de62aef7d2c86e7b6407a8a7b9be63864602321a134`. Dynamic libraries/assets and host scheduling are outside this hash scope. The source intervention was frozen in commit `6bf9dca647bffef106cc6fc4234a1f0d48891669`; the aggregate gate change is in `a364745065e96fe4e3602b8d6829732f08f8abbe`.

Validation: **36 targeted tests pass**, covering candidate/isolation safeguards, aggregate evaluation, record handling and outcome checks. Both full-UI pilots passed. The original multi-case A/B/C release acceptance, actual Greppy-agent B readiness, current-main native UNION validation and cross-platform gates remain separate and unfulfilled by this experiment.

Full raw host traces, frozen dispatches, oracle states, pilot failures and cleanup receipts are preserved under `/Users/michaelwelsch/.local/state/greppy-web-study/records-20260906/`. Compact public evidence is under `bench/web_study/evidence/records-20260906/`. All eleven task-owned Greppy runtimes and the synthetic fixture server were stopped; all fixture states were copied durably before scratch cleanup. No unrelated worktree, cache or frozen executable was removed.

The source and evidence are in draft PR13: https://github.com/metric-space-ai/greppy/pull/13 . The optional record wrapper remains a research result; the practical accepted change from this work is the corrected aggregate evaluation criterion.
