# Relevance rubric v2 development check

The first 70-record Web probe exposed numeric/rubric drift: M3 sometimes wrote
"background context" while returning relevance 2, and treated unrelated controls
as useful merely because they appeared on the same page. Rubric
`heads-rubric-2026-09-05-v2` makes the zero-based mapping explicit and distinguishes
irrelevant controls from observed task orientation and helpful evidence.

Both teachers judged the same development records again without previous labels
or model predictions. Across all 70 records, relevance disagreements fell from
51 under v1 to 5 under v2. This is a teacher-consistency result on one fixture
flow, not a model-quality, pilot-scale or production acceptance result.

The v2 matrix has M3 rows and Grok columns, in ordinal order 0/1/2/3:

| | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| 0 | 53 | 0 | 0 | 0 |
| 1 | 3 | 8 | 0 | 0 |
| 2 | 0 | 2 | 0 | 0 |
| 3 | 0 | 0 | 0 | 4 |

Five records remain ambiguous and 13 have different evidence-ID sets. All four
examples remain held: teacher agreement cannot replace the missing independently
verified evidence receipts. No label from this probe has been admitted to training.

Two first attempts under v2 were rejected: M3 returned a non-null Web severity,
and Grok returned an empty evidence list. The response schema now restricts
Web-only batches to null severity, log-only batches to the four log classes,
and every annotation to nonempty evidence and known example/target/source IDs.
Cross-example evidence membership is still checked independently after decoding.
The prompt explicitly tells a teacher to cite the target itself for an irrelevant
record instead of returning an empty list.

Only the failed examples were submitted under the refined schema. Previously
valid results were retained. Invalid annotations were neither edited nor admitted;
old failed jobs and diagnostics remain in the ledger. Cache identity binds the
changed full prompt/schema, and historical reviews can select their recorded
rubric explicitly with `admission.py --rubric`.

The ledger, explicit four-job selection, comparison and held admission report are
retained under
`/Users/michaelwelsch/.local/state/greppy-heads/2026-09-05/web-selection-rubric-v2/`.
The Python suite has 64 passing tests, including regressions for the two observed
schema/validator mismatches. The 1,000-candidate-per-domain pilot and broad corpus
stages remain outstanding.
