# Workflow cost diagnostics and the S06 regression

The heads must reduce total agent workflow costs while preserving required
information. Smaller individual responses do not establish this. The S06 Web
study is a development regression case for this requirement, not a heads-on/off
experiment: A used standard CUA and C used Greppy Web, both Luna/medium.

`tools/embedding_heads/workflow_costs.py` compares complete paired runs and keeps
provider input, provider output, serialized tool-result bytes, tool calls and
end-to-end time separate. It reports paired differences and paired percentages;
a ratio of arm medians is not substituted. A zero baseline has no defined percent
change. Missing timing stays null with an explicit coverage count.

The study importer verifies the plan hash and every planned A/C run, trial and
metadata identity, original trace byte hashes and tool-record pointers. It checks
request/response coverage, recomputes compact UTF-8 JSON result bytes and matches
provider counters to complete metadata totals. Failed, repeated, missing and
incomplete runs cannot be silently removed to improve the result. The source
summary's paired provider-token changes must reproduce.

A descriptive `heads_on_off` comparison additionally requires the same release
checksum, backend and declared agent configuration, with explicit off/on flags.
Neither comparison mode grants production eligibility. Independent source-evidence
preservation, controlled latency, representative final scenarios and the rest of
the release gates remain separate requirements.

## Reproduced S06 development result

Five Checkbox pairs, ten completed runs, both arms 5/5 correct:

| Metric | Median paired change |
| --- | ---: |
| Provider input tokens | +26.4696% |
| Provider output tokens | +57.3643% |
| Tool-result JSON bytes | -90.1040% |
| Tool calls | +2 calls; +66.6667% |
| Controlled end-to-end time | Unknown in all five pairs |

Pairs 1, 4 and 5 combine smaller results with more calls and more provider input
and output. Pairs 2 and 3 retain the standard arm's host/tool failures; those are
not head benefits and are not removed. Five repeats of one development case and
this confounded tool comparison do not establish causal head effects,
non-inferiority, timing compliance or production acceptance.

Source summary:
`/Users/michaelwelsch/.local/state/greppy-web-study/basic-series-20260905-06/summary.json`.
Plan SHA-256: `2d261e1a02ce57d59acf24ce78d7a51ab84a274f53d246713aa3971d9ac7b894`.
Reproduced, source-hash-bound report:
`/Users/michaelwelsch/.local/state/greppy-heads/2026-09-05/web-s06-workflow-costs-v3.json`.
The source study and its raw artifacts remain unchanged.

## Protected information for Web training and evaluation

These are preservation requirements, independent of a ranker's optional-detail
score. They are not automatic relevance labels for unrelated tasks.

| Information | Required distinction |
| --- | --- |
| `enabled`/`disabled`, `checked`, `value` and their observed changes | Preserve actual state; absent/unknown never becomes false or ready. |
| Outstanding result/readiness condition | Text presence or a successful wait does not prove the intended field is enabled or the task complete. |
| Current document, tab, frame/scope and usable refs | Preserve current identity; do not silently bind a stale ref to a replacement node. |
| `STALE_REF`, ambiguity and syntax/transport errors | Preserve the actual error and precise recovery context; distinguish shell generation errors from product failures. |
| Action receipts and partial chains | Preserve already executed steps, failure boundaries, remaining work and continuation handles; failure does not imply rollback. |

Development pairs in which fewer returned bytes cause an additional observation
or recovery round must be retained for error analysis. Costs include expansions,
retries and follow-up calls. An observed recovery sequence is not evidence about
private model thoughts. The heads emit scores/classes; they must not invent state,
readiness, references, causal explanations or task-success messages.

Nine regression tests cover hidden follow-up costs, paired arithmetic, unknown
timing, zero baselines, failed/incomplete/duplicate runs, configuration/backend
mismatch, malformed counters and independently recomputed metadata bytes. No new
browser run, teacher annotation or training was started for this diagnostic.
