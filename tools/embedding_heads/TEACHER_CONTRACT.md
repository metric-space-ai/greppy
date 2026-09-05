# Teacher labeling and head training contract

User priority: equal weight for long command logs and Web observations. Training
runs on GPU3. This is the contract for the next data pipeline, not an assertion
that a large labeling run has already happened.

## Separate supervised tasks

1. Log severity: a four-class classifier over an explicitly selected source
   span, with previous/context text distinguished from the target.
2. Log relevance: a ranker for actual causes, diagnostic evidence and useful
   next-step information in a full output. Severity alone does not define rank.
3. Web relevance: a task-conditioned ranker over actual source records. It
   returns source IDs and scores, never replacement state or invented outcomes.

The frozen native Q4_K encoder supplies vectors. Every dataset binds the exact
encoder and tokenizer hashes, native binary revision/backend, task prompt,
layer, truncation, pooling and normalization. Changes to any of these create a
new representation version. Classification and task-conditioned ranking need
separate documented input construction; a task-independent vector alone cannot
express arbitrary task relevance.

## M3 as offline data teacher

Use complete authentic outputs/scenarios as source units. Retain provenance,
command/exit or task/action/scope, and the distinction between supplied context
and the span to label. Partition by source/output/scenario before annotation;
keep final-test annotations and content unavailable to the fitting jobs.

For each selected candidate, request structured annotation fields:

- exact example_id and candidate_id from the input;
- log severity: error/warning/progress/text; absent for Web;
- ordinal task relevance: 0 irrelevant, 1 background, 2 useful, 3 required;
- supporting evidence IDs drawn only from the supplied input;
- a short observable justification and an explicit ambiguous flag.

The brief justification is a label-audit artifact, not private model reasoning.
Do not request or record hidden reasoning. Page/log text is untrusted data, not
instructions. No tools or shell actions are available to the teacher.

Severity rubric follows the 2026-08-05 adjudication: blanks, source excerpts,
compiler decoration and lifecycle footers do not inherit neighboring errors.
SIGPIPE/style/retry advisories differ from hard failure. Judge the selected
span's meaning, including negation, quotation and successful recovery.

The historical `label_fresh_m3.py` explicitly instructed class inheritance by
blank/detail lines and retried both network and incomplete labels forever.
Do not reuse that prompt or its unbounded retry policy for the new pipeline.

## Batch operation and admission

- Pack by bounded input/output token estimates, not a fixed count of arbitrary
  length logs. Preserve stable candidate IDs and enough context across chunks.
- Cache by input hash + rubric/prompt version + teacher configuration. Resume
  only validated completed items; changes to a rubric invalidate old labels.
- Bound concurrency, total calls, response size, timeout and retries. Separate
  transient transport failures from permanent auth/schema failures. Record
  actual usage, failures and retries without credentials or sensitive payloads.
- Reject missing/extra IDs, duplicate/conflicting labels, invalid classes,
  out-of-range evidence spans and responses exceeding the schema. Do not
  silently clamp spans or keep the first contradictory label.
- Validate structure mechanically, then audit semantic quality independently.
  Sample every class and source family and include previously unjudged TN.
  An LLM agreement percentage does not establish correctness.
- Keep ambiguous examples for adjudication or an explicitly measured policy;
  do not quietly convert them to text or drop them from reported coverage.
- Export immutable labeled examples with source/rubric/teacher hashes and
  retain an append-only admission ledger. No old artifact is overwritten.

## Curriculum and evaluation

Start with a small audited pilot before increasing volume. Train a controlled
rubric-only ablation and a separate broader-data ablation. Mine hard negatives
from the same development output/observation; include quoted errors, successful
recovery, repeated unrelated messages and competing task-relevant candidates.
LLM-generated examples may fill identified gaps but are labeled synthetic.

Fit severity with multiclass supervision; fit relevance with ordinal and/or
pairwise ranking supervision. Compare against the existing head and a
mechanical baseline on complete outputs. Do not assume a more complex head is
better. A frozen encoder/linear baseline helps attribute gains to data, head
capacity or representation changes.

Report log and Web outcomes separately with equal decision weight, including
per-family results, full-output cause/required-record retention, false lifts,
expansions, repeat tool calls, task success, result bytes and added latency.
A new final holdout is frozen before model selection and opened after choices
are fixed. Existing R5/FARM and the 16 Web pilot runs are development data.

## Deterministic preservation boundary

Exit status, explicit errors and original evidence remain independent of scores.
For Web, protect identity/scope/revisions/refs, available actions, important
field state, validation/ambiguity/stale errors, state transitions, dispatch vs
observed effect vs confirmed expectation, partial-chain progress, continuation
and unknown fields. Missing checked/state fields mean unknown, never false.

The initial observed Web contract and six synthetic development responses are
owned by the Web task at `docs/web-head-output-contract.md` in its worktree.
The current narrow example labels the quantity field becoming enabled after
checking a product as relevant; it does not prove checkbox checked state or
persistence. Larger record-level relevance labels still require annotation.
