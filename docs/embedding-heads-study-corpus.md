# Source-pinned Web study inventory

`tools/embedding_heads/study_corpus.py` indexes explicitly listed, completed
`greppy.web-study.pilot.v1` and `greppy.web-study.basic.v1` trials. It checks
artifact paths, turn identity, export SHA-256 and byte length, original source
line/byte boundaries, tool request/response pairing and equality between metadata
and the referenced original tool records. Source line numbers are not treated as
local export line numbers.

The index contains technical source pointers, hashes and bounded action receipts.
It does not copy commands, user/assistant messages, private reasoning, page text or
oracle details. The contiguous export can contain private records; only referenced
tool records are parsed. A tool-output pointer identifies the exact text block;
no substring extraction from prose or recursive search through page JSON is used.

Two source formats remain distinct:

- A complete `greppy.web-runtime.v1` envelope preserves its observed operation,
  request ID and status. Unscoped protocol errors retain an unknown operation.
- Some study adapters emitted the observation result alone. A strict observed
  field shape identifies these as `observation_result_only`; the index never
  supplies a missing operation, request ID, status or session.

An action receipt is associated only through an explicit matching session ID.
The one supported parent wrapper, `greppy.web-study.action-observe.v1`, also
provides an explicit action/observation pair. Its exact parent hash and decoded
JSON pointers are retained. This proves paired capture, not the cause of every
page change or a task outcome. Missing action verbs remain null; exit/status
contradictions and an asserted task-success value are refused. Arbitrary nested
objects are never interpreted as such a wrapper.

Missing scope, unknown protocol errors and result-only observations cannot inherit
an action from temporal proximity. Receipt success means tool completion only.
No missing `checked` or other page field is defaulted. The eventual teacher input
must be separately bound to an explicit task and independently reviewed source.
An envelope-shaped object is evidence of a captured tool result, not proof that
an arbitrary shell command actually executed a trusted browser binary.

## Development inventory, 2026-09-05

The explicit import manifest covers 16 repeated Order-Draft pilots, 34 completed
Basic-03 trials and 10 Basic-04 Dialog trials. All source episodes stay in
`development`, grouped by source family. The six unrun Address trials, unreleased
Table trials and withdrawn Office fixtures are absent. Repeated pilot 1-E is
explicitly excluded because of its adapter/install confound.

| Family | Episodes | Captured observation objects | Failed end oracle |
| --- | ---: | ---: | ---: |
| Order-Draft | 16 | 125 | 0 |
| Text | 10 | 10 | 0 |
| Checkbox | 10 | 14 | 0 |
| Address | 4 | 25 | 1 |
| Dialog | 20 | 25 | 0 |

Of 199 observations, 73 retain a full native envelope (63 inside an explicit
adapter pair) and 126 contain the result only. The 63 adapter pairs have explicit
paired action capture; the other 136 lack a verified action association.
No explicit task-goal binder has been admitted. The Address-C2 rollback
is retained as a failed outcome. An end oracle is stored separately as a hash and
boolean; it is never injected into an earlier observation's teacher context.
These counts describe recoverable direct JSON objects, not complete browser
coverage: other arms/formats and multiline tool text are not guessed into records.

Durable receipts:

- `/Users/michaelwelsch/.local/state/greppy-heads/2026-09-05/web-study-import-v1.json`
- `/Users/michaelwelsch/.local/state/greppy-heads/2026-09-05/web-study-index-v3.json`

## Prospective goal artifacts

For future series, an import entry can include `prepared_goal` with `dispatch`
and `plan` paths. `web_goal_binding.py` checks the study's versioned task-goal and
prepared-dispatch schemas, exact goal/message/plan hashes, matching run, position,
arm, case and execution configuration. The outcome trial must bind the same plan.
All trials of the same case must carry an identical goal across arms and repeats.

The resulting binding remains `prepared_not_sent`, with `delivery_verified: false`
and `admission: held`. Only hashes and source pointers enter the index; neither
the full harness prompt nor goal text is copied. Existing observation goals remain
null. Claimed delivery embedded into a prepared artifact is refused; a separate
validated delivery contract is still required. Historical plans without explicit
goals cannot be backfilled through this interface. These checks establish internal
consistency, not proof of when a file was created or that a message was sent.

Seven goal-contract tests and one integrated study-index test exercise this
boundary. No new study, dispatch or teacher call was started for these tests.

Thirteen regression tests cover original-line addressing, corrupt traces, forged
metadata, private-content exclusion, unknown/cross-session actions, unscoped
protocol errors, result-only records and missing state. No episode is admitted
for training or eligible for the final test. Task binding, privacy review,
annotations, action evidence and independent label adjudication remain required.
