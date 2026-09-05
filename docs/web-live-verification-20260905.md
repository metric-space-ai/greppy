# Prospective completion-to-oracle timing

Earlier A/C reports have no controlled duration through independent verification.
Their participant turn duration remains descriptive and is not retroactively
converted into an end-to-end measurement.

New series can preregister `prepare_series.py --live-verification`. The host then
starts `verify_on_completion.py SERIES POSITION --source HOST_ROLLOUT --turn-id ID
--evidence NEW_DIRECTORY` while that participant is still running. An already
completed turn is refused. Start it immediately after dispatch identity is
available and retain its process handle; `ready.json` records that it is armed.

The observer consumes appended rollout bytes without repeatedly parsing the
whole trace. Only record-owned task_started, turn_context and task_complete
host records count. Assistant text, another turn's completion, and partial JSONL
lines cannot complete the observation. It never reads browser state through a
participant tool or asks the agent to write measurement logs.

On completion, the observer snapshots the fixture state and executes the frozen
independent verifier against that snapshot. It retains successful and unsuccessful
oracle results, exit status, timing, source-prefix hash, and state hash. The later
`record_trial.py --live-verification RECEIPT/terminal.json` binds those facts to
the exported trial; mismatched plan, state, turn, source, or source prefix is refused.
Token extraction remains provider-only. An absent counter remains absent.

A valid time includes host task_started through verifier completion, including
agent work, retries, polling delay and verifier overhead. Pre-task dispatch queue
time is excluded and explicitly named. A completion-delivery lag above two seconds,
observer clock divergence above 0.25 seconds, changed fixture/plan/state, or missing
host completion prevents a valid verified duration. No measured overhead is
subtracted. These thresholds are preregistered, and runtime overrides must match.
The valid duration of an unsuccessful trial is still an unsuccessful trial.

This does not control competing browser/model/build load, establish cold versus
warm runtime conditions, prove three-arm comparability, or pass any efficiency
criterion. The first real participant use still requires operational validation.
Historical series and their failures remain unchanged.

Tests include streaming/partial completion, forged and foreign completion text,
unsuccessful oracle results, timeout, late host delivery, model mismatch, duplicate
boundaries, clock timezone, altered plan/fixture/source binding, and the actual
fixture verifier. A subprocess integration test joins new plan preparation, live
observation, full trial recording and missing-token handling. The harness fixture
uses synthetic host records and makes no browser-performance claim.
