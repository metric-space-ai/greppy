# Observation context for embedding heads

This is a prepared interface, not a working head-enabled browser feature. The
CLI argument/wire contract and session state are implemented; daemon RPC
registration, capture at actual action completion sites, and delivery to observe
remain with the native Web integration worker. No head is enabled automatically.

## Wire and state contract

The shared module is `crates/web-client/src/observation_context.rs`. Its snapshot
schema is `greppy.web.observation-context.v1`. A new session without an explicit
goal has `goal: null`, `goal_version: 0`, and `last_action: null`. Creation with a
goal starts at version 1. Goal text is preserved; empty/whitespace-only input or
more than 8192 UTF-8 bytes is rejected.

`web.session.set_goal` accepts a `SetGoalRequest` with `session_id`, an explicit
`goal: string | null`, and `expected_goal_version: u64`. Missing goal is rejected.
Every successful explicit set/clear increments the version, including setting
the same text again. A conflict returns `goal_version_conflict` and
`current_goal_version` without mutation. Version exhaustion also fails closed.

CLI arguments are:

- `web session create --goal "Choose the matching product"`
- `web session set-goal --session SESSION --goal "Save the draft" --expected-goal-version 1`
- `web session set-goal --session SESSION --clear --expected-goal-version 2`

The latter command requires exactly one of goal/clear and always requires a
version. These arguments do not establish daemon support in the current draft.

## Integration API

Existing `Session::new` behavior is preserved. The integration uses:

- `Session::new_with_goal(..., goal)` for optional goal initialization;
- `set_observation_goal(goal, expected_goal_version)` for compare-and-set;
- `observation_context()` for a cloned wire snapshot;
- `begin_observation_action(operation, request_id, page_id)` at action dispatch;
- `complete_observation_action(&ticket, outcome)` at completion, including failure,
  timeout or cancellation paths.

Action tickets are bound to one state instance and cannot be reused or cross
sessions. At most 16 tickets may be pending; integrations must complete failed
and cancelled attempts too. Dispatch sequence numbers increase monotonically;
the last completed receipt follows completion order, including out-of-order
completion. Each receipt retains the goal version and page identity at dispatch.

The receipt type accepts only a typed operation, bounded technical identifiers,
a sequence/version and success/failure. It has no fields for text, values,
selectors, credentials, upload contents or arbitrary page data. IDs must come
from the runtime; syntactic validation cannot attest their provenance. Request
IDs require the `wrq_` prefix; identity strings are limited to 128 ASCII
alphanumeric/punctuation bytes and reject URLs or whitespace.

Only the declared navigation, locator and tab operations can produce receipts.
Observe/inspect/status/handshake and goal updates do not overwrite last action.
Script and chain integration must identify actual underlying actions rather than
treating arbitrary script text as a receipt. A successful receipt means the tool
attempt completed; it does not prove business success or persistence.

Goal changes do not mutate profiles, limits, session lifetime, active page,
locator snapshot identity or lifecycle state. The eventual ranker must retain
the deterministic path without a goal, bind score caches to goal_version and
action context, and apply the independent model/backend release gates.

## Validation

Six shared-context tests pass and all five engine-free Session tests pass on
macOS. They cover CAS/clear/isolation, immutable failure paths, ticket fencing,
bounded pending work, reader exclusion, unknown wire fields and explicit null.
The Session test build has existing warnings in untouched daemon/supervisor
code. Both CLI parser tests also pass. Daemon integration remains unvalidated;
these unit tests are not backend calibration or agent-flow acceptance.
