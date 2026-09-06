# Native workflow experiment — 2026-09-06

Status: implementation under verification; opt-in only. No new Luna token or
end-to-end time result has been measured for this implementation.

## Evidence and hypothesis

The S08 public action/result traces show repeated confirmation and wait rounds
when an action receipt precedes the application's asynchronous state change.
The provider-round audit records those rounds without treating their entire
cost as achievable savings. See `web-feedback-round-costs-20260906.md`.

The experiment sends known navigation/action/wait steps in one typed
`web.workflow` request. The daemon executes the same native operations on one
resolved session and tab, suppresses intermediate automatic observations and
returns a final observation. This reduces external orchestration opportunities;
it does not yet establish lower provider token usage or total time.

## Interface and boundaries

- `web do --native` supports navigation, actions and declarative waits. Unsupported
  commands and incompatible explicit scopes are rejected before browser actions.
- An action with `--expect QUERY` uses the same workflow operation. Optional
  `--expect-absent` and `--expect-timeout MS` require `--expect`.
- Conditions test DOM presence, exact/regex URL or title. Presence is not
  visibility: use e.g. `css=dialog[open]` to test an open dialog. Arbitrary caller
  JavaScript predicates are not accepted by the workflow contract.
- Structural validation precedes execution; engine preflight checks CSS, XPath
  and regex syntax without matching the live page. It does not pre-prove future
  target existence, actionability, application outcome, or all key/URL dialects.
- The whole request has one deadline. A timeout stops later steps and retains
  receipts for earlier changes. There is no rollback or automatic replay.
- An operation returning is distinct from an expectation holding. Neither implies
  an independently verified business result beyond the stated condition.
- `web pw` and the old `web do` path remain available. This experiment does not
  claim that the Playwright facade already compiles into the workflow operation.
- The runtime still uses its existing DOM observation/wait primitives. An
  incremental native page index is not implemented by this change.

## Verification record

Already completed before native integration:

- Four shared Rust contract tests passed: shape, bounds, invalid reference and
  caller-code rejection; preflight excludes fill values.
- Four JavaScript contract tests passed in Node VM. These use DOM stubs and do
  not substitute for engine CSS/XPath parsing or real input delivery.
- 96 CLI web unit tests passed (capture `cli-guarded-9epc64s2`, exit 0,
  sources unchanged). This includes expectation parsing, modifier rejection,
  modal projection with ambiguity/unknown-schema fallbacks, complete background
  archive recovery, cross-session refusal and archive-write failure fallback.
  Compilation took 1m29s; tests took 0.40s; wrapper total was 127.88s. These are
  build/test timings, not agent task performance measurements.

The first corrected native compile (capture `native-capture-v3-run-1jjkq9bf`)
reached its 600-second build guard while rebuilding Servo dependencies for the
separate worktree paths. Cargo was terminated with signal 15; sources remained
unchanged and 5,096,259,584 bytes were free. No workflow test ran in this attempt.
Completed dependency artifacts are retained for the next bounded compile.

Native integration cases cover malformed later-step preflight without earlier
mutation, delayed effects, retained changes on timeout, stale references,
stale-reference absence, explicit tab identity, navigation and the whole-request
deadline. The completed engine run and retained failures are recorded below.

The second native attempt (`native-capture-v3-run-gj_97cab`) compiled the runtime
library and integration test source, but failed linking the test executable:
`clang: error: unable to make temporary file: No space left on device`.
Cargo exited 101; captured sources remained unchanged. No workflow test started.
The artifact volume still had space while the internal TMPDIR volume filled.
The attempted manual stop did not execute because the shell could no longer
create its heredoc temporary file; Cargo had already failed independently.
Both native and CLI capture helpers now check source, artifact and temporary
volumes, requiring 3 GiB at start and retaining the 2 GiB running guard. The original
single-volume guard did not cover this failure.
Only obsolete owned compiler outputs were removed (624,687,080 bytes, hashes
and no-open-handle evidence recorded); frozen candidates and trial logs remain.
The owned CLI compiler cache was subsequently relocated to the internal drive:
9,426 files / 2,905,796,559 bytes, full matching SHA manifests before/after,
no open handles, original cache path retained as a symlink. This freed space
without deleting frozen candidates. Two small guard tests covering six low-disk
scenarios passed, proving no compiler child starts when any checked volume is low.

With both reserves restored, `native-capture-v3-run-tw0hvwhd` completed compilation
in 4m50s. Both integration tests failed at supervisor socket startup, before any
workflow RPC; 0/2 passed. Runtime PID 73304 was sampled at `_dyld_start` for all
804 samples, 112 KiB footprint. The first startup failure has no equivalent
runtime sample; it must not be assigned the same exact cause without evidence.
Harness drop reported `reaped=false`; subsequent explicit process checks found
the Cargo, test and observed runtime processes absent. These failures stay in
the record. An identical runtime is copied internally for a separate startup
check; a test-only executable-path override retains the normal Cargo path by
default. No product code is changed by that relocation and no runtime test pass
is claimed yet.

The relocated-runtime run `native-capture-v3-run-2l9qm8p3` started both
supervisors and reached workflow operations. Both tests then failed on numeric
JSON assertions (`Number(1.0)` / `Number(2.0)` compared with integer JSON values).
This is a test-harness type mismatch, not a different observed numeric state.
Sources and the runtime SHA remained unchanged; both supervisors reported
`reaped=true`. The assertions now compare numerical values through `as_f64`.
The complete scenarios still require a new passing run.

The first build invocation used the wrong Cargo package name
`greppy-web-runtime`; Cargo rejected it before compilation. It is a harness
invocation error, not a product failure. The corrected package is `web-runtime`.
Its retained receipt is `native-capture-v3-run-xvqc8yqu`.

The subsequent run `native-capture-v3-run-g8zgy_jk` passed the explicit-tab,
navigation and overall-deadline scenario. The other scenario correctly stopped
at the unmet `Saved 3` expectation: my inline test handler used `value.value`
and wrote `Saved undefined`, while the input itself held `3`. The fixture and
independent value assertion now use `document.getElementById('value').value`.
An additional standard-browser probe using a data URL was blocked by browser
URL security policy; no standard-browser confirmation is claimed.

The final run `native-capture-v3-run-ckz2ukny` passed **2/2 native scenarios**
in 2.59 seconds after a 2m05s incremental compile, exit 0. Captured sources and
executables remained unchanged. It used the frozen internal runtime SHA256
`caed05280ccc134f713d7db1df1fa08abdcd8073b52c3e3b708b9f5740391d0a`.
This verifies the integration scenarios, not the complete native suite or
agent efficiency. The previous failed runs remain part of the record.

The actual CLI/runtime probe `native-workflow-cli-20260906-02` passed all
14 checks. It independently records exactly one server-side save with value 3,
confirms the delayed result in the returned workflow snapshot, verifies that a
later timeout stops the following save, and checks modal foreground output plus
full archived background recovery. Runtime stop returned `running=false` and
the HTTP fixture thread stopped. CLI SHA256:
`0a668f4cfad89b576a5875d6690929cd8061dc33022a5df04498465a24e9f1d9`;
the runtime is the same frozen `caed0528...` image. Both hashes remained unchanged.
The guarded CLI build `cli-guarded-fwwgbwtw` exited 0 with unchanged sources.

Probe 01 is retained as a harness failure: its generic command wrapper appended
`--json` after the last chain step. Greppy documents that chain flags precede
the first step, so the human output was expected; my JSON parser was wrong.
The corrected probe places `--json` immediately after `web do`.

That human output exposed a separate efficiency defect: each workflow step
repeats session/tab identity and protocol envelopes in the compact view. This
was reported with the full command, executable identity and captured output to
the existing fix worker. A compact workflow receipt is still required before
measuring the combined candidate. The observed focused modal view was 1,013
bytes, with a 2,816-byte full archive response; these are different views, not
provider token counts or an A/C performance comparison.

Acceptance remains open: new repeated Luna A/B/C comparisons, actual provider
input/output token reductions, paired time/p95, prepared-script performance,
the original development/held-out task population, and working Office fixtures.
