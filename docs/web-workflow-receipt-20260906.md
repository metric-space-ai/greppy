# Workflow receipt formatter: post-S09 verification

The fix worker changed only view.rs and new view_workflow.rs in Root's authorized
file window. Known v1 workflows now show a shared summary and ordered step lines,
without repeated matching session/tab IDs or protocol boundary fields. Action
return, expectation held, failure phase, attempted actions, completed steps and
retained effects remain explicit. Unknown envelopes fall back; unknown detail
objects remain verbatim. Compaction requires saving the complete original JSON,
and archive failure retains the original output. This is opt-in human output;
the JSON API and frozen S09 candidates are unchanged.

Root reviewed the actual diff and helper source. Verification:

- Worker's isolated harness imports the actual Root view code: 28/28 tests pass,
  including exact paginated archive reconstruction of three saved real workflow
  responses. Evidence: `/private/tmp/greppy-workflow-view-tests.flTHrX/ACCEPTANCE.md`.
- Root's compiled CLI web suite: 100/100 pass, capture `cli-guarded-gy815hp9`,
  sources unchanged, exit 0.
- Root's CLI binary build: `cli-guarded-phaamyjh`, sources unchanged, exit 0.
  Frozen CLI SHA256 `f3d06e96af0db82ce80cc74bcb33eff9cf24c5c49c806889156d154b7cd0bb21`.
- The actual CLI/runtime probe 03 passes 19 checks, including human workflow
  success/failure output, nonduplicated step identities, full archived receipts
  and current form value, independently recorded single save, and timeout stop.
  Both executable hashes remain unchanged; runtime stop reports running=false,
  and the HTTP fixture thread stops.

CLI: `/Users/michaelwelsch/.local/state/greppy-web-study/workflow-receipt-candidate-20260906/greppy`.
Runtime: the same frozen caed0528 image used before S09.
Probe: `/Users/michaelwelsch/.local/state/greppy-web-study/native-workflow-cli-20260906-03`.

The S09 reductions do not measure this formatter: it arrived after all ten
participants completed. Its provider-token effect still requires a new paired
study or controlled ablation. Page-scope default repetition, misleading observe
query handling, pending shell-result handling and asynchronous UI feedback
remain separate work. No release/default activation or overall acceptance.
