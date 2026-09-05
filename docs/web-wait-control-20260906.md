# Native Wait integration: existing-backend control

The existing S07 CLI was exercised against the preserved newer runtime on the
same small fixture prepared for the native adapter. This is a correctness
control, not an agent trial, latency comparison or efficiency acceptance.
The host had another task's Cargo build running.

CLI: `/Volumes/tmp/dev-artifacts/greppy/web-efficiency/cli-candidate-inspect-154d1a775f41/greppy`
SHA256: `154d1a775f4156c6d33742ac309e063a49563a57e16cfb6a92c97a20a8082471`.
Runtime: `/Volumes/tmp/dev-artifacts/greppy/native-select-wait-preserved.oMr7VT/web-runtime`
SHA256: `d8f437c493d7780f39bb8868da9227ce8600dadaa7a7e393936954cf1877277b`.

Run `native_wait_probe.py` with explicit `--cli`, `--runtime`, create-only
`--scratch` and `--evidence` directories, and `--backend legacy`. The default
backend remains native and requires the new Inspect contract and native Wait
result evidence. Legacy mode records the older Inspect capability without
pretending that it implements the new contract. Both paths require correct
reference invalidation; that requirement has not been relaxed.

Evidence roots: `/Users/michaelwelsch/.local/state/greppy-web-study/wait-legacy-20260906-01`
through `wait-legacy-20260906-04`. Every attempt retains raw argv, stdout, stderr,
exit codes, elapsed times, candidate hashes and cleanup results.

- Attempt 01 used worker CLI1d53, which lacks explicit tab conditions. This was
  a Root candidate-selection mistake. Its misleading suggestion to replace
  `--tab page-…` with `--max page-…` was independently reproduced and reported
  as a diagnostics bug; the suggested command fails integer parsing.
- Attempt 02 used the appropriate S07 CLI but the initial probe incorrectly
  required its Inspect output to implement a newly added feature. This was a
  control-expectation mistake. The native candidate still must expose choices.
- Attempt 03 reached reference invalidation and failed the required typed
  `STALE_REF` check. It stopped before navigation.
- Attempt 04 collected independent verdict failures and ran the remaining
  checks. It ended **exit 1**, with ten passed checks and the same failed
  reference check. The failure was not converted to success. Both candidate
  hashes remained unchanged, own runtime returned running=false and the HTTP
  thread ended.

The control confirms delayed DOM appearance, real absence, timeout with a false
verdict, continued page usability, explicit inactive-tab targeting, isolation
from another tab's state, and waiting across delayed full-document navigation.
After replacing the input, Wait on its old `@3` reference returns exit34 with
`engine_error` and a JavaScript selector SyntaxError instead of `STALE_REF`.
The replacement has a distinct `@801` ref; there is no observed false absence
success. Recovery nevertheless incorrectly suggests retry or web.doctor.

Current source `condition_expression` passes the query through the shared
`query_expression_pub` resolver. Validation accepts an `@…` string as bare CSS.
The new adapter uses the same compiler, so its reference-condition contract is
still incomplete despite passing its Boolean-response unit tests. The fix
worker received the exact reproduction and a request for shared active/stale/
removed/cross-document/cross-tab reference tests. Root has not patched around
this product bug or weakened the native acceptance check.

The native adapter still needs its executable and real integration run. The
control's successful navigation check defines behavior that the optimized
backend must preserve. No native navigation success is inferred from it.
