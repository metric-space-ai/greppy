# Independent native Ref/navigation retest

The existing `native_wait_probe.py` was run unchanged with the preserved CLI2f64
and new runtime0c5838. Evidence:
`/Users/michaelwelsch/.local/state/greppy-web-study/wait-native-20260906-02`.

Eleven positive checks: Observe/Inspect choices, strict timeout verdict and page
preservation, delayed DOM, explicit inactive tab, a negative other-tab condition,
valid absence, usable page after stale-reference refusal, URL wait across full
navigation, and unchanged binaries. Runtime and HTTP cleanup succeeded.

The new runtime resolves the previous full-navigation TIMEOUT from runtime d8f.
The overall command still exits1: the existing CLI does not yet pass the new
`condition_ref` metadata, so the typed stale-reference check remains unproven.
This is not a Page.url test; the cached synchronous getter remains separately open.

The fix worker's CLI slice e51cff91 was integrated as849eb448. Its first real build
failed E0308 in `condition_ref_selector`: `error.message` is Box<str>, while the
function returns String. Build receipt:
`/Volumes/tmp/dev-artifacts/greppy/web-efficiency/cli-guarded-_5tlwbe2/receipt.json`.
Cargo exited101, sources were unchanged, and no timeout/disk guard fired. This
compile issue was sent back to the fix worker for a narrow conversion fix. No new
CLI candidate or complete Ref acceptance is claimed from this failed build.

The worker's conversion fix was integrated as917d04d7. The resulting CLI f6ef88
with runtime4a7070 passes all 12 checks in `wait-native-20260906-03`, including
typed stale-reference refusal before absence inversion and full navigation.
CLI web tests: 88 passed; build and probe source/binary checks and cleanup passed.
This closes the tested CLI/runtime combination's Ref/navigation regression,
while Page.url, full native suite and signed release gates remain separate.
