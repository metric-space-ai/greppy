# Main integration, 2026-09-06

The owner explicitly requested bringing current development into main. This
integration preserves history; it is not a tag, package installation, release
approval, or a claim that previously independent results cover this union.

## Inputs

- origin/main `fe605caf`: existing macOS registration, FSKit attribute,
  tracker, JIT-signing and sandbox output fixes.
- `codex/web-efficiency` `537b6a8a`: native workflows, attached-page bindings,
  compact rendering and independent study evidence. Its working tree was not
  modified. S10 pair1 input regression, arm B and Office-frame/token acceptance
  remain open; readiness probes are not efficiency acceptance.
- `codex/web-post-action-state` `266b55e2`: previously tested scoped Observe,
  document/ref binding, bounded I/O deadlines, staging leases and diagnostics.
  Frozen binaries and prior receipts are preserved, not overwritten.
- `a871a0bb`: field-specific workflow preflight diagnostic. JS6/6 and a native
  engine projection17 checks passed; the newly linked runtime test is pending.
  CLI-help builds reached their 600s guard while compiling Metal dependencies;
  these attempts did not execute the help test. An earlier web-client14/14
  passed; the latest 60s rebuild also did not reach execution. No failure is
  converted to a passing test by this merge.

## Conflict decisions

- Keep Root's workflow dispatcher, capability advertisement, deferred
  observations and shared operation deadline; add scoped observations without
  changing the unscoped workflow observation path.
- Use the shared query resolver and expose both workflow and observation-scope
  client modules. Reject invalid browser grammar without graph-flag repairs.
- Keep both the CLI expectation-help regression and native scoped DOM test.
- Preserve newer probe capabilities and follow-up evidence from Root instead
  of reverting to earlier, partial descriptor/wait probes.
- Worker diagnostic conflict was formatting only; retain the same FD framing.

Unrelated historical worktrees, untracked files, installed applications and
other developers' caches are out of scope. All shipped-platform and exact-SHA
release gates still apply after integration.

## Union checks before the merge commit

- JavaScript contracts:69/69, no skipped tests, capture
  `/private/tmp/greppy-workflow-diagnostics.AyRAk9/native-workflow-run-z75_vk38`.
- Compiled web-client library:19/19, capture
  `/private/tmp/greppy-workflow-diagnostics.AyRAk9/run-tbnwv9lh`.
- Both captures exited0 with unchanged selected sources; `git diff --check`
  is clean. Full CLI and native runtime compilation/acceptance remain separate.
