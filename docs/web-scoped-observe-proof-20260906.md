# Scoped observation: independent native mechanism proof

Root independently exercised the fix worker's unchanged development CLI/runtime
pair on 2026-09-06. All 19 checks passed. Five alternating whole-page/scoped
observation pairs each returned 29,653 versus 1,322 UTF-8 stdout bytes, a 95.54%
reduction. These are bytes, not provider tokens. No agent participated in this
probe and it does not establish superiority over the standard browser.

The fixture contains 60 unrelated application regions and one working form.
The scoped view retains the current field value and save outcome, excludes the
unrelated regions, and reports its explicit scope. The same elements retain their
references across whole-page and scoped observations. A missing region returns
NO_MATCH without widening to the page; the previously observed field reference
then remains usable. A real fill updates its current value, and the save button's
event counter confirms exactly one save. No fixture state is mutated directly by
the harness. All browser interactions use the candidate CLI/runtime.

Scope observations took 0.062–0.152 seconds; whole-page observations took
0.277–0.419 seconds in this run. These are uncontrolled technical timings on a
prepared fixture, with known targets and debug binaries. They exclude agent time,
and neither estimate end-to-end savings nor satisfy time acceptance criteria.

Evidence and executable identities:

- Probe: `bench/web_study/basic_fixture/scoped_observe_probe.py`.
- Evidence: `/Users/michaelwelsch/.local/state/greppy-web-study/scope-independent-20260906-01/`;
  `context.json`, complete `calls.json`, and `terminal.json` retain commands,
  outputs, timing, identities, checks and cleanup.
- Actual CLI: `/private/tmp/greppy-scoped-observe-tests.wGSgCs/greppy-a3d6d1f`,
  SHA256 `a3d6d1f5e77e44b8187ad7611735c06427d11701e5e263a270cf0865d4ffc31e`.
- Actual runtime: `/private/tmp/greppy-scoped-observe-tests.wGSgCs/web-runtime-current`,
  SHA256 `6e925e498338e5e9d5237de62aef7d2c86e7b6407a8a7b9be63864602321a134`.
- Both executable hashes were checked before and after. The unique runtime
  reported running=false on stop; the fixture HTTP thread stopped.
- Worker source commit: `266b55e28d7cad08d8927f839b5a76ce228af5cd`, prerequisite
  `d7e6486b36f269fafcb985c88a8c01835877adff`. These changes have not been integrated
  into Root's efficiency branch by this probe.
- Worker additionally preserved an immutable development pair and receipt at
  `/private/tmp/greppy-observe-candidate-266b55e2.ti34VI/`. Receipt SHA256:
  `b211f3abce7ff474a39078846cc4c83829baacd359bf18df03d87c3f5593952f`.
  CLI uses debug/ci-test-assets; this pair is not an inference, packaged or
  signed-release candidate.

This addresses one observed waste mechanism: explicitly requested regions must
not return unrelated application state. It does not fix the additional model
rounds caused by unclear expectation syntax or intermediate action feedback.
The S10 pair with higher Greppy input remains a regression; these byte savings
cannot be subtracted from that historical run. The next expectation-diagnostics
fix is assigned to the existing bug worker on an isolated branch from Root's
current workflow implementation. Fresh paired Luna runs, including the real
Greppy-agent arm, remain necessary after integration and verification.
