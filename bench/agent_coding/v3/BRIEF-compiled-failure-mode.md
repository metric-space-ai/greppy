# v3: a compile failure at the parent is a failure, not an exclusion

Read `bench/agent_coding/v3/FINDING_compiled_languages.md` first — it contains
the measured evidence and the exact discriminating condition. This brief
implements that fix.

Work in YOUR CURRENT DIRECTORY — the launcher has put you in an isolated
worktree of the repo. Do NOT cd to any other checkout.

## The defect

`adapters/base.py` validates in this order:

```
setup                       must succeed
parent PASS_TO_PASS         must pass
apply test patch
post_patch                  must succeed   <-- raises "offline post-patch setup failed"
parent + hidden test        must fail
```

For compiled languages the hidden test for a compile-time bug **cannot build
at the parent** — that IS the intended failure. Measured on cpp-fmt: PR 4836
closes issue 4794 "FMT_COMPILE ... fails to compile", and the harness dropped
it as `registered_budget_inexecutable`. Left as is, C++/Rust/Go/Java thin out
and the corpus acquires a language bias created by the harness.

## The fix

Distinguish, AT THE PARENT ONLY:

- `setup` failing BEFORE the test patch is applied → still a real
  infrastructure exclusion, unchanged.
- `post_patch` failing AFTER the test patch is applied → the candidate failed
  for the intended reason. Record `parent_plus_test = "fail"` and
  `failure_mode = "build"`, and continue validation.

The gold side stays strict and is what protects the rule: if `gold + hidden
test` does not build AND pass, the candidate is excluded as before. A test
that compiles nowhere must never be accepted.

Add `failure_mode` (`"build"` or `"test"`) to the validation record so the
post-hoc stratification can report how many corpus tasks fail at build time
versus run time.

## Acceptance — run these and paste REAL output
- `uvx pytest bench/agent_coding/v3/ -q` — green, including new tests.
- A test proving the asymmetry: parent post-patch build failure + gold builds
  and passes → candidate VALID with `failure_mode: "build"`; parent
  post-patch build failure + gold ALSO fails to build → candidate EXCLUDED.
- A test proving a setup failure before the test patch is still an exclusion.

## FILE WHITELIST
ONLY `bench/agent_coding/v3/adapters/base.py`, `bench/agent_coding/v3/adapters/test_adapters.py`,
and `bench/agent_coding/v3/test_pipeline.py` if a ledger field needs pinning.
FORBIDDEN: the registry, the corpus contract, adapter configs, `pipeline.py`
selection logic, anything outside `bench/agent_coding/v3/`.

## Hard rules
- Do not commit unless acceptance is green. Commit message:
  `fix(v3): a compile failure at the parent is the intended failure`
- ESCAPE HATCH: if you need scope beyond the whitelist, STOP and justify.
- NO SUBAGENTS.

## REPORT TAIL (fixed form)
CHANGED: <files>
OUTPUT: <acceptance output, verbatim>
TESTS: <suite result lines>
OPEN: <what you could not do, and why>
COMMIT: <sha or "not committed">
