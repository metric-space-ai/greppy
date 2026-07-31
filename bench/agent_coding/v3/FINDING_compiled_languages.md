# Compiled languages cannot fail the way the harness requires

Found on the first real validation run (cpp-fmt, 21 harvested PRs, 2026-07-31).

## What the ledger said

```
outcomes: failed 4, not_run 17
causes:   17x not_merged_or_linked_issue        (verified genuine against the API)
           3x offline post-patch setup failed -> registered_budget_inexecutable
           1x candidate has no derivable independent behavior tests
```

Zero tasks from a repository that needs to contribute eight (six plus two
reserves).

## Why the three "budget inexecutable" rows are not an infrastructure problem

The validation sequence in `adapters/base.py` is:

```
setup            (cmake configure + build)        -> must succeed
parent baseline  (PASS_TO_PASS)                   -> must pass
apply test patch
post_patch       (cmake --build)                  -> must succeed   <-- fails here
parent + hidden test                              -> must fail
```

For an interpreted language this is right: the new test imports fine and fails
at runtime. For a COMPILED language it is usually impossible. Take the actual
candidate:

> PR 4836, issue 4794: *"FMT_COMPILE with user-defined type formatted via
> format_as fails to compile"*

The hidden test for that issue **cannot compile at the parent commit** — that
is precisely what the bug is. The harness sees the post-patch build fail,
concludes the task is not executable under the registered budget, and drops
it. The same will happen to most C++, Rust, Go and Java candidates whose fix
is a compile-time behaviour: a type trait, a generic bound, an API signature,
a missing overload.

The consequence is a silent language bias in the corpus: Python, Ruby and
JavaScript survive, compiled languages thin out — produced by the harness, not
by reality. Eight of the plan's 24 repositories are C++/Rust/Go and three more
are Java.

## The discriminating condition, which the harness already has the parts for

A post-patch build failure at the parent is the intended failure signal **iff**
the same build succeeds with the gold patch. That is exactly the pair the
validation already computes:

```
parent + hidden test   -> build fails OR tests fail   = "fail"   (both count)
gold   + hidden test   -> build succeeds AND tests pass = "pass"
```

So the fix is to distinguish, at the parent only, between

- the setup build failing BEFORE the test patch is applied — that is a real
  infrastructure failure and must stay an exclusion; and
- the build failing AFTER the test patch is applied — that is the candidate
  failing for the intended reason, and must be recorded as
  `parent_plus_test = fail` with an explicit `failure_mode: "build"`.

The gold side must remain strict: if gold+test does not build and pass, the
candidate is out. Without that asymmetry the check would accept a test that
never compiles anywhere.

`failure_mode` belongs in the ledger, and the post-hoc stratification should
report it — "how many tasks fail at build time versus at run time" is a real
property of the corpus and a reader will want it.

## Second-order finding: the linked-issue rule dominates the yield

17 of 21 candidates were excluded because the merged PR closes no unambiguously
linked issue. Spot-checked against the GitHub API: genuine (PR 4844 has no
closing reference; 4836 and 4819 do and were admitted). The rule is correct and
should stay, but it means the achievable yield per repository is roughly a
fifth of the merged PR count before validation even starts.

With a 2.5-month window fmt produced 4 admissible candidates. Eight tasks per
repository will therefore need either a wider window (the contamination floor
allows extending forward to today, not backward), repositories with stronger
issue-linking discipline, or fewer tasks per repository across more of them.
This is a corpus-feasibility question the plan should answer before Phase 4
runs for days across 24 repositories.
