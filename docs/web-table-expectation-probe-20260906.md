# Sort outcome conditions: controlled native probe, 2026-09-06

The S09 public call/result traces contained post-sort `STALE_REF` responses in four of five C trials. A new prepared technical probe isolates one factor: the outcome condition on the final sort action. It does not measure Luna performance or provider tokens.

## Design and frozen inputs

Five alternating weak/strong pairs use identical `basic-development-table-1` facts, a fresh server state and separate runtime owner per run. The unchanged S09 fixture is loaded from `/Users/michaelwelsch/.local/state/greppy-web-study/table-series-20260906-09/fixture`. No agent sees a prepared solution.

- CLI: `greppy 0.4.0`, SHA-256 `0a668f4cfad89b576a5875d6690929cd8061dc33022a5df04498465a24e9f1d9`.
- Runtime: SHA-256 `caed05280ccc134f713d7db1df1fa08abdcd8073b52c3e3b708b9f5740391d0a`.
- Evidence: `/Users/michaelwelsch/.local/state/greppy-web-study/table-expectation-probe-20260906-01`.
- Driver: `bench/web_study/basic_fixture/table_expectation_probe.py`.

Both conditions execute native region selection, capacity filtering and ascending price selection in one workflow. Only the final expectation differs:

| Condition | Final expectation | First ref click succeeds | STALE_REF | Correct item dialog |
|---|---|---:|---:|---:|
| Weak | `text=3 matching items` | 2/5 | 3/5 | 2/5 |
| Strong | `css=#price-heading[aria-sort=ascending]` | 5/5 | 0/5 | 5/5 |

The immediate next browser command clicks the cheapest item's reference from the workflow receipt. There is no intervening observation, diagnostic query or retry. The prepared driver derives the expected item from fixture facts; this is deliberately not an agent-discovery task. The endpoint is opening the correct dialog, not completing a reservation.

All ten runs applied exactly the three expected server filter mutations, with no reservation mutation. Every runtime stopped, the HTTP server thread stopped, and candidate/fixture hashes remained unchanged. The driver returned exit 0 for successful execution of the experiment; that does not turn the three failed clicks into successes.

## What the evidence establishes

All three failing weak runs returned fixture revision 2, then rejected the next reference with exit 34 and `STALE_REF`. The two successful weak runs and all five strong runs returned revision 3. Weak waits reported 6–9 ms; strong waits 12–18 ms. These are observed wait receipts, not end-to-end latency acceptance.

The fixture changes its selected option before an asynchronous server refresh replaces table rows. The number of matching items remains unchanged by sorting. Therefore the weak predicate can correctly hold before sorted rows are applied. The strong predicate observes `aria-sort`, which this fixture updates while rendering the returned state. Five trials support this mechanism; they do not prove absence of all rendering races on other applications.

Stale-reference refusal is correct. The error already gives actionable recovery guidance and supplies current page state. Changing it to silently retarget or blindly replay would damage correctness. The costly earlier observation is the target for improvement.

The returned snapshots contain controls and their selected values but no structured column-header sort state. None of the ten snapshots exposes `aria-sort`. Consequently an agent must discover a suitable completion condition through another inspection or use a weak visible text. This is an observation/efficiency gap, not evidence that the native condition evaluator falsely reports its predicate.

## Concrete next change and validation

Expose relevant table-header identity, name and sort state compactly in observations, along with a supported way to express an expectation on that state. Preserve queryable state transitions and distinguish a control's selected value from the table's applied sort state. Avoid unconditional full DOM/attribute dumps and avoid fixture-specific selectors or prescribed solution steps in Luna onboarding.

The existing worker owns product fixes. Send it the immutable evidence, command, version, exits and this classification. A proposed change must be tested for unsorted/ascending/descending states, delayed table refresh, unchanged matching-count text, stable identity until node replacement and fail-closed stale refs.

After a candidate is verified, repeat actual paired Luna trials with equal model/effort, unknown task discovery and actual provider counters. Keep A, B and C distinct. This prepared probe alone establishes neither token savings nor a general Greppy advantage.

## Reproduction

From `/Users/michaelwelsch/greppy-worktrees/web-efficiency`, with fresh scratch/evidence directories:

```sh
greppy bash-smart -- /usr/bin/python3 bench/web_study/basic_fixture/table_expectation_probe.py --cli /Users/michaelwelsch/.local/state/greppy-web-study/native-workflow-candidate-20260906/greppy --runtime /Users/michaelwelsch/.local/state/greppy-web-study/native-workflow-candidate-20260906/web-runtime --fixture /Users/michaelwelsch/.local/state/greppy-web-study/table-series-20260906-09/fixture --scratch /Volumes/tmp/dev-artifacts/greppy/web-efficiency/expectation-probe-01 --evidence /Users/michaelwelsch/.local/state/greppy-web-study/table-expectation-probe-20260906-01
```

Those directories now contain evidence and are intentionally refused for reuse. Individual exact commands, wrapper cwd/environment, stdout/stderr, elapsed time, snapshots and server states are archived per run.

## Office readiness, separate observation

The resumed Office check did not establish successful document creation. On the attempt to close the owned `officeRecheck` tab, the browser tool returned a URL-policy block referencing its generated “welsch.ctox.dev crashed unexpectedly” page. Closing is not confirmed. No alternate browser or URL was used to bypass that block. This is an environment/browser readiness limitation and is not classified as a Greppy product bug.
