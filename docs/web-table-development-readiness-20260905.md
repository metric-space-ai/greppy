# Intermediate table workflow — local readiness, 2026-09-05

The opt-in `table` development case now combines region and stock filters, price sorting, row selection, a quantity form, modal confirmation, a row disappearing from the filter, and persistence after reload. It is a synthetic inventory workflow. It does not replace the real Office work or become a held-out acceptance case.

The task requires EU items with at least three available, ascending unit price, and exactly one reservation of three units of the cheapest eligible item. Prices and initial row order vary by seed. Browser rendering selects and sorts visible rows; the independent host oracle derives the expected item from original inventory facts and verifies filter settings, one reservation, total price and stock changes. The existing four-case default stays unchanged; new series must opt in with `--cases table`. Frozen old series were not modified.

Validation:

- 39 Python tests passed, including 13 new inventory cases. They cover deterministic paired facts with varying targets, persisted success, missing-filter failures, a valid reservation of the wrong item, duplicate reservations, invalid quantities without loss of prior progress, and dialog-origin enforcement. Existing fixture, symlink-launch, export and accounting checks also pass.
- An excluded readiness run used the standard Codex in-app browser on a newly assigned local origin, fresh tab 11: `http://127.0.0.1:62316/?run_id=a63478597e0c`.
- APAC correctly produced zero matching rows and a visible empty-state message. EU, capacity and ascending price were then set in one known-action batch. The resulting rows were Cedar EUR12, Ember EUR15 and Flint EUR21.
- Reserving three Cedar units through the dialog showed `Reserved 3 × Cedar. Total: EUR 36.00.` and reduced the filtered rows from three to two. Reload retained that result, the filters and sort order. The queried browser error/warning log was empty.
- The frozen host oracle then passed all five checks at revision5/event5. The extra event is the deliberate APAC empty-state probe; it is not part of the minimal goal flow.

The readiness trace includes two failed exact-label Playwright attempts before the successful native accessibility field action: first a CDP evaluation timeout, then a no-match response. Neither changed the filter. These failures stay documented; they are not Greppy defects, participant measurements, or a basis for a timing comparison. Native accessibility actions completed the workflow.

Evidence is under `/Users/michaelwelsch/.local/state/greppy-web-study/table-readiness-20260905-01/`: `setup.json` pins the frozen source bytes, `oracle-receipt.json` contains exact verifier argv/exit/stdout/stderr, and `verified-state.json` is the final state with its SHA256 in the receipt. Disposable fixture and live server state are under `/Volumes/tmp/dev-artifacts/greppy/web-efficiency/table-readiness-20260905-01/`.

This proves the local fixture flow only. The origin was new, but this was not a newly provisioned browser profile or a Luna participant. No Greppy run, actual Greppy-agent arm B, paired token/time comparison, or production/Office readiness is established here. A prepared Greppy smoke branch is supplied but has not been executed for this case. The next paired candidate condition remains dependent on native E1 and stable references.
