# Real CTOX Spreadsheets: CSV-labelled editor readiness and recovery costs

One fresh Luna/medium standard-browser attempt created an own table, named it **Greppy Office readiness 20260906 X1**, entered four cells, and retained the visible values after reload. A1 is `Greppy X1`, B1 is `7`, C1 is `3`, and D1 shows `10` with formula `=B1+C1`. Post-reload actions select A1 and D1 only; no new content is entered. Root and a separate Luna reviewer inspected the original images and public calls.

The application labels this file **CSV**, filename `neue-tabelle-2.csv`. It runs in the real CTOX Spreadsheets editor, but this is not XLSX persistence, exported-file round-trip or independent server-oracle evidence. The own tab closes successfully; the test table remains as evidence. No preexisting table is edited or deleted.

Evidence root: `/Users/michaelwelsch/.local/state/greppy-web-study/office-readiness-20260906-03/`:

- `excel-luna/turn-01a0752c-cb26-7421-96fc-a41531e0d62c.metadata.json` and its original trace/manifest.
- `excel-media/index.json`: original tool-result images, decoded unchanged with hashes and source call IDs.
- `excel-x1-readiness.json`: outcome, provenance, explicit limitations and reconciled provider counters.

This exploratory single-arm attempt uses **33 tool requests / 34 model generations**, **2537743 input tokens** (including **2452736 cached input**) and **6354 output tokens**. Per-response provider counters sum exactly to the cumulative totals. This is retained discovery cost, not a comparison with Greppy or an accepted benchmark baseline. Startup, mistakes, observations and recovery all remain counted.

## Public interaction evidence

| Episode | Calls | Observed result and classification |
|---|---|---|
| Initial target invalidation | `call_0BpOVhLjce9LaZHIQim5gLmU` | AX click fails: No node found for given backend id. The old reference no longer resolves; precise cause is not established. |
| Ambiguous app name | `call_WIzOrsWe7J3jpHam4S2VAXA6` | Exact Tabellen matches both taskbar button and desktop icon. Strict-mode refusal supplies both matches; first() then chooses the already observed taskbar entry. |
| Own creation identity | `call_EzZWxswzKBu4EpZQ7Dv7512L`, `call_dZ99M8mYDH09uoePpZWGtDow` | Exactly one create, list changes from three to four tables; manage dialog explicitly identifies Neue Tabelle 2 before renaming. |
| AX/DOM label mismatch | `call_aMdCdg9VIseLJV8Nudc3igNx`, `call_Jvx5K9ubG8LO0ErPzXw68goV`, `call_t8frsdBoJcCKis0EpE8FeWYi`, `call_CB4fkl4CMhoyi04jZHzxdLyE` | AX reports TITEL; exact Playwright TITEL has zero matches, then a guessed aria-label also fails. DOM snapshot exposes Titel; that exact locator succeeds. This is a mismatch between observation and locator naming, with an additional guessed-selector recovery. |
| Efficient known cell sequence | `call_MsimgI9gIBfKPQsKZkgxSN9b` | One screenshot-grounded coordinate click, ordinary typing and Tab navigation fill A1:D1 in one call; Return completes formula input. |
| Save locator fallback | `call_wxF7adD6pH3fgZywqMID5YS6`, `call_Yew3OxLJzTWSDQFbYr1HNeWr` | Exact and regex page-level Save locators both return zero matches. A later AX read exposes editor Save controls in inner frame contexts. This is consistent with a frame/locator-boundary mismatch, not proof that a Save button is absent. |
| Repeated save and conflicting status scopes | `call_inBMgOQwo69QbV54CvBMy7d7`, `call_nw9q4JYanjC1ZFemNowo7UeZ`, `call_rz0cY9bXheGcQa0QZ0NvJ1Vq` | AX click on the observed spreadsheet Save button still yields Ungespeicherte Änderungen after one second. Following Meta+s and another wait, the shell says Gespeichert while the editor says Kalkulationstabelle wird gespeichert. A preceding asynchronous save may have completed; the shortcut is not proven uniquely responsible. |
| Persistence | `call_72SrkubsnrQW0HDZSkyf1oYr`, `call_qDPBI6Af4GcXv7Rjjb1CdW8I`, `call_q7Xl6IvtbhZ7S2dLLn8D0nS9` | Fresh reload retains own title and displayed row. Formula-bar reads by selecting A1 and D1 show exact text/formula despite clipped cell rendering. |
| Cleanup | `call_WLsG9rz6KvBfiRVset48lWqn` | The own tab closes, confirmed by closed_tab_id=1. |

The last screenshots still contain the editor saving message alongside the shell saved header. Readiness rests on visible content after reload, not on treating either message alone as a complete-save guarantee. The unrelated Blocks error during reload remains outside this spreadsheet outcome.

## Requirements for the next Greppy comparison

- Carry the canonical name used by the action resolver with each reference, together with the visible label when it differs. Test an uppercase-rendered label with mixed-case accessible name.
- Preserve app, frame and document identity through lookup and action. Test two Save buttons in separate editor frames and duplicate shell app launchers.
- Expose frame replacement and reference invalidation explicitly. A fallback must never target another element that later receives a recycled identifier.
- Separate shell persistence status from editor save progress. Derive completion from a defined task oracle, and retain conflicting states rather than collapsing them into a false success.
- Compare structural actions and screenshot-grounded keyboard sequences fairly. Both tools may batch the four known cell inputs; Greppy must show lower actual input and output costs over repeated matched tasks.
- Add an actual XLSX input/export case and independent native-file checks before claiming spreadsheet-format acceptance. This CSV-labelled readiness flow does not satisfy that requirement.

These are requirements inferred from a standard-browser control trace. No Greppy defect, speedup or causal token saving is asserted without reproducing it on Greppy. Worker product fixes remain separate from research and validation.
