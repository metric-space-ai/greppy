# Office readiness follow-up — 2026-09-06

Read-only check in the existing authenticated standard in-app browser at
https://welsch.ctox.dev. Shell 0.1.46-beta.1, Dokumente 1.0.0, Tabellen 1.0.1.
Opened one new browser tab and inspected the already selected documents;
no document content was edited or created. The test tab was closed afterward.

Both applications initially exposed nested editor iframes and partial controls.
That was not readiness: the table screenshot still showed loading skeletons
and an empty grid, without a usable toolbar. The document editor subsequently
showed `Dokumentfehler`, `DOCX editor konnte nicht geladen werden:` and
`PEER_UNAVAILABLE` in its visible UI. Screenshots and public accessibility
results are retained in the task's tool history.

The earlier Office blocker remains. This is not evidence of a Greppy engine
failure: the standard browser could not finish loading the selected Word file.
The table's `Gespeichert` shell status does not establish editor readiness or
successful persistence. Neither application is accepted as a working benchmark
fixture from this check. No provider token or performance result was measured.
