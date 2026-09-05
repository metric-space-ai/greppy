# Deployed Office readiness — 2026-09-05, 12:58 UTC

The real site https://welsch.ctox.dev remains unsuitable for a completed Office comparison in this check. The standard in-app browser was used for excluded readiness inspection, not a Luna participant or a timed benchmark. No document contents or cells were edited; no file was created, imported, exported or deleted.

The page was authenticated and displayed shell v0.1.46-beta.1, Dokumente v1.0.0 and Tabellen v1.0.1. Browser tab 10 was created in the existing in-app browser 1 after the earlier tab 9 was absent. The new tab is marked for handoff.

Observed UI sequence:

1. Opening Dokumente displayed the selected “Neues Dokument 2” and `Dokumentversion konnte nicht geladen werden: Version doc_0d1190f0-d6e1-482a-844c-754331fcf528_v1 konnte nicht geladen werden.`
2. Selecting the first “Neues Dokument Word · 5.9.2026” row displayed the document-editor iframe and its initialization UI, then `DOCX editor konnte nicht geladen werden: PEER_UNAVAILABLE`.
3. Activating Tabellen displayed “Neue Tabelle”, the label “Gespeichert”, and `Zu dieser Tabelle wurde keine gespeicherte Version gefunden. Bitte erneut importieren oder den Datensatz verwalten.`
4. Selecting the first “Neue Tabelle - 2026-09-04 CSV · 4.9.2026” row changed the heading to that table and retained the same missing-version error. The “Gespeichert” label alone is therefore not evidence of editable or reloadable content.

These are observed application/data readiness failures. Their underlying cause is not established. They do not show that Greppy failed to render a working Office editor, and they do not prove every document in the instance is unusable. Browser transport also had an app-server observation error and a connection timeout; retrying the same browser/Tab10 recovered. A stale accessibility click for Tabellen failed before action; a label-based Playwright locator recovered. All of this is excluded from performance acceptance.

Further Office tasks require a working document/table version and editor, an isolated test document for each participant, and usable Greppy authentication. The earlier request for an approved Greppy test-credential reference is still unanswered. No substitute local Office clone is being used.
