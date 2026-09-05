# Office readiness — 2026-09-06

Read-only check at 2026-09-05 23:40 UTC (2026-09-06 01:40 Europe/Berlin)
on https://welsch.ctox.dev, using the standard in-app browser and the existing
signed-in session. This is an application-readiness check, not a study trial.
The temporary tab was closed after inspection; no document or table was created,
edited, imported, exported or deleted.

Observed shell: v0.1.46-beta.1. Documents: v1.0.0. Tables: v1.0.1.
Existing records loaded. The document editor displayed:

> Dokumentfehler
> DOCX editor konnte nicht geladen werden: PEER_UNAVAILABLE

The table view displayed its existing record, toolbar and “Gespeichert”, but also:

> Editor konnte nicht geladen werden:
> Office RPC timed out: editor.open

An earlier loading screen or disabled iframe toolbar is not evidence that either
editor works. “Gespeichert” does not prove the editor loaded or that an edit was
persisted. The failures occur in the standard browser too; this check does not
classify them as Greppy rendering defects.

Real Office editing trials remain unqualified until both applications can open,
accept a reversible edit and independently verify persistence. Local interaction
fixtures remain development cases and cannot substitute for this acceptance.
