# Office readiness: correct the window attribution before measuring

This is a historical readiness and trace audit, not an agent benchmark or an Office acceptance result. W1 and W2 below did not establish Word save/reopen success. A later root W3 attempt did verify one own document across save and fresh navigation; see [the W3 evidence](web-office-word-readiness-20260906.md).

## W1: the reported error belongs to the spreadsheet area

Evidence directory: `/Users/michaelwelsch/.local/state/greppy-web-study/office-readiness-20260906-02/word/`. Original turn: `01a07517-f089-7c00-af3b-5524c723835d`, Luna/medium, standard in-app browser.

- `call_uIAsfm5Y24DsjA70AxPL3y5t`, response source line 45: AX235 is explicitly **Leeres Word-Dokument erstellen**, within the Dokumente container. The full desktop tree includes several other open apps and is truncated after 14093 further characters.
- `call_wIvJPcX4gSHGV4xq1lzwbS49`, response line 52: the create click returns a desktop-wide diff. Word font controls remain visible (Times New Roman), and the focused element is text entry area401. A spreadsheet blob error also appears in this diff. Temporal co-occurrence alone does not establish that the Word action caused it.
- `call_KmFrJIVX0UAGtv5QHFIU12WP`, response line 59: the first whole-body text query fails with **Unterminated string literal**. This is a script syntax error, not an editor failure.
- `call_TCCwPoX1mnCUsZoaMLkZNBy2`, response line 66: the corrected whole-body text excerpt places **Blob has no streamed chunks: sheet_…_v1_blob** below **Tabellen / Neue Tabelle / Gespeichert**. It must not be attributed to Word.
- `call_33o9goZKeDVV31v91G8uReFO`, response line 73: the own-tab close completes and prints TAB_GESCHLOSSEN.

The initial subagent conclusion that an editor/sync error prevented the Word test is withdrawn. The observed error belongs to the spreadsheet section. Word editor controls were present, but the evidence does not establish a newly created document's identity, title, input, save or reopen. The intended W1 title was never entered; its absence cannot prove that no untitled document was created. Existing documents were not subsequently modified or deleted to repair this uncertainty.

The control finding is **incorrect cross-window error attribution in the agent's global observation**, not a demonstrated Greppy defect. It supplies a concrete requirement for future scoped observations: relevant errors, action effects and references must preserve the app/dialog/frame context. Unrelated visible errors must not become a claimed result of the current action.

## W2: browser unavailable before navigation

A single corrected attempt required explicit Word-container scope and proof of a newly created document's identity before editing. Export directory: `/Users/michaelwelsch/.local/state/greppy-web-study/office-readiness-20260906-02/word-w2/`; turn `01a07523-406b-75f3-9dea-4c59035991ba`.

- `call_qLDNFUtnvPP3NZQawA2bdRQt`, response line 113: creating an own tab fails with **Browser is not available: 1**.
- `call_IGYjTKqPRRiq52L4yNnA7GHV`, response line 127: the documented availability check returns an empty browser list.

No W2 navigation, document creation, editing or save occurred. This is a browser-environment readiness failure, not evidence of a CTOX or Greppy application defect. No alternative backend was substituted.

## Local Office bundle integrity

`local-integrity.json` in the evidence root records 797 declared path/hash comparisons from source_inputs, bundle_inputs, outputs and artifacts: all matched, none missing. The comparisons include repeated paths across provenance groups; this is not a claim of 797 distinct files.

CTOX HEAD during the check was `ec3b378dd68951078ae863252025638f7fe913de`. The provenance manifest was unchanged during the audit. This read-only check neither freezes concurrently edited source files nor establishes the deployed bundle's identity. Local source/bundle consistency does not prove Word or Excel usability on welsch.ctox.dev. No CTOX source, data or deployment was changed by this audit.

## Next readiness gate

The later W3 attempt passed the Word portion of this limited readiness gate. The later [X1 spreadsheet check](web-office-spreadsheet-readiness-20260906.md) retained a CSV-labelled table's visible values and formula after reload. Independent outcome oracles and actual XLSX coverage remain prerequisites for the Office A/B/C efficiency study. Failed or inconclusive readiness attempts remain archived.
