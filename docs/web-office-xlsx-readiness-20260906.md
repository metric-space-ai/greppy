# Real XLSX readiness and control trace: X2

Luna/medium imported an actual XLSX into CTOX Spreadsheets at
https://welsch.ctox.dev, edited its own new copy, saved and reloaded. Public
post-reload cell reads establish these eight values:

| Sheet | Cell | Value after reload |
|---|---|---|
| Overview | A1 | CTOX EDIT SAVE |
| Overview | A2 | GREPPY_X2_EDITED (requested change) |
| Overview | A3 | PRESERVE_XLSX_UNRELATED_6D2A |
| Overview | B4 | 125000 |
| Overview | B5 | 98000 |
| Details | A2 | ORACLE_SHEET_DETAILS_24AF |
| Details | B4 | 42 |
| Archive | A2 | ORACLE_SHEET_ARCHIVE_B83D |

The own import was renamed `Greppy XLSX readiness 20260906 X2`; its displayed
file is `edit-save-source.xlsx`, labelled XLSX. Root inspected the public
request/result pairs and original screenshots, including the changed Overview
after reload and Archive before reload. The post-reload verification selects
cells and reads the formula bar; it does not write their contents. Only the
participant's own tab was closed; its created document remains. The result
establishes same-browser reload persistence, not independently retrieved native
XLSX export bytes, a server oracle, deployment identity or paired efficiency.

The input was independently read with the bundled artifact-tool importer before
browser use. It contains Overview, Details and Archive; its original Overview!A2
is CTOX_EDIT_CELL_ALPHA. Source:
`/Users/michaelwelsch/Documents/ctox/tests/fixtures/office/spreadsheet/edit-save.xlsx`,
4962 bytes, SHA256
`58a12bbecfc82fa256915a0d65de35999f44170eddf39ef00cc3bfc7d8a9a640`.
The importer completed successfully after a delayed module import; no Greppy
product failure was established. Input inspection recorded the source unchanged.

Evidence root:
`/Users/michaelwelsch/.local/state/greppy-web-study/office-xlsx-input-20260906-01/`.
It contains `input-inspection.json`, the exact input copy, `trace-audit.json`,
18 original decoded screenshots with hashes in `media/index.json`, and the
lossless turn export, metadata and manifest under `luna-x2/`.
Turn: `01a07543-41fe-73e2-8e24-80dbd48e4363`.

Actual provider totals reconcile across all 65 unique model generations:
7,383,335 input tokens (including 7,223,040 cached input) and 12,369 output tokens.
There are 64 public tool requests with matching responses, including setup and
coordination. No usage conflicts or missing response IDs. Root sent progress and
bounded-task steering during this exploratory run; it is not an unsteered trial,
paired baseline or evidence that either tool is more efficient. A preceding
attempt in an older agent had no browser available and performed no product
mutation; it was not silently counted as a successful browser run.

Concrete public trace findings:

- `call_Is0npyN7Q5bJmPsjytaY21Xb`: the import button opens a form, while the agent
  waits for a file chooser, which times out. The actual file field subsequently
  triggers the documented chooser and accepts the input at
  `call_NHpEGicq1T03PuGPstTaxfDl`. The browser completed the intended upload;
  the failed first call is a mistaken workflow expectation, not file corruption.
- `call_XOxjjBPm4SpFjB5sWGkSR4xX`: an unqualified iframe locator matches three
  applications. The strict error identifies each frame; the successful path uses
  `iframe[title="CTOX Spreadsheets Editor"]` and its inner editor iframe.
- `call_dTRz9ecfM3eBHrTn1NuMoeR3` and `call_yRP6KYHUU0jAPjvx6Hd9ulz1` attempt
  first/last Archive matches that are hidden. Diagnostics explicitly report
  no_visible_match. The visible sheet-list menu provides actionable entries.
  These are usage/recovery episodes in the standard control, not Greppy bugs.
- `call_J8sRYuuIWPCLGL6drjaLfnGH` reloads. Its broad state diff includes unrelated
  applications. Their errors must not be attributed to the XLSX editor.
- `call_dT2PZ8WAzNVTtDiOlveRitvx` batches all five Overview cell reads after reload;
  `call_zY8n7mzIEFoNBHRI5LEmu9Nq` reads Details and
  `call_6BgAjZURUmy4mVfyAYrXazrO` reads Archive. These responses supply the exact
  table above. `call_mKlq3yK9ZKIEp8k4KFXgxOFV` closes the own tab successfully.

Root initially suspected incorrect newline escaping from serialized requests.
The matched responses at `call_256W8yd2FzY9yuszPEegViHh` and
`call_hlG1nOsIFj8BMWk7NtbEuaLN` are only 436/412 bytes and contain the targeted
lines. That hypothesis was withdrawn; request appearance alone was insufficient.

The engineering implications are explicit frame/document identity, actual visible
actions and scoped feedback. The new Greppy scoped-observation proof only covers
regions in one document. It cannot yet be credited with solving the nested
Office-frame problem. The findings and this limitation were sent to the existing
bug worker as control requirements, separately from the active expectation-query
diagnostics bug. No hidden reasoning content was inspected or quoted.
