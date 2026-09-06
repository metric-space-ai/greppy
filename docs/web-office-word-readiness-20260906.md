# Real CTOX Word: first verified save/reload readiness

The standard in-app browser successfully created, named, edited and saved one own Word document on `https://welsch.ctox.dev/`. Its exact title and paragraph remained visible after a fresh navigation to the same URL. This is a single root readiness check using GPT-6 Astra, not a Luna comparison, independent server oracle or efficiency acceptance result.

Evidence root: `/Users/michaelwelsch/.local/state/greppy-web-study/office-readiness-20260906-03/`. `word-w3-readiness.json` identifies the original screenshots, their hashes and public calls. The original tool-result images are decoded unchanged into `word-w3-entered.jpg`, `word-w3-after-save.jpg` and `word-w3-after-reload.jpg`. The trace export is an immutable prefix of active root turn `01a07527-58fb-7970-9982-cb3d3d2622f8`, captured after this browser workflow and cleanup; it is not a completed-turn usage record.

## Outcome and ownership evidence

| Step | Public call | Evidence |
|---|---|---|
| Establish initial state | `call_MbaOwC2fLpzSSFOWjdosjv35` | Five existing files; selected editor is Neues Dokument 4. Word and spreadsheet areas are distinct. |
| Create once | `call_OdFmNkWyawiqhsy0jko2ZZWK` | Click observed Leeres Word-Dokument erstellen. No create retry. |
| Identify new document | `call_aqE7aVZ1WvThlj3ATfQOyNtm`, `call_oK39K7P7q2KolklGYVAOZNgT` | New sixth file Neues Dokument 5; exact filename neues-dokument-5.docx and its own manage button. |
| Name own file | `call_6BKafBjzCVZX34TGovWWsPJx`, `call_vESAPUK6MDLS4uOeF0GrqJGF` | Manage dialog identifies neues-dokument-5.docx. Save title Greppy Office readiness 20260906 W3. |
| Establish input target | `call_jErdIz7aFqMZYWTxrEUjI56s` | Inner Word frame now names neues-dokument-5.docx; focused settable area752 belongs to that frame. |
| Enter and inspect | `call_pWNVadE0M3y9M3COCMDKJuNt`, `call_4wg219Mjwmobp6En2Sm9UA8X` | AX setValue enters the paragraph; screenshot shows Greppy browser readiness W3 — saved and reopened. |
| Save | `call_waCc4Tc98aZxohsuFv7nZwcU`, `call_Y2qlHEIwD0VHd7a4u87ZlMUX` | One explicit click on editor Save; subsequent screenshot retains paragraph and no longer shows the saving message. |
| Fresh navigation and restored document | `call_Ahi22Z05xeFJWXQck4xippYR`, `call_uihut48oJ1Ij24O3qg8dxjc2`, `call_yuhifOwalUTB3oFaZSstMYiR` | Browser navigates anew, passes startup loader, restores the own title and rendered paragraph. No input follows navigation. |
| Close own tab | `call_tNGnLx5EhEGnXwqDlQan2Uxo` | Tab17 closes successfully. The created evidence document is retained. |

No existing document was edited or deleted. The unrelated old CSV blob error was not treated as a Word failure. This supersedes the earlier absence of Word save/reload evidence, while retaining W1's inconclusive creation and W2's browser-unavailable attempt in the historical audit.

## What the interaction reveals

1. **Creation and editor selection are separate observable transitions.** The new row appears after the create receipt, while the body still identifies the previously selected document. A successful create call or newly visible row is insufficient evidence that the input target is the new document. The study must verify row identity, selected title and inner frame before typing.
2. **Saving replaces the editor frame.** `call_H2V7DtD5c6bOxpiH6QHygCBJ` removes the previous editor subtree and returns an initializing frame with disabled controls. Later rendering restores the text. A reference must not be reused across this replacement, and the replacement must not be mistaken for data loss.
3. **The meaningful paragraph is rendered outside the observed text tree.** The AX input operation enables Undo, but the observed AX diffs do not expose the paragraph, and outer body.innerText omits the inner canvas document. Screenshots provide positive rendered-content evidence here. This does not prove that every supported accessibility or UI read method is incapable of reading the text; a later export/UI oracle is still needed.
4. **Unrelated apps dominate global updates.** Restored Research and spreadsheet windows produce updates alongside Word. This motivates measuring scoped output against the global output while preserving window/frame attribution, rather than removing apparently irrelevant lines with an unconstrained summary model.
5. **A delta is not a full tree.** The root's first manual scope projection (`call_vOhHMH1oOVajIQExNfn3YVqb`) searched for an unchanged ancestor in an AX diff and falsely printed that the window was absent from that observation. The next raw diff exposed the mistake. This recovery remains evidence of harness misuse; it is not classified as a Greppy product failure. Future scope/delta tests must cover unchanged ancestors, changed descendants and explicit baseline resets.

## Consequence for the Office study

The independently assigned real spreadsheet check also retained its values after reload; see [the CSV-labelled editor evidence](web-office-spreadsheet-readiness-20260906.md). Task-specific outcome oracles, actual XLSX round-trips and matched Luna A/B/C runs remain open. A candidate must preserve selected-document and frame identity, distinguish saving/reinitializing from a completed outcome, and provide enough evidence to avoid repeated full-tree or screenshot checks. These are hypotheses and acceptance needs derived from actual editor flows, not measured Greppy improvements yet. The twelve-task plan and both token requirements remain unchanged.
