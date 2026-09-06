# Table interaction feedback — native reproduction

This is a deterministic functional preflight, not a Luna trial or a provider-token
comparison. The fixed fixture seed is `basic-development-table-1`. Its independent
oracle requires EU, capacity >=3, ascending price, exactly one correct reservation
of three units, and the correct remaining stock after reload.

Evidence root: `/Users/michaelwelsch/.local/state/greppy-web-study`.

- `table-feedback-20260906-01`: CLI2f64/runtime d8f. Individual actions plus an
  explicit native wait pass the oracle. Exactly one Confirm was sent. However,
  the select response already reports value `ascending` while its snapshot still
  contains prices 15,25,18 (revision2). Confirm returns revision3, an actionable
  Confirm button and “No reservations yet”; the immediate independent server
  snapshot is already revision4 with the correct reservation. A native wait for
  dialog absence returns revision4 after7ms. This demonstrates an asynchronous
  intermediate observation, not by itself a false business-success assertion.
- `table-feedback-20260906-02`: CLI2f64/runtime0c5838. The two-step chain ran with
  exit0, but my harness incorrectly assumed `web do --json` was a single JSON
  document. It actually emits a compatibility stream including step records and
  final summary. The parser failure is retained; it is not a native runtime bug.
- `table-feedback-20260906-03`: same pair, human compact mode. Confirm plus native
  wait and the independent reload oracle pass, with exactly one Confirm. The
  chain returns5557bytes, including both complete page states and verbose option
  metadata. Runtime/HTTP cleanup and binary/fixture integrity checks pass.

The CLI is `/Volumes/tmp/dev-artifacts/greppy/web-efficiency/cli-guarded-s_j4y13g/greppy`,
SHA256 `2f64ac54a23912e548bcad6d969b405973541fdcaa8af3efee83c82da63e4401`.
Runtime d8f and0c5838 are pinned in each context.json, alongside the exact isolated
working directory, wrapper, commands and captured output.

## Integration defects identified

`chain_output.rs::automatic` does not recognize the new `select_choices` field.
Consequently the intended automatic-observation aggregation refuses table pages.
`view.rs` prints the field as an unknown full JSON object, repeating its schema,
counts and false flags in each state instead of a compact label/value list.

Once aggregation is restored, successful native waits also need careful treatment:
the old `held_wait` path retains a prior automatic observation and flushes it after
the explicit wait. A newer validated state from the same session/tab must remain
the current view; an older one can be archived with provenance. Legacy waits
without snapshots cannot manufacture a new state, and errors, unknown fields,
truncation and scope changes must retain their diagnostic evidence.

Both defects and the exact native reproduction were sent to the designated fix
task. `table_feedback_probe.py --delivery chain --require-compact-feedback` now
adds a separate prospective feedback gate: one current content block, visible
success and useful choice values, no obsolete Confirm state and no repeated false
choice flags. Its functional oracle remains independent. This gate does not
establish lower provider input/output tokens; fresh paired Luna trials still must.

The stricter gate was then executed against the unchanged candidate in
`table-feedback-20260906-04`: outer exit1 despite a passing functional oracle.
It detects two content blocks, obsolete confirmation state and32 redundant false
choice flags in5557bytes. All source/binary hashes and cleanup checks pass. This
retained failing receipt is the before-case for the formatter fix.
