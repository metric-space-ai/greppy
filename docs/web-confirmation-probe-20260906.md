# Confirmation feedback probe, 2026-09-06

A second controlled probe changes only the final confirmation expectation. Five alternating weak/strong pairs use the unchanged S09 CLI (`0a668f4c…`), runtime (`caed0528…`) and frozen table fixture. Fresh contexts and data are created for every run. The driver first waits for applied sorting to isolate confirmation from the previously observed stale-row problem.

| Final condition | Old dialog returned | Saved result returned | Independent correct save and reload |
|---|---:|---:|---:|
| Existing item-name text, `text=Ember` | 1/5 | 4/5 | 5/5 |
| Reservation result, `css=#reservation-status p` | 0/5 | 5/5 | 5/5 |

There is exactly one confirmation per run. The prepared driver does not retry or press Enter. The weak third run returned an old modal without saved-result text, while the immediate independent server oracle already confirmed the correct reservation. All ten later reloads showed the saved result, and server states contained exactly one reservation with the expected stock change.

This supports a feedback-timing mechanism visible in the S09 public traces. It does not show that every weak predicate fails, that native `held=true` was false, or that a specific token reduction follows. The item name was already present in the dialog, so its presence is not a save-completion proof. The strong condition is specific to the prepared fixture and must not be supplied as a solution to Luna.

Evidence: `/Users/michaelwelsch/.local/state/greppy-web-study/table-confirmation-probe-20260906-01`. Per-run calls contain exact argv, stdout/stderr, condition receipts and snapshots; independent immediate/final server states and reload evidence remain archived. Driver exit 0 means the experiment completed; individual endpoints are reported above. All ten runtimes and the HTTP thread stopped. Candidate and fixture hashes were unchanged. Provider tokens are unavailable because this is not an agent trial.

Reproduction uses `table_expectation_probe.py --stage confirmation` with the same S09 CLI/runtime/fixture arguments as the sort probe and fresh scratch/evidence directories. The executed directories were `/Volumes/tmp/dev-artifacts/greppy/web-efficiency/confirmation-probe-01` and the evidence path above. Existing evidence is never overwritten.

The worker received this result as additional evidence for the existing observation/efficiency investigation. A false business-success claim is not inferred from a correctly satisfied DOM predicate. Product work should make relevant outcome state and unfinished effects understandable without encouraging duplicate mutations.

Arm B readiness was separately rechecked: `greppy workspace status` in `/Users/michaelwelsch/Documents/greppy` returned exit 64 and a provider heartbeat stale by 139495 seconds. This is the existing environment/recovery blocker. No new Greppy-agent model run or extension activation is claimed.
