# E1 native feedback and output preflight — 2026-09-05

Acceptance remains failed. These are prepared functional checks, not Luna runs or evidence of lower input/output tokens. The real Word/Excel scope and three-arm agent comparison remain open.

The native worker's runtime `57318ead7505fdf2aa7e62a89c511bc207a9c9c9848e50e11421fc208678d399` passed five targeted cases: stable node identity, post-action state with observation failure, inspection references, keyboard references and navigation references. The independently read identity and post-action logs contain terminal exit 0, assertions reached, and clean worker/supervisor shutdown. This does not close the earlier cold-start failure or prove full native conformance.

The unsigned debug package is `/Volumes/tmp/dev-artifacts/greppy/e1-native-candidate.jzAShE/web-runtime-dist`. Its manifest binds base `52c5095c2417805e721d20169c29b4729af923e5` plus source patch `ecf450d1fcfdd64094e873b08e6ad9b1d6409b642f199f0d3ff7818b27c84971`; it explicitly records an uncommitted source tree. All nine entries in the package's SHA256SUMS passed. The root CLI was built from `67c2de4dd0dd672132c4b6cd5d2cdeb06303c93d` with CI test assets and saved separately as SHA `cdbe3663a2b40947cd56baaec8caa6ff8a7ad0c3d17301b7a336c3ff11522638`. No installed app or prior candidate was replaced.

Full preflight evidence is under `/Users/michaelwelsch/.local/state/greppy-web-study/e1-output-preflight-20260905-01`: context, fixture pins, exact argv/cwd/timestamps, raw stdout/stderr, exit codes, frozen host-state snapshots, oracle results and output hashes. Browser actions used the explicit packaged runtime. The earlier short-alias context was prepared but did not execute browser actions. Both optional compact view flags were enabled.

| Prepared check | Result | Automatic page views | Raw stdout bytes |
| --- | --- | ---: | ---: |
| Open dialog case | Exit 0; controls immediately returned | 1 | 1,334 |
| Open its modal | Exit 0; new dialog refs immediately returned | 1 | 1,813 |
| Save after coordinator analysis pause | Exit 37, session wall limit; oracle false, zero events | 0 | 381 |
| Fresh dialog chain | Exit 0; oracle true, one save event | 3 | 4,847 |
| Fresh inventory chain including reload | Exit 0; all five oracle checks true, four events | 8 | 28,151 |

These are UTF-8 byte counts, not model tokens. The failed save remains failed; fresh chains used separate fixture run IDs. The session error was `wall time exceeded (182.047946583s > 120s)`, confirming the existing E6 limit in an interaction with coordinator pauses. No speed comparison is made from these loaded-host runs. A brief overlapping CLI compilation was identified and stopped from recurring; its duration is not benchmark evidence.

The inventory flow filtered EU and at least three available, sorted price ascending, reserved three Flint for EUR 27, and reloaded. The independent oracle checked region, capacity, ordering, exactly one correct reservation and stock effects. The final returned page displayed revision 4, the saved reservation and only the remaining eligible rows. This proves the prepared intermediate fixture flow, not an unknown agent task or real Office support.

Concrete remaining output defects were sent to the existing fixing task:

- Every ordinary button repeats false/null diagnostics and label-source metadata. The root opt-in renderer now recognizes only the known v2 defaults; checked/selected/expanded=false, redaction/truncation warnings, selected values, unknown fields and unknown schema versions remain visible. Thirteen renderer tests pass. The separately saved CLI from 7022e80e (SHA 8c8902c48f48fc902e404cfb32358655f8c418dbd4d2ee66ce5f0fc3c57dece8) repeats the prepared inventory flow with all five oracle checks passing: stdout is 15,852 bytes, with checked=false/true and selected options preserved. Eight automatic page views remain.
- `web do` prints every automatic page snapshot despite compact markers. Eight views and 28,151 bytes for one inventory chain make this an executed integration finding. Automatic intermediate state needs consolidation while explicitly requested observations, errors, partial execution and machine contracts remain available.
- The modal view includes background Save/open controls without a structural dialog relationship. The standard tool's inspected modal response contains the modal text and its Save/Cancel controls.
- Native labels contain embedded option text: `Region All regionsEUUSAPAC` and `Unit price order UnsortedLow to highHigh to low`. The fixing worker confirmed the source uses the wrapping label's unfiltered textContent. Native label and modal work is separate from the root renderer.

The dialog chain's final snapshot showed revision 0 while the later independent oracle recorded revision 1. A delivered click and an observed snapshot are not a verified application result; workflows with an explicit expected result still need that condition. The inventory chain used an explicit revision wait before reload. No action was silently replayed to obtain a newer observation.

The candidate harness now also clears inherited `GREPPY_WEB_RUNTIME_DIST`, because the resolver gives it precedence over `GREPPY_WEB_RUNTIME`. Nine environment/identity tests pass. This is a coordinator harness correction, not an engine optimization. New comparisons must keep corrected transport and runtime binding constant across candidate conditions.

## Timing remains unresolved

The first inventory run took 8.97 seconds and its first repetition after field compaction took 56.93 seconds. Both have successful independent oracles; the slower result is retained. These invocations lacked per-step time markers and had uncontrolled runtime/startup state, so the delay cannot be retrospectively assigned to the formatter or dismissed as a proven environmental cause. The smaller byte count does not pass the latency gate.

Two additional diagnostic repetitions used the saved `capture-with-milestones.py` to timestamp completed steps while retaining exact stdout. The old CLI completed in 12.18 seconds (first session step: 6.49 seconds); the new CLI completed in 2.71 seconds (first session step: 0.41 seconds). Both again passed all five oracle checks. The remaining chain took 5.68 versus 2.30 seconds respectively. This old-then-new pair identifies a substantial initialization/session component to measure, but does not prove a speed win or explain the earlier 56.93-second result. All observations remain in `v2-defaults-output-comparison.json` and `milestone-probe-report.json`. Warm and startup conditions must be controlled and repeated before timing acceptance.

## Subsequent chain experiment

The separately documented [chain aggregation preflight](web-chain-aggregation-preflight-20260905.md) reduces the prepared inventory reply to one automatic page view and 2,615 stdout bytes, with all five oracle checks passing. Twenty-eight focused tests pass. It also retains an unresolved copied-binary startup failure, an explicit cache-root harness mistake, and remaining native failure-feedback issues. No new Luna token comparison or latency acceptance was performed; the efficiency verdict remains failed.
