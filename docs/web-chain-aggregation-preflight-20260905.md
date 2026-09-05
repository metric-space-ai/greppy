# Chain observation aggregation: functional preflight, 2026-09-05

**Efficiency acceptance remains failed.** No new Luna comparison or provider token measurement was performed. The prior input/output regressions remain the authoritative agent results. This experiment changes CLI feedback only; commands still execute sequentially through the existing runtime calls.

Source `2572b96ff6bd9339e2ff96cc099a2bee10f07058` passed 9 aggregation unit tests, 13 renderer tests and 6 CLI integration tests (terminal exit 0). The latter include byte-identical JSON output with and without the compact human-view flag. Both `GREPPY_WEB_VIEW=compact` and `GREPPY_WEB_CHAIN_VIEW=compact` are required; defaults are unchanged.

Within a known successful action chain, the last automatic observation is returned and earlier observations are archived privately. Explicit queries, errors, unavailable/truncated state, unknown fields, scope changes and storage failures prevent silent consolidation. Successful wait responses remain explicit. A storage failure emits every retained response; it never replays actions. The history is marked as earlier state with potentially stale references.

## Actual prepared inventory flow

The frozen table fixture and original E1 runtime package (SHA `57318ead7505fdf2aa7e62a89c511bc207a9c9c9848e50e11421fc208678d399`) were retained. Source changes were built before committing, with no edits during compilation. The resulting CLI SHA is `555cd58f17c857607d3852183280dbba790a307b5562d7a9025777f271908137` (208,828,824 bytes).

| CLI feedback condition | Automatic page views | Raw stdout bytes | Independent table checks |
| --- | ---: | ---: | --- |
| Initial compact renderer | 8 | 28,151 | 5/5 |
| Known-v2 default-field compaction | 8 | 15,852 | 5/5 |
| Default-field compaction plus chain aggregation | 1 | 2,615 | 5/5 |

These are byte counts, not input/output tokens. All invocations used a prepared solution. They do not show how often Luna would discover or use a chain, trust its feedback, request history, or fall back. The latest run reserved three Flint for EUR 27, retained region/capacity/order, reflected stock changes and persisted through reload (revision 4, exactly four events). Its observed 5.58-second duration was under concurrent native compilation and is not a latency comparison.

The last state still contains redundant receipt metadata and verbose native select labels. Successful waits retain individual output. Requesting history currently returns raw archived protocol data in bounded pages; the first page leaves 24,307 bytes. Whether an agent needs that history, and how much recovery it causes, must be measured. This is not a finished efficient interface.

## Retained startup failure and alternative path

The separately saved immutable CLI copy under `/Volumes/tmp/dev-artifacts/greppy/web-efficiency/cli-candidate-2572b96f/greppy` produced no output for 323.39 seconds and was stopped by the coordinator with SIGTERM. The child exit is -15; the Python/shell wrapper reports 241. Its fixture remained revision 0 with zero events and a failing oracle. Two process samples show only `_dyld_start+0`, a 112 KiB footprint and no loaded binary images. This supports investigating the host loader before attributing the failure to Greppy code; the exact cause is unresolved.

The subsequent functional probe used the previously executed build path `/Volumes/tmp/dev-artifacts/greppy/web-efficiency/target/debug/greppy`. Its hash matched the saved copy before and after the successful flow. This is an explicit path change, not a replacement or removal of the failed trial. The copied artifact is not ready for agent measurement. No signing/security settings were bypassed.

## Recovery and failure checks

The first history-retrieval probe mistakenly changed both the runtime executable and the cache-storage root. It correctly returned snapshot unavailable (exit 30); that harness mistake is retained. With the original cache identity and an unavailable runtime executable, the first archived page was returned successfully, explicitly marked as historical and potentially stale. The fixture's exact state hash remained unchanged. Full multi-page losslessness is covered by unit tests, not claimed from this first-page probe.

A separate dialog chain deliberately clicked an absent selector before a later Save command. Exit 34 correctly stopped at step 3 of 4, preserved attribution of the preceding observation to step 2, and left the fixture at revision 0 with zero events. No later save occurred and no rollback was claimed.

That expected failure exposed two remaining native feedback issues reported to the existing fixing task: `count=0` still recommends a narrower target, and the failure contains no fresh state. The preceding open observation only contained the initial shell before asynchronous app content loaded. A separate observe is consequently needed for recovery. These are specific diagnosis/output findings, not a claim that NO_MATCH or chain stopping is broken.

## Evidence and remaining acceptance

Create-only evidence is under `/Users/michaelwelsch/.local/state/greppy-web-study/e1-output-preflight-20260905-01`: `09-table-chain-aggregation` (stopped copy), `10-table-chain-linked-path` (successful flow), `11-chain-history-offline` (wrong cache root), `12-chain-history-same-cache` (successful history), and `13-chain-stop-proof` (expected failure). Contexts, argv, terminal results, process samples, raw stdout/stderr, frozen host-state snapshots, checks and hashes are preserved.

Next measurements require a reliably starting bound candidate, the native label/modal/locator fixes, full transport metadata, and fresh paired Luna runs. Both input and output must decrease at equal correctness. The three-arm and real Office acceptance, total time and p95 gates, and prepared-script non-regression remain open. This experiment is not activated by default or described as superior to the standard browser.
