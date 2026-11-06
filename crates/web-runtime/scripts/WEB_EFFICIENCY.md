# Complete browser-workflow efficiency measurements

`measure-web-efficiency.py` is an opt-in measurement tool. It uses existing
runtime and Playwright installations, never installs dependencies or changes host
resource limits, and does not run as a scheduler or background service.

Both engines execute the same `efficiency-workload.mjs`. Each independent task
navigates to the start page, clicks through to an article, verifies its complete
normalized text, checks context-local storage isolation/persistence, fills a
UTF-8 form, submits it, and verifies the response. The fixture server independently
checks the exact expected POST effects and rejects duplicates. A workload's own
`ok` flag cannot qualify a task without its corresponding server-side effect.
There are no retries. Failed task timings, process failures, timeouts, diagnostics
and missing evidence remain in the output; an unqualified run exits nonzero.

## Run on an admitted test host

Set `TMPDIR` and `OUTPUT_DIR` to an owned scratch directory on the required
artifact volume. Use the existing resource-admission/build path for the host.
Provide the actual existing runtime binary and Playwright package; this tool
neither builds nor downloads either engine.

Native runs create a deterministic single module beside the output JSON:
`native.json.entry.mjs` for `native.json`. The generator replaces exactly the
native adapter's relative workload import with the **unchanged** shared source.
Only the runtime's built-in `playwright` import remains; no production module
policy is relaxed. The generated entry is retained on failure and success,
never overwritten, and passed directly through the existing `--script` port.
`native_entry_provenance` records its actual SHA256, both input source hashes,
the composition version and generator hash alongside the harness git identity.
The entry is checked again after execution; modification disqualifies the run.
Preserve this entry with its receipt before cleaning the owned output directory.

The native adapter accepts exactly the shared workload's `{headless: true}`
launch intent and maps it to the runtime's fixed headless mode via `launch()`.
It rejects every other launch option or intent, including viewport/proxy options;
none are silently discarded. `native_entry_provenance.native_launch_policy`
records this mapping. Context creation and all shared task assertions are unchanged.
The native engine is not a general Playwright drop-in: `textContent()` currently
aliases `innerText()` (the fixture uses visible pre/p text), context options and
launch options are restricted, and no viewport override is used. Equal viewport
conditions must be verified separately before a reference performance comparison.

```sh
python3 crates/web-runtime/scripts/measure-web-efficiency.py \
  --engine native --runtime "$RUNTIME_BINARY" \
  --rounds 3 --tasks 2 --concurrency 1 --nodes 100 \
  --output "$OUTPUT_DIR/native.json"

PLAYWRIGHT_PACKAGE="$EXISTING_PLAYWRIGHT_PACKAGE" \
python3 crates/web-runtime/scripts/measure-web-efficiency.py \
  --engine chromium --rounds 3 --tasks 2 --concurrency 1 --nodes 100 \
  --output "$OUTPUT_DIR/chromium.json"
```

Use identical workload arguments and inherited host limits for a comparison.
Alternate engine order across independently retained samples. Document external
CPU/RAM limits and cache conditions; the receipt records available affinity and
cgroup-v2 limits but does not assert unknown limits. Parallelism and document
size are parameters, not the definition of efficiency. Existing output files
are never overwritten. The workload is bounded to at most1000 scheduled tasks
and the outer process deadline is finite.

## Interpret the receipt

- `quality_verified` requires a successful engine exit, complete receipts and
  all independent server-side effects. False receipts must not be ranked as
  successful performance results.
- `audit.independently_verified_completed_tasks` counts complete verified task
  identities only. The throughput denominator includes **all** elapsed process
  time, including failed/aborted work. Missing attempt evidence is `null`, not
  an invented zero. Raw per-task failures remain available.
- `server_monotonic_round_intervals_ms[0]` runs from process start through the
  first complete-round receipt. Later intervals measure identical repeated
  rounds with the same browser/contexts reused; no-store HTTP pages are loaded
  again for each task. Receipt transport is included for both engines. Task
  durations use the controller/Node `Date.now` clock; the server boundaries use
  the harness's monotonic clock.
- `process_cold_total_ms` covers the entire new process/browser run, all rounds
  and shutdown. This is process-cold: OS caches are neither cleared nor claimed
  cold. Process-exit observation has a20ms polling interval.
- CPU comes from `wait4` for the exact owned child and descendants that the OS
  includes after they are waited. Fixture-server and harness CPU are excluded.
  `max_individual_process_rss_bytes` is the OS maximum for an individual process,
  **not** concurrent aggregate RAM for the whole worker. It cannot establish
  process-tree RAM savings or memory-ceiling acceptance.
- On Linux, `process_family_memory` additionally samples the observed owned
  process family using `/proc/PID/smaps_rollup`. PSS counts shared mappings
  proportionally; private bytes and summed RSS are recorded separately in the
  raw `OUTPUT.memory.jsonl` artifact, bound by SHA-256. PID/starttime identities
  prevent following a reused PID. Known live reparented children remain tracked.
  Raw samples are streamed, not retained in a growing harness history.
  Samples have a minimum250ms gap; actual read windows, incomplete samples,
  sampling wall cost and monitor-thread CPU cost remain visible. Both engines
  use identical instrumentation. End-to-end timing includes its overhead;
  these are not uninstrumented speed claims. Unsupported/unreadable memory is
  unknown, not zero. Sampled peaks are lower bounds: unseen short-lived children
  and between-sample peaks can be missed. Kernel/GPU/unmapped filesystem cache,
  fixture and harness mappings are excluded. These metrics do not establish a
  full-agent memory ceiling or leak freedom.
- Server network counters are actual HTTP body bytes. They exclude headers,
  TCP/TLS overhead and IPC. Those omissions must remain explicit.
- The receipt binds the shared workload hash, launcher executable hash and
  harness source commit/dirty status. The reference reports its actual installed
  Playwright/browser version and browser executable hash. A harness commit is
  not an attestation of a separately supplied runtime binary's source commit.

This comparison covers the standalone runtime/Playwright execution boundary.
It does not exercise CLI daemon session budgets, an LLM/model/prompt, arbitrary
public websites, or full Web-Worker-Agent resource totals. Those measurements
remain necessary before claiming general efficiency leadership.

## Outcome-verification contracts

`python3 crates/web-runtime/scripts/test-measure-web-efficiency.py` checks real
fixture HTTP interactions, forged success without an effect, valid Unicode POSTs,
duplicate effects, malformed receipts, failed processes and bounded termination.
The native entry contract uses Node's parser to reject additional module imports
without executing a browser. It also loads the actual controller wrapper and runs
the generated entry only up to its deliberately refused first engine operation,
checking the exact launch mapping and rejection of additional options. This is a
controller-boundary contract, not proof of DOM/navigation/Storage/fetch behavior.
A real probe child verifies the actual `--script`
entry/hash and sends forged success receipts without a POST effect; the complete
harness must exit unsuccessfully. This probe is not a runtime acceptance result.
A mutation making `effect_verified` unconditionally true must fail
`test_claimed_success_without_post_effect_never_counts`.
