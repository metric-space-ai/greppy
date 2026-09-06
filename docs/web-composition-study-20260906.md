# Browser-result composition: measured contract gaps

6 September 2026. Development investigation, **not token, timing, native-union or release acceptance**.

The first live composition probe found a concrete obstacle to applying Greppy's local filtering tools to browser work: producers return different nested envelopes, while `web match` filters individual JSONL records. Regex matching works after an explicit investigator-written row recovery. The necessary selection facts are still split across producer outputs.

## Observed result

An isolated local page contains three visible delivery offers and one hidden offer. Prices are 7, 19, 29 and 1 respectively. A text field and Save button make usable action references observable. All browser reads use Greppy Web; no alternative browser, HTTP page reader, direct DOM mutation or application-backend shortcut was used.

| Producer | Actual record location | Direct `match 'text~/Economy/'` | Same match after explicit row recovery |
| --- | --- | --- | --- |
| scoped `observe --json` | `result.actionables` | exit 0, empty stdout | exactly Economy |
| `find css=a.offer --json` | `result.value.nodes` | exit 0, empty stdout | exactly Economy |
| `extract css=a.offer --fields text,href,attr:data-price --json` | `result.value.rows` | exit 0, empty stdout | exactly Economy |

The raw envelope is itself one valid JSONL record. It has no top-level `text`, so the empty filter result is consistent with the low-level match contract. **This is not evidence of a broken regex engine or a failed browser operation.** It is a composability/usability gap: a natural producer-to-filter pipeline does not select element records without another projection step. The inspected help describes the component commands but does not establish a direct row-producing pipeline for this task. This investigation does not claim that every possible CLI alternative has been exhausted.

The supplied flat-JSONL positive control selects the single visible offer below 10. Real captured rows then reveal a second obstacle:

| Captured record source | Action reference | Visibility field | Requested price field |
| --- | --- | --- | --- |
| observed actionables | yes | absent | absent |
| found nodes | absent | yes | absent |
| extracted rows | absent | absent | yes, as original strings |

`find` rows filtered by `visible=true` contain the three visible offers. Extracted rows filtered by `attr:data-price<10` contain **Economy and Hidden**. Adding `visible=true` to that extracted stream yields **zero rows**, because visibility was not emitted. The numeric filter correctly accepts the captured numeric strings; converting them was unnecessary.

A generic projection alone therefore does not solve “select a visible inexpensive offer and act on it.” The agent still needs another observation, a join, a more specific selector or its own page script. Joining only on text or href would be unsafe for duplicate elements and frame boundaries. No incorrect action was performed in this probe.

## Scope and exact evidence

- Source checkout for research additions: main `738867afde95af39f4fa7930b3103673908533fe`, branch `codex/web-composition-study`.
- Executed pair is the prior independently tested **scope development pair**, not the assembled main CLI/runtime. No new build was started and the pending main-native-union acceptance is not bypassed.
- CLI: `/private/tmp/greppy-scoped-observe-tests.wGSgCs/greppy-a3d6d1f`, SHA256 `a3d6d1f5e77e44b8187ad7611735c06427d11701e5e263a270cf0865d4ffc31e`.
- Runtime: `/private/tmp/greppy-scoped-observe-tests.wGSgCs/web-runtime-current`, SHA256 `6e925e498338e5e9d5237de62aef7d2c86e7b6407a8a7b9be63864602321a134`.
- Both executable hashes matched before and after the live probe. These hashes do not freeze dependencies/assets or prove source provenance for the entire process.
- A per-trial owner, workspace and runtime directory isolate the probe. Disposable data lives under `/Volumes/tmp/dev-artifacts/greppy/web-composition-study/`; older executable paths above were read unchanged, not newly created there.
- HTTP server stopped, owned runtime stop returned `running=false`, and content PID 17867 was absent after completion. No foreign runtime or build was stopped.

[Raw calls](../bench/web_study/evidence/composition-20260906-01/calls.json) preserve exact argv, stdin, stdout, stderr, exit status and observed command durations, including empty responses. [Context](../bench/web_study/evidence/composition-20260906-01/context.json) preserves executable/wrapper identity. [Terminal receipt](../bench/web_study/evidence/composition-20260906-01/terminal.json) records completion and cleanup. [Offline replay](../bench/web_study/evidence/composition-20260906-01/replay.json) preserves recovery controls and original row fields. These are synthetic local-page data; absolute historical paths identify the actual run, not portable prerequisites.

Durable originals: `/Users/michaelwelsch/.local/state/greppy-web-study/composition-20260906-01/`.

## Executable reproduction

The prepared probe is `bench/web_study/basic_fixture/composition_probe.py`. It uses the existing context preparer without prescribing any browser behavior to a task participant. There were no model participants in this probe.

```sh
TMPDIR=/Volumes/tmp/dev-artifacts/greppy/web-composition-study/tmp \
PYTHONDONTWRITEBYTECODE=1 greppy bash-smart -e PROBE -- \
python3 bench/web_study/basic_fixture/composition_probe.py \
  --cli /path/to/pinned/greppy --runtime /path/to/pinned/web-runtime \
  --scratch /Volumes/tmp/dev-artifacts/greppy/web-composition-study/new-run \
  --evidence /durable/new-evidence-directory
```

Offline replay uses `composition_replay.py --calls PATH --cli PATH --output NEW_PATH`. It maps only the three captured envelope shapes, preserves original row values and makes no visibility/reference inference. This is an investigator intervention, **not an implemented Greppy feature or participant benefit**. Failed/unknown/truncated/count-inconsistent producers are rejected instead of becoming an apparently complete empty selection.

Eight focused tests cover failed producers, unknown schemas, truncation, count mismatch, incompatible shapes, valid emptiness and preserving missing facts. Native capture and six offline real-row controls completed. One initial replay launch happened before its file-writing process had finished; Python exited 2 with file-not-found before doing any replay. After the write completed, the unchanged replay ran successfully. That was coordinator sequencing error, not a Greppy product failure or measured participant trial.

## Next experiment: a shared record view

The evidence supports testing a versioned record view across existing producers. It does not justify a new autonomous agent or a new general scripting language.

The experimental contract should:

1. Expose element records directly to local filters while preserving existing JSON compatibility. An explicit mode must distinguish rows from status/continuation metadata.
2. Carry request/session/tab/document/frame provenance and an actionable reference where one actually exists. Absence must remain explicit; no invented refs or joins by display text.
3. Permit selected attributes alongside current visibility/state facts in one observation. Preserve original values and distinguish missing fields from false/zero.
4. Retain errors, truncation, continuation and already-delivered effects even when filtering yields no records. An empty result must not hide an upstream failure or imply a complete scan.
5. Keep page text fenced as untrusted data and report states/causes neutrally. No `next`, retry plan or instructions to the calling LLM.
6. Leave the next action to the caller; revalidate target identity and current actionability when executing it.

A subsequent development task should select a visible offer under a budget, write its note, save once, and verify the saved result. Include hidden and duplicate offers, frame boundaries, DOM replacement, missing prices, producer failures and truncated sets. Compare existing tools against the experimental record view with fresh Luna/medium contexts, real usage counters and an independent outcome oracle. A parser written by the investigator must be accounted for as part of the experimental system, not misrepresented as current product capability.

## Remaining acceptance and coordination

Remote main still resolves to `738867af`; PR10 is confirmed merged, so these additions belong in a new PR. CI34018418772 was in progress; Linux sandbox34017635894 and compatibility34017635874 remain failed. The worker's latest handover still says the Linux feature-boundary fix and neutral output corrections are not implemented. The main native-union run, current tested binary pair, arm B readiness, fair A/B/C token trials and wider acceptance remain open.

The designated product-bug owner is exclusively Codex thread `01a02118-0d61-7e10-a9d4-be496fa34879`. This root owns research and evidence, not parallel product fixes. This thread has no callable Codex thread-message tool; the new composition finding has **not been delivered** to that owner. No substitute bug task, GitHub issue or message to another thread was created. The exact evidence is retained for delivery through the authorized channel.
