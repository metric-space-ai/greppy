# `greppy -p` trial — integrated agent as the coding agent

Date: 2026-08-05 · Target repo `~/greppy-data-pipeline` (58 files, 235
definitions, 148 embedded spans), deliberately not one of the six
dress-rehearsal tasks · Store prewarmed outside every measurement until
`doctor: embedding_complete: true` · Binaries built WITH the real model
assets.

Task (navigation-heavy, verifiable), identical in every run:

> Locate the adaptive rate limiter used when downloading, work out how its
> backoff grows and when it resets, then add a focused unit test for that
> reset behaviour in the project's existing test layout. Run the test to
> verify it passes.

## The measurement

Four valid runs: two models × two harness generations. "Before" is
`5aeec89` (WP15): two tools, `greppy` plus a free `bash`. "After" is
`f3f7ab4` (WP16): ONE tool, `greppy`; commands run as
`["bash-smart", "--", …]`; the prompt states the exclusivity.

| | grok-4.5 before | grok-4.5 after | MiniMax-M3 before | MiniMax-M3 after |
|---|---|---|---|---|
| Tool calls | 41 | 42 | 31 | 23 |
| …greppy | **1** | **42 (all)** | **2** | **23 (all)** |
| …navigation/read via greppy | 1 | 15 | 2 | 5 |
| …free shell | 40 | 0 | 29 | 0 |
| Failed calls | 5 | 12 | 7 | **18** |
| Wall time | 452 s | **160 s** | 518 s | 94 s |
| Outcome | test written, 2 pass | test written, passes | test written | **none — ran out of turns** |
| Tokens | not instrumented | in 29 044 / out 5 068 / cache-read 377 728 over 27 turns | not instrumented | in 17 140 / out 4 313 / cache-read 351 360 over 40 turns |

## What it shows

**Adoption is a property of the tool surface, not of the model.** With a free
`bash` next to greppy, both models navigated almost exclusively with
`rg`/`cat`/`find` — 1 of 41 and 2 of 31 calls went to greppy — although the
graph and the semantic index were fully built. Removing the alternative moved
both to 100 % greppy usage immediately. No prompt wording achieved this;
removing the option did.

**Exclusivity costs nothing and buys speed.** grok-4.5 solved the same task
**2.8× faster** (452 s → 160 s) with the same quality. The compacted
`bash-smart` output and one-call file reads replace long shell transcripts,
and the cache-read figures (≈ 350–378 k tokens) confirm the static prompt is
effectively free after the first turn.

**But adoption alone does not make a run succeed.** MiniMax-M3 reached 100 %
greppy usage and then failed the task: 18 of 23 calls errored and it
exhausted its 40 turns hunting a Python interpreter with `pytest`
(`pip install`, `uv tool list`, cache-dir juggling). grok-4.5 escaped the
same trap only by finding an unrelated virtualenv elsewhere on the host.
The repo has no ready test runner, and the write-confinement sandbox
(correctly) refuses installs outside the allowed roots — so the agent burns
turns on an environment problem it cannot solve.

## Defects this measurement exposed

1. **Environment scavenging is unbounded.** Nothing stops an agent from
   spending its whole turn budget looking for a toolchain. Options: detect a
   missing/blocked test runner early and tell the model to report instead of
   installing; surface the sandbox refusal as a clear "installs are not
   possible in this run" instead of a generic non-zero exit.
2. **`greppy greppy rg -- 429`** — M3 wrapped a greppy call inside greppy's
   own argv (first call after the prompt change). The prompt's exclusivity
   register can mislead about the passthroughs; the guard should reject a
   first argument of `greppy` with a pointed message.
3. **Graph navigation is still unused.** Post-change reads are `read-file`
   (12 and 5); `who-calls`, `brief`, `impact` were never called. Adoption of
   the *file* level happened, adoption of the *graph* level did not.
4. **`-p` cannot reach https endpoints at all** — the client is built without
   a TLS stack (localhost by construction). Correct for the doctrine, but
   `-p --help` claims "any compatible server works"; that sentence is wrong
   and must name the plain-HTTP/localhost restriction. (The M3 runs here went
   through a local test-only forwarder, not through any product path.)

## Earlier void run — kept because the failure mode is instructive

The first run of the day used a `ci-test-assets` binary (no real model
assets) and was "warmed" with `where-am-i`, which builds the graph but not
the embeddings. The agent's second call, `greppy search …`, returned the
retryable *"semantic index building — 0/148 spans"* status as a tool ERROR,
and the model never touched greppy again (2 greppy calls, 33 shell calls,
578 s). Setup error on our side; abandonment behaviour real. It is the
direct evidence behind the doctrine's first rule — a failed greppy call must
never reach the model — and behind the prewarm gate and status-not-error
handling now in WP16.

## Report tail

```
AGENT_GREPPY_CALLS=42 (grok-4.5, after) / 23 (MiniMax-M3, after)
WALL=160 (grok-4.5, after) / 94 (MiniMax-M3, after, failed run)
VERDICT=Exclusivity moved both models from ~1/40 to 100% greppy usage and cut
grok-4.5's wall time 2.8x at equal quality; it does not by itself make a run
succeed — M3 spent its whole turn budget on a missing test runner, so the
next work is bounding environment scavenging and lifting adoption from the
file level to the graph level.
```
