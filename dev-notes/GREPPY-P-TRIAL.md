# `greppy -p` trial — integrated agent as the coding agent

Date: 2026-08-05/06 · Target repo `~/greppy-data-pipeline` (58 files, 235
definitions, 148 embedded spans), deliberately not one of the six
dress-rehearsal tasks · Binaries built WITH the real model assets.

Task (navigation-heavy, verifiable), identical in every run:

> Locate the adaptive rate limiter used when downloading, work out how its
> backoff grows and when it resets, then add a focused unit test for that
> reset behaviour in the project's existing test layout. Run the test to
> verify it passes.

**Read this first: the first six runs measured a broken agent.** Two defects
made every index-backed greppy command fail inside `-p`, so the models were
reacting to a greppy that did not work, not to one they did not like. Both
are fixed; the numbers below are split accordingly.

## The two defects (found by forensics on the models' own reasoning)

1. **The seeded index was always invalid.** `-p` copied the main checkout's
   store into the worktree's store hash. `store.manifest` carries the
   original `workspace_hash`/`canonical_root`, so `read_store_manifest`
   (`crates/core/src/cache.rs:281`) rejected it: every run printed
   `invalid workspace store manifest …` and `index prewarm exited exit
   status: 73`. Copying without the manifest does not help either — graph
   rows carry a workspace identity, and a fresh index reuses **0** embeddings,
   so the copy bought nothing. Fixed in WP18: the copy is gone; each
   repository gets a stable agent worktree that keeps its own real index
   between runs and is reset to HEAD before each one.
2. **The sandbox blocked greppy's own data root.** The writable roots did not
   include `~/Library/Application Support/greppy` (Linux:
   `$XDG_DATA_HOME/greppy`), where the store's lock/lease files live. Every
   index-backed command died with `acquire lifecycle lease … Operation not
   permitted` — `where-am-i`, `search`, `who-calls`, `read`, `brief`,
   `impact`. Only `bash-smart` survived, because it needs no index. Fixed in
   WP19, with a regression test that exercises the real sandbox path.

The forensics that exposed this were the models' own words. MiniMax-M3, after
its first `where-am-i`: *"The repo appears empty"*, then *"This is a Python
project, not what greppy indexes. Let me explore directly."* It even tried to
call greppy as a plain CLI (`greppy greppy rg -- 429`) before falling back to
`grep` through `bash-smart`.

## Measurements

**G1** = two tools (greppy + free bash) · **G2** = one tool + exclusivity
prompt (WP16) · **G3** = G2 with both defects fixed (WP18+WP19). Store
prewarmed outside every measurement.

| | G1 grok | G1 M3 | G2 grok | G2 M3 | **G3 grok** |
|---|---|---|---|---|---|
| Tool calls | 41 | 31 | 42 | 23 | 57 |
| greppy navigation calls | 1 | 2 | 15* | 5* | **37** |
| …graph-level (`brief`/`who-calls`/`impact`/`read`/`search*`) | 0 | 0 | 2* | 0 | **20** |
| free shell | 40 | 29 | 0 | 0 | 0 |
| `bash-smart` executions | — | — | 27 | 17 | 20 |
| Wall time | 452 s | 518 s | 160 s | 94 s | 233 s |
| Outcome | ok | ok | ok | failed (turns) | ok |

\* G2's "navigation" was almost entirely `read-file`, which needs no index;
every index-backed call still failed.

**G3 is the first honest measurement.** The agent opened with `where-am-i`,
then `brief AdaptiveLimiter` — exactly the graph-first order the prompt
prescribes — and used `who-calls`, `impact`, `read`, `search`,
`search-symbol` and `search-pattern` throughout: 37 navigation calls, 20 of
them graph-level, zero shell navigation. Result correct
(`tests/test_adaptive_limiter.py`, verified with `uv run --with pytest`).
Tokens: 45 366 in / 7 852 out / 339 840 cache-read over 26 turns.

## What the numbers actually say

- **Adoption needs both**: the exclusive tool surface (G1→G2 killed shell
  navigation) *and* a greppy that works (G2→G3 turned nominal adoption into
  real graph navigation, 2 → 20 graph-level calls).
- **Do not read adoption from tool counts alone.** G2 looked like "100 %
  greppy" while every meaningful greppy call was failing. Count *successful
  index-backed* calls, or the number is theatre.
- **The exclusivity is not a cage** and does not need to be: `bash-smart` can
  still run `rg`. With a working index the model chose greppy anyway.

## Remaining defects (open)

1. **Proposals pick up generated files.** G3's proposal carried `uv.lock` and
   `trace_extract.egg-info/*` (434 + 49 lines) next to the 62-line test,
   because the target repo does not ignore them. `git add -A` is honest but
   noisy; consider separating obviously generated paths in the proposal
   summary.
2. **12 of 57 calls still failed**, all in the "find a working Python test
   runner" class. The host's Homebrew Python 3.14 has a broken `ensurepip`
   (reproduced with the sandbox switched off — not a sandbox issue). The
   agent recovered via `uv run --with pytest`, but only after burning turns.
3. **`-p` cannot reach https endpoints**: the client is built without a TLS
   stack. Correct for the localhost doctrine, and now stated in `--help`.

## Report tail

```
AGENT_GREPPY_CALLS=37 navigation (20 graph-level) of 57 total   [G3, grok-4.5]
WALL=233
VERDICT=With the two blocking defects fixed, the integrated agent uses greppy
the way it was designed to be used — graph-first, no shell navigation at all.
The earlier "adoption failure" was a broken index and a sandbox that locked
greppy out of its own store, not a model preference.
```
