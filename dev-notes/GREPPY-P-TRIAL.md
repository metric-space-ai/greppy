# `greppy -p` trial — integrated agent as the coding agent

Date: 2026-08-05 · Binary: `dev-0.4.0` @ `5aeec89` (WP15), built WITH the real
model assets · Model: `grok-4.5` via CLIProxyAPI (127.0.0.1:8317) · Target
repo: `~/greppy-data-pipeline` (58 files, 235 definitions, 148 embedded
spans) — deliberately not one of the six dress-rehearsal tasks.

Task (navigation-heavy, verifiable):

> Locate the adaptive rate limiter used when downloading, work out how its
> backoff grows and when it resets, then add a focused unit test for that
> reset behaviour in the project's existing test layout. Run the test to
> verify it passes.

Two runs were made. **Run 1 is void as adoption evidence** (setup error, see
below); **run 2 is the valid measurement.**

## Run 2 — valid (warm store, real assets)

Prewarm before the run, outside measurement: `greppy index` →
`embedded 148 code spans`, `doctor: embedding_complete: true`.

| | |
|---|---|
| Wall time | **452 s** (7:32) |
| Tool calls total | **41** |
| …of which `greppy` | **1** (`where-am-i`, the opening call) |
| …of which `bash` | **40** (`cat` 7, `ls` 6, `git` 6, `rg` 4, `find` 4, `python3` 3, `sed` 2, rest one-offs) |
| Failed calls | 5 (all of them the pytest hunt, see below) |
| Outcome | correct: `tests/test_adaptive_limiter.py`, two tests, both passing |
| Tokens | **not instrumented** — the CLI does not surface usage today |

**(c) Result quality.** Good and honest: the agent found the limiter, derived
the additive-ramp / halving-on-429 behaviour correctly, wrote two focused
tests that genuinely exercise the reset paths, ran them (2 passed) and
delivered them as a proposal ref. Nothing invented, verification real.

**(d) Did it fall into grep/read loops?** **Yes — it never left them.** After
the single opening `where-am-i` it navigated exclusively with `rg`, `cat`,
`find` and `sed`. It never called `search`, `search-symbol`, `who-calls`,
`brief`, `read` or `impact`, although the graph and the semantic index were
both fully built and available. Five consecutive calls were spent hunting a
working pytest interpreter across the host — the kind of environment
scavenging a sandboxed, repo-scoped tool surface is supposed to avoid.

## Run 1 — void (recorded because the failure mode is instructive)

Binary built with `--features ci-test-assets` (greppy-040 had no model
assets), and the store was "warmed" with `where-am-i`, which builds the
graph but **not** the embeddings. The agent's second call was
`greppy search "adaptive rate limiter for downloading"`, which returned the
retryable *"semantic index building — 0/148 spans"* status as a tool error.
It never touched greppy again (2 greppy calls, 33 bash calls, 578 s).

This reproduces the documented 0.2.1 bench lesson exactly: one failed greppy
call and the model abandons the whole surface, including the parts that work.
The setup error was mine; the abandonment behaviour is real.

## What this says about the adoption question

The integrated agent removes *configuration* risk — greppy is on PATH, the
prompt ships with the binary, the store is seeded into the worktree, nothing
can be wired wrongly. It does **not** by itself create *adoption*: with a
free-hand `bash` tool next to it, grok-4.5 reached for `rg`/`cat` and stayed
there, in spite of a system prompt that explicitly says to prefer one precise
relationship query over grepping and reading whole files.

Concrete follow-ups this run argues for (none implemented yet):

1. **Prewarm inside `-p`, fail-closed.** `-p` should ensure
   `embedding_complete` before the loop starts, or the first semantic query
   will hand the model an error and lose it for the rest of the run.
2. **A retryable status must not reach the model as a plain tool error.**
   Either wait and retry inside the tool, or return it as a normal result
   that says "retry in ~Ns".
3. **Adoption has to be designed, not requested.** Options worth measuring:
   drop `rg`/`find` from the reachable surface, route them through greppy,
   or restructure the prompt so navigation without greppy is the exception
   that needs justifying.
4. **Instrument token usage** in the CLI; the loop already sums it.

## Report tail

```
AGENT_GREPPY_CALLS=1
WALL=452
VERDICT=The integrated harness removes misconfiguration entirely, but on this
run it produced no more greppy adoption than pi did — the model used bash/rg
for all navigation, so adoption needs tool-surface or prompt work, not just
integration.
```
