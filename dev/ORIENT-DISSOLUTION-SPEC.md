# ORIENT dissolves — where-am-i and inventory, into NAVIGATE (0.3.0)

> **Amendment (owner decision, same day):** ONE verb, and the census lives in
> its expand packs. Each module line of `where-am-i` carries its size, its
> most-used symbols (highest incoming degree — facts, not curation) and the
> expand id whose pack is that module's full inventory, one `file:line  name
> kind` row per definition, grouped by file:
>
> ```
> $ greppy where-am-i
> /Volumes/tmp/outputs-repo — rust, 21 files, 2,655 definitions
>
> edit-src/    11 files   1,120 defs — sha256_hex, Snapshot, run_pipeline — greppy expand 3fa1c2
> cli-src/      9 files   1,535 defs — resolve_root, dispatch, Cli — greppy expand 77b0e4
>
> entry points: cli-src/main.rs
> tests: inline #[test] modules
> ```
>
> (Counts and ids illustrative until captured from the real binary.) The
> expand offer satisfies all three expand criteria: the pack holds far more
> than the line shows, no flag or other command reaches a per-module census,
> and the visible part — count plus most-used names — carries the decision to
> open it. The defs count on the line IS the announced pack size. There is no
> separate `inventory` verb.
>
> **Packs obey the size law themselves — nothing ever drops tens of thousands
> of tokens in one step.** The descent is fractal, the same shape at every
> level: repo → module → file → rows, each level one screen of
> `name  count  most-used  expand-id` lines. Only a level small enough
> (≤ ~25 definitions) delivers the full census rows directly; a file above
> the budget paginates like read-file. Every step is priced before it is
> bought, and the agent decides at every branch with the numbers in view.
> The arithmetic that forces this: an inventory row is ~11 tokens, this
> sample repo holds 2,655 definitions ≈ 29k tokens, real repos 10-100k
> definitions — context is re-billed every turn.

Owner decision 2026-07-28. The shared laws of `dev/NAV-OUTPUT-SPEC.md` apply.

## The verdicts

* **`map` dies.** It printed a repo profile — language counters, directory
  sizes, `(none detected)` filler, `try:` lines — a worse `ls` with statistics.
* **`outline` dies.** File→definitions was its one question; no rescue verb.
* **`changes` dies.** A diffuse composition (diff → symbols → callers/tests);
  the question does not earn a verb. `impact`'s undocumented `--since/--base`
  git scopes are code cleanup: remove with it.
* **`verify` dies.** The agent has a shell; running commands is its native
  ability, and a navigation/edit tool that executes foreign commands blurs its
  identity. (`edit --verify` is untouched — post-edit diagnostics belong to the
  edit receipt.)
* **`where-am-i` is new and lives in NAVIGATE.** Its purpose is measured in
  saved tool calls: the turn-1 exploration burst — `ls -R`, reading README,
  Cargo.toml/package.json, guessing test roots — becomes one call.

## where-am-i

One screen, facts only, empty categories omitted (no "(none detected)"):

```
$ greppy where-am-i
/Volumes/tmp/outputs-repo — rust, 21 files indexed

edit-src/    11 files
cli-src/      9 files
config.json

entry points: cli-src/main.rs
tests: inline #[test] modules
```

* root line: path, languages, indexed count — the index's own inventory
* one line per top-level entry, largest first; depth grows only where a single
  child dominates (`src/` alone expands one level)
* entry points from the graph (main functions, bin targets), test roots from
  the indexer's test detection, build files listed only when present
* **target state:** each directory line carries a sentence aggregated from its
  files' module docs (`//!` first lines) via the summarizer —
  `edit-src/  11 files — the edit engine: verbs, plans, transactions`.
  Until block summaries land, the factual skeleton ships alone; nothing is
  invented.
* no `try:` lines, no expand (one screen is the contract), exit 0

Prompt line (NAVIGATE):

```
  where-am-i                        the repository at one glance: layout, languages, entry
                                    points, test roots
```

## Migration

1. AGENTS.md: ORIENT section deleted; `where-am-i` line added to NAVIGATE;
   the CHAIN and footer examples do not reference ORIENT (verified).
2. prompt_contract: NAVIGATE entry count 6 → 7; new guard — `ORIENT:`,
   `outline PATH`, `verify -- CMD` and a `changes` command line must not
   reappear.
3. Code: `map`, `outline`, `changes`, `verify` subcommands removed with their
   dispatchers (`map.rs`, `changes.rs`, `verify.rs` mostly die), passthrough
   guard extended so the four dead verbs cannot become grep patterns.
   `impact`'s `--since/--base` go with `changes`.
4. `where-am-i` implemented from the existing map machinery minus the filler,
   plus entry points from the graph.
