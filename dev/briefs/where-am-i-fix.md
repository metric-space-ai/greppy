# where-am-i: three defects in the real output, and their fixes

The command was built (guard tests 11/11 green, release build clean) and run
against this repository. The census works; what it says does not survive the
output laws. Captured output: `dev/captures/where-am-i-IST.txt`.

Work in YOUR CURRENT DIRECTORY — the launcher has put you in an isolated
worktree of the repo. Do NOT cd to any other checkout. NEVER name an absolute
repo path as the place to work.

## 1. Rows that carry no answer, each with an expand offer

```
assets/                   4 files  0 defs — greppy expand a3213662ecf05448
AGENTS.md                 1 file  0 defs — greppy expand a3df2ef5c7572c4c
CITATION.cff              1 file  0 defs — greppy expand a5f6c1fc83f33567
CLAUDE.md                 1 file  0 defs — greppy expand 1bbcf2ad49dacc7d
Cargo.lock                1 file  0 defs — greppy expand b16de8bfbfb4a785
LICENSE                   1 file  0 defs — greppy expand 9b90c312fc36c50e
```

Six of 31 lines say "here is nothing", and each offers to expand that nothing.
A file with no definitions is not an orientation fact. FIX: entries with 0
definitions do not get a row; they are summed into one closing line
(`14 further files hold no definitions`), with no expand id.

## 2. The "most-used" trio is noise, which defeats the hub

```
crates/           294 files  10,772 defs — Fixture::store, Fixture::repo, Node
bench/            111 files   1,526 defs — out, result, options
training/qwen35/   20 files     202 defs — r, w, accepted_raw
```

The three names per module are the one thing that gives an agent its bearings,
and they name test fixtures and single-letter locals. `out`, `result`, `r`, `w`
are not what `crates/` or `bench/` is about. FIX: rank by INCOMING graph edges
(how often a definition is actually referenced), not by whatever the store
returns first; exclude definitions with no incoming edges from the trio; if a
module has no referenced definitions, the trio is omitted rather than filled
with noise. Verify on this repo: `crates/` must surface names a reader of the
codebase would recognize.

## 3. Non-code definitions inflate every number

`Cargo.toml 1 file 62 defs — license, categories, clap` counts TOML keys;
markdown files count `Section::…` entries; the headline "12,889 definitions"
mixes code with prose headings and config keys. FIX: the census counts code
definitions; documentation sections and config keys are reported separately
(e.g. a closing line `docs: 21 files, 118 sections · config: 6 files, 94 keys`)
so the headline number means what a reader assumes it means.

## Acceptance — run these and paste REAL output
- `cargo test -p greppy --test prompt_contract` — 11/11 green.
- `cargo run -p greppy -- where-am-i` in this repo — paste the FULL output.
  It must contain no 0-definition row, no expand id for empty content, and the
  `crates/` trio must be referenced code symbols.
- `cargo run -p greppy -- where-am-i --json | head -40` — the JSON carries the
  same numbers as the text.

## FILE WHITELIST
- ONLY `crates/cli/src/` files that implement where-am-i and its census, plus
  `crates/cli/tests/` for new tests.
- FORBIDDEN: `AGENTS.md`, `crates/cli/tests/prompt_contract.rs`, every other
  command's dispatch, the `--json` shapes of other commands.

## Hard rules
- Do not commit unless the acceptance block is green. Commit message:
  `fix(where-am-i): rows that answer, symbols that are referenced, numbers that mean what they say`
- ESCAPE HATCH: if you believe you need scope beyond the whitelist, STOP and
  justify it in the report. Never widen on your own.
- NO SUBAGENTS.

## REPORT TAIL (fixed form, at the very end)
CHANGED: <files>
OUTPUT: <the acceptance commands' real output, verbatim>
TESTS: <suite result lines>
OPEN: <what you could not do, and why>
COMMIT: <sha(s) or "not committed">
