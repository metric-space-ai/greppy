# Adding a language to greppy — an all-in-one agent playbook

This document is a **self-contained prompt for a coding agent**. Given the name of
one programming language, scripting language, config format, or markup format, an
agent that follows every step below will implement extraction for it, verify it to
the project's correctness bar, and open a pull request — **without needing the
private C reference implementation** that the core team uses internally.

> **You are the agent.** Read this whole file first, then execute it top to bottom
> for your target language. Do not skip the verification steps. If you cannot reach
> the bar honestly, say so and open a draft PR describing the exact remaining gap —
> never label a language "supported" on a hunch.

> **How to run this.** Every command below is run **synchronously**: issue it, wait
> for it to finish, read its output, then decide the next step. Do **not** poll,
> sleep-loop, or wait for a background event — there are none. A release build
> (`cargo build -p greppy --release`) takes a few minutes the first time and
> after edits; that is normal — let it complete, don't retry it in a loop. The one
> long-lived artifact you produce is your extractor code; everything else is a
> one-shot command whose stdout you read.

---

## 0. What greppy extracts, and what "done" means

greppy indexes a repository into a graph of **nodes** (code symbols + structural
containers) and **edges** (relationships). Your job is to make a new language
produce a *correct* graph.

**Node labels** you may emit (free-form strings; these are the established ones):
`Function`, `Method`, `Class`, `Interface`, `Type`, `Enum`, `Struct`, `Union`,
`Module`, `Package`, `Namespace`, `Variable`, `Field`, `Constant`, `Decorator`.
Structural nodes — `Project`, `File`, `Folder`, and a per-file `Module` — are added
for you by the indexer; you do **not** emit them.

**Edge types** you may emit:
`CALLS`, `DEFINES`, `DEFINES_METHOD`, `IMPORTS`, `USAGE`, `WRITES`, `INHERITS`,
`IMPLEMENTS`, `DECORATES`, `RAISES`. Structural edges (`CONTAINS_FILE`,
`CONTAINS_FOLDER`) and semantic ones (`SIMILAR_TO`, `CONFIGURES`, `EMITS`) are added
by later passes.

**The correctness bar (evidence-verified — no C oracle required).**
A language is `supported` when, on a **fresh multi-file fixture you write yourself**,
greppy's extracted graph is *provably correct* by independent evidence:

1. **Every** definition, call, import, and top-level usage that actually exists in
   your fixture source is represented by a node/edge of the right label/type with
   the right endpoints — verified by cross-checking against `ripgrep` (§6) and by
   reading the source, **not** by trusting the extractor.
2. There are **no spurious** nodes/edges (nothing the source doesn't support).
3. Qualified names, line spans, and endpoints match what a human reading the code
   would say is correct.
4. Structural containment (`Folder`/`File`/`Module` + `CONTAINS_*`/`DEFINES`) is
   consistent with the directory tree.

This bar is *stricter* in spirit than "matches some reference tool": it is
**correctness against the source itself**. If greppy is more correct than any
other tool, that is a pass, not a deviation.

---

## 1. Set up and orient (5 minutes)

```bash
# Build the CLI once (release; the binary is target/release/greppy).
cargo build -p greppy --release
BIN="$PWD/target/release/greppy"
```

Read one example of each shape so you copy the right pattern:

- Structural-only (no defs/calls) — `crates/parser/src/langs/json.rs`
- Spec-driven with real defs — `crates/parser/src/langs/toml.rs`
- Filename-detected (no extension) — `crates/parser/src/langs/just.rs`
- A bespoke extractor — search `crates/parser/src/extract.rs` for `fn extract_xml`
  or `fn extract_toml`.

Key files you will touch (most languages need only the first two or three):

| File | Purpose |
|---|---|
| `crates/discover/src/language.rs` | Map file extension / filename / shebang → language |
| `crates/parser/src/langs/<lang>.rs` | Register the language: grammar, spec, tree-sitter queries |
| `crates/parser/Cargo.toml` | Add the `tree-sitter-<lang>` grammar dependency |
| `crates/parser/src/extract.rs` | *Only if* the declarative spec can't match the source — a bespoke `extract_<lang>` |
| `crates/indexer/src/lib.rs` | *Only if* imports/inheritance need custom resolution |
| `LANGUAGE_SUPPORT.md` | Status row + count line |

The language list is auto-discovered by `build.rs` — you do **not** edit any `mod.rs`.

---

## 2. Write the fixture AND the expected-graph golden — BEFORE coding

Create a realistic, adversarial, **multi-file** fixture *outside* the repo (never
commit it), e.g. `/tmp/fix-<lang>/`, with nested folders. Exercise everything the
language really has: definitions of each kind, calls (including method/qualified
calls), imports (including cross-file and directory imports), variables/fields,
inheritance, and **traps** — things that look like code but aren't (values inside
strings, fenced code blocks, comments) so you can prove they're *not* extracted.

Then, **by reading your own fixture**, hand-write an `EXPECTED.md` golden next to it:
list the exact node labels+names and edge types+endpoints you expect, and the total
counts. This is your oracle. Writing it before you look at any tool output keeps you
honest.

```
/tmp/fix-<lang>-EXPECTED.md   # keep the golden OUTSIDE the indexed dir (see warning)
/tmp/fix-<lang>/              # <- this is what you `index`
  src/a.<ext>
  src/util/b.<ext>
```

> **Trap:** do NOT put `EXPECTED.md` (or any `.md`/`.markdown`) *inside* the fixture
> directory you index — Markdown is itself a supported language, so `index .` will
> walk your golden and inflate `stats` with an extra `File`/`Module` (and `Section`
> nodes). Keep the golden a sibling of the fixture dir, or in a subdir you don't
> index. The same applies to any stray file whose extension greppy supports.

**Crucial when writing EXPECTED.md — `stats` shows the RESOLVED graph, not raw
extraction.** Your extractor emits raw nodes/edges, but the indexer then **drops**
edges whose endpoint doesn't resolve. `greppy stats` reflects that post-resolution
graph, so write your golden at the *resolved* level:
- A **`CALLS`** edge survives only when the callee resolves to a **defined node** (a
  function/method that exists in the indexed graph). Calls to language builtins,
  operators, or external/library functions you didn't define are parsed but produce
  **no** surviving edge. Do not count `print(...)`, `+`, `tostring(...)`, etc.
- An **`IMPORTS`** edge survives only when its symbol resolves to an *importable*
  node — a definition-like label (Function/Class/Interface/Type/Enum/Struct/Trait/
  TypeAlias/Module/File/Folder). An import that binds a plain `Variable`, or points
  at an external package with no file in your fixture, may be **dropped**. Write your
  fixture so cross-file imports point at files/symbols that exist, then assert those.
- A `CALLS` edge is attributed to the **enclosing callable** (the function the call
  sits inside), which becomes the edge source.

If `stats` shows fewer calls/imports than your raw reading, that is usually
**correct resolution**, not a bug — reconcile your golden to the resolved level
before "fixing" anything. (The resolver rules live in `crates/indexer/src/lib.rs`;
you shouldn't need to change them.)

Make the fixture a git repo (`git init`) — the indexer keys on a repo root.

---

## 3. Wire detection

In `crates/discover/src/language.rs`:

- Add your extension(s) to `language_for_ext` (lowercased, no dot), **or**
- Add exact filename(s) to `language_for_filename` for extension-less files
  (like `justfile`), **or**
- Add a shebang interpreter to `language_from_shebang` for scripts.
- Add a variant to the `DetectedLanguage` enum if the routing needs one.

**Pitfall — hidden dotfiles:** files beginning with `.` (e.g. `.mytoolrc`) are
dropped by the discovery walker unless the basename is on the `SPECIAL_HIDDEN_FILENAMES`
allowlist in `crates/discover/src/lib.rs`. Only add a dotfile there if it maps to
your language, and **never** add data files that commonly hold secrets (`.env`, etc.)
unless that language is genuinely being supported — greppy must not walk secrets
as a side effect.

**Pitfall — filename detection lives in TWO places.** A language detected by exact
filename (e.g. `BUILD`, `WORKSPACE`, `justfile`) must be listed **both** in
`discover/src/language.rs` (`language_for_filename`) *and* in the `filenames:` array
of its `LangDef` in `crates/parser/src/langs/<lang>.rs` (§4). Detection can appear to
work via one while the other is empty — populate both or you get a latent gap.

**Pitfall — shared extensions:** if your extension overlaps another language
(`.svg` is XML-ish; `.cfc` is CFML vs CFScript), match exactly which language the
project intends and don't over-claim files another language owns.

---

## 4. Register the language (declarative spec first)

Create `crates/parser/src/langs/<lang>.rs`. Add the grammar to
`crates/parser/Cargo.toml` (`tree-sitter-<lang> = "..."`).

**First, discover the grammar's node kinds** — your queries need the exact
tree-sitter node-type names. Use the parse-tree dumper (no C reference needed):
`crates/parser/examples/dump_ts.rs` prints the syntax tree of a source snippet so
you can see what the grammar calls a function definition, a call, an import, etc.
Run it (`cargo run -p greppy-parser --example dump_ts -- <file>` — check the
example's own `--help`/source for exact args) on a small sample and read off the
kinds you will match.

Copy the shape from `toml.rs`, filling in a `LangSpec` (see
`crates/parser/src/spec.rs`):

- `name: NameStrategy::Capture` — your tree-sitter `def_query` captures the name
  node as `@name`; its parent becomes the definition node. (`CStructural` /
  `RAssign` exist for C/C++ and R.)
- `defs: &[DefRule::…]` — one rule per definition kind. Use `DefRule::func(kind)`
  for free callables, `DefRule::method(kind)` for owned methods, `DefRule::ty(kind, "Class")`
  for types/tables.
- `owner_kinds` — the enclosing node kinds that own methods (empty if none).
- `calls: CallSpec { skip_callees }` — callee names to ignore (builtins).
- `imports: ImportStrategy::…` — pick the closest existing strategy, or `Bash`
  (inert) if the language has no imports.
- Write the `def_query` / `call_query` / `import_query` tree-sitter queries.

`inventory::submit!(LangDef { name, extensions, filenames, grammar, spec, def_query,
call_query, import_query })` self-registers it.

**Many languages are done here.** Build and go to §5.

---

## 5. Index the fixture and compare to your golden — the core loop

```bash
cargo build -p greppy --release
cd /tmp/fix-<lang> && "$BIN" index . >/dev/null && "$BIN" stats
```

`stats` prints node counts by label and edge counts by type (plain text — there is
no `--json` on `stats`). Compare **every line** to your `EXPECTED.md`. For endpoints,
spot-check specific symbols. These are the exact working invocations — pass a
**qualified name** (`<relpath>::<Label>::<name>`, as printed by `callees`/`who-calls`
themselves):

```bash
"$BIN" callees   "src/a.ext::Function::main"     # outgoing CALLS of main
"$BIN" who-calls "src/a.ext::Function::helper"   # incoming CALLS and USAGE edges
"$BIN" who-calls "src/a.ext::Class::MyType"      # incoming usage/type evidence
"$BIN" search-graph --name "pkg::MyClass::method" --json   # look up ONE known qname
```

Notes learned the hard way:
- `callees`/`who-calls` for an existing symbol with **zero** matching edges say
  so and exit 0. A missing or ambiguous symbol exits non-zero and names the
  resolution problem. Confirm the node exists via `stats` when diagnosing the
  distinction.
- `search-graph --name ""` / label-listing flags do **not** enumerate all nodes;
  `stats` is your enumeration tool. Use `search-graph --name` only to fetch one
  qname you already expect.

Iterate query → source → spec until `stats` and the spot-checks match your golden
exactly. Add fixture cases for anything you got wrong and re-verify.

---

## 6. Prove completeness with ripgrep (the C-free "gate")

For every definition/call/import kind in your fixture, cross-check that greppy
accounts for all of them — this replaces diffing against a reference tool:

```bash
# Example: every `func` definition in the fixture must be a Function/Method node.
rg -n '^\s*(pub\s+)?func\s+\w+' /tmp/fix-<lang> | wc -l      # source count
"$BIN" stats | grep -E 'Function|Method'                    # graph count — must reconcile
```

Do this per kind (defs, calls, imports). Any mismatch is either a missing extraction
(fix it) or a deliberate, *documented* exclusion (write down why in the PR). "The
numbers reconcile against the raw source" is the evidence that stands in for a C
diff.

---

## 7. When the spec isn't enough — two escape hatches

**7a. A new `ImportStrategy` variant.** The existing strategies each read a specific
capture shape (e.g. `ImportStrategy::Lua` reads a call node's `arguments` field). If
your language's import form doesn't fit any of them — a very common case — add a new
scoped variant to the `ImportStrategy` enum in `crates/parser/src/spec.rs` plus its
expander arm, gated so it only fires for your capture (e.g. on the callee/keyword
name) and is inert for every other language. `spec.rs` is a **shared file** — §8's
"never widen silently" rule applies: keep the new arm strictly conditioned on your
language and re-verify neighbors.

**7b. A bespoke extractor.** If the declarative spec cannot reproduce the correct
graph at all (the grammar models several construct kinds as one node, or names need
custom resolution), write a
`fn extract_<lang>(source, file_path) -> Result<ExtractionResult>` in
`crates/parser/src/extract.rs` and dispatch to it from `extract()` (guard on
`d.name == "<lang>"`). Return `ExtractionResult { nodes, edges }` with the right
labels/types. Keep the guard strictly scoped to your language.

---

## 8. Indexer passes — only if imports/inheritance need resolution

Most import styles resolve via the shared `resolve_file_imports`. Add a dedicated
pass in `crates/indexer/src/lib.rs` **only** if your language resolves imports
unusually — e.g. to a **`Folder`** node rather than a file's `Module` (real cases:
Perl `use lib 'lib'` → `Folder:lib`; SCSS `@use '../abstracts/x'` → `Folder:abstracts`;
Beancount/Nix directory includes). Gate the pass to your file extension so it is a
no-op for every other language, and place it in the existing pass sequence. Same for
inheritance/implements routing if the language has class hierarchies.

**Rule — never widen a shared change silently.** If you touch a shared file
(`extract.rs` dispatcher, `indexer/src/lib.rs`, `discover/src/lib.rs`, `spec.rs`),
scope the change to your language and re-verify 2–3 neighbor languages that share
that file still produce identical graphs (index their fixtures, compare `stats`).

**Rule — leave structural folder edges to the indexer.** The `Folder`/`File`/
`Module` nodes and the `CONTAINS_FILE`/`CONTAINS_FOLDER`/`DEFINES` containment spine
are built for you by the structural pass — your extractor does **not** emit them.
Do not hand-add or hand-remove folder edges to make a count come out a particular
way. If your fixture has deeply nested directories and the `CONTAINS_FOLDER` count
looks off by one, that is a known structural nuance in how the containment chain is
built on deep trees, not something your language extractor should fix: note it in
the PR and leave it for maintainer review. (Maintainers keep folder-edge
construction uniform across all languages via
`prune_c_walk_folder_edges_shared_langs` in `crates/indexer/src/lib.rs`; a
contributor should never need to touch it.) Focus your verification on the nodes and
edges your extractor actually produces — defs, calls, imports, usages.

---

## 9. Non-regression + green build (mandatory before PR)

```bash
# All parser/indexer tests must pass.
cargo test -p greppy-parser -p greppy-indexer
```

**Lint/format — check for *new* problems, not a clean baseline.** The tree may carry
pre-existing clippy warnings and rustfmt diffs in code you never touched; a blanket
`cargo clippy … -D warnings` / `cargo fmt --all --check` can fail on that debt and
does **not** mean you broke anything. What matters is that **your additions** are
clean:

```bash
# Format only the files you changed:
rustfmt --edition 2021 --check crates/parser/src/langs/<lang>.rs
# Clippy on the crate you touched; confirm no NEW warning names your files:
cargo clippy -p greppy-parser --no-deps 2>&1 | grep -i '<lang>' || echo "no new lints in my files"
```

If a wider `clippy`/`fmt` is red, confirm every red line is in a file you did not
edit before proceeding.

**Non-regression — prove you didn't change other languages.** Index a few
established fixtures (or any repo in those languages) and confirm `greppy stats` is
identical before and after your change. At minimum re-check the neighbor languages
that share any file you touched (§8) — e.g. if you added an `ImportStrategy` variant
or edited `spec.rs`, index a couple of unrelated languages and confirm their graphs
are byte-identical.

Add a `#[test]` for your language near the extractor (pattern: a `const SRC`,
`extract(...)`, then `assert_eq!` on node/edge counts + labels). See the existing
tests at the bottom of `crates/parser/src/extract.rs`.

---

## 10. Update the status doc

In `LANGUAGE_SUPPORT.md`: flip your language's row to
`… | ✅ **PASS** — <one-line what it extracts + how verified> | **supported**`,
and bump the `**N supported**` count line (add your language name to the list).

Only claim `supported` if §5, §6, and §9 all hold on a fresh adversarial fixture.
Otherwise use `verified` (fixture green but not adversarially stressed) or leave it
`wired`/`bespoke` and describe the gap.

---

## 11. Open the pull request

Commit **only** `crates/` + `LANGUAGE_SUPPORT.md` (never the fixture, never `.vendor`,
never a debug `examples/dump_*.rs`). Push a branch and open a PR whose body states:

- What the language needs (the graph model: which node/edge kinds, why).
- The fixture you used and the **evidence** it's correct: the `stats` output, the
  ripgrep completeness reconciliation (§6), and the neighbor re-checks (§8).
- Any deliberate, documented exclusion (with the reason).
- Confirmation that `cargo test`, `clippy`, and `fmt` are green.

A reviewer should be able to re-run your fixture and see the same graph. If you fell
short of `supported`, open a **draft** PR with the exact remaining diff — an honest
partial is welcome; a false "supported" is not.

---

### Quick reference — the whole loop

```
write fixture + EXPECTED.md  ->  wire detection  ->  register lang (spec + queries)
  ->  cargo build  ->  index fixture + `stats`/`callees`/`who-calls` vs golden
  ->  ripgrep completeness reconcile  ->  (bespoke extractor / indexer pass if needed)
  ->  cargo test + clippy + fmt + neighbor non-regression  ->  flip LANGUAGE_SUPPORT.md
  ->  commit crates/ only  ->  open PR with the evidence
```
