# read / read-smart / read-file — output specification (0.3.0)

Normative. The shared laws of `dev/NAV-OUTPUT-SPEC.md` apply: no justification,
no instruction, count only what is missing, no duplicated identity, source
verbatim, never answer a question that was not asked. Output is English.

## The law of this family

`read` prints file bytes, verbatim — every interpreted view of code lives in
`brief`, `impact` and `search`. (Design rule for us; it does NOT appear in the
prompt.) One verb, one behaviour: the bare `read` never truncates, never folds,
never summarizes. Reliability comes from verbs having exactly one behaviour,
not from flags modulating a trusted one — which is why the lossy view is its
own verb whose name announces the trade: `read-smart` is interpreted and
gapped; `read-file` bypasses the symbol system.

## The surface

```
READ:
  read S [S …]              the source code of S; --head M and --tail N for only its
                            first M and last N lines
  read-smart S [S …]        the source code of S, nested blocks below --depth N folded
                            into one-line semantic descriptions; default 1
  read-file PATH [PATH …]   the files; paginated at 400 lines unless --lines A:B or --all

  --handle                  also print a handle naming exactly the span that was printed
```

This block replaces the READ section of AGENTS.md verbatim. Deleted with it:
`--context N` (its documented purpose — "so its doc comment comes along" — is
now default behaviour), the `--symbol` force flag and the guess heuristic
"a name that is also a path on disk is read as the file" (read takes symbols,
read-file takes paths; nothing is guessed), `--lines` on symbols, and the
`--level/--no-code` drafts.

## read

**Whole, always.** No size threshold, no auto-degrade. Naming the symbol is
the consent. The earlier ~120-line head-cut violated the expand law (a pack
must hold far more than what is shown) and, worse, made the bare verb
unpredictable.

**The span includes the definition's documentation, both directions.** Rust
`///` and attributes above the head, Python docstrings after the signature
(already inside the span). The header names the range actually printed:

```
$ greppy read parse_path
edit-src/data.rs:27-71  parse_path
/// Parses a `$.a.b[0]` data path into segments.
fn parse_path(path: &str) -> Result<Vec<Seg>> {
…43 verbatim lines…
}
[exit 0]
```

Header `file:start-end  name`, two spaces, then exactly those file lines,
byte-identical: no line numbers, no prefixes, no trailing blank line. Several
symbols print as blocks separated by one blank line; a symbol that does not
resolve gets its group message and the others still print (partial delivery,
exit 1).

**`--head M` and `--tail N`** are explicit partial loads and combinable:
`read S --head 30 --tail 10` prints two blocks — the first 30 and the last 10
lines, each under its own `file:start-end  name` header, nothing between. The
numbers are the agent's own; nothing is hidden by policy.

## read-smart

**The folding rule is structural and exceptionless — that is the trust.**
Counted from the function body, every block that opens below `--depth N`
(default 1) folds into one gap line. At default the top level stays raw and
every nested block folds — including a three-line `if`: rule purity over
micro-savings, the same trade as byte-exact `read`.

**The gap line** carries the range, the sentence, and the full command:

```
    … 38-68 ⟨sentence⟩ — greppy expand 4f21c8
```

* the leading `…` is the only non-source marker in the output; every line is
  mechanically classifiable (source vs. gap) without lookahead
* the sentence is generated on demand by the span summarizer
  (`summarize_definition_span` accepts arbitrary spans; measured 0.3–0.9 s per
  block on the GPU build) — it describes exactly the folded lines
* the offer is the complete command, copyable as printed

**Expands chain.** Filling a gap returns those lines verbatim; blocks below
the depth inside them fold again with their own ids. The descent is arbitrary
and every step is priced before it is bought. Packs store address plus content
hash and relocate-or-refuse on drift.

The reference output for the spec is captured from the real summarizer at
acceptance time — no invented sentences in examples.

## read-file

Classic reading. Whole file up to 400 lines; past that the first 400 and a
fact line with the continuation as a full command:

```
23,976 more lines — greppy expand 91ab03f2 continues at 401
```

Each expand delivers the next 400 and ends with the next offer. `--lines A:B`
prints exactly those lines (header without a name — it is not a definition);
`--all` prints everything at once. Several PATHs print as blocks. Exit codes:
missing file → the OS truth (`no such file: PATH`), exit 1.

## Failures — the NAV helpers, not a second language

* unknown symbol → `no symbol \`x\`` plus `similar names:` when any, exit 1
  (today: `read: no definition found for \`zzz_nix\``, exit 10 — the `read:`
  prefix and the out-of-contract exit code both die)
* ambiguous → `` `run` is 5 definitions `` plus one address per line, exit 1
  (today read silently picks one)
* `read-smart` on a kind with no body (struct, enum) → the whole definition,
  no gaps: nothing to fold is not an error

## --handle

Only with the flag, one line, format C: versioned binary, base64url, 128-bit
digests — target ≤ 70 chars against today's ~340-char `geh1:` JSON blob, which
is pure prefill the model must copy. When output was cut (`--head`, pages),
the handle covers the shown part only — never the rest.
