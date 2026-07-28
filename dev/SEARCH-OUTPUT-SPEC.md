# search / search-symbol / search-pattern — output specification (0.3.0)

Normative. The shared laws of `dev/NAV-OUTPUT-SPEC.md` apply: no justification,
no instruction, count only what is missing, no duplicated identity, source
verbatim, never answer a question that was not asked. Output is English.

## The surface

The three commands `search-code`, `search-symbols` and `semantic-search` are one
axis — what the caller already knows — spelled as three unrelated names. They
become one family; 0.3.0 is a breaking release and there are no aliases, no
deprecation lines, no legacy paths:

```
search "WHAT IT DOES"       definitions, by meaning — the default verb
search-symbol NAME          definitions whose name contains NAME
search-pattern REGEX        every text occurrence matching REGEX
```

* The bare name goes to meaning because that is the safe default: a meaning
  query that happens to be a name still finds the name; a name query that is
  actually a description finds nothing.
* `search` joins everything after the verb into one query. The shell strips
  quotes before greppy sees anything, so
  `greppy search retry a failed request` works; the quotes in the prompt are
  a signal to the agent ("this is prose"), not mechanics.
* `search-pattern`, not `search-code`: "pattern" says regex, and "code" was
  wrong anyway — it matches comments, strings and config, which is precisely
  why one calls it.
* The removed names get the passthrough guard that `find-usages` and
  `references` already have. `greppy semantic-search X` must be an unknown
  subcommand, not a grep for "semantic-search" in a file called X.
* The grep-compatible mode needs no flag: its spelling is grep's own.
  `greppy -rn PATTERN PATH` stays byte-identical passthrough. A `--grep` flag
  would be a second spelling of an existing thing.

## Result lines

The family base line is the NAV row: `file:line  name`, two spaces between
fields, a trailing `test` for hits that live in tests. Each command adds only
what its question needs.

**search — the question is "what does it do", so the line carries the
sentence** (the navigation hint, first letter lowercased, no trailing period,
after ` — `; a node with no hint gets no dash). Order is similarity rank —
NOT file grouping; the rank is the information. Real output shape, from the
sample repo:

```
$ greppy search restrict a value to a range
cli-src/lib.rs:9998  edit_parse_line_range
cli-src/lib.rs:22205  vector_exact_scan_exceeds_limit — returns the candidate limit when it exceeds total
cli-src/lib.rs:8312  parse_read_line_range — parses a file range for read
[exit 0]
```

Eight hits at most. No count line — there is no true total, everything is
somewhat similar. No expand offer: `--code` is the flag for sources. No spans
in the address, no flattened signatures (today's output prints
`fn edit_publish( root_path: ..., )` as one line — text that exists in no
file), no indentation.

While the embedding index is still building: one status line with progress and
ETA, exit 1. Never partial hits.

**search-symbol — the question is "what is this name", so the line carries the
kind** at the end, from `nav_kind_word`:

```
$ greppy search-symbol data_set
edit-src/data.rs:81  data_set  function
[exit 0]
```

The query must be CONTAINED in the name. That single rule removes today's
fragment noise (`data_set` matched `toml_set_scalar` — shared fragment "set").
Exact name equality first, then by name length, then by address.

**search-pattern — the question is "where does this text occur", so the line is
the address plus the enclosing definition**; a hit at file level is the bare
address:

```
$ greppy search-pattern "data path must start"
edit-src/data.rs:32  parse_path
[exit 0]
```

Grouped by file, fewest first; past 25 hits the distribution line plus five
rows, exactly as in NAV. `--code` prints each matched line verbatim, the
address naming exactly what follows. No handles, no `NNNNN |` prefixes, no
whole enclosing definitions (today one hit prints a 68-line test function).

## The miss cascade (search-symbol)

Stages fire only when every earlier stage is empty; only the best stage is
shown, and its label states how much it is worth — a typo hit is near-certain,
a meaning hit is a guess:

1. exact name
2. case/underscore normalization (`dataSet` → `data_set`)
3. edit distance ≤ 2 (`data_sett` → `data_set`)
4. shared identifier words, best tier only (`parseJsonPath` → `parse_path`)
5. meaning — labelled `closest by meaning:` and carrying sentences, not kinds

```
$ greppy search-symbol dataSet
no definition named `dataSet`

similar names:
edit-src/data.rs:81  data_set  function
[exit 1]
```

When even stage 5 finds nothing above the floor, the first line stands alone.
`search` itself has no cascade: it IS the last stage.

`search-pattern` with zero hits prints `no matches`; when the case-insensitive
run WOULD hit, one computed fact follows — `case-insensitive: 12 matches` — a
number, not an instruction.

## Exit codes: grep's convention, deliberately

`0` for at least one hit, `1` for none, `64` for a malformed invocation. This
differs from NAV, where `no callers` is an answer with exit 0 — and both are
right: for search commands grep's convention is burned into every model, and
the prompt already promises it for the passthrough. Two families, two trained
conventions, each the native one.

## Prompt section (target state)

```
SEARCH:
  search "WHAT IT DOES"       definitions that do what you describe, found by
                              meaning — for when you do not know the name
  search-symbol NAME          definitions whose name contains NAME, with kind
  search-pattern REGEX        every text occurrence, comments and strings
                              included, with the definition it sits in

  --kind K          only function, method, class, struct, enum or trait results; a text match
                    counts by the definition it sits in
  --code            also print the source at each result
  --all             every result instead of the first few

> **Amendment (owner rule): footer flags hold for every command in the
> section.** `--kind` therefore also applies to `search-pattern`, counting a
> text match by the definition it sits in. `--fixed` is not a footer flag any
> more but part of `search-pattern`'s syntax: `search-pattern REGEX [--fixed]`.
> The prompt block above in this spec is superseded by AGENTS.md, which is
> already updated and guarded by prompt_contract.rs.
```
