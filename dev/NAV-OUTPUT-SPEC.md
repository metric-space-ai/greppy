# who-calls / callees — output specification (0.3.0)

> **Amendment 2026-07-28.** `find-usages` is deleted: no alias, no deprecation
> line, no legacy path. Measured on `Snapshot` in the sample repo it reported
> 15 of 43 real references and zero imports, while promising "calls, uses,
> imports" in the prompt. A rename driven by it breaks the build.
>
> `who-calls` absorbs what was true in it: it walks `CALLS` **and** `USAGE`
> edges, both of which are already in the store. Consequently the wrong-kind
> refusal below does NOT apply to `who-calls` any more — a struct has real
> references and gets them. It still applies to `callees`, which asks what a
> symbol calls, and a struct calls nothing.
>
> Import edges and `Type::method` path references (22 of the 43) do not exist
> in the graph at all. Adding them is indexer work and is specified separately;
> until it lands, neither the prompt nor the output may claim completeness.

Normative. Where this document and the code disagree, the code is wrong.
All tool output is English. No German, ever.

## Laws that apply to both commands

1. **No justification, no instruction.** The output never explains itself, never
   suggests a command, never argues for a flag. Lines like `try: …`,
   `this sample usually answers the question`, `pass --all only if you truly
   need every site` are deleted, not reworded.
2. **Count only what is missing.** A trailing `— 2 callers` under two visible
   lines is packaging. A number appears only when something is not shown.
3. **Numbers are true numbers.** A count is never capped to the display limit.
   (`50 source match(es)` for 337 real occurrences was a false statement.)
4. **No duplicated identity.** The path appears once per line. No `Function::`
   kind prefix, no qualified name that repeats the file already in the address.
5. **Source is verbatim.** Any source line printed is byte-identical to the file:
   no dedent, no reformat, no prefix, no line numbers, no fence. The address
   above it names the range, and exactly that many file lines follow — for every
   result, without exception. Identical blocks are NOT merged.
6. **Exit 0 only for an answer.** `no callers` is an answer (0). Not found,
   ambiguous, wrong kind: 1. Exit 64 is not used by these commands.
7. **Never answer a question that was not asked.** No silent resolution of an
   ambiguous name, no substring matching, no textual fallback dump.

## Result-line format

```
<file>:<line>  <name>
```

two spaces between fields. `<name>` is the short symbol name. A caller/callee
that is a test carries a third field:

```
<file>:<line>  <name>  test
```

Fields are positional and splittable: neither a path nor a symbol name contains
whitespace.

## Ordering

Group by file; files with the FEWEST results first; within a file by ascending
line. Deterministic, no scoring. On a skewed distribution this puts the outliers
at the top, which is where the information is.

## Size

* total ≤ 25 → print every result line, nothing else.
* total > 25 → print, in this order:
  1. `N callers: <file> <n>, <file> <n>, …`  (files fewest-first; the same
     ordering as the result lines)
  2. a blank line
  3. the first 5 result lines
  and nothing more. The count says something is missing; that `--all` exists is
  documented in the system prompt, once, and never repeated in output.
* `--all` prints every result and suppresses the summary line.

There is **no expand offer** on these two commands. Everything an expand pack
could carry is one flag away (`--all`, `--code`), so the offer would be a second
spelling of an existing flag plus a handle to remember.

## `--code`

For each result, in place of the plain result line:

```
<file>:<start>-<end>  <name>[  test]
<exactly (end - start + 1) verbatim file lines>
```

A single-line span prints `<file>:<line>` and one line. A blank line separates
results. There is no cap, no budget, no truncation: the unflagged call already
told the agent how many results there are.

What the span covers differs by command, because the reported location differs:

* **who-calls** → the statement enclosing the call site, inside the caller.
  NOT the caller's body. (Today it prints the caller's body from its first line
  and truncates after 30, so the call site itself lands in the cut-off part.)
* **callees** → the callee's own definition span.

## Addresses

* **who-calls** → the call site inside the caller. The caller's name is the
  answer; where it calls from is the new information.
* **callees** → the callee's definition. The caller is known; where the callee
  lives is the new information.

Single-symbol and multi-symbol invocations MUST use the same address semantics.
(Today `who-calls A B` prints definition starts while `who-calls A` prints call
sites.)

## Several symbols

```
<symbol>
  <indented answer lines>
<symbol>
  <indented answer lines>
```

Queried symbol at column 0, its answer indented by two. A symbol that does not
resolve gets its own group with the message it would get alone. **Partial
results are always delivered**; one bad symbol never discards the others.
Exit 0 if every queried symbol resolved, otherwise 1.

A single symbol gets no header and no indentation.

## Failure outputs

Checked in this order. The kind check runs BEFORE the ambiguity check.

**Wrong kind** — the name resolves, but not to something callable:
```
`Snapshot` is a struct, not a function  edit-src/txn.rs:25
[exit 1]
```
Kind word from the index label: struct, enum, enum variant, field, variable,
trait, type. No usage count — it would not change the agent's next move.

Today `who-calls Snapshot` answers `(no callers)` with exit 0, and
`callees Snapshot` invents a callee with exit 0. Both are wrong answers wearing
the clothes of right ones.

**Ambiguous** — two or more callable definitions share the name:
```
`run` is 5 definitions
cli-src/changes.rs:146
cli-src/main.rs:25
cli-src/map.rs:47
cli-src/trial.rs:168
cli-src/verify.rs:665
[exit 1]
```
The name is omitted from the candidate lines: it is the same for all of them,
and what distinguishes them is the file, which the address already carries.
The agent disambiguates with `file.rs::name`.

**No symbol**:
```
no symbol `xyzzy_frobnicate`
[exit 1]
```
If similar names exist, they follow after a blank line:
```
no symbol `dataSet`

similar names:
edit-src/data.rs:81  data_set
[exit 1]
```
Similar means: edit distance ≤ 2 on the raw name, or equal after lowercasing and
removing underscores (this covers `dataSet` / `DataSet` / `data_sett`), or
sharing at least two identifier words with the query where the best tier alone is
shown (`parseJsonPath` → `parse_path`, sharing `parse` and `path`).
**Substring matches are never candidates** (`data` must not suggest `metadata`).
When nothing qualifies, no block follows — the absence is the statement.

**No results**:
```
no callers
[exit 0]
```
```
no callees
[exit 0]
```

## Graph correctness

`callees json_missing_path_refuses` currently reports
`cli-src/inference_daemon.rs::LimitedFrame::write` because the body calls
`std::fs::write`. A call to a symbol not defined in this repository must produce
no edge at all, and must never be attributed to an unrelated same-named symbol.
That the graph covers this repository only belongs in the system prompt, not as
a footer under every answer.

## Deleted from these two commands

* `try: greppy …` lines
* `— N callers` / `— N callees`
* `(no callers)` / `(no callees)` with parentheses
* `unresolved textual candidates:` and its listing
* `… and N more (M shown of T total — this sample usually answers …)`
* `Expand: greppy expand …` offers
* `Function::` / `Class::` prefixes and the repeated path
