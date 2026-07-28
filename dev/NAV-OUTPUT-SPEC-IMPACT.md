# impact — output specification (0.3.0)

Normative. Where this document and the code disagree, the code is wrong.
Output is English. The shared laws of `dev/NAV-OUTPUT-SPEC.md` apply here too:
no justification, no instruction, count only what is missing, no duplicated
identity, source verbatim, exit 0 only for an answer.

## What is wrong today

Measured on the sample repo, `greppy impact parse_path`:

```
hop 1 edit-src/data.rs::Function::data_set edit-src/data.rs:81-209
hop 1 edit-src/data.rs::Function::data_delete edit-src/data.rs:218-324
hop 2 cli-src/lib.rs::Function::dispatch_edit_inner cli-src/lib.rs:8827-9443
hop 2 edit-src/data.rs::Function::json_set_preserves_formatting edit-src/data.rs:764-787
…
Expand: greppy expand 3aacf17fdbdb9918  (prepared evidence: 15 spans)
```

1. **It flattens a tree into levels.** At `hop 3 dispatch_edit` nobody can tell
   whether the path runs through `data_set` or through `data_delete`. That
   parentage is the only thing `impact` has that `who-calls` run six times does
   not, and it is thrown away.
2. **Nothing is marked as a test**, although the system prompt promises "and the
   tests among it". Seven of the fifteen reached symbols are tests.
3. `hop N ` is a six-character prefix on every line for a number that changes
   six times in fifteen rows.
4. The path is printed twice per line, with a `Function::` segment between.
5. An expand offer under a result that is already shown in full.

## The shape

Indentation is the path. One node per line, its children indented by two:

```
$ greppy impact parse_path
edit-src/data.rs:81  data_set — loads a snapshot and classifies JSON or TOML content
  cli-src/lib.rs:8827  dispatch_edit_inner — dispatches an EditCommand to a grammar handler
    cli-src/lib.rs:8817  dispatch_edit — dispatches an EditCommand and returns the result or an error code
      cli-src/lib.rs:3310  dispatch_subcommand — routes subcommand variants to handlers by Command variant
        cli-src/lib.rs:3239  dispatch — dispatches a Cli to its subcommand or greppy passthrough handler
          cli-src/lib.rs:24468  dispatch_to_code — dispatches a Cli to code and returns the result
            cli-src/lib.rs:1846  run_os — dispatches the command line based on the first argument
          cli-src/lib.rs:26147  trace_invalid_direction_is_a_usage_error  test
      cli-src/lib.rs:24613  edit_symbol_subprocess_helper  test
  edit-src/data.rs:764  json_set_preserves_formatting  test
  edit-src/data.rs:790  json_missing_path_refuses  test
  edit-src/data.rs:799  json_array_index  test
  edit-src/data.rs:817  ensure_is_idempotent  test
  edit-src/data.rs:834  toml_set_scalar  test
  edit-src/data.rs:854  yaml_scalar_set  test
edit-src/data.rs:218  data_delete — handles deletion of a file, returning a certificate
  cli-src/lib.rs:8827  dispatch_edit_inner  (above)
[exit 0]
```

Everything above is real: the addresses, the names and the sentences all come
from the index of the sample repo, and the parentage from `who-calls` on each
node. It is the reference for what a correct implementation prints.

Read off it what a flat list cannot say: a change to `parse_path` runs through
`data_set` **and** `data_delete`, both meet again at `dispatch_edit_inner`, and
all six tests hang off the `data_set` branch — `data_delete` is untested.

## Rules

**Line format.** `<file>:<line>  <name>` — the same row as `who-calls` and
`callees`, from `nav_short_name`, with the definition line as the address. A
test carries a trailing `  test`.

**The sentence.** After the name, ` — ` and the node's navigation hint. The
separator is exactly space-em-dash-space and appears nowhere else on the line,
so everything before it is a symbol the agent can pass to another command.
A node with no hint gets no separator and no sentence.

**Tests get no sentence.** The name of a test states what it checks; its hint
would restate it.

**Indentation is the hop.** Two spaces per level. No `hop N` prefix, no level
headers: the depth is visible.

**Every node appears once.** A node already printed is repeated as
`<file>:<line>  <name>  (above)` with no children and no sentence. Cycles
therefore terminate without a depth limit, and the CLI spine — which every
symbol in this repo reaches — is written out once instead of once per branch.

**Order.** Children in the order of `who-calls` / `callees` for that node:
by file with the fewest results first, then by line. Deterministic.

**Size.** Up to 25 nodes the whole tree is printed. Past that, the shape leads:
`N reached: <file> <n>, …` (files fewest-first), a blank line, then the first
five top-level branches with their subtrees pruned to the same budget. No expand
offer — `--depth` and `--all` reach everything the pack could carry.

**`--direction outgoing`** walks callees instead of callers; everything else is
identical.

**`--depth N`** limits the levels. The default stays 6.

## Failures

Identical to `who-calls`, and through the same helpers:

* not callable in this direction → `` `X` is a struct, not a function  file:line ``, exit 1
* several definitions → `` `X` is N definitions `` plus one address per line, exit 1
* no symbol → `no symbol `X`` plus `similar names:` when there are any, exit 1
* nothing reached → `nothing reached`, exit 0

## Data the current implementation does not have

The walk records the distance of a node, not the edge it arrived on. The tree
needs the parent. Collecting has to change, not only printing — this is the
substantial part of the work, and it must not be faked by re-deriving the
parentage with a second query per node.

## Known defect this must not paper over

`impact data_set --direction outgoing` reports
`cli-src/inference_daemon.rs::LimitedFrame::write` because `data_set`'s body
calls `std::fs::write`. It is a false edge from same-name resolution across
unrelated types, and the transitive walk turns one false edge into dozens of
false descendants. Fixing the resolver is a separate job; `impact` must not
compensate for it by filtering.
