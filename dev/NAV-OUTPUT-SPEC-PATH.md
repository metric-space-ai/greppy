# path — output specification (0.3.0)

Normative. The shared laws of `dev/NAV-OUTPUT-SPEC.md` apply. Output is English.

## What `path` is for, and why it is not `impact`

`impact A` walks everything a change to `A` reaches. `path --from A --to B`
walks only the branches that end at `B`: **every leaf of its output is the
target.** That is the whole difference, and it is what makes the output small
enough to answer a question `impact` on a hub cannot.

The answer is a tree of **edges**, not of nodes. A path is made of call sites,
and a call site is the only address in this family that a caller can actually
edit to break the connection. `callees` says where a callee is defined,
`impact` says how far a change reaches, `brief` says what something does — only
`path` says *where the chain hangs*.

## Shape

The start symbol at its definition, then one line per call, indented by the
step. The address is the call site inside the parent; the name is what is
called there.

```
$ greppy path --from data_set --to parse_path
edit-src/data.rs:81  data_set
  edit-src/data.rs:99  parse_path
[exit 0]
```

Several paths become one tree with the common prefix written once:

```
$ greppy path --from data_set --to sha256_hex
edit-src/data.rs:81  data_set
  edit-src/data.rs:90  planned_precondition_refusal_for
    edit-src/verbs.rs:95  planned_preconditions_hold
      edit-src/verbs.rs:82  sha256_hex
  edit-src/data.rs:185  apply_in_memory
    edit-src/txn.rs:99  sha256_hex
  edit-src/data.rs:200  run_pipeline_public
    edit-src/verbs.rs:2213  run_pipeline
      edit-src/verbs.rs:2280  apply_in_memory
        edit-src/txn.rs:99  sha256_hex
      edit-src/verbs.rs:2316  sha256_hex
      edit-src/verbs.rs:2395  planned_preconditions_hold
        edit-src/verbs.rs:82  sha256_hex
      edit-src/verbs.rs:2403  publish_atomic
        edit-src/publish.rs:79  sha256_hex
[exit 0]
```

Every line above is real: the six paths and every call-site line come from the
sample repo. This is the reference a correct implementation must reproduce.

Fourteen lines for six paths; side by side they would be twenty-four with the
common head repeated three times. And the shape answers the question `path`
exists for: **no single line cuts this.** Three independent branches leave
`data_set` at 90, 185 and 200. A tree that forks only further down would name
its cut point in one line.

**No sentences.** They are what made an earlier draft look like `impact`. Here
the location carries the answer; meaning is `brief`'s job.

**Order.** Children by ascending call-site line. Deterministic.

**A shared suffix is repeated, not deduplicated.** `apply_in_memory ->
sha256_hex` appears on two branches because it is walked twice. Collapsing it
would break the rule that every branch is a complete path.

**Cap.** The number of simple paths grows exponentially. At most eight are
walked; when the walk hits the cap the last line of the output is
`at least 8 paths shown`, so the tree never claims a completeness nobody
computed.

## Non-answers

```
$ greppy path --from parse_path --to data_set
no path from parse_path to data_set
[exit 0]
```
Today this prints nothing at all with exit 0, which reads as "the tool did
nothing" and cannot be told apart from a failure.

```
$ greppy path --from run --to parse_path
`run` is 5 definitions
cli-src/changes.rs:146
cli-src/main.rs:25
cli-src/map.rs:47
cli-src/trial.rs:168
cli-src/verify.rs:665
[exit 1]
```
Today: empty output, exit 0. An ambiguous endpoint makes the question
unanswerable, and `who-calls` already refuses it — through the same helpers,
`nav_refuse_ambiguous` and `nav_refuse_non_callable`.

```
$ greppy path --from data_set --to xyzzy_frobnicate
no symbol `xyzzy_frobnicate`
[exit 1]
```
Today: `(symbol not found: xyzzy_frobnicate)`. Same message as everywhere else,
via `nav_report_missing`.

`--from A --to A` is the question "is there a call cycle through A". It is
answered by a tree that returns to `A`, or by `no path from A to A`. The single
node that is printed today implies a path that does not exist.

## Exit codes

* 0 — an answer, including `no path`
* 1 — a well-formed call with no answer: no symbol, ambiguous endpoint
* 64 — a malformed invocation. `--from A B` already yields clap's
  `unexpected argument 'B' found` plus the usage line; that is correct and stays.

Several endpoints per side are NOT supported. A path has one start and one end;
a cross product of endpoints is several questions, needs grouping, a shared
expand and a per-pair count, and there is no evidence any agent asks for it.

## Interface lies to remove

* `--edge USES` and `--edge TYPE_REF` are accepted, but the store holds `USAGE`
  and `TYPE_ASSIGN`. They match nothing and report exit 0. Rename or drop.
* `--code` is documented in its own help as "Accepted for agent ergonomics —
  no-op". A flag the agent pays tokens for and that does nothing.

## The graph underneath is not sound

Measured on the sample repo:

```
self CALLS edges                                   11   (4 checked, 4 false)
CALLS edges whose target name exists in >1 file   314
CALLS edges total                                2401
```

All four checked self-edges are misresolutions: `parse` -> `<Self as
Parser>::parse()`, `new` -> `Vec::new()`, `definition_end_idx` -> the
same-named function in `greppy_core`. 13% of all call edges point at a name
that exists in more than one file, i.e. each was resolved by picking one
candidate. `path` must not compensate for this by filtering; fixing the
resolver is a separate job.
