# brief — output specification (0.3.0)

Normative. The shared laws of `dev/NAV-OUTPUT-SPEC.md` apply. Output is English.

## What `brief` is, and what it is not

Today `brief` prints the definition, then `-- CALLERS (n) --`, then
`-- REFERENCES (n) --`, then `-- CALLS (n) --`. That is `who-calls` and
`callees` in one call with ASCII bars around them, and it is byte-identical to
0.2.1 — every output literal of `dispatch_brief` is unchanged between `v0.2.1`
and HEAD.

`brief` is not a bundle of the other commands. It is **the body of a function,
sketched**: a shortened version of the real source in which the working lines
are replaced by what happens there, with every symbol name kept real so the
agent can carry it to the next command. It is the orientation aid and the hub;
`who-calls` and `callees` give addresses, `brief` gives understanding.

## Shape

```
$ greppy brief data_set
Sets a value at a JSON, TOML or YAML path in a file and returns a certificate.

edit-src/data.rs:81-88
pub fn data_set(
    workspace_root: &Path,
    file: &Path,
    path: &str,
    value_json: &str,
    _ensure: bool,
    options: &VerbOptions,
) -> Result<Certificate> {
   89  Snapshot::read — reads the file and its hash
   90  planned_precondition_refusal_for — returns a certificate and stops if a precondition fails
   99  parse_path — turns "$.a.b[0]" into segments
  100  refuses a --value-json that is not valid JSON
  103  reads the file extension
  108  match extension
  109    json — json_value_spans, classify_spans
  117    toml — toml_value_spans, json_to_toml, classify_spans
  128    yaml — yaml_scalar_spans, yaml_scalar, classify_spans
  132    else — refuses anything but .json, .yaml, .yml, .toml
  140  match lookup
  141    missing — single_refusal_certificate, Status::NotFound
  151    ambiguous — single_status_certificate, Status::Ambiguous
  162    unique — keeps start, end and the replacement text
  165  single_status_certificate — Status::AlreadySatisfied if the bytes already match
  176  builds one PlannedOp for that byte range
  185  apply_in_memory — applies it to the snapshot without writing
  186  match extension — re-parses the projected document before anything is written
  200  run_pipeline_public — writes and returns the certificate
}

called by dispatch_edit_inner and 6 tests
expand 4dcc534d — the call tree below data_set sketched, 47 functions, 640 lines
[exit 0]
```

129 body lines become 22. Every line number, symbol name and branch above is
real, taken from `edit-src/data.rs:81-209` of the sample repo.

## Rules

**The sentence comes first.** One line, no address — the only line in the output
without one, which is how it is recognised. It is the definition's doc comment
when there is one, and the generated navigation hint otherwise: an authored
sentence beats a generated one.

**Then the address and the head, verbatim.** The head runs from the first
attribute (`#[test]`, `#[derive]`, a decorator, an annotation — they change
behaviour and belong to the interface) through the line that opens the body.
`where` clauses and multi-line return types are therefore included without a
special rule. Doc comments are NOT part of the head; they became the sentence.
The address names the range and exactly those file lines follow, byte for byte.
Measured in this repo, 289 of 1247 function signatures span several lines, so
one-line is the short case, not the normal one.

**Then the sketch, one line per step.**

```
<line>  <symbol> — <what happens there>
<line>  <what happens there>
```

The separator is exactly space-em-dash-space and appears nowhere else on the
line: everything before it is a symbol the agent can pass to `brief`,
`who-calls`, `read`. A line without a symbol carries only the description, and
that is the signal that there is nothing here to follow.

**Indentation is the source's own.** The branches of a `match` sit one level in,
so the shape of the function is visible without a word describing it.

**The closing brace closes the sketch.** It is the last line of the head's
block, and it makes the sketch read as code.

**Then one line for the callers.** Aggregated, not listed: `called by
dispatch_edit_inner and 6 tests`. They are not part of the body, and the full
list is `who-calls`.

**Then expand, when it is worth it.** The pack is the same sketch for every
function this one calls, recursively, each function once, a repeat rendered as
the name plus `(above)`. That is the case where expand earns its line: the
follow-up is not a flag, it is twelve separate `brief` calls. The offer states
the number of functions and the number of lines, both computed while preparing
the pack.

## Kinds other than functions

A struct, enum or trait has no body to sketch: its fields, variants or method
signatures ARE its interface. For those the head is the whole definition, the
sentence stays, and there is no sketch and no closing brace.

## Failures

Identical to `who-calls`, through the same helpers:

* several definitions → `` `run` is 5 definitions `` plus one address per line,
  exit 1. Today `brief run` prints five complete function bodies, about 250
  lines, where `who-calls run` refuses.
* no symbol → `` no symbol `x` `` plus `similar names:` when there are any, exit 1
* a name that resolves but has nothing to sketch → the kind line, exit 1

## What has to be built underneath

The per-symbol navigation hints exist and cover every node — checked on twenty
symbols. The branch lines come from the parser. What does NOT exist is a
sentence for the lines that contain no call (100, 103 and 176 in the example
above): those need a generation step greppy does not have today. Until it does,
such lines are omitted rather than invented, and the sketch is the calls and the
branches only.

Two of twenty sampled hints are malformed — they return the source line instead
of a sentence:

```
json_set_preserves_formatting|        fn json_set_preserves_formatting() {
toml_set_scalar|                      fn toml_set_scalar() {
```

That is not an inaccurate summary, it is a defect in generation, and in `brief`
the sentence is the merchandise. Validation at index time — the hint must be a
sentence and must not be a copy of a line inside the span — belongs to the
generator, not to this output.
