`greppy` is this system's tool for coding tasks. Written like grep (`greppy PATTERN [FILE]`,
`greppy -n …`) it searches like grep and prints byte-identical output; `greppy rg …` does the same
for ripgrep syntax. Everything else is listed below.

Throughout: S is a symbol name — a function, method, class or type. Where a name is defined more
than once, qualify it with its file: `edit-src/data.rs::run`. H is a handle, printed by
`--handle`. A:B is a line range, 1-based, both ends included. A result is printed as
`file:line  name`; a result that lives in a test carries a trailing `test`.

SEARCH:
  search "WHAT IT DOES"             definitions that do what you describe, found by meaning —
                                    for when you do not know the name: "restrict a value to a
                                    range", "retry a failed request"
  search-symbol NAME                definitions whose name contains NAME, with their kind
  search-pattern REGEX              every text occurrence, comments and strings included, with
                                    the definition it sits in

  --fixed           search-pattern: the argument is literal text, not a regular expression
  --kind KIND       search, search-symbol: function, method, class, struct, enum, trait
  --code            also print the source at each reported location
  --all             every result instead of the first few

NAVIGATE:
  who-calls S                       every place that uses S: calls, imports, type references
  callees S                         what S uses, and where those are defined
  brief S                           what S does in one sentence, its signature, then its body
                                    sketched: one line per step with the symbol used there and
                                    what happens
  impact S                          how far a change to S reaches, as a tree of callers, each
                                    with what it does, tests marked
  path --from A --to B              every call chain from A to B, as a tree of the call sites
                                    they hang on

  who-calls and callees answer for several symbols at once: `who-calls A B C`.

  --code            also print the source at each reported location
  --all             every result instead of the first few
  --depth N         impact: how many steps to walk instead of 6
  --direction incoming|outgoing     impact: which way to walk instead of incoming

ORIENT:
  map [PATH]                        which languages, modules, test roots and build+test commands
  outline PATH                      one line per definition in that file: kind, signature, span
  changes                           what you have changed so far, grouped by symbol, with the
                                    callers and tests each change affects
  verify -- CMD                     runs CMD, and again against the base revision in a separate
                                    worktree, and reports what CMD breaks that it did not break
                                    before. Nothing is stashed and nothing is checked out.

  --base REV        changes, verify: compare against REV instead of the base revision

READ:
  read S [S …]                      the source of those definitions
  read PATH [PATH …]                those files whole
  read PATH --lines A:B             those lines of that file

  A name that is also a path on disk is read as the file; write `--symbol S` to force the symbol.

  --handle          also print a handle marking exactly what was printed — if the output was cut,
                    the handle covers the shown part only, never the rest. Give it to an edit as
                    `--target H`: the edit changes that span or nothing, and refuses if the file
                    changed in between.
  --context N       read S: also the N lines above the definition, so its doc comment comes along

EDIT: an edit applies completely or changes nothing. It reports the file, the span it wrote, the
resulting text, and a handle for that new span, so nothing needs reading back.

  replace WHERE --content TEXT | --content-file F     put the new text in place of WHERE
  insert  WHERE --after|--before --content TEXT | --content-file F
  delete  WHERE                     remove what WHERE points at
  patch   WHERE --patch-file F      apply a unified diff. Its line numbers may count from the
                                    start of the file or from the start of WHERE — whichever the
                                    context lines confirm. Paths inside the diff are ignored.

  WHERE is exactly one of:
    --file F --old TEXT | --old-file F        that exact text; by default it must occur once
    --file F --pattern REGEX                  what the regular expression matches
    --file F --lines A:B                      those lines
    --file F                                  the whole file
    --symbol S                                the whole definition of S
    --symbol S --body                         only its body; the signature stays as it is
    --target H                                the span a handle marks
  `insert` takes --symbol, --lines or --target; the others take any of them.

  Whole files:
    write --file F --content-file F2          create that file; `replace --file F` needs one
                                              that already exists
    move --file A --to B                      move or rename it, and update the imports naming it
    remove --file F                           delete it, and report what still references it

  These do language work that replacing text cannot:
    rename --symbol S --to N        rename S and every reference to it; reports what it could
                                    not resolve
    rename --in S --call A --to B   make S call B where it called A
    change-signature --symbol S --spec '{"params":[…],"returns":"…"}'
                                    change a signature and every call site with it
    ensure-import --file F --module M [--name N]     add an import if the file is missing it
    ensure-method --symbol CLASS --name N --content-file F   add a method the class lacks
    ensure-argument --symbol S --call C --arg A      add an argument to a call inside S
    ensure-annotation --symbol S --annotation A      add an annotation to a definition
    data set --file F --path '$.a.b' --value-json V  set a value in JSON, TOML or YAML
    data delete --file F --path '$.a.b'              remove one

  apply --plan F                    many edits as one single change; the plan is
                                    {"operations":[{"verb":"replace","file":"a.rs","old":"x",
                                    "new":"y"}, …]} and every verb above may appear in it
  undo                              reverse the last edit, if the file still looks the way
                                    that edit left it
  recover                           finish or roll back an edit that was interrupted

  --content TEXT    the new text, for short single-line text
  --content-file F  the new text from a file; --patch-file, --plan and --old-file work the same
                    way. Write `-` for the name to read it from the pipe, which keeps the edit in
                    one call and stops the shell from mangling quotes and backticks:

                      greppy edit replace --symbol S --body --content-file - <<'EOF'
                      {
                          …the new body, verbatim…
                      }
                      EOF

  --expect N        require exactly N matches instead of one, and change nothing otherwise
  --verify          after writing, run the build or linter for the touched files and report the
                    diagnostics against symbols and spans
  --dry-run         report what it would write and write nothing
  --report FILE     write the full record of the edit to a file

CHAIN — every command takes its input from the pipe, so one result goes straight into the next:
  greppy search-symbol NAME --json | greppy read -
  greppy who-calls S --json | greppy brief -
  greppy edit apply --plan -

ON EVERY COMMAND:
  --path P          only results under that file or directory
  --json            the same answer as data, with exact counts
  --all             every result; without it long results are cut and the output ends with the
                    exact command that continues them
  --limit N         at most N results; --offset K starts at the Kth
  --root DIR        work on a different repository; greppy finds the current one by itself
  --help            the syntax of that command, with a working example

A question with an empty answer — a symbol that exists but has no callers — prints nothing and
exits 0. A question that cannot be answered — a symbol or file that does not exist, a selector
that matches nothing — says why and exits non-zero, and an edit in that situation writes nothing.
In grep-compatible form the exit codes are grep's: 0 for a match, 1 for none.
