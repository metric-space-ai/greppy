`greppy` holds this repository as a graph: every definition, what it calls, what uses it,
and a meaning index over its source. Invoked like grep (`greppy PATTERN [FILE]`,
`greppy -n …`) it stays grep — byte-identical output, grep's exit codes; `greppy rg …`
the same for ripgrep syntax.

Throughout: S is a symbol — a function, method, class or type; a name defined more than
once is qualified with its file, `edit-src/data.rs::run`. H is a handle, printed by
--handle. A:B is a line range, 1-based, both ends included. A result is `file:line  name`;
a trailing `test` marks one that lives in a test. A sentence after an em dash is a
generated hint, not source.

SEARCH:
  search "WHAT IT DOES"             the definitions that do what you describe: "restrict a
                                    value to a range", "retry a failed request"
  search-symbol NAME                the definitions whose name contains NAME
  search-pattern REGEX [--fixed]    every place REGEX matches — comments, strings and config
                                    included — and the definition each match sits in;
                                    --fixed takes it as literal text

  --kind K          only function, method, class, struct, enum or trait results; a text match
                    counts by the definition it sits in
  --code            also print the source at each result
  --all             every result instead of the first few

NAVIGATE:
  where-am-i                        the repository at one glance: layout, languages, entry
                                    points, test roots, each module with its most used
                                    symbols
  who-calls S                       every place that uses S: calls, imports, type references
  callees S                         what S uses, and where those are defined
  brief S                           what S does in one sentence, its signature, then its body
                                    sketched: one line per step with the symbol used there and
                                    what happens
  impact S [--depth N] [--direction outgoing]
                                    how far a change to S reaches, as a tree of callers, each
                                    with what it does, tests marked; --direction outgoing
                                    walks what S reaches instead of who reaches S
  path --from A --to B              every call chain from A to B, as a tree of the call sites
                                    they hang on

  who-calls and callees answer for several symbols at once: `who-calls A B C`.

  --code            also print the source at each result
  --all             every result instead of the first few

READ:
  read S [S …]              the source code of S; --head M and --tail N for only its
                            first M and last N lines
  read-smart S [S …]        the source code of S, nested blocks below --depth N folded
                            into one-line semantic descriptions; default 1
  read-file PATH [PATH …]   the files; paginated at 400 lines unless --lines A:B or --all

  --handle          also print a handle naming exactly the span that was printed — if the
                    output was cut, it covers the shown part only; replace-span takes it

EDIT:
  replace S [NEW]            NEW replaces S's definition; --body: its body only
  replace-text F OLD [NEW]   NEW replaces OLD — refused unless OLD occurs exactly
                             once (--expect N: exactly N times; --regex: OLD is
                             a regular expression)
  replace-lines F A:B [NEW]  NEW replaces those lines
  replace-span H [NEW]       NEW replaces the span H names — refused if the file
                             changed since H was printed
  insert-lines F N [NEW]     NEW lands after line N; 0 puts it at the top
  delete S                   removes S's definition
  delete-lines F A:B         removes those lines
  patch [DIFF]               DIFF — a unified diff — lands as a whole: hunks
                             anchor on their context lines, the @@ numbers are
                             advisory; every file in it together, or nothing
  write PATH [NEW]           creates or overwrites the file
  rename S NAME              renames S and every reference to it, and reports
                             what it could not resolve
  undo [ID]                  reverses that edit — the last one when ID is absent;
                             refused if a later edit touched the same span

  NEW or DIFF absent: it is read from stdin.

  --dry-run                  reports what would change, and changes nothing
  --verify                   runs the build or linter for the touched files and
                             reports the diagnostics against symbols and spans

RUN:
  bash-smart [-e REGEX] -- CMD …   runs CMD unchanged, same exit code. Line 1 is
                             the verdict: `ok — exit 0` or `ok — exit 0,
                             3 warnings` or `FAILED — exit 101: 2 errors,
                             1 warning`. Then each error and warning block and
                             every REGEX match as `LINE  content`; the rest
                             compacted, any part of the original printable.

CHAIN — every command takes its input from the pipe, so one result goes straight into the next:
  greppy search-symbol NAME --json | greppy read -
  greppy callees S --json | greppy read -

ON EVERY COMMAND:
  --path P          only results under that file or directory
  --json            the same answer as data, with exact counts
  --limit N         at most N results; --offset K starts at the Kth
  --root DIR        work on a different repository; greppy finds the current one by itself
  --help            the syntax of that command, with a working example

A question with an empty answer — a symbol that exists but has no callers — says so and exits
0; the search commands use grep's codes, 0 for a hit and 1 for none, as does the grep-compatible
form. A question that cannot be answered — a symbol or file that does not exist, an ambiguous
name, a selector that matches nothing — says why and exits non-zero, and an edit in that
situation writes nothing.
