This project has `greppy`, a local code-navigation and code-editing tool over a
symbol graph and an on-device semantic index. Ordinary grep invocations are
delegated byte-for-byte to the real system grep, but Greppy must not be
installed or invoked as a global grep alias.

CODE-NAVIGATION COMMANDS. SYMBOL is a function / method / class / type name.
They return resolved results as `qualified_name file:line`, not text matches:
  greppy who-calls SYMBOL        the callers of SYMBOL (incoming calls)
  greppy callees SYMBOL          the functions SYMBOL calls (outgoing calls)
  greppy find-usages SYMBOL      every reference to SYMBOL (calls, uses, imports)
  greppy brief SYMBOL            SYMBOL's definition plus its callers and callees, in one call
  greppy impact SYMBOL           the transitive set of code a change to SYMBOL reaches
  greppy search-symbols NAME     definitions whose name matches NAME (a name or fragment)
  greppy path --from A --to B    a call chain from symbol A to symbol B, if one exists

SEMANTIC SEARCH — use when you do NOT know the symbol's name:
  greppy semantic-search "PLAIN-ENGLISH DESCRIPTION"
      Describe the behaviour or code you are looking for in plain English
      (e.g. "restrict a value to a range", "retry a failed HTTP request").
      Returns the closest-matching definitions by meaning (signature + file:line).
      While first-use embeddings are still building, returns a retryable status
      with the active backend, progress, and ETA instead of partial/empty hits.

EXPAND — get the full source in one call instead of opening files by hand:
  greppy expand ID
      who-calls / callees / impact / semantic-search may end their output with a
      line `Expand: greppy expand <id>`. Run it to print the prepared evidence
      pack — the full source of the top matches, bundled — in a single call,
      instead of reading each file:line yourself.

READ — exact source plus a handle for editing it:
  greppy read SYMBOL [FILE]
      SYMBOL's definition span, byte-precise. With --handle it also returns an
      edit handle pinning the file, byte range, and content hashes — pass it to
      the edit commands below. FILE (or --path FILE) disambiguates when SYMBOL
      resolves in several files. Prefer this over opening whole files.
  greppy read PATH [--lines A:B]
      An existing file path reads the FILE (numbered lines; --lines for a
      range). With --handle the range becomes an edit handle too — so
      replace-span / patch-span work on file regions, not only on symbols.

ORIENT — the project at a glance, the working tree by meaning:
  greppy map [PATH]
      One screen of orientation: languages with index coverage, top-level
      modules, test roots, build/test commands, vendored/generated dirs,
      largest subtrees. Use this instead of ls/find exploration.
  greppy changes [--base REV]
      The current diff grouped by SYMBOLS: changed/new/deleted definitions,
      signature changes, their direct callers, and affected tests — split
      strictly into known_impacted and unknown_or_unindexed.

VERIFY — baseline-vs-after test comparison without touching your worktree:
  greppy verify [--baseline REV] -- COMMAND
      Runs COMMAND in the current tree AND against REV (default HEAD) in an
      isolated temporary worktree — never stash, never checkout. Classifies
      test cases: newly_failed / fixed / preexisting_failed /
      infrastructure_error. Exit 0 = nothing newly broken; 21 = newly_failed;
      22 = infrastructure error. Use this instead of stashing to check
      whether a failure is yours or preexisting.

EDIT — transactional, hash-guarded, all-or-nothing. Every edit verifies its own
result and emits a certificate; on failure nothing is written and the error names
the next step. A successful certificate's result_span IS the written state — never
re-read a file to confirm an edit.

The workflow is read → edit: `greppy read SYM --handle` pins the exact span and its
content hashes; pass that handle (or just --symbol) to an edit verb. Five cover
almost everything:
  greppy edit replace-body --symbol S --content-file F   replace a definition's body with the code in F
  greppy edit text-cas --file F --old '…' --new '…'      exact text replacement; add --expect N when it occurs N times
  greppy edit apply --plan P                             many edits as ONE atomic transaction; a plan is just
                                                         {"operations":[{"file":"a.rs","old":"x","new":"y"}]}
  greppy edit rename-symbol --symbol S --new-name N       rename S and every reference across the workspace at once
  greppy edit change-signature --symbol S --spec '{…}'    change a signature and every call site in one transaction

When a change spans many call sites, prefer rename-symbol / change-signature — one
transaction beats many text edits. `greppy edit --help` lists every verb and
`greppy edit VERB --help` prints a working example. `greppy edit recover` restores a
crashed transaction.

FLAGS (append to any command above):
  --code            include each result's source lines (so no separate read is needed)
  --all             return every result (turn off the default truncation)
  --json            machine-readable output with exact counts
  --root DIR        run against a repo other than the current directory
  --kind KIND       (search-symbols) restrict to function|method|class|struct|enum|trait
  --direction incoming|outgoing, --depth N   (impact) which way and how far to walk
  --from A --to B   (path) the two endpoint symbols
  --report FILE     (edit) write the full certificate to FILE; stdout stays compact
  --limit N         cap the number of results (alias --max)
  --max-bytes N, --offset K   budget the output; truncation prints total,
                    shown, and the exact continuation command — never pipe
                    greppy through head/tail, budgeting keeps JSON valid

Prefer these over grepping a symbol name and reading every hit: who-calls /
callees / impact answer relationship questions directly, and semantic-search
finds code you cannot name. Prefer the edit commands over hand-editing files:
they verify their own result, so a red certificate costs nothing and a green
one needs no re-read.

Treat returned source paths, exact spans, signatures, graph relations, and
edit certificates as evidence. The indented English sentence below a function
signature is a local Qwen navigation hint. Read the source and verify changes
with builds and tests.
