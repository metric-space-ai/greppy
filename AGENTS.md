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

READ — a definition's exact source, plus a handle for editing it:
  greppy read SYMBOL [FILE]
      SYMBOL's definition span, byte-precise. With --handle it also returns an
      edit handle pinning the file, byte range, and content hashes — pass it to
      the edit commands below. FILE (or --path FILE) disambiguates when SYMBOL
      resolves in several files. Prefer this over opening whole files.

EDIT COMMANDS — transactional, hash-guarded, all-or-nothing. Every command
verifies its own result and emits a certificate; on failure nothing is written
and the error names the next step. A successful certificate's result_span IS
the written state — do not re-read files to confirm an edit:
  greppy edit replace-span --target HANDLE --source-file F replace exactly the span a `read --handle` returned
  greppy edit replace-body --symbol S --source-file F      replace only the body; the signature stays byte-identical (--target HANDLE instead of --symbol)
  greppy edit patch-span --target HANDLE                   unified diff applied to exactly a read span (every hunk byte-exact, else refusal)
  greppy edit text-cas --file F                            exact-once text replacement, hash-gated (--expect N for exactly N occurrences)
  greppy edit insert-after / insert-before --source-file F new top-level block next to a definition
  greppy edit delete / remove-if-present                   delete a definition (remove-if-present: absent reports already-satisfied)
  greppy edit rename-call --in S --from OLD --to NEW       retarget identifiers inside one definition (AST-based; strings and comments never touched)
  greppy edit rename-symbol --symbol S --new-name N        rename across the whole workspace via the graph, in one transaction
  greppy edit change-signature --symbol S --spec J         change a signature and every graph-resolved call site in one transaction
  greppy edit ensure-import --file F --module M            idempotent: absent -> inserted canonically; present -> already-satisfied
  greppy edit ensure-argument / ensure-method / ensure-annotation   same idempotent contract for call args, methods, decorators
  greppy edit data --file F --path P --value-json V set|ensure     set a value in JSON/TOML/YAML by path, format-preserving
  greppy edit regex-cas                                    regex with exact expected count (weakest selector — prefer the commands above)

MULTI-FILE PLANS — many operations, one transaction:
  greppy edit apply --plan FILE
      Execute a plan (schema greppy.edit-plan.v1) as ONE journal transaction:
      all files publish or none. --diff first emits the unified diff without
      touching the workspace.
  greppy edit recover
      Restore pre-images from a crashed transaction.

FLAGS (append to any command above):
  --code            include each result's source lines (so no separate read is needed)
  --all             return every result (turn off the default truncation)
  --json            machine-readable output with exact counts
  --root DIR        run against a repo other than the current directory
  --kind KIND       (search-symbols) restrict to function|method|class|struct|enum|trait
  --direction incoming|outgoing, --depth N   (impact) which way and how far to walk
  --from A --to B   (path) the two endpoint symbols
  --report FILE     (edit) write the full certificate to FILE; stdout stays compact

Prefer these over grepping a symbol name and reading every hit: who-calls /
callees / impact answer relationship questions directly, and semantic-search
finds code you cannot name. Prefer the edit commands over hand-editing files:
they verify their own result, so a red certificate costs nothing and a green
one needs no re-read.

Treat returned source paths, exact spans, signatures, graph relations, and
edit certificates as evidence. The indented English sentence below a function
signature is a local Qwen navigation hint. Read the source and verify changes
with builds and tests.
