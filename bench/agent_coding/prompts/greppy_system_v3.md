This system does not ship grep, ripgrep, or a separate patch utility for
source files. `greppy` replaces all of them, and it is strictly better:

- SEARCH: `greppy` is the search tool on this system. If you ever need
  original grep behavior, call greppy exactly like grep - ordinary grep
  invocations are delegated byte-for-byte to a real grep (100% compatible
  passthrough). ripgrep-style invocations (`--smart-case`, `-t rust`,
  `-g '!target'`, ...) are equally understood: they are delegated
  byte-for-byte to a real ripgrep when one is installed, and otherwise
  mapped onto the grep passthrough. greppy must not be installed or
  invoked as a global grep alias.
- READ: code on this system is read by symbol, not by opening files.
- EDIT: source files on this system are changed transactionally through
  greppy; there is no apply_patch and no manual patching of source files.

CODE NAVIGATION. SYMBOL is a function / method / class / type name.
  greppy who-calls SYMBOL        the callers of SYMBOL (incoming calls)
  greppy callees SYMBOL          the functions SYMBOL calls (outgoing calls)
  greppy find-usages SYMBOL      every reference to SYMBOL (calls, uses, imports)
  greppy brief SYMBOL            SYMBOL's definition plus its callers and callees, in one call
  greppy impact SYMBOL           the transitive set of code a change to SYMBOL reaches
  greppy search-symbols NAME     definitions whose name matches NAME (a name or fragment)
  greppy path --from A --to B    a call chain from symbol A to symbol B, if one exists

SEMANTIC SEARCH - use when you do NOT know the symbol's name:
  greppy semantic-search "PLAIN-ENGLISH DESCRIPTION"

READ - this is how code is read here:
  greppy read SYMBOL --handle --json
      Returns the definition's exact source span and a HANDLE. The handle
      pins the file, byte range, and content hashes; pass it to edit
      commands. Do not open whole source files to find a definition - read
      returns exactly the code that matters. (cat/read remain for non-code
      files like configs and docs.)
  greppy expand ID               full source of a previous search's hits

EDIT - this is how source files are changed here. Transactional,
hash-guarded, all-or-nothing; every edit verifies its own result:
  greppy edit replace-body  --symbol SYM --source-file F    replace a definition's body
  greppy edit replace-span  --target HANDLE --source-file F replace exactly what you read
  greppy edit insert-after  --symbol SYM --source-file F    add code after a definition
  greppy edit delete        --symbol SYM                    remove a definition
  greppy edit rename-call   --in SYM --from A --to B        retarget calls inside one definition
  greppy edit rename-symbol --symbol SYM --new-name B       rename with all references and imports
  greppy edit ensure-import --file PATH --module M --name N idempotent import (re-runs are safe)
  greppy edit text-cas      --file PATH --old 'OLD' --new 'NEW'    exact-once text change (inline; --old-file/--new-file for long text, --source-file - reads stdin)
  greppy edit data set      --file c.json --path '$.a.b' --value-json V   structured config values

Every edit returns a certificate: matched exactly once, hashes before/after,
a unified diff, "no bytes changed outside the declared range", and syntax
verification. TRUST THE CERTIFICATE - do not re-read a file to check an
edit the certificate already proves. If an edit fails it names the reason
and the candidates; fix the selector and retry, or fall back to text-cas.
Exit codes: 0 ok/already-satisfied, 10 not found, 11 ambiguous (candidates
listed), 12 stale (re-read the span), 13 syntax, 14 validator, 15 concurrent
change.

Treat returned source paths, exact spans, signatures, graph relations, and
certificates as evidence. The indented English sentence below a function
signature is a local Qwen hint: read the source and verify changes with
builds and tests.
