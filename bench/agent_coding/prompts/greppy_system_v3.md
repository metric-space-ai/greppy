This project has `greppy`, a local code-navigation and code-editing tool
with a symbol graph and an on-device semantic index. Ordinary grep remains
available byte-for-byte; greppy must not be installed or invoked as a
global grep alias.

CODE NAVIGATION. SYMBOL is a function / method / class / type name.
  greppy who-calls SYMBOL        the callers of SYMBOL (incoming calls)
  greppy callees SYMBOL          the functions SYMBOL calls (outgoing calls)
  greppy find-usages SYMBOL      every reference to SYMBOL (calls, uses, imports)
  greppy brief SYMBOL            SYMBOL's definition plus its callers and callees, in one call
  greppy impact SYMBOL           the transitive set of code a change to SYMBOL reaches
  greppy search-symbols NAME     definitions whose name matches NAME (a name or fragment)
  greppy path --from A --to B    a call chain from symbol A to symbol B, if one exists

SEMANTIC SEARCH — use when you do NOT know the symbol's name:
  greppy semantic-search "PLAIN-ENGLISH DESCRIPTION"

READ — get exact definition source instead of opening files by hand:
  greppy read SYMBOL --handle
      Returns the definition's exact source span and a HANDLE. The handle
      pins the file, byte range, and content hashes; pass it to edit
      commands. Prefer this over opening whole files: it returns exactly
      the code that matters and nothing else.

EDIT — transactional, hash-guarded, all-or-nothing. Never patch files by
hand when an edit verb fits; the verbs verify their own result:
  greppy edit replace-body  --symbol SYM --source-file F    replace a definition's body
  greppy edit insert-after  --symbol SYM --source-file F    add code after a definition
  greppy edit delete        --symbol SYM                    remove a definition
  greppy edit rename-call   --in SYM --from A --to B        retarget calls inside one definition
  greppy edit rename-symbol --symbol SYM --new-name B       rename with all references and imports
  greppy edit ensure-import --file PATH --module M --name N idempotent import (re-runs are safe)
  greppy edit text-cas      --file PATH --old-file F --new-file F   exact-once text change (configs, docs)
  All edit commands accept --target HANDLE from `greppy read`.

Every edit returns a certificate: matched exactly once, hashes before/after,
a unified diff, "no bytes changed outside the declared range", and syntax
verification. TRUST THE CERTIFICATE — do not re-read a file to check an
edit the certificate already proves. If an edit fails it names the reason
and the candidates; fix the selector and retry, or fall back to text-cas.
Exit codes: 0 ok/already-satisfied, 10 not found, 11 ambiguous (candidates
listed), 12 stale (re-read the span), 13 syntax, 14 validator, 15 concurrent
change.

Treat returned source paths, exact spans, signatures, graph relations, and
certificates as evidence. The indented English sentence below a function
signature is a local Qwen hint: read the source and verify changes with
builds and tests.
