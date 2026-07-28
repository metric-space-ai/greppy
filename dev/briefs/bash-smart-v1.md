Build bash-smart v1 — the training-free layers per dev/SMART-BASH-SPEC.md
(read it first; the classifier head is explicitly NOT in scope). Work in YOUR
CURRENT DIRECTORY — the launcher has put you in an isolated worktree. Do NOT
cd to any other checkout.

SURFACE:
  bash-smart -- CMD …    runs CMD untouched (argv after --, no shell unless
                         a single string is given), exit code passed through.

BEHAVIOUR (all in a NEW module crates/cli/src/bash_smart.rs):
1. Capture stdout+stderr streams separately, store the full raw output in the
   expand-pack store with a content hash. Under ~80 total lines: print
   everything verbatim, done.
2. Above: first 20 lines, then the collapse/lift block, then a gap line with
   the full continuation command (`… N lines — greppy expand ID continues at
   L`), then the last 30 lines. On exit != 0 the tail widens to 60. stderr is
   printed verbatim in full when it is ≤ 40 lines, else it gets the same
   regime.
3. Repetition collapse before anything else: consecutive identical lines and
   template-identical lines (equal after digits/hex/paths are masked) become
   one line plus `… N weitere \`…\`-Zeilen` — arithmetic only.
4. Novelty lift: embed the collapsed unique lines with the EXISTING embedder
   batch API (16er batches). A line whose vector is far from the wall's
   centroid (top-k by distance, k small, with a floor so uniform walls lift
   nothing) is lifted verbatim with its ORIGINAL line number. Daemon cold or
   embedder unavailable → skip silently, skeleton stands.
5. Byte gate: any lifted line is verified byte-identical at its claimed line
   in the stored raw output before display; mismatches are dropped.
6. Expand: opening the id pages the raw output 400 lines at a time, each page
   ending with the next continuation command. Relocate-or-refuse on hash
   drift, like every other pack.

ACCEPTANCE — run and paste real output:
  greppy bash-smart -- echo hi                          # verbatim, exit 0
  greppy bash-smart -- sh -c 'seq 1 500'                # skeleton + gap + tail
  greppy expand <id>                                    # lines 21-420, next id
  greppy bash-smart -- sh -c 'yes hello | head -300'    # collapse to one line + count
  greppy bash-smart -- sh -c 'for i in $(seq 200); do echo routine line $i; done; echo XYZZY-UNEXPECTED-994; for i in $(seq 200); do echo routine tail $i; done'
      # the odd line lifted with its number (or, without daemon: skeleton only —
      # state which case your run hit)
  greppy bash-smart -- false                            # exit 1 passed through
  cargo build --release && cargo test -p greppy --lib
  cargo test -p greppy --test prompt_contract           # stays green — AGENTS.md untouched
FILE WHITELIST: crates/cli/src/bash_smart.rs (new), cli_surface.rs, lib.rs
(dispatch + SUBCOMMANDS only), crates/cli/tests/bash_smart.rs (new).
FORBIDDEN: AGENTS.md, prompt_contract.rs, nav.rs, search.rs, read.rs, edit.rs.
Do not commit unless green. Commit message:
  feat(bash-smart): the training-free v1 — skeleton, collapse, novelty, byte gate
ESCAPE HATCH: need more scope? STOP and justify. NO SUBAGENTS.
REPORT TAIL: CHANGED / OUTPUT (verbatim) / TESTS / OPEN / COMMIT
