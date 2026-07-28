Rebuild the EDIT surface to the shape in AGENTS.md's EDIT section (already
final and guarded by prompt_contract). Work in YOUR CURRENT DIRECTORY — the
launcher has put you in an isolated worktree. Do NOT cd to any other checkout.

THE SURFACE (AGENTS.md is normative):
  replace S [NEW]              replace-text F OLD [NEW]     replace-lines F A:B [NEW]
  replace-span H [NEW]         insert-lines F N [NEW]       delete S
  delete-lines F A:B           patch [DIFF]                 write PATH [NEW]
  rename S NAME                undo [ID]
Footer flags --dry-run / --verify on every verb. NEW/DIFF absent → stdin;
stdin empty or a TTY → usage error (a forgotten payload must never become a
silent deletion). Positionals take allow_hyphen_values; `--` works.

COMMIT IN GREEN SLICES, in this order — a timeout must still land value:
 1. replace S / replace-text / replace-lines / replace-span / write, with
    receipts and refusals (below), positional signatures, stdin rule. The old
    machinery behind edit replace exists in crates/cli/src/edit.rs — this is a
    re-surface, not a rewrite.
 2. delete S / delete-lines / insert-lines (thin wrappers over the same
    pipeline; delete = empty replacement, insert = zero-width span after N).
 3. rename S NAME (positional re-surface of edit rename --symbol/--to),
    undo [ID] (the journal already holds transaction ids; bare = last).
 4. patch [DIFF]: hunks anchor on context lines, @@ numbers advisory, all
    files in the diff land atomically or nothing does (extend
    apply_unified_patch_exact; multi-file via the plan pipeline).
 5. The `edit` prefix and every dead verb (insert, delete under `edit`,
    move, remove, ensure-*, change-signature, data, apply, recover,
    --content, --content-file, --old-file, --pattern, --target) are removed;
    all dead names go into unknown_verb_refusal so none becomes a grep
    pattern. clap variants, SUBCOMMANDS, dispatch arms, usage table updated.

RECEIPTS AND REFUSALS (the certificate):
  applied edit-src/data.rs:682  e7f3a2      ← span + short transaction id, one line
  would apply edit-src/data.rs:682          ← every --dry-run says "would", never "applied"
    (edit rename --dry-run saying "applied" is a live bug — fix it in passing)
  OLD occurs 0 times — nothing written      ← refusals name the fact, never echo OLD
  refused: the edit would break the file's syntax — nothing written
  applied, already as sent  F:span          ← NEW identical to current content
An edit that lands is byte-exact CAS; no resulting text is echoed.

ACCEPTANCE — run and paste real output (sandbox: cp -r the sample repo first,
never edit /Volumes/tmp/outputs-repo in place):
  greppy replace skip_ws 'fn skip_ws(text: &str, mut pos: usize) -> Option<usize> { None }'
  greppy replace-text config.json '"port":8081' '"port":9090'
  greppy replace-text config.json 'zzz' 'x'        # occurs 0 times, exit != 0
  greppy replace-lines neu.txt 1:1 'x'
  greppy insert-lines neu.txt 0 'top'
  greppy delete-lines neu.txt 1:1
  greppy patch <<'EOF' … a real two-file diff … EOF
  greppy rename unquote_key_v2 unquote_key_v3 --dry-run   # "would", not "applied"
  greppy undo                                             # names what it reversed
  greppy edit replace --file x --old a --content b        # unknown subcommand
  cargo build --release && cargo test -p greppy --lib
  cargo test -p greppy --test prompt_contract             # must stay green
FILE WHITELIST: crates/cli/src/{edit.rs,cli_surface.rs,lib.rs,emit.rs},
crates/cli/tests/*.rs (not prompt_contract.rs), lib_tests.rs.
FORBIDDEN: AGENTS.md, prompt_contract.rs, nav.rs, search.rs, read.rs.
JSON: --json shapes of surviving operations stay frozen; dead verbs' JSON dies.
Commit message per slice:
  feat(edit): <slice> — the trained surface, receipts that never lie
ESCAPE HATCH: need more scope? STOP and justify in the report. NO SUBAGENTS.
REPORT TAIL: CHANGED / OUTPUT (verbatim) / TESTS / OPEN / COMMIT(s)
