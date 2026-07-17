# v0.3.0 (draft — numbers land with the measured gate-v4 run)

greppy extends from navigation to the full agent loop: **read exactly the
context you resolved, change exactly the context you read, trust the
certificate instead of re-reading.**

## Read

- `greppy read SYMBOL [--handle]` returns the definition's byte-exact source
  span from the live file. With `--handle` it returns a stateless edit
  handle pinning file hash, byte range, target hash, structural signature
  fingerprint, and grammar identity — the CAS ticket for every edit verb.
  Ambiguous symbols return the qualified candidate list (exit 11); a stale
  index refuses instead of pinning a shifted span.

## Edit — transactional, hash-guarded, all-or-nothing

Every verb runs one pipeline: live parse → CAS preconditions → in-memory
apply against a single snapshot (overlaps rejected, high-to-low offsets) →
reparse (no new ERROR/MISSING nodes) → byte-isolation proof → optional
formatter policy → atomic publish → incremental store refresh → a
machine-checkable certificate (`greppy.edit-certificate.v1`) with named
guarantee levels. No fuzzy application exists anywhere, permanently.

Verbs: `replace-body`, `replace-span`, `patch-span` (unified diff, fuzz 0),
`insert-after/-before`, `delete`, `rename-call`, `rename-symbol` (graph-
backed, cross-file, one journal transaction), `change-signature` (parameters
node replacement + call-site review checklist), `ensure-import`,
`ensure-method`, `ensure-argument`, `ensure-annotation`, `remove-if-present`
(all `ensure-*` idempotent: re-runs report `already-satisfied`), `text-cas`,
`regex-cas` (marked weakest selector), `data set/ensure` (JSON span-
tokenizer, TOML via lossless document model, YAML scalar paths).

Multi-file: `greppy edit apply --plan` executes `greppy.edit-plan.v1`
documents with four publish modes — `atomic` (single file), `journal`
(all-or-nothing with pre-image journal + `greppy edit recover` for crashed
transactions), `patch` (diff only), `shadow-worktree` (validate in an
isolated copy, publish only when every argv validator passes).

Exit codes are a registered contract (0/10/11/12/13/14/15/16/17/20); every
failure carries the next step (candidates, changed file, failing
postcondition).

## Registered gate (v4, measured before release)

- billed-cost ratio greppy-edit/explorer on solved pairs: **≤ 0.80**
- post-edit re-reads: **≤ 0.1 per edit**
- partially applied edits: **0 by construction**
- exact-McNemar correctness parity: unchanged hard gate

## Prompts

The v3 prompt set (`greppy_system_v3`, `mscc_skill_v3`) teaches the loop
resolve → read span → edit transactionally → trust the certificate; v2
remains pinned for all v0.2.x evidence.
