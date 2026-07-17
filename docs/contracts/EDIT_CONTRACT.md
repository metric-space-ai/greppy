# greppy read & edit — CLI contract (v0.3.0)

This file is the registered contract for the 0.3.0 read/edit surface. The
JSON schemas next to it (`edit-plan.v1.schema.json`,
`edit-certificate.v1.schema.json`) are normative; this file binds the CLI to
them. Changes require the same re-registration discipline as benchmark
thresholds: documented rationale, version bump, owner sign-off.

## Principles (binding)

1. No fuzzy application of any kind, permanently. Selectors match exactly
   the declared cardinality or fail with the qualified candidate list.
2. Compare-and-swap end to end: plans bind file/target hashes and signature
   fingerprints; all hashes are re-verified immediately before publish.
3. The store addresses; the live file decides. A stale index can fail an
   edit (exit 12), never corrupt one.
4. Certificates instead of re-reads: every operation emits a
   `greppy.edit-certificate.v1` document; guarantee levels are named, never
   scored.
5. Failure is a next step: every non-zero exit carries machine-readable
   context (candidates, changed file, failing postcondition).

## Commands

### Read

```
greppy read SYMBOL [--handle]
```

(Contract revision 2026-07-17: `expand ID --handle` was dropped before
implementation — expand packs bundle several nodes, and a handle binds
exactly one span. `greppy read` is the single handle issuer; reading a hit
from a previous search is `read` on its qualified name.)

`--handle` returns an opaque versioned token binding
`{workspace_root, path, file_sha256, byte_range, target_sha256,
signature_fingerprint, grammar_id, grammar_version}` (base64url, prefix
`geh1:`). Handles are stateless; every component is re-verified at use.

### Edit verbs

```
greppy edit replace-body    --symbol Q.SYM | --target HANDLE   --source-file F
greppy edit replace-span    --target HANDLE                    --source-file F
greppy edit patch-span      --target HANDLE --patch-file F     (fuzz 0, hunks inside target)
greppy edit insert-after    --symbol Q.SYM | --target HANDLE   --source-file F
greppy edit insert-before   --symbol Q.SYM | --target HANDLE   --source-file F
greppy edit delete          --symbol Q.SYM | --target HANDLE
greppy edit rename-call     --in Q.SYM --from NAME --to NAME [--expect N]
greppy edit rename-symbol   --symbol Q.SYM --new-name NAME [--backend graph|lsp]
greppy edit change-signature --symbol Q.SYM --spec sig.json [--backend graph|lsp]
greppy edit ensure-import   --file PATH --module M [--name N]
greppy edit ensure-method   --symbol CLASS --spec method.json
greppy edit ensure-argument --symbol Q.SYM --call NAME --arg SPEC
greppy edit ensure-annotation --symbol Q.SYM --annotation A
greppy edit remove-if-present --symbol Q.SYM | --target HANDLE
greppy edit text-cas        --file PATH (--old S --new S | --old-file F --new-file F) [--expect 1]
greppy edit regex-cas       --file PATH --pattern RE --replacement S --expect N
greppy edit data set|ensure --file PATH --path JSONPATH --value-json V
greppy edit apply           --plan plan.json [--publish atomic|journal|patch|shadow-worktree]
greppy edit recover         [--workspace ROOT]      (journal crash recovery)
```

Common flags: `--json` (default when stdout is not a tty), `--report FILE`,
`--diff FILE`, `--dry-run`, `--at PATH:LINE` (symbol disambiguation),
`--expect N|exactly-one|zero`.

(Revision 2026-07-17, from K3 reasoning traces: `text-cas` accepts inline
`--old`/`--new` strings — agents reach for that form first and only then
create temp files — and every `--source-file` accepts `-` for stdin so
heredocs work. Pure surface addition; semantics, hashes, and exit codes
unchanged.)

## Exit codes (binding)

| Code | Meaning | Certificate `status` |
|---:|---|---|
| 0 | applied or already satisfied | `applied` / `already-satisfied` |
| 10 | target not found | `not-found` |
| 11 | target ambiguous (candidates listed) | `ambiguous` |
| 12 | plan/file hash stale | `stale` |
| 13 | syntax or postcondition failure | `invalid-result` |
| 14 | validator failed | `validation-failed` |
| 15 | concurrent modification detected | `stale` |
| 16 | publish / I-O failure | `publish-failed` |
| 17 | unsafe path or symlink situation | `publish-failed` |
| 20 | invalid edit specification | (report emitted if spec was readable) |

## Publish modes and their guarantees

| Mode | Guarantee |
|---|---|
| `atomic` | strict single-file atomicity: tmp + fsync + rename + dir fsync; preserves mode, encoding, line endings; rejects symlinks, path traversal, hardlink surprises; `ReplaceFileW` on Windows |
| `journal` | logical all-or-nothing across files: pre-image journal, apply-all-or-rollback, crash-recoverable via `greppy edit recover` |
| `patch` | no workspace mutation; unified diff + certificate only |
| `shadow-worktree` | apply + validate in an isolated worktree, then journal-publish into the real workspace after re-verifying all input hashes under a workspace lock |

## Formatter policy

`none` (default) / `selected-range` / `file` (explicit argv;
`permit_changes_outside_target` required to widen scope). A widened scope is
always reported (`formatter_expanded_change_scope: true`), never silent.

## Benchmark metrics registered with this contract (gate v4)

Measured by `bench/agent_coding` on the paired task set, third arm
`greppy-edit` (v3 prompt set):

- `provider_cost_ratio` (greppy-edit / explorer, solved pairs): **≤ 0.80**
- `post_edit_source_opens_per_edit` (source opens of a file the same agent
  already edited in the same task): **≤ 0.1**
- `partial_apply_incidents`: **0** (by construction; measured anyway)
- exact-McNemar correctness parity vs explorer: hard gate, unchanged
- diagnostics (not gate metrics): edit-verb usage share, failures by exit
  code, ambiguity rate, certificate count

Prompt set v3 (`bench/agent_coding/prompts/greppy_system_v3.md`,
`mscc_skill_v3.md`) is pinned by hash in every run manifest; v2 remains
pinned for all v0.2.x evidence.

Arm tool surface (revision 2026-07-17, before any measured gate-v4 run):
the `greppy-edit` arm runs pi with `--tools bash` — no builtin
read/edit/write. Rationale from trace forensics: the displacement prompt
("there is no apply_patch") is visibly false while a builtin `edit` tool
sits in the palette, and the agent then ignores greppy entirely (0 greppy
calls in the greppy-arm serde trace); the MSCC panel shows 78-87% greppy
adoption exactly where the displacement claim is true. Tool surfaces are
part of the arm definition, recorded per arm in the manifest
(`tools_per_arm`), and identical across both arms' *capabilities*: bash
can still read and write files, so the arm loses no ability, only the
contradiction. Explorer and greppy arms keep `bash,read,edit,write`.
Thresholds are unchanged.
