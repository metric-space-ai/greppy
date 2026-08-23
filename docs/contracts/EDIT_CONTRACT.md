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
6. Certificates are compact on stdout: status, exit code, match/target
   counts, changed byte ranges, and the unified diff. Heavy evidence
   (before/after node text, per-postcondition detail, validator output) is
   written only to `--report FILE`. The certificate must stay cheaper than
   the re-read it replaces.

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
greppy edit rename-symbol   --symbol Q.SYM --new-name NAME [--backend graph|lsp] [--expect-residual N]
greppy edit change-signature --symbol Q.SYM --spec sig.json [--backend graph|lsp] [--expect-residual N]
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

Residual postcondition (binding, rename-symbol / change-signature): after
publish, the workspace occurrence count of the old name in same-language code
files must equal the declared residue (`--expect-residual N`, default 0);
mismatch is exit 13 and the count is reported in the certificate.

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

**Task bank (re-registered 2026-07-21, owner decision, before any measured
run on this bank):** `bench/agent_coding/tasks_v2.json`
(sha256 `8dd46a943f94fb0f…`) — 41 tasks derived from real commits of the six
pinned repositories: parent commit as the working state, the real commit's
test diff applied as the failing specification, the real issue/commit intent
as the task text, the code diff hidden. 18 serious multi-file tasks
(80–800 changed lines) plus 23 mechanical ones. The previous v1 bank
(30 single-file two-line mutation reverts) is **retired**: its tasks contain
no navigation or transaction surface, so it structurally cannot measure the
edit thesis — its gate-v4 run (2026-07-21, gpu3) is archived as evidence of
correctness parity (30/30/30) and the post-edit re-read rate (0.088), not of
cost. Thresholds below are UNCHANGED from the original registration.

Measured by `bench/agent_coding` on the paired task set, third arm
`greppy-edit` (harness-v4 registered treatment):

- `provider_cost_ratio` (greppy-edit / explorer, solved pairs): **≤ 0.80**
- `post_edit_source_opens_per_edit` (source opens of a file the same agent
  already edited in the same task): **≤ 0.1**
- exact-McNemar correctness parity vs explorer: hard gate, unchanged
- minimum sample: **30** complete pairs and **20** both-solved pairs
- diagnostics (not gate metrics): tool-call, source-open, input-token,
  navigation-arm provider-cost, and solved-pair wall-time ratios

The shipped `AGENTS.md` manual and each arm's treatment/full-system prompt are
pinned separately by hash in every run manifest. Harness v4 records explorer,
Greppy navigation, and Greppy-edit prompt identities; a resume refuses any
prompt, binary, model, task-bank, platform, or gate-contract mismatch.

Arm tool surface (re-registered 2026-08-23 before the next measured release
run): all three arms receive the identical explicit Pi palette
`bash,read,edit,write`. Greppy-edit's requirement to edit through Greppy is a
preregistered treatment, not a hidden capability restriction. The older
2026-07-17 bash-only arm definition is retired and its evidence is not combined
with harness-v4 evidence. `tools_per_arm` and every full prompt hash are part of
the manifest identity. Thresholds are unchanged.

Agent cutoff semantics (re-registered 2026-08-24 after harness-v3 demonstrated
that full fresh-session timeout replays could themselves reach the same task
budget): each task's fixed timeout is a compute cutoff, not an infrastructure
failure. The rule applies identically to explorer, Greppy, and Greppy-edit. If
Pi reaches the cutoff after at least one completed turn and without a
provider-reported error, the harness kills the process group and independently
tests the exact worktree snapshot left at the cutoff. That single attempt is a
valid measurement: a passing snapshot is correct and a failing snapshot is an
ordinary correctness loss. Its full tokens, tool calls, source opens, edits,
turns, and wall time remain charged; there is no timeout replay. A zero-turn
cutoff or provider-reported error remains invalid and fails closed.
Provider/harness infrastructure retries remain separately bounded and are not
measurements. Harness-v2/v3 results cannot resume under this contract.

## Verify (binding, v0.3.0)

```
greppy verify [--baseline REV] [--timeout SECONDS] [--json] [--no-cache] -- <test-command...>
```

`verify` runs the command in the current working tree first, then runs the same
argv against the committed `REV` (default `HEAD`) in a detached temporary Git
worktree. Greppy never stashes, checks out, writes the index, or otherwise
mutates the user's worktree to create the baseline. The report attests a digest
of index entries plus live bytes for every tracked path before and after both
runs. Temporary baseline worktrees are force-removed even after command failure
or timeout.

A baseline worktree may symlink an existing, Git-ignored `.tox`, `.venv`,
`venv`, `.nox`, `node_modules`, or `target` directory from the corresponding
repository-root or command-directory location. The exact relative mirror list
is printed and included in `greppy.verify-report.v1`. No dependency installer
is run. Baseline results are cached under the workspace's Greppy store with a
key derived from the resolved revision, exact command argv, and mirror list;
`--no-cache` bypasses that result.

Exit `0` means no newly failed test and no infrastructure error, `21` means at
least one `newly_failed` test, and `22` means an after/baseline command,
collection/build, timeout, environment, or worktree infrastructure error.
Infrastructure takes precedence over test-failure exit `21`. Supported v1
per-test parsers are pytest, `go test`, `cargo test`, and Jest/Vitest. Unknown
frameworks are explicitly reported as `framework: "unknown"` and compared at
command-exit level rather than being represented as an empty passing suite.
