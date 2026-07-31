# Coding bench: three defects that make it invalid as 0.3.0 evidence

`bench/agent_coding/run_benchmark.py` measures whether an agent can edit a
pinned repository so an independent test passes. It is the ONLY harness that
can prove the 0.3.0 READ/EDIT surface. Three defects currently invalidate it.
Repair them; do NOT rebuild the harness or touch the task bank — the 41 tasks
(pinned commits, mutations, test patches, setup commands) are validated
substance.

Work in YOUR CURRENT DIRECTORY — the launcher has put you in an isolated
worktree of the repo. Do NOT cd to any other checkout.

## Slice 1 — edit detection is blind to every 0.3.0 verb

```python
if re.search(r"greppy[^\n]*\bedit\b", command):
    edit_calls += 1
    m = re.search(r"--file\s+([\w./-]+)", command)
```

0.3.0's verbs are `replace`, `replace-text`, `replace-lines`, `replace-span`,
`insert-lines`, `delete`, `delete-lines`, `patch`, `write`, `rename`, `undo`
(see AGENTS.md, section EDIT) — none contains "edit", so `edit_calls` stays 0.
The file is a POSITIONAL argument, not `--file`, so `edited_files` stays empty
and post-edit re-reads are never attributed.

Worse, the gate divides by that zero:

```python
reread_rate = post_edit_rereads / edit_calls_total if edit_calls_total else 0.0
reread_pass = reread_rate <= POST_EDIT_REREADS_MAX
```

With no observed edit the rate is 0.0 and the gate passes VACUOUSLY.

FIX: detect the real verb set (anchored so `greppy read` is not mistaken for
an edit); extract the touched file from the verb's actual argument shape per
AGENTS.md; and make the re-read check FAIL, not pass, when the greppy-edit arm
produced zero observed greppy edits — a gate that passes without observation
is worse than no gate. State that reason in the gate output.

## Slice 2 — the arms do not get the same tools

```python
ARM_TOOLS = {
    "explorer": "bash,read,edit,write",
    "greppy":   "bash,read,edit,write",
    "greppy-edit": "bash",
}
```

The cost comparison then measures the palette cut as much as it measures
greppy. FIX: every arm gets the SAME palette (`bash,read,edit,write`); the
only intended difference between arms stays the system prompt, exactly as the
harness README claims ("The only intended prompt delta is the preregistered
navigation treatment"). Record the palette in the manifest as it already does,
and note the change where the harness documents its contract.

## Slice 3 — the solution is reachable from inside the workspace

`clone_pinned_repository` runs `git clone --mirror` (full upstream history,
all refs) and the agent's workspace is a `git worktree` on that mirror. The
agent has `bash`, so `git log --all -p` finds the real fix. The task id makes
it trivial: `expected_task_id()` is `{repo}-{type}-{commit[:12]}`, i.e. the
FIX commit's own prefix.

FIX: the agent's workspace must contain exactly ONE commit — the pinned tree
plus the mutation — and no upstream refs, reflog, or remotes. Suggested shape
(adapt if you find a cleaner one): materialize the pinned tree into a fresh
directory (`git archive` from the mirror, or worktree then detach), then
`git init` + one commit there; keep the harness's own mirror for its diffing
and verification OUTSIDE the agent's view. Setup commands and tests must still
run, so a working `.git` with one commit is preferred over deleting `.git`.
Also ensure the task id is never written into the agent's workspace or prompt.

Prove it in the acceptance: from inside a prepared agent workspace,
`git log --all --oneline | wc -l` is 1, and `git log --all` does not contain
the fix commit prefix from the task id.

## Acceptance — run these and paste REAL output
- `python3 -m py_compile bench/agent_coding/run_benchmark.py`
- `python3 bench/agent_coding/test_benchmark.py` (or the file's own runner) — green.
- A unit-level demonstration for slice 1: feed the parser a synthetic pi turn
  containing `greppy replace-text src/a.rs 'OLD' 'NEW' --root .` and show
  `edit_calls == 1` and the file attributed; and a case with zero edits showing
  the gate FAILS with its stated reason.
- Slice 3 proof as described above, on one real task
  (`--tasks bench/agent_coding/tasks_v2.json --task zod-cross-cutting-change-32ae1cd86c1b --validate-only`
  is cheap; if a full prepare is needed, use `--arms greppy-edit`).

## FILE WHITELIST
ONLY `bench/agent_coding/run_benchmark.py`, `bench/agent_coding/test_benchmark.py`,
and `bench/agent_coding/README.md` (contract note for slice 2).
FORBIDDEN: the task banks (`tasks_v2.json`, `tasks_hard.json`, `*.jsonl`),
`bench/agent_efficiency/**`, `AGENTS.md`, `crates/**`.

## Hard rules
- Do not commit unless acceptance is green. Commit message per slice:
  `fix(coding-bench): <slice>`
- ESCAPE HATCH: if you believe you need scope beyond the whitelist, STOP and
  justify it in the report. Never widen on your own.
- NO SUBAGENTS.

## REPORT TAIL (fixed form, at the very end)
CHANGED: <files>
OUTPUT: <the acceptance commands' real output, verbatim>
TESTS: <suite result lines>
OPEN: <what you could not do, and why>
COMMIT: <sha(s) or "not committed">
