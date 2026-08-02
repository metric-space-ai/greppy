#!/usr/bin/env python3
"""Fail if a baseline arm invoked greppy.

The comparison only means something if the baselines genuinely lack the tool
under test. On 2026-07-31 they did not: `_greppy_on_path_env()` documented
"only the greppy arm gets it" while the call site passed the same environment
to every arm, so the uncoached `explorer` baseline invoked greppy in 88 of 115
tasks (461 verb calls) and the restricted `grep` arm in 41. The contrast was
greppy against a partly-greppy baseline, which understates the difference.

That was invisible in the aggregate row — it only shows in the executed
commands. This checker reads them and refuses a run that repeats it.

    python3 assert_arm_isolation.py <raw-dir> [--allow greppy,plus]

Exit 0 when every non-allowed arm is clean, 1 otherwise.
"""
import json
import pathlib
import re
import sys

GREPPY_CALL = re.compile(r"(?:^|[\s;|&(])greppy\b")


def executed_commands(trace: pathlib.Path):
    for line in trace.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") != "tool_execution_start":
            continue
        args = event.get("args") or {}
        command = args.get("command")
        if isinstance(command, str):
            yield command


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    raw_dir = pathlib.Path(sys.argv[1])
    allowed = {"greppy", "plus"}
    if "--allow" in sys.argv:
        allowed = set(sys.argv[sys.argv.index("--allow") + 1].split(","))

    offenders: dict[str, list[tuple[str, str]]] = {}
    checked = 0
    for task_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for trace in sorted(task_dir.glob("*.jsonl")):
            arm = trace.stem
            if arm in allowed:
                continue
            checked += 1
            for command in executed_commands(trace):
                if GREPPY_CALL.search(command):
                    offenders.setdefault(arm, []).append(
                        (task_dir.name, command.strip()[:90]))

    if not offenders:
        print(f"arm isolation ok: {checked} baseline traces, no greppy invocation")
        return 0

    print("ARM ISOLATION VIOLATED — the baseline had the tool under test",
          file=sys.stderr)
    for arm, hits in sorted(offenders.items()):
        tasks = {task for task, _ in hits}
        print(f"  {arm}: {len(hits)} greppy calls across {len(tasks)} tasks",
              file=sys.stderr)
        for task, command in hits[:5]:
            print(f"    {task}: {command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
