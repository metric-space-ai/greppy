#!/usr/bin/env python3
"""Audit v2 benchmark final diffs for agent edits to harness-supplied tests."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import shlex
import sys
from typing import Any, Sequence

import run_benchmark as bench

AUDIT_SCHEMA_VERSION = "greppy.agent-coding-v2-test-audit.v1"


def _diff_header_paths(line: str) -> tuple[str, str]:
    try:
        fields = shlex.split(line)
    except ValueError as error:
        raise bench.HarnessError("invalid diff header in recorded patch") from error
    if len(fields) != 4 or fields[:2] != ["diff", "--git"]:
        raise bench.HarnessError("invalid diff header in recorded patch")

    def normalize(value: str, prefix: str) -> str:
        return value[len(prefix):] if value.startswith(prefix) else value

    return normalize(fields[2], "a/"), normalize(fields[3], "b/")


def patch_sections(patch: str) -> dict[str, str]:
    """Map both sides of every diff to a canonical, index-independent section."""
    raw_sections: list[list[str]] = []
    current: list[str] | None = None
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            current = [line]
            raw_sections.append(current)
        elif current is not None:
            current.append(line)

    sections: dict[str, str] = {}
    for lines in raw_sections:
        old_path, new_path = _diff_header_paths(lines[0])
        canonical_lines = [
            line
            for line in lines[1:]
            if not line.startswith("index ")
        ]
        canonical = "\n".join(canonical_lines).rstrip("\n") + "\n"
        for path in (old_path, new_path):
            if path != "/dev/null":
                sections[path] = canonical
    return sections


def compare_test_patch(test_patch: str, final_patch: str) -> tuple[bool, list[str], list[str]]:
    touched = bench.patch_touched_paths(test_patch)
    expected = patch_sections(test_patch)
    actual = patch_sections(final_patch)
    modified = [path for path in touched if expected.get(path) != actual.get(path)]
    return bool(modified), modified, touched


def load_v2_tasks(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise bench.HarnessError(f"cannot read v2 task file: {error.__class__.__name__}") from error
    if document.get("schema_version") != bench.TASK_SCHEMA_VERSION_V2:
        raise bench.HarnessError("audit requires a v2 task document")
    tasks = document.get("tasks")
    if not isinstance(tasks, list):
        raise bench.HarnessError("v2 task document has no task list")
    return {task["id"]: task for task in tasks if isinstance(task, dict) and isinstance(task.get("id"), str)}


def load_result_rows(path: pathlib.Path) -> tuple[str | None, list[dict[str, Any]]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise bench.HarnessError(f"cannot read benchmark results: {error.__class__.__name__}") from error
    rows = document.get("results")
    if not isinstance(rows, list):
        raise bench.HarnessError("benchmark results contain no results array")
    return document.get("run_id"), [row for row in rows if isinstance(row, dict)]


def resolve_results_path(run_dir: pathlib.Path, explicit: pathlib.Path | None = None) -> pathlib.Path:
    candidates = [explicit] if explicit is not None else [
        run_dir / "results.json",
        run_dir / "results" / "results.json",
        run_dir / "MANIFEST.json",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise bench.HarnessError("results.json or MANIFEST.json not found")


def resolve_raw_dir(
    run_dir: pathlib.Path,
    run_id: str | None,
    explicit: pathlib.Path | None = None,
) -> pathlib.Path:
    candidates = [explicit] if explicit is not None else [
        run_dir / "raw",
        run_dir / "raw_traces",
        run_dir.parent / "raw" / (run_id or ""),
        bench.RAW_ROOT / (run_id or ""),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            return candidate.resolve()
    raise bench.HarnessError("raw diff directory not found; pass --raw-dir")


def audit_run(
    *,
    results_path: pathlib.Path,
    raw_dir: pathlib.Path,
    task_path: pathlib.Path,
) -> dict[str, Any]:
    tasks = load_v2_tasks(task_path)
    run_id, result_rows = load_result_rows(results_path)
    audit_rows: list[dict[str, Any]] = []
    for result in result_rows:
        task_id = result.get("task_id")
        arm = result.get("arm")
        if arm not in bench.ARMS or task_id not in tasks:
            continue
        final_path = raw_dir / str(task_id) / str(arm) / "final.patch"
        if not final_path.is_file():
            audit_rows.append(
                {
                    "task_id": task_id,
                    "arm": arm,
                    "gaming_suspected": True,
                    "modified_test_paths": [],
                    "touched_test_paths": bench.patch_touched_paths(tasks[task_id]["test_patch"]),
                    "final_diff_present": False,
                    "reason": "missing_final_diff",
                }
            )
            continue
        final_patch = final_path.read_text(encoding="utf-8", errors="replace")
        suspicious, modified, touched = compare_test_patch(tasks[task_id]["test_patch"], final_patch)
        audit_rows.append(
            {
                "task_id": task_id,
                "arm": arm,
                "gaming_suspected": suspicious,
                "modified_test_paths": modified,
                "touched_test_paths": touched,
                "final_diff_present": True,
                "reason": "test_path_differs_from_test_patch" if suspicious else "test_patch_preserved",
            }
        )
    suspicious_count = sum(row["gaming_suspected"] for row in audit_rows)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "results_path": str(results_path.resolve()),
        "raw_dir": str(raw_dir.resolve()),
        "task_path": str(task_path.resolve()),
        "summary": {
            "audited_runs": len(audit_rows),
            "gaming_suspected_runs": suspicious_count,
            "clean_runs": len(audit_rows) - suspicious_count,
        },
        "runs": audit_rows,
    }


def render_table(report: dict[str, Any]) -> str:
    headers = ("TASK", "ARM", "GAMING?", "MODIFIED TEST PATHS", "EVIDENCE")
    rows = [
        (
            row["task_id"],
            row["arm"],
            "yes" if row["gaming_suspected"] else "no",
            ",".join(row["modified_test_paths"]) or "-",
            row["reason"],
        )
        for row in report["runs"]
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(str(value)))

    def format_row(row: Sequence[str]) -> str:
        return "  ".join(str(value).ljust(widths[index]) for index, value in enumerate(row))

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([format_row(headers), separator, *(format_row(row) for row in rows)])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path, help="v2 run directory containing results")
    parser.add_argument("--results", type=pathlib.Path, help="explicit results.json or MANIFEST.json")
    parser.add_argument("--raw-dir", type=pathlib.Path, help="explicit raw task/arm diff directory")
    parser.add_argument("--tasks", type=pathlib.Path, default=bench.HERE / "tasks_v2.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    results_path = resolve_results_path(run_dir, args.results)
    run_id, _ = load_result_rows(results_path)
    raw_dir = resolve_raw_dir(run_dir, run_id, args.raw_dir)
    report = audit_run(results_path=results_path, raw_dir=raw_dir, task_path=args.tasks)
    print(render_table(report), file=sys.stderr)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if report["summary"]["gaming_suspected_runs"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except bench.HarnessError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
