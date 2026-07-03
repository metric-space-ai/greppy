#!/usr/bin/env python3
"""Validate R7 benchmark regression/router classes.

The class file is part of the acceptance contract: every synthetic LLM-100 task
must belong to exactly one primary router class so forensics can report wins and
regressions by the known failure modes, not only by aggregate token factors.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_TASKS = HERE / "tasks.json"
DEFAULT_CLASSES = HERE / "task_classes.json"

REQUIRED_CLASSES = {
    "direct_similarity": {"role": "embedding_candidate", "min_count": 19},
    "hybrid_seed_graph": {"role": "embedding_candidate", "min_count": 5},
    "literal_control": {"role": "avoid_embedding", "min_count": 9},
    "graph_control": {"role": "avoid_embedding", "min_count": 20},
}

KNOWN_BAD_DIRECT_SIMILARITY = {
    "t006",
    "t007",
    "t030",
    "t033",
    "t039",
    "t043",
    "t052",
    "t058",
    "t059",
    "t092",
    "t093",
    "t095",
    "t098",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", type=pathlib.Path, default=DEFAULT_TASKS)
    ap.add_argument("--classes", type=pathlib.Path, default=DEFAULT_CLASSES)
    args = ap.parse_args()

    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    class_doc = json.loads(args.classes.read_text(encoding="utf-8"))
    errors = validate(tasks, class_doc)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 2

    classes = class_doc["classes"]
    print(f"task classes ok: {len(tasks)} tasks, {len(classes)} classes")
    for name in sorted(classes):
        cls = classes[name]
        print(f"  {name}: {len(cls.get('ids', []))} tasks, role={cls.get('role')}")
    return 0


def validate(tasks: list[dict[str, Any]], class_doc: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if class_doc.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    classes = class_doc.get("classes")
    if not isinstance(classes, dict):
        return errors + ["classes must be an object"]

    task_ids = {str(t.get("id")) for t in tasks}
    seen: dict[str, str] = {}
    for name, req in REQUIRED_CLASSES.items():
        cls = classes.get(name)
        if not isinstance(cls, dict):
            errors.append(f"missing required class {name}")
            continue
        role = cls.get("role")
        if role != req["role"]:
            errors.append(f"class {name} role must be {req['role']!r}, got {role!r}")
        ids = cls.get("ids")
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            errors.append(f"class {name} ids must be a string list")
            continue
        if len(ids) < int(req["min_count"]):
            errors.append(f"class {name} has {len(ids)} ids, expected >= {req['min_count']}")
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        for tid in duplicates:
            errors.append(f"class {name} contains duplicate task id {tid}")
        for tid in ids:
            if tid not in task_ids:
                errors.append(f"class {name} references unknown task id {tid}")
            owner = seen.get(tid)
            if owner is not None:
                errors.append(f"task {tid} appears in both {owner} and {name}")
            seen[tid] = name

    missing = sorted(task_ids - set(seen))
    extra = sorted(set(seen) - task_ids)
    if missing:
        errors.append("tasks missing from task_classes.json: " + ", ".join(missing))
    if extra:
        errors.append("unknown task ids in task_classes.json: " + ", ".join(extra))

    direct = set(classes.get("direct_similarity", {}).get("ids", []))
    hard = set(classes.get("direct_similarity", {}).get("hard_negative_ids", []))
    if not hard:
        errors.append("direct_similarity.hard_negative_ids must be non-empty")
    if not hard <= direct:
        errors.append(
            "direct_similarity.hard_negative_ids must be a subset of direct_similarity.ids"
        )
    missing_known_bad = sorted(KNOWN_BAD_DIRECT_SIMILARITY - hard)
    if missing_known_bad:
        errors.append(
            "direct_similarity.hard_negative_ids missing known bad tasks: "
            + ", ".join(missing_known_bad)
        )

    return errors


if __name__ == "__main__":
    raise SystemExit(main())
