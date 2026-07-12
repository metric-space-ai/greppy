#!/usr/bin/env python3
"""Attach conservative answer-quality grades to benchmark results.

This is not a semantic judge. It checks an agent's final answer against the
mechanically known hints from tasks.json.

Modes:

* smoke: loose triage; never accepted for speed claims unless explicitly forced.
* mechanical: stricter symbol/file/count/path checks; can be accepted when
  --accept-mechanical is supplied. This still remains a benchmark-specific
  mechanical gate, not a general proof of semantic correctness.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results.json"
DEFAULT_TASKS = HERE / "tasks.json"


AGENT_FIELDS = {"grep", "greppy", "explorer", "plus", "gemma"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=pathlib.Path, default=DEFAULT_RESULTS)
    ap.add_argument("--tasks", type=pathlib.Path, default=DEFAULT_TASKS)
    ap.add_argument("--output", type=pathlib.Path)
    ap.add_argument("--agents", default="grep,greppy,explorer,plus,gemma")
    ap.add_argument("--mode", choices=("smoke", "mechanical"), default="smoke")
    ap.add_argument(
        "--accept-smoke",
        action="store_true",
        help=(
            "mark smoke-pass grades as accepted_for_speed_claim. Use only for "
            "controlled experiments; default is intentionally false."
        ),
    )
    ap.add_argument(
        "--accept-mechanical",
        action="store_true",
        help=(
            "mark strict mechanical-pass grades as accepted_for_speed_claim. "
            "This is intended for the synthetic 100-task bench where tasks.json "
            "is the ground-truth contract."
        ),
    )
    args = ap.parse_args()

    wanted_agents = {a.strip() for a in args.agents.split(",") if a.strip()}
    tasks = {t["id"]: t for t in json.loads(args.tasks.read_text(encoding="utf-8"))}
    rows = json.loads(args.results.read_text(encoding="utf-8"))
    graded = 0
    for row in rows:
        task = tasks.get(row.get("id"))
        if not task:
            continue
        for agent in sorted(wanted_agents & AGENT_FIELDS):
            run = row.get(agent)
            if not isinstance(run, dict) or "answer" not in run:
                continue
            run["quality"] = grade_answer(
                task,
                run.get("answer", ""),
                mode=args.mode,
                accept_smoke=args.accept_smoke,
                accept_mechanical=args.accept_mechanical,
            )
            graded += 1

    out = args.output or args.results
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"graded {graded} answers -> {out}")


def grade_answer(
    task: dict[str, Any],
    answer: str,
    mode: str,
    accept_smoke: bool,
    accept_mechanical: bool,
) -> dict[str, Any]:
    chk = task.get("check", {})
    if mode == "mechanical":
        return grade_answer_mechanical(task, answer, accept_mechanical)
    required = expected_terms(chk)
    found = []
    missing = []
    answer_norm = canonical(answer)
    for term in required:
        if canonical(term) in answer_norm:
            found.append(term)
        else:
            missing.append(term)
    forbidden_found = find_forbidden_terms(chk, answer_norm)

    count_status = count_check(chk, answer)
    required_total = len(required) + (1 if count_status == "missing" else 0)
    found_total = len(found) + (1 if count_status == "ok" else 0)
    score = 1.0 if required_total == 0 else found_total / required_total
    verdict = (
        "pass"
        if not missing and count_status != "missing" and not forbidden_found
        else "partial"
    )
    if found_total == 0 and required_total > 0:
        verdict = "fail"
    if forbidden_found:
        verdict = "fail"

    accepted = bool(accept_smoke and verdict == "pass")
    return {
        "grader": "ground_truth_smoke_v1",
        "accepted_for_speed_claim": accepted,
        "score": round(score, 4),
        "verdict": verdict,
        "found": found,
        "missing": missing,
        "forbidden_found": forbidden_found,
        "count_status": count_status,
        "note": (
            "Smoke grade only. It checks expected symbols/files/count hints in "
            "the final answer and is not a full semantic correctness proof."
        ),
    }


def grade_answer_mechanical(
    task: dict[str, Any], answer: str, accept_mechanical: bool
) -> dict[str, Any]:
    chk = task.get("check", {})
    kind = chk.get("kind")
    answer_norm = canonical(answer)

    # SWE-QA-style floor: the gold answer names many identifiers/files, but a
    # correct agent answer need only hit a FLOOR of them (min_hits), not all.
    # Terms are rg-verified against the pinned commit at generation time, so
    # each is a real, checkable anchor. This keeps grading mechanical on
    # genuinely open-ended "how/why" questions.
    if kind == "floor_terms":
        terms = [t for t in chk.get("terms", []) if t]
        min_hits = int(chk.get("min_hits", 1))
        found = [t for t in terms if canonical(t) in answer_norm]
        forbidden_found = find_forbidden_terms(chk, answer_norm)
        hits = len(found)
        if forbidden_found:
            verdict = "fail"
        elif hits >= min_hits:
            verdict = "pass"
        elif hits > 0:
            verdict = "partial"
        else:
            verdict = "fail"
        return {
            "grader": "ground_truth_mechanical_v1",
            "accepted_for_speed_claim": bool(accept_mechanical and verdict == "pass"),
            "score": round(hits / max(min_hits, 1), 4),
            "verdict": verdict,
            "found": found,
            "missing": [t for t in terms if t not in found],
            "forbidden_found": forbidden_found,
            "count_status": "not_applicable",
            "extra_failures": (
                [f"forbidden term present: {term}" for term in forbidden_found]
                + ([] if hits >= min_hits else [f"hit {hits}/{min_hits} required floor terms"])
            ),
            "note": (
                "SWE-QA floor grade: answer must name at least min_hits of the "
                "rg-verified gold identifiers/files."
            ),
        }

    required = expected_terms(chk)
    found = []
    missing = []
    for term in required:
        if canonical(term) in answer_norm:
            found.append(term)
        else:
            missing.append(term)

    extra_failures: list[str] = []
    forbidden_found = find_forbidden_terms(chk, answer_norm)
    extra_failures.extend(
        f"forbidden term present: {term}" for term in forbidden_found
    )
    count_status = count_check_mechanical(chk, answer, found)
    count_required = chk.get("min_count") is not None
    count_passed = count_status in {"ok", "inferred_from_members", "not_applicable"}
    if count_status == "missing":
        extra_failures.append(f"missing count >= {chk.get('min_count')}")

    path_order_required = kind == "path"
    path_order_passed = True
    if path_order_required:
        path_order_passed = ordered_path_present(chk, answer_norm)
        if not path_order_passed:
            extra_failures.append("path endpoints not in source-to-target order")

    required_total = len(required) + (1 if count_required else 0) + (
        1 if path_order_required else 0
    )
    found_total = len(found)
    if count_required and count_passed:
        found_total += 1
    if path_order_required and path_order_passed:
        found_total += 1
    score = 1.0 if required_total == 0 else found_total / required_total

    if forbidden_found:
        verdict = "fail"
    elif not missing and not extra_failures:
        verdict = "pass"
    elif found_total == 0:
        verdict = "fail"
    else:
        verdict = "partial"

    accepted = bool(accept_mechanical and verdict == "pass")
    return {
        "grader": "ground_truth_mechanical_v1",
        "accepted_for_speed_claim": accepted,
        "score": round(score, 4),
        "verdict": verdict,
        "found": found,
        "missing": missing,
        "forbidden_found": forbidden_found,
        "count_status": count_status,
        "extra_failures": extra_failures,
        "note": (
            "Mechanical benchmark grade. It is accepted only when "
            "--accept-mechanical is used and all required symbols/files/counts "
            "pass for this tasks.json check descriptor."
        ),
    }


def expected_terms(chk: dict[str, Any]) -> list[str]:
    kind = chk.get("kind")
    terms: list[str] = []
    if kind in {"who_calls", "callees", "find_usages"}:
        terms.extend(chk.get("expect_members", []))
        if chk.get("symbol"):
            terms.append(chk["symbol"])
    elif kind == "path":
        terms.extend([chk.get("frm", ""), chk.get("to", "")])
    elif kind in {"search_code", "search_symbols"}:
        terms.append(chk.get("query", ""))
        terms.append(chk.get("expect_file", ""))
    return [t for t in terms if t]


def find_forbidden_terms(chk: dict[str, Any], answer_norm: str) -> list[str]:
    """Return explicit hard negatives mentioned in the candidate answer.

    Benchmark tasks use these only for symbols that would assert a wrong edge
    direction or confuse direct and transitive scope. A hit is a hard quality
    failure: false graph evidence is more harmful to an agent than an omitted
    optional result.
    """
    return [
        term
        for term in chk.get("forbid_terms", [])
        if term and canonical(term) in answer_norm
    ]


def count_check(chk: dict[str, Any], answer: str) -> str:
    min_count = chk.get("min_count")
    if min_count is None:
        return "not_applicable"
    numbers = [int(n) for n in re.findall(r"\b\d+\b", answer)]
    if any(n >= int(min_count) for n in numbers):
        return "ok"
    return "missing"


def count_check_mechanical(chk: dict[str, Any], answer: str, found_terms: list[str]) -> str:
    min_count = chk.get("min_count")
    if min_count is None:
        return "not_applicable"
    numbers = [int(n) for n in re.findall(r"\b\d+\b", answer)]
    if any(n >= int(min_count) for n in numbers):
        return "ok"
    expected_members = chk.get("expect_members", [])
    if min_count <= len(expected_members) and all(m in found_terms for m in expected_members):
        return "inferred_from_members"
    return "missing"


def ordered_path_present(chk: dict[str, Any], answer_norm: str) -> bool:
    frm = canonical(chk.get("frm", ""))
    to = canonical(chk.get("to", ""))
    if not frm or not to:
        return False
    frm_pos = answer_norm.find(frm)
    to_pos = answer_norm.find(to)
    return frm_pos >= 0 and to_pos >= 0 and frm_pos < to_pos


def canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


if __name__ == "__main__":
    main()
