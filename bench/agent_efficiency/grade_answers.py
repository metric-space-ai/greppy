#!/usr/bin/env python3
"""Attach conservative answer-quality grades to benchmark results.

This is not a semantic judge. It checks an agent's final answer against the
mechanically known hints from the tasks file (default tasks_v2.json, corpus v2;
the v1 tasks.json format is still accepted if such a file is passed).

Modes:

* smoke: loose triage; never accepted for speed claims unless explicitly forced.
* mechanical: stricter symbol/file/count/path checks; can be accepted when
  --accept-mechanical is supplied. This still remains a benchmark-specific
  mechanical gate, not a general proof of semantic correctness.

Corpus-v2 FLOOR semantics (check.semantics == "floor", candidates.json):
the C oracle undercounts, so every expected item is a SUBSET floor -- every
expect_member (caller name) must appear in the answer, every file_evidence
path must appear, and the answer's stated total (or the full inferred member
set) must reach min_count. An agent listing MORE true callers than the floor
passes; a missing floor item or a total below the floor does not (partial
credit is never accepted). ``path`` checks require the whole chain
frm -> via... -> to to appear in order; ``impact`` checks require the named
transitively impacted members. v1 kinds (the 19 controls) grade exactly as
before.

Self-test (no API cost): python3 grade_answers.py --self-test
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results.json"
DEFAULT_TASKS = HERE / "tasks_v2.json"


AGENT_FIELDS = {"grep", "grepplus", "explorer", "plus"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=pathlib.Path, default=DEFAULT_RESULTS)
    ap.add_argument("--tasks", type=pathlib.Path, default=DEFAULT_TASKS)
    ap.add_argument("--output", type=pathlib.Path)
    ap.add_argument("--agents", default="grep,grepplus,explorer,plus")
    ap.add_argument("--mode", choices=("smoke", "mechanical"), default="smoke")
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="run the built-in v1/v2 grading self-test and exit (no results file needed)",
    )
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
            "This is intended for the synthetic 100-task bench where the tasks "
            "file is the ground-truth contract."
        ),
    )
    args = ap.parse_args()

    if args.self_test:
        raise SystemExit(self_test())

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

    count_status = count_check(chk, answer)
    required_total = len(required) + (1 if count_status == "missing" else 0)
    found_total = len(found) + (1 if count_status == "ok" else 0)
    score = 1.0 if required_total == 0 else found_total / required_total
    verdict = "pass" if not missing and count_status != "missing" else "partial"
    if found_total == 0 and required_total > 0:
        verdict = "fail"

    accepted = bool(accept_smoke and verdict == "pass")
    return {
        "grader": "ground_truth_smoke_v1",
        "accepted_for_speed_claim": accepted,
        "score": round(score, 4),
        "verdict": verdict,
        "found": found,
        "missing": missing,
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

    required = expected_terms(chk)
    found = []
    missing = []
    for term in required:
        if canonical(term) in answer_norm:
            found.append(term)
        else:
            missing.append(term)

    extra_failures: list[str] = []
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
            extra_failures.append(
                "path chain members not present in source-to-target order"
            )

    required_total = len(required) + (1 if count_required else 0) + (
        1 if path_order_required else 0
    )
    found_total = len(found)
    if count_required and count_passed:
        found_total += 1
    if path_order_required and path_order_passed:
        found_total += 1
    score = 1.0 if required_total == 0 else found_total / required_total

    if not missing and not extra_failures:
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
        "count_status": count_status,
        "extra_failures": extra_failures,
        "note": (
            "Mechanical benchmark grade. It is accepted only when "
            "--accept-mechanical is used and all required symbols/files/counts "
            "pass for this tasks-file check descriptor."
        ),
    }


def expected_terms(chk: dict[str, Any]) -> list[str]:
    kind = chk.get("kind")
    terms: list[str] = []
    if kind in {"who_calls", "callees", "find_usages", "impact"}:
        # v2 floor semantics: expect_members is a caller-name SUBSET floor
        # (the answer may contain MORE true callers); file_evidence entries
        # are caller sites the oracle could only name by file path. "impact"
        # (v2 blast-radius) requires the named impacted members just like
        # who_calls requires the named callers.
        # The QUERIED symbol itself is deliberately NOT required: it is the
        # question's subject, not evidence. A terse correct answer such as
        # "**Caller:** `serialize_variant` in ser.rs:421 — Total: 1" (r008)
        # must pass without echoing "serialize_externally_tagged_variant"
        # back. Callers/members/file evidence remain required.
        terms.extend(chk.get("expect_members", []))
        terms.extend(chk.get("file_evidence", []) or [])
    elif kind == "path":
        # v2 chains carry intermediate hops in `via`; v1 has frm/to only.
        terms.extend(path_chain(chk))
    elif kind in {"search_code", "search_symbols"}:
        terms.append(chk.get("query", ""))
        terms.append(chk.get("expect_file", ""))
    return [t for t in terms if t]


def path_chain(chk: dict[str, Any]) -> list[str]:
    """Ordered chain frm -> via... -> to (via is empty for v1 checks)."""
    via = chk.get("via") or []
    return [chk.get("frm", ""), *via, chk.get("to", "")]


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
    # Floor inference: the oracle's min_count counts named members plus the
    # unnamed callers recorded as file_evidence paths (corpus v2). Only when
    # EVERY floor item appears in the answer and the floor is covered by the
    # listed items can the count be inferred without a stated total.
    floor_items = list(chk.get("expect_members", [])) + list(
        chk.get("file_evidence", []) or []
    )
    if min_count <= len(floor_items) and floor_items and all(
        m in found_terms for m in floor_items
    ):
        return "inferred_from_members"
    return "missing"


def ordered_path_present(chk: dict[str, Any], answer_norm: str) -> bool:
    """The whole chain frm -> via... -> to must appear IN ORDER.

    Greedy earliest-occurrence subsequence scan over the canonical answer:
    each member must occur after the END of the previous match, so an answer
    that skips an intermediate hop (e.g. names only frm and to, where the via
    member is a prefix of to) does not pass.
    """
    members = [canonical(m) for m in path_chain(chk)]
    if not all(members) or len(members) < 2:
        return False
    pos = 0
    for member in members:
        i = answer_norm.find(member, pos)
        if i < 0:
            return False
        pos = i + len(member)
    return True


def canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


# ---------------------------------------------------------------------------
# self-test: dry grading of crafted answers, one pass + failure modes per
# corpus-v2 check kind (floor semantics) plus v1 control regressions.
# ---------------------------------------------------------------------------
def self_test() -> int:
    who_calls_v2 = {
        "check": {
            "kind": "who_calls", "symbol": "allow_transparent",
            "expect_members": ["check_transparent"], "file_evidence": [],
            "min_count": 1, "semantics": "floor",
        }
    }
    # r008 (tasks_v2.json, serde): the echo-bug regression check. The queried
    # symbol must NOT be a required term — a terse, correct answer names the
    # caller and the total without repeating the question's subject.
    who_calls_r008 = {
        "check": {
            "kind": "who_calls",
            "symbol": "serialize_externally_tagged_variant",
            "expect_members": ["serialize_variant"], "file_evidence": [],
            "min_count": 1, "semantics": "floor",
        }
    }
    who_calls_big_floor = {
        "check": {
            "kind": "who_calls", "symbol": "processInput",
            "expect_members": ["runStage"], "file_evidence": [],
            "min_count": 17, "semantics": "floor",
        }
    }
    who_calls_file_evidence = {
        "check": {
            "kind": "who_calls", "symbol": "get_lit_str",
            "expect_members": ["parse_lit_into_expr_path", "parse_lit_into_ty"],
            "file_evidence": ["serde_derive/src/internals/attr.rs"],
            "min_count": 3, "semantics": "floor",
        }
    }
    impact_v2 = {
        "check": {
            "kind": "impact", "symbol": "parseSyncInternal",
            "direction": "incoming",
            "expect_members": ["safeParse", "handleResult"],
            "file_evidence": [], "min_count": 8, "semantics": "floor",
        }
    }
    path_v2 = {
        "check": {
            "kind": "path", "frm": "pretend_used",
            "via": ["pretend_fields_used"],
            "to": "pretend_fields_used_struct_packed", "semantics": "floor",
        }
    }
    search_symbols_v2 = {
        "check": {
            "kind": "search_symbols", "query": "apply_to_field",
            "expect_file": "serde_derive/src/internals/case.rs",
            "semantics": "floor",
        }
    }
    search_code_v1 = {
        "check": {
            "kind": "search_code", "query": "to_minor_units",
            "expect_file": "money.py",
        }
    }
    who_calls_v1 = {
        "check": {
            "kind": "who_calls", "symbol": "compute_checksum",
            "expect_members": ["normalize_record", "merge_checksums"],
            "min_count": 70,
        }
    }
    path_v1 = {"check": {"kind": "path", "frm": "run_pipeline",
                         "to": "compute_checksum"}}

    cases = [
        # (name, task, answer, expect_accepted)
        ("who_calls v2 pass (floor met, total stated)", who_calls_v2,
         "The function allow_transparent has exactly one caller: "
         "check_transparent. Total: one caller (1).", True),
        ("who_calls v2 pass (MORE than floor)", who_calls_v2,
         "allow_transparent is called by check_transparent and also by "
         "derive_helper -- two (2) callers in total.", True),
        ("who_calls v2 FAIL floor (stated total below min_count)",
         who_calls_big_floor,
         "processInput is called by runStage. I found five (5) callers "
         "in total.", False),
        ("who_calls v2 FAIL subset (expect_member missing)", who_calls_v2,
         "allow_transparent has callers spread across the crate; "
         "three (3) in total.", False),
        ("who_calls r008 pass (terse answer, subject NOT echoed)",
         who_calls_r008,
         "**Caller:** `serialize_variant` in ser.rs:421 — Total: 1", True),
        ("who_calls r008 FAIL negative control (wrong caller, subject echoed)",
         who_calls_r008,
         "serialize_externally_tagged_variant has exactly one caller: "
         "serialize_adjacently_tagged_variant in ser.rs:421 — Total: 1",
         False),
        ("who_calls v2 pass (file_evidence + inferred floor)",
         who_calls_file_evidence,
         "get_lit_str callers: parse_lit_into_expr_path, parse_lit_into_ty, "
         "plus one closure caller in serde_derive/src/internals/attr.rs.", True),
        ("who_calls v2 FAIL subset (file_evidence path missing)",
         who_calls_file_evidence,
         "get_lit_str callers: parse_lit_into_expr_path, parse_lit_into_ty.",
         False),
        ("impact v2 pass (members + total >= floor)", impact_v2,
         "Blast radius of parseSyncInternal: twelve (12) functions depend on "
         "it transitively, including safeParse, handleResult and parseAsync.",
         True),
        ("impact v2 FAIL floor (total below min_count)", impact_v2,
         "Changing parseSyncInternal affects safeParse and handleResult -- "
         "three (3) callers total.", False),
        ("impact v2 FAIL subset (impacted member missing)", impact_v2,
         "Changing parseSyncInternal impacts about twenty (20) functions, "
         "mainly safeParse and its wrappers.", False),
        ("path v2 pass (full chain in order)", path_v2,
         "The chain is pretend_used -> pretend_fields_used -> "
         "pretend_fields_used_struct_packed.", True),
        ("path v2 FAIL floor (via hop skipped)", path_v2,
         "pretend_used reaches pretend_fields_used_struct_packed directly.",
         False),
        ("path v2 FAIL subset (chain reversed)", path_v2,
         "pretend_fields_used_struct_packed is called by pretend_fields_used, "
         "which is called by pretend_used.", False),
        ("search_symbols v2 pass (symbol + file)", search_symbols_v2,
         "The routine is apply_to_field in serde_derive/src/internals/case.rs "
         "(line 82).", True),
        ("search_symbols v2 FAIL floor (file missing)", search_symbols_v2,
         "The routine is apply_to_field, implemented in the derive internals.",
         False),
        ("search_symbols v2 FAIL subset (symbol not named)", search_symbols_v2,
         "The conversion lives in serde_derive/src/internals/case.rs.", False),
        # v1 controls must keep grading exactly as before
        ("search_code v1 pass", search_code_v1,
         "In to_minor_units in app/core/money.py: int(round(amount * 100)).",
         True),
        ("search_code v1 FAIL (file missing)", search_code_v1,
         "The conversion is done by to_minor_units.", False),
        ("who_calls v1 pass (count stated)", who_calls_v1,
         "compute_checksum is called from normalize_record, merge_checksums "
         "and 70 other call sites (72 total).", True),
        ("who_calls v1 FAIL (count below min_count)", who_calls_v1,
         "compute_checksum is called by normalize_record and merge_checksums "
         "-- 2 callers.", False),
        ("path v1 pass (frm before to)", path_v1,
         "run_pipeline -> stage_svc -> compute_checksum.", True),
        ("path v1 FAIL (to before frm only)", path_v1,
         "compute_checksum is at the end; the entry point run_pipeline was "
         "not traced.", False),
    ]

    failures = 0
    for name, task, answer, expect_accepted in cases:
        grade = grade_answer_mechanical(task, answer, accept_mechanical=True)
        accepted = grade["accepted_for_speed_claim"]
        ok = accepted == expect_accepted
        status = "ok  " if ok else "FAIL"
        print(f"[self-test] {status} {name}: verdict={grade['verdict']} "
              f"accepted={accepted} expected_accepted={expect_accepted}")
        if not ok:
            failures += 1
            print(f"            missing={grade['missing']} "
                  f"count={grade['count_status']} "
                  f"extra={grade['extra_failures']}")
    total = len(cases)
    print(f"[self-test] {total - failures}/{total} cases behaved as expected")
    return 1 if failures else 0


if __name__ == "__main__":
    main()
