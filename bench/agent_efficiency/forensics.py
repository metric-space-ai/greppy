#!/usr/bin/env python3
"""Generate mandatory post-benchmark forensics for greppy agent runs.

This report is intentionally stricter than the aggregate reporter in
run_bench.py. Token savings alone are never accepted as product evidence here:
quality evidence must exist, output cost is weighted separately, and every
regression becomes optimization backlog.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
from dataclasses import dataclass
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results.json"
DEFAULT_TASK_CLASSES = HERE / "task_classes_v2.json"


@dataclass
class Factors:
    ctx: float | None
    input: float | None
    output: float | None
    weighted: float | None
    tool_calls: float | None
    wall: float | None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=pathlib.Path, default=DEFAULT_RESULTS)
    ap.add_argument("--baseline", default="grep")
    ap.add_argument("--candidate", default="plus")
    ap.add_argument("--output", type=pathlib.Path)
    ap.add_argument("--task-classes", type=pathlib.Path, default=DEFAULT_TASK_CLASSES)
    ap.add_argument("--output-weight", type=float, default=4.0)
    ap.add_argument(
        "--enforce",
        action="store_true",
        help="exit non-zero when mandatory acceptance evidence is missing",
    )
    args = ap.parse_args()

    rows = json.loads(args.results.read_text(encoding="utf-8"))
    task_classes = load_task_classes(args.task_classes)
    report = build_report(
        rows=rows,
        results_path=args.results,
        baseline=args.baseline,
        candidate=args.candidate,
        output_weight=args.output_weight,
        task_classes=task_classes,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)

    if args.enforce and has_blocking_evidence_gap(
        rows, args.baseline, args.candidate, task_classes
    ):
        raise SystemExit(2)


def build_report(
    rows: list[dict[str, Any]],
    results_path: pathlib.Path,
    baseline: str,
    candidate: str,
    output_weight: float,
    task_classes: dict[str, dict[str, Any]],
) -> str:
    comparable = [r for r in rows if r.get(baseline) and r.get(candidate)]
    factor_rows = [(r, compute_factors(r, baseline, candidate, output_weight)) for r in comparable]
    quality_rows = [
        r
        for r in comparable
        if quality_score(r, baseline) is not None and quality_score(r, candidate) is not None
    ]
    accepted_quality_rows = [
        r for r in comparable if quality_not_worse(r, baseline, candidate) is True
    ]
    control_violations = [
        r
        for r in comparable
        if class_role(class_for_task(r, task_classes), task_classes) == "avoid_embedding"
        and candidate_uses_vector(r, candidate)
    ]
    strong_wins = [
        (r, f)
        for r, f in factor_rows
        if is_strong_win(f) and quality_not_worse(r, baseline, candidate) is True
    ]
    unaccepted_speed_wins = [
        (r, f)
        for r, f in factor_rows
        if is_strong_win(f) and quality_not_worse(r, baseline, candidate) is not True
    ]
    regressions = [(r, f) for r, f in factor_rows if is_regression(f)]

    lines: list[str] = []
    lines.append(f"# greppy benchmark forensics - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Results: `{results_path}`")
    lines.append(f"- Baseline: `{baseline}`")
    lines.append(f"- Candidate: `{candidate}`")
    lines.append(f"- Output cost weight: `{output_weight:g}x`")
    lines.append(f"- Comparable tasks: {len(comparable)}/{len(rows)}")
    if task_classes:
        covered = sum(1 for r in comparable if str(r.get("id")) in task_classes)
        lines.append(f"- Task classes loaded: {covered}/{len(comparable)} comparable tasks")
    lines.append("")

    lines.append("## Acceptance verdict")
    lines.append("")
    if not comparable:
        lines.append("BLOCKED: no comparable rows for baseline and candidate.")
    elif len(quality_rows) != len(comparable):
        lines.append(
            "NOT ACCEPTED: accepted quality evidence is missing for "
            f"{len(comparable) - len(quality_rows)}/{len(comparable)} comparable tasks."
        )
        lines.append(
            "Token or tool-call gains in this report are therefore only "
            "optimization hints, not 10x or production-ready evidence."
        )
    elif len(accepted_quality_rows) != len(comparable):
        lines.append(
            "NOT ACCEPTED: at least one candidate run is qualitatively "
            "worse than the baseline."
        )
    elif control_violations:
        lines.append(
            "NOT ACCEPTED: the candidate uses embedding/vector search on "
            f"{len(control_violations)} control tasks that mandate exact/local/graph-only."
        )
    else:
        lines.append(
            "QUALITY GATE PRESENT: all comparable tasks have "
            "machine-readable quality scores and the candidate is not worse."
        )
    lines.append("")

    append_metric_summary(lines, factor_rows)
    append_class_summary(lines, factor_rows, baseline, candidate, task_classes)
    append_task_tables(lines, strong_wins, unaccepted_speed_wins, regressions)
    append_per_task_forensics(lines, factor_rows, baseline, candidate, task_classes)
    append_backlog(lines, factor_rows, baseline, candidate, task_classes)
    append_required_next_steps(lines)
    return "\n".join(lines) + "\n"


def append_metric_summary(lines: list[str], factor_rows: list[tuple[dict[str, Any], Factors]]) -> None:
    lines.append("## Metric summary")
    lines.append("")
    lines.append("Higher is better; factor = baseline / candidate.")
    lines.append("")
    lines.append("| Metric | Median | Mean | n |")
    lines.append("|---|---:|---:|---:|")
    for field in ("ctx", "input", "output", "weighted", "tool_calls", "wall"):
        vals = [getattr(f, field) for _, f in factor_rows if getattr(f, field) is not None]
        if vals:
            lines.append(
                f"| {field} | {statistics.median(vals):.2f}x | "
                f"{statistics.mean(vals):.2f}x | {len(vals)} |"
            )
        else:
            lines.append(f"| {field} | n/a | n/a | 0 |")
    lines.append("")


def append_class_summary(
    lines: list[str],
    factor_rows: list[tuple[dict[str, Any], Factors]],
    baseline: str,
    candidate: str,
    task_classes: dict[str, dict[str, Any]],
) -> None:
    if not task_classes:
        return
    lines.append("## Regression / router classes")
    lines.append("")
    lines.append(
        "These classes are machine-readable R7 gates from `task_classes_v2.json`. "
        "`embedding_candidate` may use vector/Gemma; `avoid_embedding` is "
        "a control set where vector/Gemma use is a router violation."
    )
    lines.append("")
    lines.append("| Class | Role | n | ctx med | weighted med | calls med | accepted wins | regressions | vector-control violations |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    by_class: dict[str, list[tuple[dict[str, Any], Factors]]] = {}
    for row, factors in factor_rows:
        cls = class_for_task(row, task_classes)
        by_class.setdefault(cls, []).append((row, factors))
    for cls in sorted(by_class):
        rows = by_class[cls]
        role = class_role(cls, task_classes)
        ctx = median_factor(rows, "ctx")
        weighted = median_factor(rows, "weighted")
        calls = median_factor(rows, "tool_calls")
        wins = sum(
            1
            for row, factors in rows
            if is_strong_win(factors) and quality_not_worse(row, baseline, candidate) is True
        )
        regressions = sum(1 for _, factors in rows if is_regression(factors))
        vector_violations = sum(
            1
            for row, _ in rows
            if role == "avoid_embedding" and candidate_uses_vector(row, candidate)
        )
        lines.append(
            f"| `{cls}` | `{role}` | {len(rows)} | {fmt_factor(ctx)} | "
            f"{fmt_factor(weighted)} | {fmt_factor(calls)} | {wins} | "
            f"{regressions} | {vector_violations} |"
        )
    lines.append("")


def append_task_tables(
    lines: list[str],
    strong_wins: list[tuple[dict[str, Any], Factors]],
    unaccepted_speed_wins: list[tuple[dict[str, Any], Factors]],
    regressions: list[tuple[dict[str, Any], Factors]],
) -> None:
    lines.append("## Winners and losers")
    lines.append("")
    lines.append(
        f"- Accepted strong wins: {len(strong_wins)} "
        "(only where quality is not worse)."
    )
    lines.append(
        f"- Unaccepted speed wins: {len(unaccepted_speed_wins)} "
        "(gain visible, but quality evidence is missing or worse)."
    )
    lines.append(f"- Regression / risk tasks: {len(regressions)}")
    lines.append("")

    append_compact_table(lines, "Accepted strong wins", strong_wins[:20])
    append_compact_table(lines, "Unaccepted speed wins", unaccepted_speed_wins[:20])
    append_compact_table(lines, "Regression / risk tasks", regressions[:30])


def append_compact_table(
    lines: list[str], title: str, rows: list[tuple[dict[str, Any], Factors]]
) -> None:
    lines.append(f"### {title}")
    lines.append("")
    if not rows:
        lines.append("_None._")
        lines.append("")
        return
    lines.append("| Task | ctx | weighted | calls | output | Question |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for r, f in rows:
        lines.append(
            f"| `{r.get('id')}` | {fmt_factor(f.ctx)} | {fmt_factor(f.weighted)} | "
            f"{fmt_factor(f.tool_calls)} | {fmt_factor(f.output)} | {one_line(r.get('q', ''))} |"
        )
    lines.append("")


def append_per_task_forensics(
    lines: list[str],
    factor_rows: list[tuple[dict[str, Any], Factors]],
    baseline: str,
    candidate: str,
    task_classes: dict[str, dict[str, Any]],
) -> None:
    lines.append("## Per-task forensics")
    lines.append("")
    lines.append(
        "Each row names the likely relevant optimization class. "
        "Raw paths are listed when `run_bench.py --save-raw` stored them."
    )
    lines.append("")
    for r, f in factor_rows:
        base = r[baseline]
        cand = r[candidate]
        lines.append(f"### `{r.get('id')}` - {one_line(r.get('q', ''))}")
        lines.append("")
        cls = class_for_task(r, task_classes)
        if cls != "unclassified":
            lines.append(f"- Class: `{cls}` / role `{class_role(cls, task_classes)}`.")
        lines.append(
            f"- Cost: ctx {base.get('ctx_tok')} -> {cand.get('ctx_tok')} "
            f"({fmt_factor(f.ctx)}), output {base.get('output')} -> {cand.get('output')} "
            f"({fmt_factor(f.output)}), calls {base.get('tool_calls')} -> "
            f"{cand.get('tool_calls')} ({fmt_factor(f.tool_calls)})."
        )
        lines.append(f"- Quality: {quality_line(r, baseline, candidate)}")
        lines.append(
            f"- Diagnosis: {diagnose_task(r, f, baseline, candidate, task_classes)}"
        )
        raw = r.get("raw_paths", {})
        if raw:
            if baseline in raw:
                lines.append(f"- Raw baseline: `{raw[baseline]}`")
            if candidate in raw:
                lines.append(f"- Raw candidate: `{raw[candidate]}`")
                commands = extract_commands(pathlib.Path(raw[candidate]))
                if commands:
                    lines.append("- Candidate path:")
                    for command in commands[:8]:
                        lines.append(f"  - `{shorten_command(command)}`")
                    if len(commands) > 8:
                        lines.append(f"  - ... {len(commands) - 8} more commands")
        lines.append("")


def append_backlog(
    lines: list[str],
    factor_rows: list[tuple[dict[str, Any], Factors]],
    baseline: str,
    candidate: str,
    task_classes: dict[str, dict[str, Any]],
) -> None:
    buckets: dict[str, set[str]] = {
        "quality_gate": set(),
        "router_literal_fast_path": set(),
        "output_contract": set(),
        "over_iteration": set(),
        "semantic_noise_or_ranking": set(),
        "embedding_on_control": set(),
        "embedding_candidate_not_compressed": set(),
        "latency": set(),
    }
    for r, f in factor_rows:
        tid = str(r.get("id"))
        cls = class_for_task(r, task_classes)
        role = class_role(cls, task_classes)
        if quality_not_worse(r, baseline, candidate) is None:
            buckets["quality_gate"].add(tid)
        if f.ctx is not None and f.ctx < 1.0:
            buckets["semantic_noise_or_ranking"].add(tid)
        if f.output is not None and f.output < 1.0:
            buckets["output_contract"].add(tid)
        if f.tool_calls is not None and f.tool_calls < 1.0:
            buckets["over_iteration"].add(tid)
        if looks_literal_or_local(r.get("q", "")) and is_regression(f):
            buckets["router_literal_fast_path"].add(tid)
        if role == "avoid_embedding" and candidate_uses_vector(r, candidate):
            buckets["embedding_on_control"].add(tid)
        if (
            role == "embedding_candidate"
            and (f.tool_calls is None or f.tool_calls < 1.5)
            and quality_not_worse(r, baseline, candidate) is not False
        ):
            buckets["embedding_candidate_not_compressed"].add(tid)
        if f.wall is not None and f.wall < 1.0:
            buckets["latency"].add(tid)

    lines.append("## Optimization backlog")
    lines.append("")
    lines.append(
        "- P0 quality gate: a machine-readable grader or human-adjudicated "
        "`quality` fields must be in `results.json`, otherwise speed wins stay unaccepted."
    )
    lines.append(format_bucket("P1 literal/local router", buckets["router_literal_fast_path"]))
    lines.append(format_bucket("P1 output contract too broad", buckets["output_contract"]))
    lines.append(format_bucket("P1 agent keeps iterating", buckets["over_iteration"]))
    lines.append(format_bucket("P1 semantic noise / ranking", buckets["semantic_noise_or_ranking"]))
    lines.append(format_bucket("P1 embedding on control tasks", buckets["embedding_on_control"]))
    lines.append(
        format_bucket(
            "P1 embedding candidates without path compression",
            buckets["embedding_candidate_not_compressed"],
        )
    )
    lines.append(format_bucket("P2 latency", buckets["latency"]))
    lines.append("")
    lines.append("## Long-horizon research backlog")
    lines.append("")
    lines.append(
        "- Build a hard-negative set from all regression tasks: exact/local "
        "questions must not trigger semantic expansion."
    )
    lines.append(
        "- Ranking calibration: exact grep > exact symbol > graph-local > vector; "
        "embedding may only surface for concept-to-symbol/owner/helper intents."
    )
    lines.append(
        "- If EmbeddingGemma keeps missing semantically on concept tasks despite the "
        "router: domain-specific fine-tuning or an adapter only after a "
        "separate retrieval eval with hard negatives."
    )
    lines.append(
        "- Determine the ANN/exact threshold empirically: p50/p95 query latency, "
        "memory, hit quality, deterministic top-K stability."
    )
    lines.append(
        "- Minimize the agent prompt: output tokens are more expensive; every plus line "
        "must reduce a next meaningful read/graph step."
    )
    lines.append("")


def append_required_next_steps(lines: list[str]) -> None:
    lines.append("## Required next bench")
    lines.append("")
    lines.append("1. Run `run_bench.py --save-raw --agents grep,plus,greppy`.")
    lines.append("2. Add quality scores per agent/task or run the grader.")
    lines.append("3. Run `forensics.py --baseline grep --candidate <agent> --enforce`.")
    lines.append("4. Only accepted strong wins may be reported as progress.")
    lines.append("")


def compute_factors(row: dict[str, Any], baseline: str, candidate: str, output_weight: float) -> Factors:
    b = row[baseline]
    c = row[candidate]
    return Factors(
        ctx=ratio(b.get("ctx_tok"), c.get("ctx_tok")),
        input=ratio(b.get("input"), c.get("input")),
        output=ratio(b.get("output"), c.get("output")),
        weighted=ratio(weighted_cost(b, output_weight), weighted_cost(c, output_weight)),
        tool_calls=ratio(b.get("tool_calls"), c.get("tool_calls")),
        wall=ratio(b.get("wall_s"), c.get("wall_s")),
    )


def weighted_cost(row: dict[str, Any], output_weight: float) -> float | None:
    if row.get("input") is None or row.get("output") is None:
        return None
    return float(row.get("input", 0)) + output_weight * float(row.get("output", 0))


def ratio(base: Any, cand: Any) -> float | None:
    try:
        b = float(base)
        c = float(cand)
    except (TypeError, ValueError):
        return None
    if c <= 0:
        return None
    return b / c


def quality_score(row: dict[str, Any], agent: str) -> float | None:
    # A graded partial/fail verdict is a score, not missing evidence
    # (BENCHMARK_CONTRACT.md: only absent grades void a run; same owner-decided
    # reading as release_gate.quality()). accepted_for_speed_claim keeps its
    # role for speed claims elsewhere in this module.
    q = row.get(agent, {}).get("quality")
    if isinstance(q, dict) and q.get("score") is not None:
        return float(q["score"])
    if row.get(agent, {}).get("correct") is not None:
        return 1.0 if row[agent]["correct"] else 0.0
    return None


def quality_not_worse(row: dict[str, Any], baseline: str, candidate: str) -> bool | None:
    b = quality_score(row, baseline)
    c = quality_score(row, candidate)
    if b is None or c is None:
        return None
    return c >= b


def has_blocking_evidence_gap(
    rows: list[dict[str, Any]],
    baseline: str,
    candidate: str,
    task_classes: dict[str, dict[str, Any]],
) -> bool:
    comparable = [r for r in rows if r.get(baseline) and r.get(candidate)]
    if not comparable:
        return True
    for row in comparable:
        # Blocking means EVIDENCE is missing (a row without a grade on either
        # arm) or a router violation - matching this flag's documented purpose
        # ("exit non-zero when mandatory acceptance evidence is missing").
        # A graded row where the candidate scored worse is NOT an evidence
        # gap: per-row losses are adjudicated by the pre-registered paired
        # correctness gates in release_gate.py, which tolerate losses up to a
        # one-sided exact test at alpha 0.05. A zero-loss rule here would
        # contradict those registered gates.
        if quality_not_worse(row, baseline, candidate) is None:
            return True
        cls = class_for_task(row, task_classes)
        if class_role(cls, task_classes) == "avoid_embedding" and candidate_uses_vector(
            row, candidate
        ):
            return True
    return False


def is_strong_win(f: Factors) -> bool:
    return (
        (f.ctx is not None and f.ctx >= 1.5)
        and (f.weighted is not None and f.weighted >= 1.1)
        and (f.tool_calls is not None and f.tool_calls >= 1.5)
    )


def is_regression(f: Factors) -> bool:
    return any(
        value is not None and value < 0.9
        for value in (f.ctx, f.output, f.weighted, f.tool_calls, f.wall)
    )


def diagnose_task(
    row: dict[str, Any],
    f: Factors,
    baseline: str,
    candidate: str,
    task_classes: dict[str, dict[str, Any]],
) -> str:
    cls = class_for_task(row, task_classes)
    role = class_role(cls, task_classes)
    if row.get(candidate, {}).get("error"):
        return f"Candidate error: {row[candidate]['error']}"
    if role == "avoid_embedding" and candidate_uses_vector(row, candidate):
        return (
            "Router violation: a control task must not use embedding/vector search; "
            "force the exact grep/symbol/graph fast path."
        )
    if quality_not_worse(row, baseline, candidate) is None:
        return "Quality missing; no speed acceptance possible."
    if role == "embedding_candidate" and f.tool_calls is not None and f.tool_calls < 1.5:
        return (
            "Embedding candidate class without sufficient path compression; "
            "check top-K/brief/context routing or output stop signal."
        )
    if is_regression(f) and looks_literal_or_local(row.get("q", "")):
        return "Literal/local task regressed; the router must force the exact/grep fast path."
    if f.tool_calls is not None and f.tool_calls < 1.0:
        return "Agent keeps iterating after candidate output; output gives no clear stop/read signal."
    if f.output is not None and f.output < 1.0:
        return "Output tokens regress; the plus answer is too broad or too explanatory."
    if f.ctx is not None and f.ctx < 1.0:
        return "Search context regresses; ranking/semantic hits return too much or the wrong context."
    if is_strong_win(f):
        return "Strong path-compression candidate; freeze it as a regression test."
    return "Mixed effect; inspect the raw path first before fine-tuning/routing."


def load_task_classes(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for class_name, cls in (doc.get("classes") or {}).items():
        if not isinstance(cls, dict):
            continue
        role = str(cls.get("role", "unknown"))
        for tid in cls.get("ids", []) or []:
            if isinstance(tid, str):
                out[tid] = {"class": class_name, "role": role}
    return out


def class_for_task(row: dict[str, Any], task_classes: dict[str, dict[str, Any]]) -> str:
    entry = task_classes.get(str(row.get("id")))
    if not entry:
        return "unclassified"
    return str(entry.get("class") or "unclassified")


def class_role(class_name: str, task_classes: dict[str, dict[str, Any]]) -> str:
    for entry in task_classes.values():
        if entry.get("class") == class_name:
            return str(entry.get("role") or "unknown")
    return "unknown"


def candidate_uses_vector(row: dict[str, Any], candidate: str) -> bool:
    commands: list[str] = []
    raw = row.get("raw_paths", {})
    if isinstance(raw, dict) and candidate in raw:
        commands = extract_commands(pathlib.Path(raw[candidate]))
    answer = str(row.get(candidate, {}).get("answer", ""))
    haystack = "\n".join(commands + [answer]).lower()
    triggers = [
        "embeddinggemma",
        "embeddinggemma_context.py",
        "--embedding-gguf",
        "--embedding-model-dir",
        "vector search",
        "semantic_results",
    ]
    return any(t in haystack for t in triggers)


def median_factor(rows: list[tuple[dict[str, Any], Factors]], field: str) -> float | None:
    values = [getattr(f, field) for _, f in rows if getattr(f, field) is not None]
    if not values:
        return None
    return statistics.median(values)


def quality_line(row: dict[str, Any], baseline: str, candidate: str) -> str:
    b = quality_score(row, baseline)
    c = quality_score(row, candidate)
    if b is None or c is None:
        details = []
        for agent in (baseline, candidate):
            q = row.get(agent, {}).get("quality")
            if isinstance(q, dict):
                accepted = q.get("accepted_for_speed_claim") is True
                details.append(
                    f"{agent}: {q.get('verdict', 'unknown')} score={q.get('score')} "
                    f"accepted={accepted}"
                )
        if details:
            return "not accepted (" + "; ".join(details) + ")"
        return "missing"
    verdict = "not worse" if c >= b else "worse"
    return f"{candidate}={c:.3f}, {baseline}={b:.3f} -> {verdict}"


def looks_literal_or_local(question: str) -> bool:
    q = question.lower()
    triggers = [
        "exact",
        "literal",
        "where is",
        "find where",
        "show code",
        "return code",
        "definition",
        "line",
        "amount * 100",
        "operator",
    ]
    return any(t in q for t in triggers)


def extract_commands(path: pathlib.Path) -> list[str]:
    if not path.exists():
        return []
    commands: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "tool_execution_start":
            args = event.get("args") or {}
            command = args.get("command")
            if isinstance(command, str):
                commands.append(command)
    return commands


def shorten_command(command: str, limit: int = 180) -> str:
    command = " ".join(command.split())
    if len(command) <= limit:
        return command
    return command[: limit - 3] + "..."


def format_bucket(title: str, task_ids: set[str]) -> str:
    if not task_ids:
        return f"- {title}: no current tasks."
    return f"- {title}: {', '.join(f'`{t}`' for t in sorted(task_ids))}."


def fmt_factor(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}x"


def one_line(text: str, limit: int = 90) -> str:
    text = " ".join(str(text).split()).replace("|", "\\|")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


if __name__ == "__main__":
    main()
