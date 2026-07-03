#!/usr/bin/env python3
"""Gate-P0 metric verification: independently recompute every benchmark metric
from raw Pi JSONL and compare against the harness-recorded rows.

Two legs (Codex-Review P2-1: claims stated precisely):
  1. The INDEPENDENT recompute leg (`recompute()`) is a second implementation
     written without importing run_bench.py — a harness parser bug cannot
     silently self-confirm there (PLAN_10X §6 P0.3, GESAMTZIEL §2.3).
  2. The parser-vs-parser leg (`_harness_parse()`) DOES deliberately import
     run_bench.py to obtain the harness's own values on identical raw input;
     the comparison is between the two implementations.

Default agents: explorer,grep,grepplus — the gate baseline `explorer` MUST be
verified (Codex-Review P0-1); a verification that skips the gate denominator
is worthless.

Tolerances:
  ctx_chars / ctx_tok / input / output / tool_calls / turns / variable_input:
      EXACT match required.
  wall_s: row value vs .meta.json subprocess wall, |delta| <= 0.15s
      (row rounds to 0.1s, meta to 0.001s; no other source exists in the raw).

Usage:
  python3 verify_metrics.py --run-dir <dir with results.json + raw/> \
      [--sample N] [--artifact OUT.md]

Exit codes: 0 = all sampled rows verified; 2 = any mismatch or missing data.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

# Same key set the harness sums for prompt-side usage; re-stated here on
# purpose (a harness typo in these keys must FAIL verification, not inherit).
PROMPT_KEYS = ("input", "cacheRead", "cacheWrite", "cacheWrite1h", "cacheWrite5m")


def recompute(jsonl_text: str) -> dict:
    """Independent re-derivation of all metrics from one Pi JSONL transcript."""
    ctx_chars = 0
    tool_calls = 0
    inputs, outputs, prompt_inputs = [], [], []
    for line in jsonl_text.splitlines():
        try:
            obj = json.loads(line)
        except (ValueError, json.JSONDecodeError):
            continue
        if obj.get("type") != "turn_end":
            continue
        results = obj.get("toolResults") or []
        tool_calls += len(results)
        for tr in results:
            for chunk in tr.get("content") or []:
                if isinstance(chunk, dict) and chunk.get("type") == "text":
                    ctx_chars += len(chunk.get("text", ""))
        usage = (obj.get("message") or {}).get("usage") or {}
        inputs.append(int(usage.get("input", 0) or 0))
        outputs.append(int(usage.get("output", 0) or 0))
        prompt_inputs.append(sum(int(usage.get(k, 0) or 0) for k in PROMPT_KEYS))
    return {
        "ctx_chars": ctx_chars,
        "ctx_tok": round(ctx_chars / 4),
        "input": sum(inputs),
        "output": sum(outputs),
        "tool_calls": tool_calls,
        "turns": len(inputs),
        "variable_input": sum(prompt_inputs[1:]),  # base turn removed
    }


EXACT_KEYS = ("ctx_chars", "ctx_tok", "input", "output", "tool_calls", "turns",
              "variable_input")


def _harness_parse(jsonl_text: str) -> dict:
    """Parse the same raw JSONL with the HARNESS implementation (run_bench.py).

    This is the parser-vs-parser leg: two independently written extractors must
    agree on identical raw input for every metric, including the
    base-prompt-neutral fields old result rows do not carry."""
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import run_bench  # noqa: PLC0415 (deliberate late import, harness leg only)
    return run_bench.parse_pi_jsonl(jsonl_text)


def verify_task(task_dir: pathlib.Path, row: dict, strict_rows: bool,
                agents: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Return (failures, warnings) for one task's agents."""
    failures: list[str] = []
    warnings: list[str] = []
    for agent in agents:
        jsonl = task_dir / f"{agent}.jsonl"
        recorded = row.get(agent)
        if not jsonl.exists() or not isinstance(recorded, dict):
            failures.append(f"{task_dir.name}/{agent}: raw or row missing")
            continue
        raw_text = jsonl.read_text(encoding="utf-8", errors="replace")
        mine = recompute(raw_text)
        harness = _harness_parse(raw_text)
        for key in EXACT_KEYS:
            # Leg 1: parser vs parser (always possible, covers ALL keys).
            if int(harness.get(key, -1)) != int(mine[key]):
                failures.append(
                    f"{task_dir.name}/{agent}: {key} harness-parser="
                    f"{harness.get(key)} independent={mine[key]}"
                )
            # Leg 2: recorded pipeline row vs independent recomputation.
            if key not in recorded or recorded[key] is None:
                msg = f"{task_dir.name}/{agent}: row lacks '{key}'"
                (failures if strict_rows else warnings).append(msg)
                continue
            if int(recorded[key]) != int(mine[key]):
                failures.append(
                    f"{task_dir.name}/{agent}: {key} row={recorded[key]} "
                    f"recomputed={mine[key]}"
                )
        meta_path = jsonl.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if abs(float(recorded.get("wall_s", 0)) - float(meta.get("wall_s", 0))) > 0.15:
                failures.append(
                    f"{task_dir.name}/{agent}: wall_s row={recorded.get('wall_s')} "
                    f"meta={meta.get('wall_s')} (>0.15s)"
                )
    return failures, warnings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="directory containing results.json and raw/<task>/")
    ap.add_argument("--sample", type=int, default=10,
                    help="number of tasks to verify (sorted ids, deterministic)")
    ap.add_argument("--artifact", default=None,
                    help="write a markdown verification artifact here")
    ap.add_argument("--strict-rows", action="store_true",
                    help="missing row fields are failures (required for P1+ runs)")
    ap.add_argument("--agents", default="explorer,grep,grepplus",
                    help="comma-separated agents to verify; the gate baseline "
                         "explorer is in the default on purpose (P0-1)")
    args = ap.parse_args()
    agents = tuple(a.strip() for a in args.agents.split(",") if a.strip())

    run_dir = pathlib.Path(args.run_dir)
    results_path = run_dir / "results.json"
    raw_root = run_dir / "raw"
    if not results_path.exists() or not raw_root.is_dir():
        sys.exit(f"need {results_path} and {raw_root}/")

    rows = {r["id"]: r for r in json.loads(results_path.read_text(encoding="utf-8"))}
    task_dirs = sorted(
        d for d in raw_root.iterdir()
        if d.is_dir() and d.name in rows
        and all((d / f"{a}.jsonl").exists() for a in agents)
    )[: args.sample]
    if len(task_dirs) < args.sample:
        print(f"WARN: only {len(task_dirs)} verifiable tasks found "
              f"(wanted {args.sample})", file=sys.stderr)
    if not task_dirs:
        sys.exit("no verifiable tasks (need raw/<id>/{grep,grepplus}.jsonl)")

    all_failures: list[str] = []
    all_warnings: list[str] = []
    checked = 0
    for td in task_dirs:
        fails, warns = verify_task(td, rows[td.name], args.strict_rows, agents)
        all_failures.extend(fails)
        all_warnings.extend(warns)
        checked += 1
        status = "OK " if not fails else "FAIL"
        print(f"  {status} {td.name}")

    # per agent: EXACT_KEYS x2 legs (parser-vs-parser + row) + wall_s
    n_metrics = checked * len(agents) * (len(EXACT_KEYS) * 2 + 1)
    verdict = "PASS" if not all_failures else "FAIL"
    summary = (
        f"verify_metrics: {verdict} — {checked} tasks x {len(agents)} agents "
        f"({','.join(agents)}), {n_metrics} metric checks, "
        f"{len(all_failures)} mismatches, {len(all_warnings)} warnings "
        f"(exact keys: {', '.join(EXACT_KEYS)}; wall_s tol 0.15s)"
    )
    print(summary)
    for f in all_failures:
        print(f"  MISMATCH: {f}")
    for w in all_warnings:
        print(f"  WARN:     {w}")

    if args.artifact:
        art = pathlib.Path(args.artifact)
        lines = [
            "# Gate-P0 metric verification (independent recomputation)",
            "",
            f"Run dir: `{run_dir}`  ",
            f"Tasks verified: {checked} (deterministic: sorted ids)  ",
            f"Verdict: **{verdict}**",
            "",
            "Leg 1: independent re-implementation (recompute(), written without "
            "run_bench import) of ctx_chars/ctx_tok, input/output, "
            "variable_input (base turn removed), tool_calls, turns. "
            "Leg 2: parser-vs-parser — deliberately imports run_bench.py to "
            "compare both implementations on identical raw input. "
            "wall_s cross-checked against .meta.json. "
            f"Agents verified: {', '.join(agents)} (incl. gate baseline).",
            "",
            f"Result: {summary}",
            "",
        ]
        if all_failures:
            lines += ["## Mismatches", ""] + [f"- {f}" for f in all_failures]
        art.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"artifact written: {art}")

    sys.exit(0 if not all_failures else 2)


if __name__ == "__main__":
    main()
