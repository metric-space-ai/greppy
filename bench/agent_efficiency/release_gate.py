#!/usr/bin/env python3
"""Evaluate the pre-registered Greppy v0.2 agent-efficiency release gates."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any


def quality(row: dict[str, Any], agent: str) -> float | None:
    result = row.get(agent)
    if not isinstance(result, dict):
        return None
    evidence = result.get("quality")
    if isinstance(evidence, dict) and evidence.get("accepted_for_speed_claim") is True:
        value = evidence.get("score")
        return float(value) if value is not None else None
    if result.get("correct") is not None:
        return 1.0 if result["correct"] else 0.0
    return None


def one_sided_loss_probability(losses: int, wins: int) -> float:
    """Exact P(X >= losses), X~Binomial(losses+wins, 0.5)."""
    total = losses + wins
    if total == 0 or losses <= wins:
        return 1.0
    numerator = sum(math.comb(total, k) for k in range(losses, total + 1))
    return numerator / (2**total)


def metric_ratio(
    rows: list[dict[str, Any]], baseline: str, candidate: str, metric: str
) -> tuple[float | None, int]:
    base_total = 0.0
    candidate_total = 0.0
    count = 0
    for row in rows:
        base = row.get(baseline)
        cand = row.get(candidate)
        if not isinstance(base, dict) or not isinstance(cand, dict):
            continue
        left, right = base.get(metric), cand.get(metric)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            continue
        base_total += float(left)
        candidate_total += float(right)
        count += 1
    if count == 0 or base_total <= 0:
        return None, count
    return candidate_total / base_total, count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--baseline", default="explorer")
    parser.add_argument("--candidate", default="greppy")
    args = parser.parse_args()

    rows = json.loads(args.results.read_text(encoding="utf-8"))
    comparable = [
        row
        for row in rows
        if isinstance(row.get(args.baseline), dict)
        and isinstance(row.get(args.candidate), dict)
    ]
    missing_quality = 0
    losses = wins = ties = 0
    for row in comparable:
        base = quality(row, args.baseline)
        cand = quality(row, args.candidate)
        if base is None or cand is None:
            missing_quality += 1
        elif cand < base:
            losses += 1
        elif cand > base:
            wins += 1
        else:
            ties += 1

    structural = [row for row in comparable if row.get("type") == "locate"]
    tool_ratio, tool_n = metric_ratio(
        structural, args.baseline, args.candidate, "tool_calls"
    )
    open_ratio, open_n = metric_ratio(
        structural, args.baseline, args.candidate, "source_open_calls"
    )
    input_ratio, input_n = metric_ratio(
        structural, args.baseline, args.candidate, "variable_input"
    )
    loss_p = one_sided_loss_probability(losses, wins)
    observed_not_lower = wins >= losses
    checks = {
        "all_rows_have_accepted_quality": missing_quality == 0 and bool(comparable),
        "candidate_observed_correctness_not_lower": observed_not_lower,
        "no_significant_correctness_regression": loss_p >= 0.05,
        "tool_calls_at_least_20_percent_lower": tool_ratio is not None and tool_ratio <= 0.80,
        "source_opens_at_least_20_percent_lower": open_ratio is not None and open_ratio <= 0.80,
        "structural_variable_input_at_least_20_percent_lower": input_ratio is not None
        and input_ratio <= 0.80,
    }
    report = {
        "schema_version": "greppy.agent-release-gate.v2",
        "baseline": args.baseline,
        "candidate": args.candidate,
        "comparable_rows": len(comparable),
        "structural_rows": len(structural),
        "quality": {
            "missing": missing_quality,
            "candidate_wins": wins,
            "candidate_losses": losses,
            "ties": ties,
            "candidate_observed_correctness_not_lower": observed_not_lower,
            "one_sided_regression_p": loss_p,
        },
        "ratios_candidate_over_baseline": {
            "tool_calls": {"ratio": tool_ratio, "rows": tool_n},
            "source_open_calls": {"ratio": open_ratio, "rows": open_n},
            "variable_input": {"ratio": input_ratio, "rows": input_n},
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
