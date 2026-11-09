#!/usr/bin/env python3
"""Audit a fixed CLI/native comparison without launching any measured work.

python3 bench/analyze_web_comparison.py --protocol protocol.json --results results.json \
    --receipt-root receipts/ --output analysis.json

This verifies consistency of recorded evidence, not its independent authenticity,
registration time, host reservation, actual runtime image, or global efficiency.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_EXACT_PAIRS = 20


class InvalidComparison(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise InvalidComparison(reason)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON member")
        result[key] = value
    return result


def _bad_constant(_value):
    raise InvalidComparison("non-finite JSON constant")


def _finite_float(value):
    value = float(value)
    require(math.isfinite(value), "non-finite JSON number")
    return value


def load_json(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    require(len(raw) <= MAX_JSON_BYTES, "JSON document exceeds bounded input size")
    return json.loads(raw, object_pairs_hook=_unique_object,
                      parse_constant=_bad_constant, parse_float=_finite_float), hashlib.sha256(raw).hexdigest()


def positive(value):
    return type(value) in (float, int) and math.isfinite(value) and value > 0


def integer(value, expected):
    return type(value) is int and value == expected


def visible_host_snapshot(envelope):
    # End-of-run observations cannot establish stable or reserved capacity.
    snapshot = {key: envelope.get(key) for key in (
        "platform", "logical_cpus", "cpu_affinity", "sustained_cpu_capacity_upper_bound")}
    visible = envelope.get("cgroup_v2_observation") or {}
    snapshot.update({key: visible.get(key) for key in (
        "cpu_quota_upper_bound", "cpu_quota_upper_bound_ratio",
        "memory_max_upper_bound_bytes", "swap_max_upper_bound_bytes")})
    return snapshot


def latency_statistics(ratios):
    """Conditional exact paired sign-flip test; exact order-statistic coverage."""
    require(1 <= len(ratios) <= MAX_EXACT_PAIRS and all(map(positive, ratios)),
            "invalid bounded ratio series")
    logs = [math.log(value) for value in ratios]
    observed = abs(math.fsum(logs))
    extreme = sum(
        abs(math.fsum(sign * value for sign, value in zip(signs, logs))) >= observed - 1e-12
        for signs in itertools.product((-1, 1), repeat=len(logs))
    )
    ordered = sorted(ratios)
    size = len(ordered)
    interval, coverage = None, None
    for rank in range(1, (size + 1) // 2 + 1):
        candidate_coverage = 1 - 2 * sum(math.comb(size, i) for i in range(rank)) / 2 ** size
        if candidate_coverage >= 0.95:
            interval = [ordered[rank - 1], ordered[size - rank]]
            coverage = candidate_coverage
    p_value = extreme / 2 ** size
    conclusion = "inconclusive"
    if interval is not None and p_value <= 0.05:
        if interval[1] < 1:
            conclusion = "conditional evidence of lower latency"
        elif interval[0] > 1:
            conclusion = "conditional evidence of higher latency"
    return {
        "paired_median_candidate_over_control": statistics.median(ratios),
        "paired_median_change_percent": 100 * (statistics.median(ratios) - 1),
        "paired_median_ratio_interval": interval,
        "interval_coverage_under_iid_continuous_pairs": coverage,
        "exact_randomization_p_two_sided_sharp_zero_effect": p_value,
        "randomization_test_statistic": "abs(sum(log(candidate_i / control_i)))",
        "observed_randomization_test_statistic": observed,
        "randomization_null_distribution": "all 2^n paired sign changes; absolute statistic >= observed, tolerance 1e-12",
        "conclusion": conclusion,
    }


def validate_tasks(receipt, config):
    audit, workload = receipt["audit"], receipt["workload"]
    count = config["rounds"] * config["tasks"] * config["concurrency"]
    require(audit["all_tasks_verified"] is True and audit["error"] is None
            and integer(audit["scheduled_tasks"], count)
            and integer(audit["reported_attempts"], count)
            and integer(audit["independently_verified_completed_tasks"], count)
            and audit["reported_failed_attempts"] == [], "incomplete independent task audit")
    rounds = workload["rounds"]
    require(len(rounds) == config["rounds"], "missing or extra workload round")
    durations = []
    for index, row in enumerate(rounds):
        tasks = row["tasks"]
        round_count = config["tasks"] * config["concurrency"]
        require(type(row["round"]) is int and row["round"] == index
                and row["attempted"] == round_count
                and row["succeeded"] == round_count
                and len(tasks) == round_count, "incomplete workload round")
        require(sorted((task["worker"], task["task"]) for task in tasks)
                == [(worker, task) for worker in range(config["concurrency"])
                    for task in range(config["tasks"])],
                "duplicate or missing worker/task identity")
        for task in tasks:
            require(type(task["round"]) is int and task["round"] == index
                    and type(task["task"]) is int
                    and type(task["worker"]) is int
                    and 0 <= task["worker"] < config["concurrency"]
                    and task["ok"] is True and task["error"] is None
                    and type(task["duration_ms"]) in (int, float)
                    and math.isfinite(task["duration_ms"]) and task["duration_ms"] >= 0,
                    "failed or malformed task hidden by summary")
            durations.append(task["duration_ms"])
    require(sorted(audit["verified_task_durations_ms"]) == sorted(durations),
            "audit and workload durations disagree")


def kernel_resource_statistics(attempts, control, candidate):
    """Keep kernel charge observations separate from sampled PSS and wall time."""
    metrics = (
        ("cpu_usage_seconds", "owned CPU time"),
        ("memory_peak_bytes", "owned memory charge peak"),
    )
    comparisons = {}
    for metric, description in metrics:
        errors, values = [], {}
        for identity, kernel in attempts:
            reason = None
            if not isinstance(kernel, dict):
                reason = "owned kernel accounting missing"
            elif not (kernel.get("schema") == "greppy.kernel-resource-group.v1"
                      and kernel.get("fresh_group") is True
                      and kernel.get("populated") is False
                      and kernel.get("target_entry_observed") is True
                      and kernel.get("group_removed") is True
                      and isinstance(kernel.get("memory_events"), dict)
                      and type(kernel["memory_events"].get("oom_kill")) is int
                      and kernel["memory_events"]["oom_kill"] == 0):
                reason = "fresh completed owned group without OOM not verified"
            value = kernel.get(metric) if isinstance(kernel, dict) else None
            if reason is None:
                valid = (type(value) is int and value > 0 if metric == "memory_peak_bytes"
                         else type(value) in (int, float) and math.isfinite(value) and value > 0)
                if not valid:
                    reason = "positive kernel metric missing or invalid"
            if reason is not None:
                errors.append({"pair": identity[0], "position": identity[1],
                               "label": identity[2], "reason": reason})
            else:
                values.setdefault(identity[0], {})[identity[2]] = value
        comparison = {"qualified": not errors, "unqualified_attempts": errors,
                      "planned_attempts": len(attempts), "pairs": [], "statistics": None}
        if not errors:
            pairs = [{"pair": pair, "control": value[control], "candidate": value[candidate],
                      "ratio": value[candidate] / value[control]}
                     for pair, value in values.items()]
            stats = latency_statistics([pair["ratio"] for pair in pairs])
            stats["conclusion"] = stats["conclusion"].replace("latency", description)
            comparison.update(pairs=pairs, statistics=stats)
        comparisons[metric] = comparison
    return {"scope": "fresh owned cgroup CPU and memory charges through exit and cleanup; "
                     "includes charged kernel/cache and descendants; shared or preexisting pages "
                     "may be charged elsewhere; not sampled PSS or all physical RAM",
            "comparisons": comparisons}


def analyze(protocol_path, results_path, receipt_root, control="old", candidate="current"):
    plan, plan_sha = load_json(protocol_path)
    result, result_sha = load_json(results_path)
    require(plan["schema"] in ("greppy.cli-native-paired-protocol.v1",
                               "greppy.cli-native-paired-protocol.v2")
            and plan["status"] == "planned", "unsupported fixed protocol")
    coherent = plan["schema"] == "greppy.cli-native-paired-protocol.v2"
    require(result["protocol_sha256"] == plan_sha, "protocol hash mismatch")
    require(result["status"] == "fixed_schedule_completed", "comparison did not complete")
    require(plan["primary_metric"] == "complete process_cold_total_ms including explicit owner stop",
            "unsupported preregistered primary metric")
    practical = plan.get("practical_effect_percent")
    require(practical is None or (type(practical) in (int, float)
            and math.isfinite(practical) and 0 <= practical < 100),
            "invalid declared practical effect threshold")
    labels = {control, candidate}
    require(len(labels) == 2 and set(plan["runtime_sha256"]) == labels,
            "comparison requires two declared versions")
    if coherent:
        cli_hashes = plan["cli_sha256"]
        require(type(cli_hashes) is dict and set(cli_hashes) == labels
                and all(type(value) is str and len(value) == 64
                        and set(value) <= set("0123456789abcdef")
                        for value in cli_hashes.values()), "invalid version-bound CLI hashes")
    else:
        cli_hashes = {label: plan["cli_sha256"] for label in labels}
    schedule = plan["schedule"]
    require(1 <= len(schedule) <= MAX_EXACT_PAIRS, "exact comparison pair count exceeds bound")
    require(all(type(order) is list and len(order) == 2 and set(order) == labels
                for order in schedule), "invalid paired schedule")
    expected = [(pair, position, label) for pair, order in enumerate(schedule)
                for position, label in enumerate(order)]
    rows = result["runs"]
    require(len(rows) == len(expected), "missing or extra planned attempt")
    config = {key: plan["configuration"][key] for key in ("rounds", "tasks", "concurrency", "nodes")}
    require(all(type(value) is int and value > 0 for value in config.values()),
            "invalid workload configuration")
    root = Path(receipt_root).resolve()
    pairs, states, receipt_names, common = {}, set(), set(), None
    inputs, resource_attempts = [], []
    for row, identity in zip(rows, expected):
        require(type(row["pair"]) is int and type(row["position"]) is int
                and (row["pair"], row["position"], row["label"]) == identity,
                "planned attempt order or identity mismatch")
        require(row["status"] == "recorded" and integer(row["launcher_exit_code"], 0)
                and row["quality_verified"] is True and row["quality_error"] is None
                and integer(row["cleanup_exit_code"], 0)
                and not row.get("cleanup_recovery_required", False),
                "failed or recovered attempt cannot become success-only speedup")
        owners = row["owner_status"]
        require(integer(row["new_state_count"], 1) and len(owners) == 1
                and owners[0]["running"] is False and owners[0]["owned"] is False,
                "official post-cleanup state is incomplete")
        name = Path(row["receipt"]).name
        require(name.endswith(".json") and name not in receipt_names,
                "duplicate or invalid receipt name")
        path = root / name
        require(not path.is_symlink() and path.resolve().is_relative_to(root)
                and path.is_file(), "receipt escapes explicit evidence directory")
        receipt_names.add(name)
        receipt, digest = load_json(path)
        require(digest == row["receipt_sha256"], "receipt hash mismatch")
        require(receipt["schema"] == "greppy.web-efficiency.measurement.v1"
                and receipt["engine"] == "cli"
                and receipt["quality_verified"] is True and receipt["quality_error"] is None,
                "unverified raw measurement")
        metrics, provenance = receipt["metrics"], receipt["cli_provenance"]
        require(receipt["configuration"] == row["configuration"] == config,
                "workload configuration changed")
        require(receipt["executable_sha256"] == provenance["sha256"] == cli_hashes[identity[2]]
                and provenance["runtime_sha256"] == plan["runtime_sha256"][identity[2]],
                "configured executable identity mismatch")
        require(receipt["source_provenance"] == row["source"]
                and row["source"]["harness_commit"] == plan["source"]
                and row["source"]["harness_source_status"] == "", "measurement source changed")
        require(integer(metrics["exit_code"], 0) and metrics["timed_out"] is False
                and integer(metrics["cleanup_exit_code"], 0), "raw execution or cleanup failed")
        require(positive(metrics["process_cold_total_ms"])
                and row["wall_ms"] == metrics["process_cold_total_ms"],
                "invalid or inconsistent complete process time")
        validate_tasks(receipt, config)
        require(row["independently_verified_tasks"] == (
                    config["rounds"] * config["tasks"] * config["concurrency"]),
                "summary task count disagrees")
        cleanup = json.loads(metrics["cleanup_stdout_tail"])
        require(cleanup["schema"] == "greppy.web-runtime.v1" and cleanup["status"] == "ok"
                and cleanup["result"]["running"] is False
                and cleanup["result"]["owned"] is False
                and cleanup["result"]["run_id"] == owners[0]["run_id"],
                "raw stop and independent post-state disagree")
        state = provenance["private_state_directory"]
        require(state not in states, "fresh process context was reused")
        states.add(state)
        entry = receipt["native_entry_provenance"]
        invariant = {
            "host_snapshot": visible_host_snapshot(receipt["host_envelope"]),
            "workload": receipt["workload_sha256"],
            "adapter": entry["adapter_sha256"], "generator": entry["generator_sha256"],
            "launch": entry["native_launch_policy"], "surface": receipt["execution_surface"],
            "cold_scope": receipt["cold_scope"],
        }
        require(row["workload_sha256"] == entry["workload_sha256"] == invariant["workload"],
                "workload provenance disagrees")
        if common is None:
            common = invariant
        require(common == invariant, "visible host capacity, workload or execution surface changed across versions")
        pairs.setdefault(identity[0], {})[identity[2]] = metrics["process_cold_total_ms"]
        resource_attempts.append((identity, metrics.get("kernel_resource_accounting")))
        inputs.append({"pair": identity[0], "position": identity[1], "label": identity[2],
                       "receipt_sha256": digest})
    ratios = [pair[candidate] / pair[control] for pair in pairs.values()]
    stats = latency_statistics(ratios)
    practical_supported = (practical is not None and stats["paired_median_ratio_interval"] is not None
        and stats["paired_median_ratio_interval"][1] <= 1 - practical / 100
        and stats["exact_randomization_p_two_sided_sharp_zero_effect"] <= 0.05)
    report = {
        "schema": "greppy.cli-native-paired-audit.v1",
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": plan_sha, "results_sha256": result_sha,
        "input_receipts": inputs, "control": control, "candidate": candidate,
        "independent_unit": "fixed process pairs; rounds/tasks are not independent repetitions",
        "recorded_attempt_integrity_verified": True,
        "visible_host_snapshot": common["host_snapshot"],
        "pairs": [{"pair": index, "control_ms": pair[control], "candidate_ms": pair[candidate],
                   "ratio": ratios[index]} for index, pair in pairs.items()],
        **stats,
        "declared_practical_effect_percent": practical,
        "declared_practical_latency_effect_supported_under_assumptions": practical_supported,
        "assumptions": "sign-flip inference assumes independent randomized within-pair order, sharp zero effect and no version interference; median interval also assumes iid continuous pair effects",
        "limitations": "offline consistency only; no proof of preregistration time, randomization generation, authenticity, host reservation or interval-stable resource limits; configured hashes are not observed runtime-image proof",
        "whole_agent_resource_efficiency_proved": False,
        "whole_family_cpu_ram_comparison": None,
        "scope": plan["scope"],
    }
    if coherent:
        report["schema"] = "greppy.cli-native-paired-audit.v2"
        report["cli_sha256_by_version"] = cli_hashes
        report["configuration"] = config
        report["verified_tasks_per_attempt"] = (
            config["rounds"] * config["tasks"] * config["concurrency"])
        report["owned_kernel_resource_comparison"] = kernel_resource_statistics(
            resource_attempts, control, candidate)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--receipt-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--control", default="old")
    parser.add_argument("--candidate", default="current")
    args = parser.parse_args()
    try:
        report = analyze(args.protocol, args.results, args.receipt_root,
                         args.control, args.candidate)
    except InvalidComparison as error:
        parser.exit(2, "invalid comparison: " + str(error) + "\n")
    except (OSError, KeyError, TypeError, ValueError, OverflowError):
        parser.exit(2, "invalid comparison: incomplete, inconsistent or unsupported recorded evidence\n")
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        try:
            with args.output.open("x") as stream:
                stream.write(encoded)
        except OSError:
            parser.exit(2, "comparison output must be a new writable file; inputs are preserved\n")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
