#!/usr/bin/env python3
"""Prove offline comparison guards reject omissions, forged summaries and leaks."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "bench/analyze_web_comparison.py"
TEST = "bench/test_web_comparison.py"
CASES = [
    ('require(len(rows) == len(expected), "missing or extra planned attempt")',
     'require(True, "missing or extra planned attempt")',
     "test_missing_planned_attempt_cannot_be_dropped",
     "missing planned attempt was silently dropped"),
    ('and receipt["quality_verified"] is True and receipt["quality_error"] is None',
     'and True and receipt["quality_error"] is None',
     "test_raw_quality_failure_cannot_be_hidden_by_summary",
     "raw failure was hidden by successful summary"),
    ('require(digest == row["receipt_sha256"], "receipt hash mismatch")',
     'require(True, "receipt hash mismatch")',
     "test_changed_receipt_bytes_require_the_recorded_hash",
     "changed receipt bytes were accepted"),
    ('and owners[0]["running"] is False and owners[0]["owned"] is False',
     'and True',
     "test_independent_post_state_must_be_stopped",
     "live post-cleanup scope was accepted"),
    ('"host_snapshot": visible_host_snapshot(receipt["host_envelope"]),',
     '"host_snapshot": None,',
     "test_changed_visible_host_capacity_cannot_be_hidden",
     "changed visible capacity was accepted"),
    ('require(math.isfinite(value), "non-finite JSON number")',
     'require(True, "non-finite JSON number")',
     "test_overflow_host_capacity_cannot_be_accepted",
     "overflow host capacity was accepted"),
    ('with args.output.open("x") as stream:',
     'with args.output.open("w") as stream:',
     "test_cli_cannot_overwrite_input_evidence",
     "analysis overwrote its input evidence"),
]


CASES += [
    [
        "receipt[\"executable_sha256\"] == provenance[\"sha256\"] == cli_hashes[identity[2]]",
        "receipt[\"executable_sha256\"] == provenance[\"sha256\"] == provenance[\"sha256\"]",
        "CoherentComparisonContract.test_coherent_pair_binds_cli_per_version_and_keeps_resource_units",
        "wrong CLI version was accepted"
    ],
    [
        "\"qualified\": not errors",
        "\"qualified\": True",
        "CoherentComparisonContract.test_missing_memory_does_not_drop_pairs_or_invalidate_wall_and_cpu",
        "incomplete memory was qualified"
    ],
    [
        "require(sorted((task[\"worker\"], task[\"task\"]) for task in tasks)\n                == [(worker, task) for worker in range(config[\"concurrency\"])\n                    for task in range(config[\"tasks\"])],\n                \"duplicate or missing worker/task identity\")",
        "require(True, \"duplicate or missing worker/task identity\")",
        "CoherentComparisonContract.test_duplicate_worker_cannot_replace_a_planned_worker",
        "missing concurrent worker was accepted"
    ]
]


def run(case=None):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    # Same-length source faults cannot be masked by bytecode from another case.
    with tempfile.TemporaryDirectory(prefix="comparison-bytecode-") as cache:
        env["PYTHONPYCACHEPREFIX"] = cache
        command = [sys.executable, "-B", "-X", "pycache_prefix=" + cache, TEST]
        if case:
            command.append(case if "." in case else "ComparisonContract." + case)
        return subprocess.run(command, cwd=ROOT, env=env, text=True,
                              capture_output=True, timeout=30)


def positive():
    result = run()
    if result.returncode != 0:
        raise AssertionError("comparison contract failed:\n" + result.stdout + result.stderr)


def main():
    positive()
    original = SOURCE.read_text()
    for old, fault, case, marker in CASES:
        if original.count(old) != 1:
            raise AssertionError("comparison source target is not unique")
        try:
            SOURCE.write_text(original.replace(old, fault))
            result = run(case)
            if result.returncode != 1 or marker not in result.stdout + result.stderr:
                raise AssertionError("wrong comparison counter failure:\n" +
                                     result.stdout + result.stderr)
            print("mutation: " + marker + " rejected by contract", flush=True)
        finally:
            SOURCE.write_text(original)
        positive()
    subprocess.run(["git", "diff", "--exit-code", "--", str(SOURCE)],
                   cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
