#!/usr/bin/env python3
"""Offline evidence-integrity contracts; modeled JSON is not a browser fixture.

An explicit GREPPY_COMPARISON_EVIDENCE_DIR uses existing real, immutable
measurement records instead; only private copies are modified by these tests.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from analyze_web_comparison import InvalidComparison, analyze, latency_statistics


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def modeled_records(root):
    config = dict(rounds=1, tasks=1, concurrency=1, nodes=10)
    source = dict(harness_commit="a" * 40, harness_source_status="")
    plan = dict(schema="greppy.cli-native-paired-protocol.v1", status="planned",
                source=source["harness_commit"], cli_sha256="d" * 64,
                runtime_sha256=dict(old="e" * 64, current="f" * 64),
                configuration=config, schedule=[["old", "current"] for _ in range(6)],
                primary_metric="complete process_cold_total_ms including explicit owner stop",
                practical_effect_percent=5,
                scope="modeled offline records only; no browser execution")
    write_json(root / "protocol.json", plan)
    rows = []
    for pair, order in enumerate(plan["schedule"]):
        for position, label in enumerate(order):
            name = f"{pair}-{position}-{label}.json"
            owner = dict(running=False, owned=False, run_id=f"run_{pair}_{position}")
            wall = 100.0 if label == "old" else 90.0
            receipt = dict(
                schema="greppy.web-efficiency.measurement.v1", engine="cli",
                quality_verified=True, quality_error=None, configuration=config,
                host_envelope=dict(platform="modeled", logical_cpus=1,
                    cpu_affinity=None, sustained_cpu_capacity_upper_bound=1),
                executable_sha256=plan["cli_sha256"], source_provenance=source,
                workload_sha256="c" * 64, execution_surface="modeled CLI record",
                cold_scope="modeled record; no runtime",
                cli_provenance=dict(sha256=plan["cli_sha256"],
                    runtime_sha256=plan["runtime_sha256"][label],
                    private_state_directory=f"/modeled/{pair}-{position}"),
                native_entry_provenance=dict(adapter_sha256="a"*64,
                    generator_sha256="b"*64, workload_sha256="c"*64,
                    native_launch_policy={"version": "modeled"}),
                metrics=dict(exit_code=0, timed_out=False, cleanup_exit_code=0,
                    process_cold_total_ms=wall, cleanup_stdout_tail=json.dumps(
                        dict(schema="greppy.web-runtime.v1", status="ok", result=owner))),
                audit=dict(all_tasks_verified=True, error=None, scheduled_tasks=1,
                    reported_attempts=1, independently_verified_completed_tasks=1,
                    reported_failed_attempts=[], verified_task_durations_ms=[1]),
                workload={"rounds":[dict(round=0, attempted=1, succeeded=1,
                    tasks=[dict(round=0, worker=0, task=0, duration_ms=1, ok=True, error=None)])]})
            write_json(root / name, receipt)
            rows.append(dict(pair=pair, position=position, label=label, receipt=name,
                status="recorded", launcher_exit_code=0, new_state_count=1,
                owner_status=[owner], quality_verified=True, quality_error=None,
                receipt_sha256=digest(root/name), workload_sha256="c"*64,
                configuration=config, source=source, wall_ms=wall,
                cleanup_exit_code=0, independently_verified_tasks=1))
    write_json(root / "results.json", dict(protocol_sha256=digest(root/"protocol.json"),
        status="fixed_schedule_completed", runs=rows))


class ComparisonContract(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="comparison-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        evidence = os.environ.get("GREPPY_COMPARISON_EVIDENCE_DIR")
        if evidence:
            original = Path(evidence)
            result = json.loads((original/"results.json").read_text())
            names = {"protocol.json", "results.json"}
            names.update(Path(row["receipt"]).name for row in result["runs"])
            for name in names:
                shutil.copyfile(original/name, self.root/name)
        else:
            modeled_records(self.root)
        self.result_path = self.root/"results.json"
        self.result = json.loads(self.result_path.read_text())

    def audit(self):
        return analyze(self.root/"protocol.json", self.result_path, self.root)

    def first_receipt(self):
        return self.root/Path(self.result["runs"][0]["receipt"]).name

    def test_valid_records_never_invent_whole_agent_resource_coverage(self):
        report = self.audit()
        self.assertTrue(report["recorded_attempt_integrity_verified"])
        self.assertFalse(report["whole_agent_resource_efficiency_proved"])
        self.assertIsNone(report["whole_family_cpu_ram_comparison"])

    def test_missing_planned_attempt_cannot_be_dropped(self):
        self.result["runs"] = self.result["runs"][:-2]
        write_json(self.result_path, self.result)
        with self.assertRaises(InvalidComparison, msg="missing planned attempt was silently dropped"):
            self.audit()

    def test_raw_quality_failure_cannot_be_hidden_by_summary(self):
        path = self.first_receipt()
        receipt = json.loads(path.read_text())
        receipt["quality_verified"] = False
        write_json(path, receipt)
        self.result["runs"][0]["receipt_sha256"] = digest(path)
        write_json(self.result_path, self.result)
        with self.assertRaises(InvalidComparison, msg="raw failure was hidden by successful summary"):
            self.audit()

    def test_changed_receipt_bytes_require_the_recorded_hash(self):
        path = self.first_receipt()
        path.write_bytes(path.read_bytes() + b"\n ")
        with self.assertRaises(InvalidComparison, msg="changed receipt bytes were accepted"):
            self.audit()

    def test_independent_post_state_must_be_stopped(self):
        self.result["runs"][0]["owner_status"][0]["running"] = True
        write_json(self.result_path, self.result)
        with self.assertRaises(InvalidComparison, msg="live post-cleanup scope was accepted"):
            self.audit()


    def test_changed_visible_host_capacity_cannot_be_hidden(self):
        path = self.first_receipt()
        receipt = json.loads(path.read_text())
        receipt["host_envelope"]["logical_cpus"] += 1
        write_json(path, receipt)
        self.result["runs"][0]["receipt_sha256"] = digest(path)
        write_json(self.result_path, self.result)
        with self.assertRaises(InvalidComparison, msg="changed visible capacity was accepted"):
            self.audit()

    def test_cli_cannot_overwrite_input_evidence(self):
        before = digest(self.result_path)
        result = subprocess.run([sys.executable, "-B",
            str(Path(__file__).with_name("analyze_web_comparison.py")),
            "--protocol", str(self.root/"protocol.json"), "--results", str(self.result_path),
            "--receipt-root", str(self.root), "--output", str(self.result_path)],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 2, "analysis overwrote its input evidence")
        self.assertEqual(digest(self.result_path), before, "analysis overwrote its input evidence")


class StatisticalContract(unittest.TestCase):
    def test_exact_small_sample_inference_keeps_its_actual_coverage(self):
        six = latency_statistics([0.9] * 6)
        self.assertEqual(six["exact_randomization_p_two_sided_sharp_zero_effect"], 2 / 64)
        self.assertEqual(six["interval_coverage_under_iid_continuous_pairs"], 62 / 64)
        self.assertEqual(six["paired_median_ratio_interval"], [0.9, 0.9])
        five = latency_statistics([0.9] * 5)
        self.assertIsNone(five["paired_median_ratio_interval"])
        self.assertEqual(five["conclusion"], "inconclusive")


if __name__ == "__main__":
    unittest.main(verbosity=2)
