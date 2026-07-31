#!/usr/bin/env python3
"""Network-free standard-library tests for the agent coding harness."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import run_benchmark as bench


def git(cwd: pathlib.Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


class GitFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="agent-coding-test-")
        self.root = pathlib.Path(self.tempdir.name)
        self.source = self.root / "source"
        self.source.mkdir()
        git(self.source, "init", "-q")
        git(self.source, "config", "user.name", "Benchmark Test")
        git(self.source, "config", "user.email", "benchmark@example.invalid")
        (self.source / "value.txt").write_text("old\n", encoding="utf-8")
        (self.source / "test_guard.py").write_text(
            "import pathlib\nassert pathlib.Path('value.txt').read_text() == 'old\\n'\n",
            encoding="utf-8",
        )
        git(self.source, "add", "value.txt", "test_guard.py")
        git(self.source, "commit", "-qm", "fixture")
        self.commit = git(self.source, "rev-parse", "HEAD")
        self.backing = self.root / "repo.git"
        git(self.root, "clone", "--mirror", "--no-local", str(self.source), str(self.backing))
        git(self.root, "--git-dir", str(self.backing), "remote", "remove", "origin")

    def tearDown(self) -> None:
        self.tempdir.cleanup()


class PatchTests(GitFixture):
    PATCH = """diff --git a/value.txt b/value.txt
--- a/value.txt
+++ b/value.txt
@@ -1 +1 @@
-old
+new
"""

    def test_patch_applies_and_binary_diff_is_hashed(self) -> None:
        worktree_path = self.root / "patch-worktree"
        with bench.temporary_worktree(self.backing, self.commit, worktree_path, 10) as worktree:
            bench.apply_mutation(worktree, self.PATCH, 10)
            self.assertEqual((worktree / "value.txt").read_text(encoding="utf-8"), "new\n")
            (worktree / "asset.bin").write_bytes(b"\x00\x01\xff\x00")
            diff = bench.capture_binary_diff(worktree, self.commit, 10)
            self.assertIn(b"+new", diff)
            self.assertIn(b"GIT binary patch", diff)
            self.assertRegex(bench.sha256_bytes(diff), r"^[0-9a-f]{64}$")

    def test_invalid_patch_does_not_modify_worktree(self) -> None:
        worktree_path = self.root / "bad-patch-worktree"
        with bench.temporary_worktree(self.backing, self.commit, worktree_path, 10) as worktree:
            with self.assertRaises(bench.HarnessError):
                bench.apply_mutation(worktree, self.PATCH.replace("-old", "-missing"), 10)
            self.assertEqual((worktree / "value.txt").read_text(encoding="utf-8"), "old\n")


class WorktreeTests(GitFixture):
    def test_repository_clone_resolves_exact_pinned_commit(self) -> None:
        (self.source / "value.txt").write_text("gold\n", encoding="utf-8")
        git(self.source, "commit", "-qam", "gold solution")
        gold_commit = git(self.source, "rev-parse", "HEAD")
        clone_parent = self.root / "clone-parent"
        clone_parent.mkdir()
        task = {
            "repository": {"url": str(self.source), "commit": self.commit},
            "timeout_seconds": 10,
        }
        backing = bench.clone_pinned_repository(task, clone_parent)
        resolved = git(clone_parent, "--git-dir", str(backing), "rev-parse", "HEAD")
        self.assertEqual(resolved, self.commit)
        self.assertEqual(git(clone_parent, "--git-dir", str(backing), "remote"), "")
        self.assertEqual(git(clone_parent, "--git-dir", str(backing), "rev-list", "--all", "--count"), "1")
        hidden = subprocess.run(
            ["git", "--git-dir", str(backing), "cat-file", "-e", f"{gold_commit}^{{commit}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertNotEqual(hidden.returncode, 0)
        with bench.temporary_worktree(backing, self.commit, self.root / "sealed-worktree", 10) as worktree:
            self.assertEqual(git(worktree, "log", "--all", "--oneline").count("\n"), 0)
            self.assertEqual(git(worktree, "remote"), "")

    def test_worktree_is_removed_after_exception(self) -> None:
        worktree_path = self.root / "cleanup-worktree"
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with bench.temporary_worktree(self.backing, self.commit, worktree_path, 10):
                self.assertTrue(worktree_path.is_dir())
                raise RuntimeError("intentional")
        self.assertFalse(worktree_path.exists())
        listing = git(self.root, "--git-dir", str(self.backing), "worktree", "list", "--porcelain")
        self.assertNotIn(str(worktree_path), listing)


class SetupLifecycleTests(GitFixture):
    PATCH = """diff --git a/value.txt b/value.txt
--- a/value.txt
+++ b/value.txt
@@ -1 +1 @@
-old
+new
"""

    @staticmethod
    def value_test_command() -> list[str]:
        return [
            sys.executable,
            "-c",
            "import pathlib,sys; sys.exit(0 if pathlib.Path('value.txt').read_text() == 'old\\n' else 1)",
        ]

    def task(self, *, setup_commands: list[list[str]] | None = None) -> dict[str, object]:
        return {
            "id": "setup-lifecycle",
            "repository": {"url": str(self.source), "commit": self.commit},
            "setup_commands": setup_commands if setup_commands is not None else [],
            "mutation_patch": self.PATCH,
            "user_task": "Restore the old value.",
            "test_command": self.value_test_command(),
            "timeout_seconds": 10,
        }

    def test_setup_nonzero_is_redacted_recorded_and_fails(self) -> None:
        secret = "setup-output-secret"
        task = self.task(
            setup_commands=[
                [
                    sys.executable,
                    "-c",
                    "import os,sys; print(os.getenv('MINIMAX_API_KEY', 'api-missing')); "
                    "print(os.getenv('BENCH_OUTPUT_SECRET', 'output-missing')); sys.exit(7)",
                ]
            ]
        )
        raw_dir = self.root / "raw-setup-failure"
        with mock.patch.dict(
            os.environ,
            {"MINIMAX_API_KEY": "provider-key-must-not-leak", "BENCH_OUTPUT_SECRET": secret},
        ):
            with self.assertRaises(bench.SetupCommandError) as raised:
                bench.run_setup_commands(
                    task=task,
                    worktree=self.source,
                    store_dir=self.root / "failure-store",
                    raw_dir=raw_dir,
                    secrets=["provider-key-must-not-leak", secret],
                )
        summary = raised.exception.summary
        self.assertFalse(summary["success"])
        self.assertRegex(summary["summary_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(summary["commands"][0]["return_code"], 7)
        self.assertRegex(summary["commands"][0]["status_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(summary["commands"][0]["output_sha256"], r"^[0-9a-f]{64}$")
        output = (raw_dir / "setup-00.log").read_text(encoding="utf-8")
        self.assertIn("api-missing", output)
        self.assertIn("<redacted>", output)
        self.assertNotIn(secret, output)
        self.assertNotIn("provider-key-must-not-leak", output)
        failure_row = bench.sanitized_failure_row("sample", "explorer", raised.exception)
        self.assertEqual(failure_row["setup"], summary)

    def test_setup_timeout_is_recorded_and_fails(self) -> None:
        task = self.task(setup_commands=[[sys.executable, "-c", "pass"]])
        timed_out = bench.ProcessResult(None, b"partial", b"", 1.0, True)
        with mock.patch.object(bench, "run_process", return_value=timed_out):
            with self.assertRaises(bench.SetupCommandError) as raised:
                bench.run_setup_commands(
                    task=task,
                    worktree=self.source,
                    store_dir=self.root / "timeout-store",
                    raw_dir=self.root / "raw-setup-timeout",
                    secrets=[],
                )
        self.assertTrue(raised.exception.summary["commands"][0]["timed_out"])
        self.assertIsNone(raised.exception.summary["commands"][0]["return_code"])

    def test_preflight_requires_clean_pass_then_mutated_failure(self) -> None:
        task = self.task(setup_commands=[[sys.executable, "-c", "import pathlib; assert pathlib.Path('value.txt').read_text() == 'old\\n'"]])
        preflight = bench.run_mutation_preflight(
            task,
            self.backing,
            self.root / "valid-preflight",
            self.root / "raw-valid-preflight",
            [],
        )
        self.assertTrue(preflight["valid"])
        self.assertTrue(preflight["setup"]["success"])
        self.assertEqual(preflight["clean_test"]["return_code"], 0)
        self.assertNotEqual(preflight["mutated_test"]["return_code"], 0)
        self.assertRegex(preflight["mutation_diff_sha256"], r"^[0-9a-f]{64}$")

    def test_preflight_rejects_clean_source_test_failure_before_mutation(self) -> None:
        task = self.task()
        task["test_command"] = [sys.executable, "-c", "raise SystemExit(4)"]
        preflight = bench.run_mutation_preflight(
            task,
            self.backing,
            self.root / "invalid-preflight",
            self.root / "raw-invalid-preflight",
            [],
        )
        self.assertFalse(preflight["valid"])
        self.assertEqual(preflight["failure_kind"], "clean_source_test_failed")
        self.assertEqual(preflight["clean_test"]["return_code"], 4)
        self.assertIsNone(preflight["mutated_test"])
        self.assertNotIn("mutation_diff_sha256", preflight)

    def test_v2_preflight_records_infrastructure_classification_and_proof(self) -> None:
        task = self.task()
        task["task_bank"] = "v2"
        task["test_command"] = [
            sys.executable,
            "-c",
            "print(\"ModuleNotFoundError: No module named 'app'\"); raise SystemExit(1)",
        ]
        preflight = bench.run_mutation_preflight(
            task,
            self.backing,
            self.root / "v2-infra-preflight",
            self.root / "raw-v2-infra-preflight",
            [],
        )
        self.assertFalse(preflight["valid"])
        self.assertEqual(preflight["failure_kind"], "preflight_infra_failure")
        classification = preflight["patched_failure_classification"]
        self.assertEqual(classification["signature"], "import_error_without_collected_tests")
        self.assertIn("ModuleNotFoundError", classification["proof_line"])

    def test_arm_setup_precedes_mutation_and_is_excluded_from_agent_wall(self) -> None:
        setup = [
            sys.executable,
            "-c",
            "import pathlib,time; assert pathlib.Path('value.txt').read_text() == 'old\\n'; time.sleep(0.2)",
        ]
        task = self.task(setup_commands=[setup])
        preflight = bench.run_mutation_preflight(
            task,
            self.backing,
            self.root / "timing-preflight",
            self.root / "raw-timing-preflight",
            [],
        )
        self.assertTrue(preflight["valid"])

        agent_metrics = {
            "wall_seconds": 0.01,
            "success": True,
            "turns": 1,
            "reported_error": False,
            "tool_calls": 0,
            "source_opens": 0,
            "input_tokens": 1,
            "output_tokens": 1,
        }
        agent_process = bench.ProcessResult(0, b"", b"", 0.01, False)
        with mock.patch.object(bench, "run_pi_agent", return_value=(agent_metrics, agent_process)):
            row = bench.run_arm(
                arm="explorer",
                task=task,
                backing=self.backing,
                task_tmp=self.root / "timed-arm",
                raw_dir=self.root / "raw-timed-arm",
                pi_bin=pathlib.Path(sys.executable),
                greppy_bin=pathlib.Path(sys.executable),
                warm_greppy=False,
                expected_mutation_hash=preflight["mutation_diff_sha256"],
                secrets=["not-logged"],
            )
        self.assertTrue(row["setup"]["excluded_from_agent_wall"])
        self.assertGreaterEqual(row["setup"]["wall_seconds"], 0.15)
        self.assertEqual(row["agent"]["wall_seconds"], 0.01)
        self.assertTrue(row["valid"])

    def test_warmup_includes_the_gated_greppy_edit_arm(self) -> None:
        task = self.task()
        preflight = bench.run_mutation_preflight(
            task,
            self.backing,
            self.root / "warm-preflight",
            self.root / "raw-warm-preflight",
            [],
        )
        greppy = self.root / "fake-greppy"
        greppy.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        greppy.chmod(0o755)
        metrics = {
            "wall_seconds": 0.01,
            "success": True,
            "turns": 1,
            "reported_error": False,
            "tool_calls": 0,
            "source_opens": 0,
            "input_tokens": 1,
            "output_tokens": 1,
        }
        with mock.patch.object(
            bench,
            "run_pi_agent",
            return_value=(metrics, bench.ProcessResult(0, b"", b"", 0.01, False)),
        ):
            row = bench.run_arm(
                arm="greppy-edit",
                task=task,
                backing=self.backing,
                task_tmp=self.root / "warm-arm",
                raw_dir=self.root / "raw-warm-arm",
                pi_bin=pathlib.Path(sys.executable),
                greppy_bin=greppy,
                warm_greppy=True,
                expected_mutation_hash=preflight["mutation_diff_sha256"],
                secrets=[],
            )
        self.assertTrue(row["warmup"]["enabled"])
        self.assertEqual(row["warmup"]["return_code"], 0)

    def test_v2_arm_restores_agent_modified_test_before_final_run(self) -> None:
        test_patch = """diff --git a/test_guard.py b/test_guard.py
--- a/test_guard.py
+++ b/test_guard.py
@@ -1,2 +1,2 @@
 import pathlib
-assert pathlib.Path('value.txt').read_text() == 'old\\n'
+assert pathlib.Path('value.txt').read_text() == 'new\\n'
"""
        task = self.task()
        task.update(
            {
                "task_bank": "v2",
                "test_patch": test_patch,
                "mutation_patch": test_patch,
                "test_command": [sys.executable, "test_guard.py"],
            }
        )
        with bench.temporary_worktree(
            self.backing, self.commit, self.root / "reset-hash-worktree", 10
        ) as hash_worktree:
            bench.apply_mutation(hash_worktree, test_patch, 10)
            expected_hash = bench.sha256_bytes(
                bench.capture_binary_diff(hash_worktree, self.commit, 10)
            )

        def weaken_test(**kwargs):
            worktree = kwargs["worktree"]
            (worktree / "test_guard.py").write_text(
                "import pathlib\nassert pathlib.Path('value.txt').read_text() == 'old\\n'\n",
                encoding="utf-8",
            )
            metrics = {
                "wall_seconds": 0.01,
                "success": True,
                "turns": 1,
                "reported_error": False,
                "tool_calls": 1,
                "source_opens": 1,
                "input_tokens": 1,
                "output_tokens": 1,
            }
            return metrics, bench.ProcessResult(0, b"", b"", 0.01, False)

        with mock.patch.object(bench, "run_pi_agent", side_effect=weaken_test):
            row = bench.run_arm(
                arm="explorer",
                task=task,
                backing=self.backing,
                task_tmp=self.root / "reset-arm",
                raw_dir=self.root / "raw-reset-arm",
                pi_bin=pathlib.Path(sys.executable),
                greppy_bin=pathlib.Path(sys.executable),
                warm_greppy=False,
                expected_mutation_hash=expected_hash,
                secrets=[],
            )
        self.assertTrue(row["test_files_modified_by_agent"])
        self.assertEqual(row["test_files_modified_by_agent_paths"], ["test_guard.py"])
        self.assertFalse(row["correctness"])
        self.assertNotEqual(row["test"]["return_code"], 0)
        final_patch = (self.root / "raw-reset-arm" / "final.patch").read_text(encoding="utf-8")
        self.assertIn("+assert pathlib.Path('value.txt').read_text() == 'new", final_patch)


class V2PreflightClassificationTests(unittest.TestCase):
    def classify(self, command: list[str], output: str, paths: list[str] | None = None):
        return bench.classify_v2_patched_failure(command, output.encode(), paths or ["tests/test_feature.py"])

    def test_accepts_each_supported_framework_failure_signature(self) -> None:
        cases = [
            (["python3", "-m", "pytest", "-q"], "collected 2 items\ntest_case FAILED\n1 failed\n", "pytest_failed_with_collected_tests"),
            (["go", "test", "./..."], "--- FAIL: TestFeature (0.00s)\n", "go_test_fail_marker"),
            (["go", "test", "./..."], "FAIL\texample.test/pkg\t0.01s\n", "go_test_fail_marker"),
            (["cargo", "test"], "test result: FAILED. 0 passed; 1 failed\n", "cargo_test_failed_result"),
            (["pnpm", "exec", "vitest", "run"], "✕ rejects invalid input\n", "vitest_jest_failure_marker"),
            (["npx", "jest"], "Tests: 1 failed, 2 passed\n", "vitest_jest_failure_marker"),
            (["mvn", "test"], "Tests run: 4, Failures: 1, Errors: 0\n", "surefire_nonzero_failures"),
        ]
        for command, output, signature in cases:
            with self.subTest(signature=signature, output=output):
                classification = self.classify(command, output)
                self.assertEqual(classification["verdict"], "test_failure")
                self.assertEqual(classification["signature"], signature)
                self.assertTrue(classification["proof_line"])

    def test_pytest_requires_a_positive_collected_tests_line(self) -> None:
        classification = self.classify(
            ["python3", "-m", "pytest", "-q"],
            "FAILED tests/test_feature.py::test_case - assertion failed\n",
        )
        self.assertEqual(classification["verdict"], "preflight_infra_failure")
        self.assertEqual(classification["signature"], "no_framework_failure_evidence")

    def test_import_failure_without_collected_tests_is_infrastructure(self) -> None:
        classification = self.classify(
            ["python3", "-m", "pytest", "-q"],
            "ImportError while importing test module\nModuleNotFoundError: No module named 'app'\n",
        )
        self.assertEqual(classification["signature"], "import_error_without_collected_tests")
        self.assertIn("ImportError", classification["proof_line"])

    def test_patched_test_compile_error_is_a_legitimate_cargo_failure(self) -> None:
        classification = self.classify(
            ["cargo", "test"],
            "error[E0425]: missing value\n --> tests/test_feature.rs:8:5\nerror: could not compile `demo` (test)\n",
            ["tests/test_feature.rs"],
        )
        self.assertEqual(classification["verdict"], "test_failure")
        self.assertEqual(classification["signature"], "cargo_patched_test_compile_failure")
        self.assertIn("tests/test_feature.rs", classification["proof_line"])

    def test_production_compile_error_is_infrastructure(self) -> None:
        classification = self.classify(
            ["cargo", "test"],
            "error[E0425]: missing value\n --> src/lib.rs:8:5\nerror: could not compile `demo`\n",
            ["tests/test_feature.rs"],
        )
        self.assertEqual(classification["verdict"], "preflight_infra_failure")
        self.assertEqual(classification["signature"], "compile_failure_outside_patched_tests")

    def test_pure_infrastructure_signatures_are_rejected(self) -> None:
        cases = [
            ("pytest: command not found\n", "command_not_found"),
            (".venv/bin/python: No such file or directory\n", "missing_file_or_interpreter"),
            ("python: No module named pytest\n", "missing_python_module"),
            ("/usr/bin/python: bad interpreter\n", "bad_interpreter"),
            ("tool cannot execute: required file not found\n", "cannot_execute"),
            ("'pytest' is not recognized as an internal or external command\n", "windows_command_not_found"),
        ]
        for output, signature in cases:
            with self.subTest(signature=signature):
                classification = self.classify(["python3", "-m", "pytest"], output)
                self.assertEqual(classification["verdict"], "preflight_infra_failure")
                self.assertEqual(classification["signature"], signature)

    def test_spawn_error_is_rejected_with_proof(self) -> None:
        classification = bench.classify_v2_patched_failure(
            ["missing-pytest"], b"test process could not start: FileNotFoundError\n",
            ["tests/test_feature.py"], spawn_error=True,
        )
        self.assertEqual(classification["signature"], "test_process_spawn_error")
        self.assertIn("FileNotFoundError", classification["proof_line"])


def result_row(
    task_id: str,
    arm: str,
    *,
    passed: bool,
    tools: int,
    inputs: int,
    wall: float,
    source_opens: int = 1,
    valid: bool = True,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "arm": arm,
        "valid": valid,
        "correctness": passed,
        "agent": {
            "tool_calls": tools,
            "source_opens": source_opens,
            "input_tokens": inputs,
            "uncached_input_tokens": inputs,
            "output_tokens": inputs // 10,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "wall_seconds": wall,
            "edit_calls": 2,
            "post_edit_source_opens": 0,
        },
    }


class GradingTests(unittest.TestCase):
    def test_gate_requires_provider_cost_noninferiority(self) -> None:
        task_ids = [f"t{i}" for i in range(30)]
        rows: list[dict[str, object]] = []
        for task_id in task_ids:
            rows.extend(
                [
                    result_row(task_id, "explorer", passed=True, tools=10, source_opens=5, inputs=1000, wall=10),
                    result_row(task_id, "greppy", passed=True, tools=8, source_opens=4, inputs=800, wall=8),
                    result_row(task_id, "greppy-edit", passed=True, tools=8, source_opens=4, inputs=800, wall=8),
                ]
            )
        grade = bench.grade_results(rows, task_ids)
        self.assertTrue(grade["passed"])
        self.assertEqual(grade["cost_on_solved_pairs"]["greppy_edit_to_explorer_provider_cost"], 0.8)
        self.assertEqual(grade["cost_on_solved_pairs"]["threshold_ratio"], 0.80)
        # Token-Ratios bleiben Diagnose, keine Gate-Metrik
        self.assertEqual(grade["token_ratios_on_solved_pairs"]["greppy_to_explorer_input_tokens"], 0.8)
        self.assertFalse(grade["token_ratios_on_solved_pairs"]["is_gate_metric"])

    def test_gate_fails_when_greppy_costs_more_dollars(self) -> None:
        task_ids = [f"t{i}" for i in range(30)]
        rows: list[dict[str, object]] = []
        for task_id in task_ids:
            rows.extend(
                [
                    result_row(task_id, "explorer", passed=True, tools=10, source_opens=5, inputs=1000, wall=10),
                    result_row(task_id, "greppy", passed=True, tools=8, source_opens=4, inputs=1050, wall=8),
                    result_row(task_id, "greppy-edit", passed=True, tools=8, source_opens=4, inputs=1050, wall=8),
                ]
            )
        grade = bench.grade_results(rows, task_ids)
        self.assertGreater(grade["cost_on_solved_pairs"]["greppy_edit_to_explorer_provider_cost"], 1.0)
        self.assertFalse(grade["cost_on_solved_pairs"]["passes"])
        self.assertFalse(grade["passed"])

    def test_zero_edit_denominator_is_na_and_fails_gate(self) -> None:
        task_ids = [f"t{i}" for i in range(30)]
        rows: list[dict[str, object]] = []
        for task_id in task_ids:
            explorer = result_row(task_id, "explorer", passed=True, tools=10, inputs=1000, wall=10)
            candidate = result_row(task_id, "greppy-edit", passed=True, tools=8, inputs=700, wall=8)
            candidate["agent"]["edit_calls"] = 0
            rows.extend([explorer, candidate])
        grade = bench.grade_results(rows, task_ids)
        self.assertIsNone(grade["edit_loop"]["post_edit_reread_rate"])
        self.assertEqual(grade["edit_loop"]["denominator_status"], "not_applicable_no_greppy_edits")
        self.assertFalse(grade["edit_loop"]["passes"])
        self.assertFalse(grade["greppy_edit_adoption"]["passes"])
        self.assertFalse(grade["passed"])

    def test_gate_requires_edit_adoption_on_eighty_percent_of_solved_tasks(self) -> None:
        task_ids = [f"t{i}" for i in range(30)]
        rows: list[dict[str, object]] = []
        for index, task_id in enumerate(task_ids):
            explorer = result_row(task_id, "explorer", passed=True, tools=10, inputs=1000, wall=10)
            candidate = result_row(task_id, "greppy-edit", passed=True, tools=8, inputs=700, wall=8)
            candidate["agent"]["edit_calls"] = 1 if index < 23 else 0
            rows.extend([explorer, candidate])
        grade = bench.grade_results(rows, task_ids)
        self.assertEqual(grade["greppy_edit_adoption"]["adoption_rate"], round(23 / 30, 4))
        self.assertFalse(grade["greppy_edit_adoption"]["passes"])
        self.assertFalse(grade["passed"])

    def test_one_task_cannot_pass_the_benchmark(self) -> None:
        rows = [
            result_row("t1", "explorer", passed=True, tools=10, source_opens=5, inputs=1000, wall=10),
            result_row("t1", "greppy", passed=True, tools=1, source_opens=1, inputs=100, wall=1),
            result_row("t1", "greppy-edit", passed=True, tools=1, source_opens=1, inputs=100, wall=1),
        ]
        grade = bench.grade_results(rows, ["t1"])
        self.assertFalse(grade["sample_size"]["passes"])
        self.assertFalse(grade["passed"])

    def test_gate_requires_at_least_twenty_solved_pairs(self) -> None:
        task_ids = [f"t{i}" for i in range(30)]
        rows: list[dict[str, object]] = []
        for index, task_id in enumerate(task_ids):
            passed = index < 19
            rows.extend(
                [
                    result_row(task_id, "explorer", passed=passed, tools=10, source_opens=5, inputs=1000, wall=10),
                    result_row(task_id, "greppy", passed=passed, tools=8, source_opens=4, inputs=800, wall=8),
                    result_row(task_id, "greppy-edit", passed=passed, tools=8, source_opens=4, inputs=800, wall=8),
                ]
            )
        grade = bench.grade_results(rows, task_ids)
        self.assertEqual(grade["complete_pair_count"], 30)
        self.assertEqual(grade["solved_pair_count"], 19)
        self.assertFalse(grade["sample_size"]["passes"])
        self.assertFalse(grade["passed"])

    def test_exact_paired_test_detects_significant_regression(self) -> None:
        rows: list[dict[str, object]] = []
        task_ids = [f"t{i}" for i in range(5)]
        for task_id in task_ids:
            rows.extend(
                [
                    result_row(task_id, "explorer", passed=True, tools=2, inputs=100, wall=1),
                    result_row(task_id, "greppy", passed=False, tools=1, inputs=50, wall=0.1),
                    result_row(task_id, "greppy-edit", passed=False, tools=1, inputs=50, wall=0.1),
                ]
            )
        grade = bench.grade_results(rows, task_ids)
        self.assertFalse(grade["correctness"]["no_significant_regression"])
        self.assertEqual(grade["correctness"]["one_sided_exact_mcnemar_p"], 0.03125)
        self.assertFalse(grade["passed"])

    def test_observed_loss_fails_when_exact_alarm_is_not_significant(self) -> None:
        task_ids = [f"t{i}" for i in range(30)]
        rows: list[dict[str, object]] = []
        for index, task_id in enumerate(task_ids):
            rows.extend(
                [
                    result_row(task_id, "explorer", passed=True, tools=10, source_opens=5, inputs=1000, wall=10),
                    result_row(
                        task_id,
                        "greppy",
                        passed=index != 0,
                        tools=8,
                        source_opens=4,
                        inputs=800,
                        wall=8,
                    ),
                    result_row(
                        task_id,
                        "greppy-edit",
                        passed=index != 0,
                        tools=8,
                        source_opens=4,
                        inputs=800,
                        wall=8,
                    ),
                ]
            )
        grade = bench.grade_results(rows, task_ids)
        self.assertEqual(grade["correctness"]["one_sided_exact_mcnemar_p"], 0.5)
        self.assertTrue(grade["correctness"]["no_significant_regression"])
        self.assertFalse(grade["correctness"]["greppy_observed_correctness_not_lower"])
        self.assertFalse(grade["passed"])

    def test_failed_pair_never_receives_wall_time_credit(self) -> None:
        rows = [
            result_row("solved", "explorer", passed=True, tools=10, inputs=100, wall=1),
            result_row("solved", "greppy", passed=True, tools=8, inputs=100, wall=2),
            result_row("solved", "greppy-edit", passed=True, tools=8, inputs=100, wall=2),
            result_row("failed", "explorer", passed=True, tools=100, inputs=1000, wall=100),
            result_row("failed", "greppy", passed=False, tools=1, inputs=10, wall=0.01),
            result_row("failed", "greppy-edit", passed=False, tools=1, inputs=10, wall=0.01),
        ]
        grade = bench.grade_results(rows, ["solved", "failed"])
        self.assertEqual(grade["solved_pair_count"], 1)
        self.assertEqual(grade["wall_time_on_solved_pairs_only"]["credited_greppy_wins"], 0)
        self.assertFalse(grade["failed_tests_receive_speed_credit"])


class ContractTests(unittest.TestCase):
    def test_legacy_prompt_file_cannot_copy_the_command_vocabulary(self) -> None:
        prompt_path = bench.HERE / "prompts" / "greppy_system_v3.md"
        prompt_text = prompt_path.read_text(encoding="utf-8")
        self.assertNotIn("greppy ", prompt_text.lower())
        self.assertIn(bench.AGENTS_MD, bench.GREPPY_POLICY_TEMPLATE)
        self.assertIn(bench.AGENTS_MD, bench.GREPPY_EDIT_POLICY_TEMPLATE)

    def test_arm_validity_requires_success_even_when_turns_exist(self) -> None:
        self.assertFalse(bench.agent_result_is_valid({"success": False, "turns": 3, "timed_out": True}))
        self.assertFalse(bench.agent_result_is_valid({"success": False, "turns": 3, "return_code": 1}))
        self.assertTrue(bench.agent_result_is_valid({"success": True, "turns": 1, "return_code": 0}))

    def test_publishable_manifest_includes_platform_and_versions(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent-coding-manifest-") as tmp_name:
            root = pathlib.Path(tmp_name)
            executable = root / "fake-tool"
            executable.write_text("#!/bin/sh\nprintf 'fake-tool 1.2.3\\n'\n", encoding="utf-8")
            executable.chmod(0o755)
            task = {
                "id": "sample",
                "repository": {"url": "https://example.invalid/repo.git", "commit": "a" * 40},
                "setup_commands": [["python3", "-m", "pip", "install", "-e", "."]],
                "mutation_patch": "diff --git a/a b/a\n",
                "user_task": "Fix it.",
                "test_command": ["true"],
                "timeout_seconds": 60,
            }
            document = {"schema_version": bench.TASK_SCHEMA_VERSION, "tasks": [task]}
            task_path = root / "tasks.json"
            task_path.write_text(json.dumps(document), encoding="utf-8")
            manifest = bench.build_base_manifest(
                run_id="test-run",
                task_path=task_path,
                task_document=document,
                tasks=[task],
                pi_bin=executable,
                greppy_bin=executable,
                warm_greppy=False,
            )
            self.assertEqual(manifest["executables"]["pi"]["version"], "fake-tool 1.2.3")
            self.assertEqual(manifest["executables"]["greppy"]["version"], "fake-tool 1.2.3")
            self.assertRegex(manifest["greppy_source"]["git_commit"], r"^[0-9a-f]{40}$")
            self.assertIsInstance(manifest["greppy_source"]["tracked_worktree_dirty"], bool)
            self.assertIn("greppy_source", bench.RESUME_IDENTITY_FIELDS)
            self.assertTrue(manifest["platform"]["operating_system"])
            self.assertTrue(manifest["platform"]["architecture"])
            self.assertRegex(manifest["tasks"][0]["setup_commands_sha256"], r"^[0-9a-f]{64}$")
            self.assertIn("setup_contract", bench.RESUME_IDENTITY_FIELDS)
            prompt = manifest["prompt_contract"]
            self.assertEqual(prompt["shipped_agents_md_sha256"], bench.sha256_text(bench.AGENTS_MD))
            self.assertRegex(prompt["greppy_edit_full_system_sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue(prompt["identical_builtin_tool_palette_per_arm"])
            self.assertGreater(prompt["greppy_edit_minus_explorer_utf8_bytes"], 0)
            self.assertEqual(len(set(bench.ARM_TOOLS.values())), 1)
            self.assertIn("provider-dollar", manifest["gate_preregistration"]["efficiency"])
            self.assertIn("zero edit calls is N/A and fails", manifest["gate_preregistration"]["post_edit_rereads"])
            changed_setup = json.loads(json.dumps(manifest))
            changed_setup["tasks"][0]["setup_commands_sha256"] = "0" * 64
            with self.assertRaisesRegex(bench.HarnessError, "tasks"):
                bench.validate_resume_identity(manifest, changed_setup)

    def test_resume_rejects_changed_identity_and_duplicate_rows(self) -> None:
        current = {field: {"value": field} for field in bench.RESUME_IDENTITY_FIELDS}
        previous = json.loads(json.dumps(current))
        bench.validate_resume_identity(previous, current)
        previous["prompt_contract"] = {"value": "changed"}
        with self.assertRaisesRegex(bench.HarnessError, "prompt_contract"):
            bench.validate_resume_identity(previous, current)

        row = {
            "schema_version": bench.RESULT_SCHEMA_VERSION,
            "task_id": "sample",
            "arm": "explorer",
        }
        self.assertEqual(bench.validate_resume_rows([row], ["sample"]), [row])
        with self.assertRaisesRegex(bench.HarnessError, "duplicate"):
            bench.validate_resume_rows([row, dict(row)], ["sample"])
        with self.assertRaisesRegex(bench.HarnessError, "selected task set"):
            bench.validate_resume_rows([{**row, "task_id": "other"}], ["sample"])

    def test_schema_and_runtime_validator_agree_on_minimal_task(self) -> None:
        schema = json.loads((bench.HERE / "task.schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema_version"]["const"], bench.TASK_SCHEMA_VERSION)
        document = {
            "schema_version": bench.TASK_SCHEMA_VERSION,
            "tasks": [
                {
                    "id": "sample",
                    "repository": {"url": "/tmp/repo", "commit": "a" * 40},
                    "setup_commands": [],
                    "mutation_patch": "diff --git a/a b/a\n",
                    "user_task": "Fix the regression.",
                    "test_command": ["python3", "-m", "unittest"],
                    "timeout_seconds": 60,
                }
            ],
        }
        self.assertEqual(bench.validate_task_document(document)[0]["id"], "sample")
        setup_schema = schema["$defs"]["task"]["properties"]["setup_commands"]
        self.assertEqual(setup_schema["items"]["type"], "array")
        self.assertEqual(setup_schema["items"]["minItems"], 1)

    def test_setup_commands_reject_shell_strings_and_empty_argv(self) -> None:
        task = {
            "id": "sample",
            "repository": {"url": "/tmp/repo", "commit": "a" * 40},
            "setup_commands": ["python3 -m pip install -e ."],
            "mutation_patch": "diff --git a/a b/a\n",
            "user_task": "Fix the regression.",
            "test_command": ["python3", "-m", "unittest"],
            "timeout_seconds": 60,
        }
        document = {"schema_version": bench.TASK_SCHEMA_VERSION, "tasks": [task]}
        with self.assertRaisesRegex(bench.HarnessError, "non-empty argv array"):
            bench.validate_task_document(document)
        task["setup_commands"] = [[]]
        with self.assertRaisesRegex(bench.HarnessError, "non-empty argv array"):
            bench.validate_task_document(document)

    def test_secret_redaction_and_metric_parsing(self) -> None:
        secret = "sk-never-log-this"
        event = {
            "type": "turn_end",
            "toolResults": [{"content": [{"type": "text", "text": secret}]}],
            "message": {
                "usage": {"input": 100, "output": 20, "cacheRead": 10},
                "content": [
                    {"type": "toolCall", "name": "read", "arguments": {"path": "src/lib.rs"}}
                ],
            },
        }
        raw = (json.dumps(event) + "\n").encode()
        redacted = bench.redact(raw, [secret])
        self.assertNotIn(secret.encode(), redacted)
        metrics = bench.parse_pi_jsonl(redacted)
        self.assertEqual(metrics["input_tokens"], 110)
        self.assertEqual(metrics["uncached_input_tokens"], 100)
        self.assertEqual(metrics["output_tokens"], 20)
        self.assertEqual(metrics["tool_calls"], 1)
        self.assertEqual(metrics["source_opens"], 1)

    def test_current_greppy_read_and_edit_verbs_are_structurally_measured(self) -> None:
        commands = [
            "/opt/bench/greppy --root . read-file src/lib.rs --lines 1:20",
            "greppy --root . replace-text --expect 1 src/lib.rs old new",
            "greppy --root . read-file ./src/lib.rs",
            "greppy --root . rename src/model.rs::OldName NewName",
            "printf 'greppy replace fake'",  # text is not a command invocation
        ]
        events = []
        for command in commands:
            events.append(
                {
                    "type": "turn_end",
                    "toolResults": [{}],
                    "message": {
                        "usage": {},
                        "content": [{"type": "toolCall", "name": "bash", "arguments": {"command": command}}],
                    },
                }
            )
        metrics = bench.parse_pi_jsonl("\n".join(json.dumps(event) for event in events).encode())
        self.assertEqual(metrics["greppy_read_calls"], 2)
        self.assertEqual(metrics["edit_calls"], 2)
        self.assertEqual(metrics["post_edit_source_opens"], 1)
        self.assertEqual(metrics["source_opens"], 2)

    def test_every_shipped_edit_verb_is_recognised(self) -> None:
        commands = [
            "greppy --root . replace src/lib.rs::run body",
            "greppy --root . replace-text src/lib.rs old new",
            "greppy --root . replace-lines src/lib.rs 1:2 body",
            "greppy --root . replace-span handle body",
            "greppy --root . insert-lines src/lib.rs 3 body",
            "greppy --root . delete src/lib.rs::run",
            "greppy --root . delete-lines src/lib.rs 1:2",
            "greppy --root . patch change.diff",
            "greppy --root . write src/new.rs body",
            "greppy --root . rename src/lib.rs::Old New",
            "greppy --root . undo",
        ]
        events = [
            {
                "type": "turn_end",
                "toolResults": [{}],
                "message": {
                    "usage": {},
                    "content": [{"type": "toolCall", "name": "bash", "arguments": {"command": command}}],
                },
            }
            for command in commands
        ]
        metrics = bench.parse_pi_jsonl("\n".join(json.dumps(event) for event in events).encode())
        self.assertEqual(metrics["edit_calls"], len(bench.GREPPY_EDIT_VERBS))

    def test_command_sets_are_extracted_from_the_runtime_shipped_manual(self) -> None:
        extracted = bench._commands_from_shipped_manual(bench.AGENTS_MD)
        self.assertTrue(extracted["READ"])
        self.assertTrue(extracted["EDIT"])
        self.assertEqual(
            extracted["READ"], frozenset({"read", "read-smart", "read-file"})
        )
        self.assertEqual(
            extracted["EDIT"],
            frozenset({
                "replace", "replace-text", "replace-lines", "replace-span",
                "insert-lines", "delete", "delete-lines", "patch", "write",
                "rename", "undo",
            }),
        )

    def test_navigation_code_output_counts_as_source_open(self) -> None:
        event = {
            "type": "turn_end",
            "toolResults": [{}],
            "message": {
                "usage": {},
                "content": [{
                    "type": "toolCall", "name": "bash",
                    "arguments": {"command": "greppy --root . who-calls parse --code"},
                }],
            },
        }
        metrics = bench.parse_pi_jsonl((json.dumps(event) + "\n").encode())
        self.assertEqual(metrics["source_opens"], 1)
        self.assertEqual(metrics["greppy_read_calls"], 0)

    def test_balanced_arm_orders_are_exact_and_nav_arm_does_not_split_pair(self) -> None:
        for count, expected in ((41, (21, 20)), (144, (72, 72))):
            task_ids = [f"task-{index:03d}" for index in range(count)]
            orders = bench.balanced_arm_orders(list(reversed(task_ids)))
            explorer_first = sum(order.index("explorer") < order.index("greppy-edit") for order in orders.values())
            edit_first = count - explorer_first
            self.assertEqual((explorer_first, edit_first), expected)
            self.assertTrue(all(order[-1] == "greppy" for order in orders.values()))

    def test_builtin_edits_are_fallbacks_not_greppy_adoption(self) -> None:
        event = {
            "type": "turn_end",
            "toolResults": [{}],
            "message": {
                "usage": {},
                "content": [
                    {"type": "toolCall", "name": "edit", "arguments": {"path": "src/lib.rs"}},
                    {"type": "toolCall", "name": "read", "arguments": {"path": "src/lib.rs"}},
                ],
            },
        }
        metrics = bench.parse_pi_jsonl((json.dumps(event) + "\n").encode())
        self.assertEqual(metrics["fallback_edit_calls"], 1)
        self.assertEqual(metrics["edit_calls"], 0)
        self.assertEqual(metrics["post_edit_source_opens"], 1)

    def test_all_provider_retry_attempts_are_billed(self) -> None:
        failed = {
            "input_tokens": 100,
            "uncached_input_tokens": 100,
            "output_tokens": 10,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reported_error": True,
        }
        final = {
            "input_tokens": 200,
            "uncached_input_tokens": 200,
            "output_tokens": 20,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reported_error": False,
            "edit_calls": 1,
        }
        aggregate = bench.aggregate_provider_attempts(final, [failed, final])
        self.assertEqual(aggregate["provider_attempts"], 2)
        self.assertEqual(aggregate["uncached_input_tokens"], 300)
        self.assertEqual(aggregate["output_tokens"], 30)
        self.assertEqual(aggregate["edit_calls"], 1)
        self.assertAlmostEqual(
            aggregate["provider_cost_usd_all_attempts"],
            bench.provider_cost_usd(failed) + bench.provider_cost_usd(final),
        )


class TaskBankV2Tests(unittest.TestCase):
    def _v2_doc(self, **overrides):
        task = {
            "id": "hugo-reported-bugfix-abc123",
            "class": "S",
            "type": "reported-bugfix",
            "repository": {"url": "https://github.com/gohugoio/hugo", "commit": "a" * 40},
            "setup_commands": [],
            "test_patch": "diff --git a/x_test.go b/x_test.go\n+test\n",
            "user_task": "Fix the reported behavior.",
            "test_command": ["go", "test", "./..."],
            "timeout_seconds": 1800,
        }
        task.update(overrides)
        return {"schema_version": "greppy.agent-coding-tasks.v2", "tasks": [task]}

    def test_v2_document_validates_and_normalizes(self):
        tasks = bench.validate_task_document(self._v2_doc())
        self.assertEqual(tasks[0]["task_bank"], "v2")
        self.assertEqual(tasks[0]["mutation_patch"], tasks[0]["test_patch"])

    def test_v2_rejects_mutation_patch_field(self):
        doc = self._v2_doc()
        doc["tasks"][0]["mutation_patch"] = "x"
        with self.assertRaises(bench.HarnessError):
            bench.validate_task_document(doc)

    def test_v2_rewrites_stale_flask_venv_python(self):
        stale = "/private/tmp/whatever/validation-v2/venvs/flask/bin/python3"
        doc = self._v2_doc(test_command=[stale, "-m", "pytest", "-q"])
        tasks = bench.validate_task_document(doc)
        self.assertEqual(
            tasks[0]["test_command"][0], bench.FLASK_LOCAL_PYTHON
        )

    def test_v1_documents_still_validate(self):
        doc = self._v2_doc()
        doc["schema_version"] = "greppy.agent-coding-tasks.v1"
        t = doc["tasks"][0]
        del t["class"]; del t["type"]
        t["mutation_patch"] = t.pop("test_patch")
        tasks = bench.validate_task_document(doc)
        self.assertEqual(tasks[0]["task_bank"], "v1")

class AuditV2RunTests(unittest.TestCase):
    TEST_PATCH = (
        "diff --git a/pkg/x_test.go b/pkg/x_test.go\n"
        "index 0000000..1111111 100644\n"
        "--- a/pkg/x_test.go\n"
        "+++ b/pkg/x_test.go\n"
        "@@ -1,1 +1,2 @@\n"
        " package pkg\n"
        "+func TestNew(t *testing.T) { t.Fatal(\"discriminates\") }\n"
    )

    def _fixture(self, tmp, final_patches):
        import audit_v2_run as audit

        tasks_path = tmp / "tasks.json"
        tasks_path.write_text(json.dumps({
            "schema_version": bench.TASK_SCHEMA_VERSION_V2,
            "tasks": [{"id": task_id, "test_patch": self.TEST_PATCH}
                      for task_id in final_patches],
        }), encoding="utf-8")
        results_path = tmp / "results.json"
        results_path.write_text(json.dumps({
            "run_id": "audit-fixture",
            "results": [{"task_id": task_id, "arm": "greppy-edit"}
                        for task_id in final_patches],
        }), encoding="utf-8")
        raw_dir = tmp / "raw"
        for task_id, final_patch in final_patches.items():
            arm_dir = raw_dir / task_id / "greppy-edit"
            arm_dir.mkdir(parents=True)
            if final_patch is not None:
                (arm_dir / "final.patch").write_text(final_patch, encoding="utf-8")
        return audit.audit_run(
            results_path=results_path, raw_dir=raw_dir, task_path=tasks_path
        )

    def test_preserved_defused_and_missing_diffs_are_classified(self):
        defused = self.TEST_PATCH.replace(
            "+func TestNew(t *testing.T) { t.Fatal(\"discriminates\") }",
            "+func TestNew(t *testing.T) {}",
        )
        with tempfile.TemporaryDirectory() as tmp_name:
            report = self._fixture(pathlib.Path(tmp_name), {
                "task-clean": self.TEST_PATCH,
                "task-defused": defused,
                "task-lost": None,
            })
        rows = {row["task_id"]: row for row in report["runs"]}
        self.assertFalse(rows["task-clean"]["gaming_suspected"])
        self.assertEqual(rows["task-clean"]["reason"], "test_patch_preserved")
        self.assertTrue(rows["task-defused"]["gaming_suspected"])
        self.assertEqual(rows["task-defused"]["modified_test_paths"], ["pkg/x_test.go"])
        self.assertTrue(rows["task-lost"]["gaming_suspected"])
        self.assertEqual(rows["task-lost"]["reason"], "missing_final_diff")
        self.assertEqual(report["summary"]["gaming_suspected_runs"], 2)

    def test_index_lines_do_not_trigger_suspicion(self):
        reindexed = self.TEST_PATCH.replace(
            "index 0000000..1111111 100644", "index 2222222..3333333 100644"
        )
        with tempfile.TemporaryDirectory() as tmp_name:
            report = self._fixture(
                pathlib.Path(tmp_name), {"task-reindexed": reindexed}
            )
        self.assertFalse(report["runs"][0]["gaming_suspected"])


if __name__ == "__main__":
    unittest.main()
