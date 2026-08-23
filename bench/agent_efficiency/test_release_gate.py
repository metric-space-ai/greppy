import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import release_gate
import run_bench
import verify_metrics


def result(correct: bool, *, tools: int, opens: int, variable_input: int) -> dict:
    return {
        "correct": correct,
        "tool_calls": tools,
        "source_open_calls": opens,
        "variable_input": variable_input,
    }


class ReleaseGateTests(unittest.TestCase):
    def evaluate(self, rows: list[dict]) -> tuple[int, dict]:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "results.json"
            output = root / "gate.json"
            source.write_text(json.dumps(rows), encoding="utf-8")
            with mock.patch.object(
                sys,
                "argv",
                [
                    "release_gate.py",
                    "--results",
                    str(source),
                    "--output",
                    str(output),
                ],
            ):
                status = release_gate.main()
            return status, json.loads(output.read_text(encoding="utf-8"))

    def test_observed_loss_fails_even_when_exact_alarm_is_not_significant(self) -> None:
        status, report = self.evaluate(
            [
                {
                    "id": "t1",
                    "type": "locate",
                    "explorer": result(True, tools=10, opens=5, variable_input=1000),
                    "greppy": result(False, tools=8, opens=4, variable_input=800),
                }
            ]
        )
        self.assertEqual(status, 2)
        self.assertEqual(report["quality"]["one_sided_regression_p"], 0.5)
        self.assertFalse(report["checks"]["candidate_observed_correctness_not_lower"])

    def test_equal_observed_correctness_and_twenty_percent_savings_pass(self) -> None:
        rows = [
            {
                "id": f"t{index}",
                "type": "locate",
                "explorer": result(True, tools=10, opens=5, variable_input=1000),
                "greppy": result(True, tools=8, opens=4, variable_input=800),
            }
            for index in range(4)
        ]
        status, report = self.evaluate(rows)
        self.assertEqual(status, 0)
        self.assertTrue(report["passed"])

    def test_provider_error_voids_coverage_without_contaminating_ratios(self) -> None:
        rows = [
            {
                "id": "healthy",
                "type": "locate",
                "explorer": result(True, tools=10, opens=5, variable_input=1000),
                "greppy": result(True, tools=7, opens=3, variable_input=700),
            },
            {
                "id": "rate-limited",
                "type": "locate",
                "explorer": result(True, tools=10, opens=5, variable_input=1000),
                "greppy": {
                    **result(False, tools=41, opens=16, variable_input=184883),
                    "error": "429 rate limit",
                },
            },
        ]
        status, report = self.evaluate(rows)
        self.assertEqual(status, 2)
        self.assertEqual(report["schema_version"], "greppy.agent-release-gate.v4")
        self.assertEqual(report["invalid_agent_rows"], ["rate-limited"])
        self.assertEqual(report["valid_agent_rows"], 1)
        self.assertFalse(report["checks"]["all_rows_have_valid_agent_results"])
        self.assertEqual(report["ratios_candidate_over_baseline"]["variable_input"], {
            "ratio": 0.7,
            "rows": 1,
        })

    def test_arm_order_is_reproducible_and_not_fixed(self) -> None:
        agents = ["grep", "greppy", "explorer"]
        orders = {
            tuple(run_bench.deterministic_agent_order(f"task-{index}", agents))
            for index in range(100)
        }
        self.assertGreater(len(orders), 1)
        for order in orders:
            self.assertEqual(set(order), set(agents))
        self.assertEqual(
            run_bench.deterministic_agent_order("task-7", agents),
            run_bench.deterministic_agent_order("task-7", agents),
        )
        self.assertEqual(run_bench.prompt_contract()["arm_order"], run_bench.ARM_ORDER_VERSION)

    def test_greppy_prompt_routes_direct_questions_to_one_shot_commands(self) -> None:
        prompt = run_bench.gp_sys("/ignored")
        self.assertEqual(
            run_bench.BENCHMARK_PROMPT_VERSION,
            "greppy-agents-md-0.3.2-single-semantic-query",
        )
        self.assertIn("greppy who-calls NAME`", prompt)
        self.assertIn("greppy impact NAME`", prompt)
        self.assertIn('"what depends on"', prompt)
        self.assertIn("never run\n`search-symbol` first", prompt)
        self.assertIn("greppy search-symbol NAME`", prompt)
        self.assertIn("Add `--code` only when", prompt)
        self.assertIn("do not add it merely to confirm", prompt)
        self.assertIn("Stop after a successful command", prompt)
        self.assertIn("run exactly one `greppy search`", prompt)
        self.assertIn("Do not paraphrase the question", prompt)
        self.assertIn("`greppy read` accepts symbols", prompt)

    def test_variable_input_excludes_each_arms_fixed_prompt_on_every_turn(self) -> None:
        def transcript(prompt_inputs: list[int]) -> str:
            rows = []
            for prompt_input in prompt_inputs:
                rows.append(
                    json.dumps(
                        {
                            "type": "turn_end",
                            "toolResults": [],
                            "message": {
                                "content": [],
                                "usage": {"input": prompt_input},
                            },
                        }
                    )
                )
            return "\n".join(rows)

        compact = transcript([1000, 1100, 1250])
        long_fixed_prompt = transcript([3000, 3100, 3250])
        for parser in (run_bench.parse_pi_jsonl, verify_metrics.recompute):
            self.assertEqual(parser(compact)["variable_input"], 350)
            self.assertEqual(parser(long_fixed_prompt)["variable_input"], 350)


if __name__ == "__main__":
    unittest.main()


def graded(verdict: str, score: float, *, tools: int = 10, opens: int = 5,
           variable_input: int = 1000) -> dict:
    return {
        "quality": {
            "grader": "ground_truth_mechanical_v1",
            "verdict": verdict,
            "score": score,
            "accepted_for_speed_claim": verdict == "pass",
        },
        "tool_calls": tools,
        "source_open_calls": opens,
        "variable_input": variable_input,
    }


class GradedQualityContractTests(ReleaseGateTests):
    """BENCHMARK_CONTRACT.md: only genuinely missing grades void a run; a
    graded partial/fail is a comparison datapoint, not a coverage gap."""

    def test_partial_baseline_counts_as_candidate_win_not_missing(self) -> None:
        rows = [
            {
                "id": f"t{index}",
                "type": "locate",
                "explorer": graded("partial", 0.5),
                "greppy": graded("pass", 1.0, tools=7, opens=3, variable_input=700),
            }
            for index in range(4)
        ]
        status, report = self.evaluate(rows)
        self.assertEqual(status, 0)
        self.assertTrue(report["checks"]["all_rows_have_accepted_quality"])
        self.assertEqual(report["quality"]["missing"], 0)
        self.assertEqual(report["quality"]["candidate_wins"], 4)

    def test_candidate_partial_against_pass_baseline_is_a_loss(self) -> None:
        status, report = self.evaluate(
            [
                {
                    "id": "t1",
                    "type": "locate",
                    "explorer": graded("pass", 1.0),
                    "greppy": graded("partial", 0.5, tools=7, opens=3,
                                     variable_input=700),
                }
            ]
        )
        self.assertEqual(status, 2)
        self.assertEqual(report["quality"]["candidate_losses"], 1)
        self.assertFalse(report["checks"]["candidate_observed_correctness_not_lower"])

    def test_absent_grade_still_voids_the_run(self) -> None:
        status, report = self.evaluate(
            [
                {
                    "id": "t1",
                    "type": "locate",
                    "explorer": {"tool_calls": 10, "source_open_calls": 5,
                                 "variable_input": 1000},
                    "greppy": graded("pass", 1.0, tools=7, opens=3,
                                     variable_input=700),
                }
            ]
        )
        self.assertEqual(status, 2)
        self.assertEqual(report["quality"]["missing"], 1)
        self.assertFalse(report["checks"]["all_rows_have_accepted_quality"])
