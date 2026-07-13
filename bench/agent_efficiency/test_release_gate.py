import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import release_gate
import run_bench


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


if __name__ == "__main__":
    unittest.main()
