#!/usr/bin/env python3
"""Regression tests for mechanical graph-answer precision grading."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from grade_answers import grade_answer


class GradeAnswerPrecisionTests(unittest.TestCase):
    def test_forbidden_graph_term_is_a_hard_failure(self) -> None:
        task = {
            "check": {
                "kind": "callees",
                "symbol": "normalize_record",
                "expect_members": ["clamp_value", "compute_checksum"],
                "forbid_terms": ["process_svc00"],
                "min_count": 2,
            }
        }
        grade = grade_answer(
            task,
            "normalize_record calls clamp_value, compute_checksum, and process_svc00.",
            mode="mechanical",
            accept_smoke=False,
            accept_mechanical=True,
        )

        self.assertEqual(grade["verdict"], "fail")
        self.assertFalse(grade["accepted_for_speed_claim"])
        self.assertEqual(grade["forbidden_found"], ["process_svc00"])

    def test_exact_graph_answer_still_passes(self) -> None:
        task = {
            "check": {
                "kind": "callees",
                "symbol": "normalize_record",
                "expect_members": ["clamp_value", "compute_checksum"],
                "forbid_terms": ["process_svc00"],
                "min_count": 2,
            }
        }
        grade = grade_answer(
            task,
            "normalize_record has 2 direct callees: clamp_value and compute_checksum.",
            mode="mechanical",
            accept_smoke=False,
            accept_mechanical=True,
        )

        self.assertEqual(grade["verdict"], "pass")
        self.assertTrue(grade["accepted_for_speed_claim"])
        self.assertEqual(grade["forbidden_found"], [])


if __name__ == "__main__":
    unittest.main()
