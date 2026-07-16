import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import pathlib
import tempfile
import unittest
from unittest import mock


SCRIPT = pathlib.Path(__file__).with_name("summary_quality.py")
SPEC = importlib.util.spec_from_file_location("greppy_summary_quality", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUMMARY_QUALITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY_QUALITY)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SummaryQualityGateTests(unittest.TestCase):
    def documents(self, root: pathlib.Path, *, helpful: int, anti: int):
        case_ids = [f"sq{index:03d}" for index in range(200)]
        cases = root / "cases.json"
        results = root / "results.json"
        judgments = root / "judgments.json"
        output = root / "gate.json"

        cases.write_text(
            json.dumps(
                {
                    "schema_version": SUMMARY_QUALITY.CASES_SCHEMA,
                    "cases": [
                        {
                            "id": case_id,
                            "repo": "fixture",
                            "file_path": "src/lib.rs",
                        }
                        for case_id in case_ids
                    ],
                }
            ),
            encoding="utf-8",
        )
        results.write_text(
            json.dumps(
                {
                    "schema_version": SUMMARY_QUALITY.RESULTS_SCHEMA,
                    "cases_sha256": sha256(cases),
                    "records": [
                        {
                            "id": case_id,
                            "summary": ["Useful purpose hint."],
                            "mechanical_flags": [],
                            "error": None,
                        }
                        for case_id in case_ids
                    ],
                }
            ),
            encoding="utf-8",
        )
        judgments.write_text(
            json.dumps(
                {
                    "schema_version": SUMMARY_QUALITY.JUDGMENTS_SCHEMA,
                    "judge_prompt_version": SUMMARY_QUALITY.JUDGE_PROMPT_VERSION,
                    "cases_sha256": sha256(cases),
                    "results_sha256": sha256(results),
                    "verdicts": [
                        {
                            "id": case_id,
                            "utility": (
                                "anti_helpful" if index < anti
                                else "helpful" if index < anti + helpful
                                else "barely_helpful"
                            ),
                            "invented_symbols": [],
                            "signature_echo": False,
                        }
                        for index, case_id in enumerate(case_ids)
                    ],
                }
            ),
            encoding="utf-8",
        )
        return argparse.Namespace(
            cases=cases,
            results=results,
            judgments=judgments,
            output=output,
        )

    def run_gate(self, args: argparse.Namespace) -> tuple[int, dict]:
        with mock.patch.object(
            SUMMARY_QUALITY,
            "source_for",
            return_value="fn fixture() { useful_symbol(); }",
        ), contextlib.redirect_stdout(io.StringIO()):
            return_code = SUMMARY_QUALITY.gate(args)
        return return_code, json.loads(args.output.read_text(encoding="utf-8"))

    def test_registered_threshold_boundaries_pass(self):
        with tempfile.TemporaryDirectory() as raw:
            args = self.documents(pathlib.Path(raw), helpful=170, anti=10)
            return_code, report = self.run_gate(args)

        self.assertEqual(return_code, 0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["helpful_or_better_rate"], 0.85)
        self.assertEqual(report["anti_helpful_rate"], 0.05)
        self.assertTrue(all(report["checks"].values()))

    def test_one_anti_helpful_result_over_the_limit_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            args = self.documents(pathlib.Path(raw), helpful=189, anti=11)
            return_code, report = self.run_gate(args)

        self.assertEqual(return_code, 2)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["anti_helpful_at_most_5_percent"])

    def test_digest_mismatch_and_signature_echo_fail(self):
        with tempfile.TemporaryDirectory() as raw:
            args = self.documents(pathlib.Path(raw), helpful=200, anti=0)
            judgments = json.loads(args.judgments.read_text(encoding="utf-8"))
            judgments["results_sha256"] = "0" * 64
            judgments["verdicts"][0]["signature_echo"] = True
            args.judgments.write_text(json.dumps(judgments), encoding="utf-8")
            return_code, report = self.run_gate(args)

        self.assertEqual(return_code, 2)
        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["evidence_digests_match"])
        self.assertFalse(report["checks"]["no_signature_echoes"])


class SummaryQualityJudgeTests(unittest.TestCase):
    def valid_response(self) -> dict:
        return {
            "prompt_version": SUMMARY_QUALITY.JUDGE_PROMPT_VERSION,
            "verdicts": [
                {
                    "id": "sq030",
                    "utility": "helpful",
                    "invented_symbols": [],
                    "signature_echo": False,
                    "reason": "correct purpose",
                },
                {
                    "id": "sq031",
                    "utility": "barely_helpful",
                    "invented_symbols": [],
                    "signature_echo": True,
                    "reason": "only echoes signature",
                },
            ],
        }

    def test_response_requires_exact_ids_in_exact_order(self):
        items = [{"id": "sq030"}, {"id": "sq031"}]
        rows = SUMMARY_QUALITY.validate_judge_response(self.valid_response(), items)
        self.assertEqual([row["id"] for row in rows], ["sq030", "sq031"])

        wrong = self.valid_response()
        wrong["verdicts"][0]["id"] = "sq001"
        with self.assertRaisesRegex(RuntimeError, "wrong IDs"):
            SUMMARY_QUALITY.validate_judge_response(wrong, items)

    def test_response_rejects_missing_typed_fields(self):
        response = self.valid_response()
        response["verdicts"][0]["utility"] = "yes"
        with self.assertRaisesRegex(RuntimeError, "invalid utility"):
            SUMMARY_QUALITY.validate_judge_response(
                response, [{"id": "sq030"}, {"id": "sq031"}]
            )


if __name__ == "__main__":
    unittest.main()
