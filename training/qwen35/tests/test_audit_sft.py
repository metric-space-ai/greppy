#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audit_sft


def accepted_raw(source="def count():\n    return 1", repo="example/project", license_value=None):
    return {
        "lang": "python",
        "repo": repo,
        "path": "module.py",
        "source": source,
        "license": ["MIT"] if license_value is None else license_value,
        "hexsha": "a" * 40,
        "docstring_stripped": False,
        "summary": ["Returns one."],
        "tokens": 42,
    }


def dropped_raw(source="def dropped():\n    pass"):
    row = accepted_raw(source=source)
    row.pop("tokens")
    row["summary"] = None
    row["dropped"] = True
    row["last_errors"] = ["invalid format"]
    return row


def sft_for(raw):
    return {
        "prompt": audit_sft.PROMPT_PREFIX + raw["source"].strip() + audit_sft.PROMPT_SUFFIX,
        "completion": "\n".join(raw["summary"]) + audit_sft.COMPLETION_SUFFIX,
        "lang": raw["lang"],
        "repo": raw["repo"],
    }


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def write_jsonl(self, name, rows):
        path = self.root / name
        path.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def run_audit(self, raw_rows, sft_rows, extra_denylists=()):
        raw = self.write_jsonl("raw.jsonl", raw_rows)
        sft = self.write_jsonl("sft.jsonl", sft_rows)
        return audit_sft.audit([raw], [sft], extra_denylists)

    def test_success_is_deterministic_and_emits_only_aggregates(self):
        raw = accepted_raw()
        report = self.run_audit([raw, dropped_raw()], [sft_for(raw)])
        again = self.run_audit([raw, dropped_raw()], [sft_for(raw)])
        self.assertEqual(report, again)
        self.assertEqual(
            report["row_histogram"],
            {
                "accepted_raw": 1,
                "distinct_accepted_prompts": 1,
                "duplicate_raw_sources": 0,
                "dropped_raw": 1,
                "mapped_sft": 1,
                "raw_total": 2,
                "sft_total": 1,
            },
        )
        self.assertEqual(report["repository_histogram"], {"example/project": 1})
        self.assertEqual(report["language_histogram"], {"python": 1})
        self.assertEqual(report["license_histogram"], {'["MIT"]': 1})
        self.assertEqual(len(report["aggregate_sha256"]), 64)
        rendered = json.dumps(report)
        self.assertNotIn(raw["source"], rendered)
        self.assertNotIn(raw["summary"][0], rendered)

    def test_unmapped_prompt_fails(self):
        raw = accepted_raw()
        other = accepted_raw(source="def other():\n    return 2")
        with self.assertRaisesRegex(audit_sft.AuditError, "unmapped SFT prompt"):
            self.run_audit([raw], [sft_for(other)])

    def test_duplicate_raw_source_is_deduplicated_by_sft_provenance(self):
        selected = accepted_raw(repo="example/selected")
        duplicate = accepted_raw(repo="example/duplicate")
        report = self.run_audit([selected, duplicate], [sft_for(selected)])
        self.assertEqual(report["row_histogram"]["accepted_raw"], 2)
        self.assertEqual(report["row_histogram"]["distinct_accepted_prompts"], 1)
        self.assertEqual(report["row_histogram"]["duplicate_raw_sources"], 1)

    def test_duplicate_raw_source_with_same_provenance_fails(self):
        raw = accepted_raw()
        with self.assertRaisesRegex(audit_sft.AuditError, "ambiguous"):
            self.run_audit([raw, dict(raw)], [sft_for(raw)])

    def test_duplicate_sft_prompt_fails(self):
        raw = accepted_raw()
        sft = sft_for(raw)
        with self.assertRaisesRegex(audit_sft.AuditError, "duplicate SFT prompt"):
            self.run_audit([raw], [sft, dict(sft)])

    def test_normalized_prompt_collision_fails(self):
        first = accepted_raw(source="def same():\n    pass")
        second = accepted_raw(source="  def same():\n    pass  ")
        with self.assertRaisesRegex(audit_sft.AuditError, "normalized prompt collision"):
            self.run_audit([first, second], [sft_for(first)])

    def test_missing_license_fails(self):
        raw = accepted_raw(license_value=[])
        with self.assertRaisesRegex(audit_sft.AuditError, "missing license"):
            self.run_audit([raw], [sft_for(raw)])

    def test_unexpected_schema_fails(self):
        raw = accepted_raw()
        raw["surprise"] = True
        with self.assertRaisesRegex(audit_sft.AuditError, "unexpected raw schema"):
            self.run_audit([raw], [sft_for(raw)])

    def test_default_summary_quality_repository_guard_fails(self):
        raw = accepted_raw(repo="https://github.com/Pallets/Flask.git")
        with self.assertRaisesRegex(audit_sft.AuditError, "pallets/flask"):
            self.run_audit([raw], [sft_for(raw)])

    def test_additional_denylist_guard_fails(self):
        raw = accepted_raw()
        denylist = self.root / "extra.txt"
        denylist.write_text("example/project\n", encoding="utf-8")
        with self.assertRaisesRegex(audit_sft.AuditError, "example/project"):
            self.run_audit([raw], [sft_for(raw)], [denylist])

    def test_label_mismatch_fails_without_printing_label(self):
        raw = accepted_raw()
        sft = sft_for(raw)
        sft["completion"] = "Invents a different label.<|im_end|>"
        with self.assertRaisesRegex(audit_sft.AuditError, "SFT label mismatch"):
            self.run_audit([raw], [sft])


if __name__ == "__main__":
    unittest.main()
