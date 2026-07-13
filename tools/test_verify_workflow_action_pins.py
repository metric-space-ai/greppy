from __future__ import annotations

import pathlib
import tempfile
import unittest

import tools.verify_workflow_action_pins as verifier


class WorkflowActionPinTests(unittest.TestCase):
    def workflow(self, content: str) -> pathlib.Path:
        temporary = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".yml", delete=False
        )
        self.addCleanup(pathlib.Path(temporary.name).unlink, missing_ok=True)
        with temporary:
            temporary.write(content)
        return pathlib.Path(temporary.name)

    def test_commit_pins_and_local_actions_pass(self):
        path = self.workflow(
            "steps:\n"
            "  - uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5\n"
            "  - uses: ./.github/actions/local\n"
        )

        self.assertEqual(verifier.verify([path]), [])

    def test_mutable_tag_fails_with_location(self):
        path = self.workflow("steps:\n  - uses: actions/checkout@v4\n")

        errors = verifier.verify([path])

        self.assertEqual(len(errors), 1)
        self.assertIn(f"{path}:2", errors[0])
        self.assertIn("actions/checkout@v4", errors[0])


if __name__ == "__main__":
    unittest.main()
