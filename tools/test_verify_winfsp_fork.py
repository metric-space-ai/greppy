from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools import verify_winfsp_fork as verifier


class WinFspForkVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.patch_paths: list[Path] = []
        patch_records = []
        for relative, expected_files in verifier.EXPECTED_PATCH_FILES.items():
            patch_path = self.root / relative
            patch_path.parent.mkdir(exist_ok=True)
            sections = []
            for path in sorted(expected_files):
                sections.append(
                    f"diff --git a/{path} b/{path}\n"
                    f"--- a/{path}\n"
                    f"+++ b/{path}\n"
                    "@@ -1 +1 @@\n-old\n+new\n"
                )
            patch_path.write_text("".join(sections), encoding="utf-8")
            self.patch_paths.append(patch_path)
            patch_records.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
                    "modified_files": sorted(expected_files),
                }
            )
        self.patch_path = self.patch_paths[0]
        self.manifest = {
            "schema": "greppy.winfsp-transport-upstream.v1",
            "repository": verifier.EXPECTED_REPOSITORY,
            "tag": verifier.EXPECTED_TAG,
            "commit": verifier.EXPECTED_COMMIT,
            "tag_object": verifier.EXPECTED_TAG_OBJECT,
            "license": verifier.EXPECTED_LICENSE,
            "patches": patch_records,
        }
        self.write_manifest()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_manifest(self) -> None:
        (self.root / "upstream.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )

    def test_accepts_exact_minimal_fork(self) -> None:
        result = verifier.verify(self.root)
        self.assertTrue(result["release_eligible_source"])
        self.assertEqual(result["commit"], verifier.EXPECTED_COMMIT)
        self.assertEqual(result["modified_files"], sorted(verifier.EXPECTED_FILES))

    def test_rejects_changed_source_identity_or_license(self) -> None:
        for key, replacement in (
            ("commit", "0" * 40),
            ("tag_object", "0" * 40),
            ("license", "GPL-3.0-only"),
        ):
            with self.subTest(key=key):
                original = self.manifest[key]
                self.manifest[key] = replacement
                self.write_manifest()
                with self.assertRaises(verifier.ForkError):
                    verifier.verify(self.root)
                self.manifest[key] = original
        self.write_manifest()

    def test_rejects_patch_tampering(self) -> None:
        self.patch_path.write_text(
            self.patch_path.read_text(encoding="utf-8") + "# tampered\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(verifier.ForkError, "SHA-256 mismatch"):
            verifier.verify(self.root)

    def test_rejects_extra_or_undeclared_file(self) -> None:
        changed = copy.deepcopy(self.manifest)
        changed["patches"][0]["modified_files"].append("src/extra.c")
        self.manifest = changed
        self.write_manifest()
        with self.assertRaisesRegex(verifier.ForkError, "minimal approved set"):
            verifier.verify(self.root)

    def test_rejects_traversal_and_symlinks(self) -> None:
        self.manifest["patches"][0]["path"] = "../outside.patch"
        self.write_manifest()
        with self.assertRaisesRegex(verifier.ForkError, "traverse parents"):
            verifier.verify(self.root)

        original_path = self.manifest["patches"][0]["path"]
        self.manifest["patches"][0]["path"] = "patches/link.patch"
        self.write_manifest()
        (self.root / "patches/link.patch").symlink_to(self.patch_path)
        with self.assertRaisesRegex(verifier.ForkError, "unexpected patch path"):
            verifier.verify(self.root)
        self.manifest["patches"][0]["path"] = original_path

    def test_rejects_binary_add_delete_or_duplicate_sections(self) -> None:
        variants = (
            self.patch_path.read_text(encoding="utf-8") + "GIT binary patch\n",
            "--- a/inc/winfsp/fsctl.h\n+++ b/inc/winfsp/other.h\n",
            "--- a/inc/winfsp/fsctl.h\n+++ b/inc/winfsp/fsctl.h\n"
            "--- a/inc/winfsp/fsctl.h\n+++ b/inc/winfsp/fsctl.h\n",
        )
        for text in variants:
            with self.subTest(text=text[-30:]):
                self.patch_path.write_text(text, encoding="utf-8")
                self.manifest["patches"][0]["sha256"] = hashlib.sha256(
                    self.patch_path.read_bytes()
                ).hexdigest()
                self.write_manifest()
                with self.assertRaises(verifier.ForkError):
                    verifier.verify(self.root)
