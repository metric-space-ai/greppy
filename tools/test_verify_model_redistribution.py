#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_model_redistribution as verifier


class VerifyModelRedistributionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.root = Path(self._temporary_directory.name)
        (self.root / "models").mkdir()
        (self.root / "licenses").mkdir()
        (self.root / "docs").mkdir()

        self.asset = self._write("models/model.gguf", b"model weights\n")
        self.license = self._write("licenses/model-license.txt", b"license terms\n")
        self.provenance = self._write(
            "docs/model-provenance.json",
            json.dumps(
                {
                    "schema_version": "greppy.model-provenance.v1",
                    "release_ready": True,
                }
            ).encode("utf-8"),
        )
        self.modifications = self._write("docs/model-modifications.md", b"quantized to Q4_K_M\n")
        self.lock_path = self.root / "licenses" / "MODEL-REDISTRIBUTION.lock.json"

    def _write(self, relative_path: str, content: bytes) -> Path:
        path = self.root / relative_path
        path.write_bytes(content)
        return path

    def _record(self, path: Path) -> dict[str, object]:
        content = path.read_bytes()
        return {
            "path": path.relative_to(self.root).as_posix(),
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
        }

    def _manifest(self, *, release_ready: bool = True, model_ready: bool = True) -> dict[str, object]:
        return {
            "schema": verifier.SCHEMA,
            "version": verifier.VERSION,
            "release_ready": release_ready,
            "models": [
                {
                    "id": "example/model-q4",
                    "release_ready": model_ready,
                    "assets": [self._record(self.asset)],
                    "license": [self._record(self.license)],
                    "provenance": [self._record(self.provenance)],
                    "modifications": [self._record(self.modifications)],
                }
            ],
        }

    def _write_manifest(self, manifest: dict[str, object]) -> None:
        self.lock_path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_good_manifest_passes_integrity_and_release_modes(self) -> None:
        self._write_manifest(self._manifest())

        self.assertEqual(verifier.verify_lock(self.lock_path, self.root), [])
        self.assertEqual(verifier.verify_lock(self.lock_path, self.root, release=True), [])
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(verifier.main([str(self.lock_path), "--root", str(self.root), "--release"]), 0)

    def test_tampered_asset_reports_digest_mismatch(self) -> None:
        self._write_manifest(self._manifest())
        self.asset.write_bytes(b"tamper weights\n")

        errors = verifier.verify_lock(self.lock_path, self.root)

        self.assertTrue(any("SHA256 mismatch" in error for error in errors), errors)

    def test_traversal_path_is_rejected(self) -> None:
        manifest = self._manifest()
        manifest["models"][0]["assets"][0]["path"] = "../outside.gguf"  # type: ignore[index]
        self._write_manifest(manifest)

        errors = verifier.verify_lock(self.lock_path, self.root)

        self.assertTrue(any("normalized relative path" in error for error in errors), errors)

    def test_unresolved_release_passes_integrity_but_fails_release_gate(self) -> None:
        self._write_manifest(self._manifest(release_ready=False, model_ready=False))

        self.assertEqual(verifier.verify_lock(self.lock_path, self.root), [])
        release_errors = verifier.verify_lock(self.lock_path, self.root, release=True)

        self.assertTrue(any("global release_ready is false" in error for error in release_errors), release_errors)
        self.assertTrue(any("models[0]" in error and "release_ready is false" in error for error in release_errors))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(verifier.main([str(self.lock_path), "--root", str(self.root), "--release"]), 1)

    def test_release_gate_reads_provenance_release_state(self) -> None:
        self._write_manifest(self._manifest())
        document = json.loads(self.provenance.read_text(encoding="utf-8"))
        document["release_ready"] = False
        self.provenance.write_text(json.dumps(document), encoding="utf-8")
        manifest = self._manifest()
        self._write_manifest(manifest)

        self.assertEqual(verifier.verify_lock(self.lock_path, self.root), [])
        release_errors = verifier.verify_lock(self.lock_path, self.root, release=True)

        self.assertTrue(any("provenance" in error and "not release_ready" in error for error in release_errors))

    def test_report_is_bound_to_lock_digest_commit_and_release_mode(self) -> None:
        self._write_manifest(self._manifest())
        report_path = self.root / "report.json"
        commit = "a" * 40

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = verifier.main(
                [
                    str(self.lock_path),
                    "--root",
                    str(self.root),
                    "--release",
                    "--git-commit",
                    commit,
                    "--report",
                    str(report_path),
                ]
            )

        self.assertEqual(result, 0)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["schema_version"], verifier.REPORT_SCHEMA)
        self.assertEqual(report["mode"], "release")
        self.assertTrue(report["passed"])
        self.assertEqual(report["git_commit"], commit)
        self.assertEqual(report["lock_path"], "licenses/MODEL-REDISTRIBUTION.lock.json")
        self.assertEqual(report["lock_sha256"], hashlib.sha256(self.lock_path.read_bytes()).hexdigest())
        self.assertEqual(report["models"][0]["asset_sha256s"], [self._record(self.asset)["sha256"]])

    def test_failed_release_still_writes_failure_report(self) -> None:
        self._write_manifest(self._manifest(release_ready=False, model_ready=False))
        report_path = self.root / "report.json"

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = verifier.main(
                [
                    str(self.lock_path),
                    "--root",
                    str(self.root),
                    "--release",
                    "--report",
                    str(report_path),
                ]
            )

        self.assertEqual(result, 1)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertFalse(report["passed"])
        self.assertTrue(any("release_ready is false" in error for error in report["errors"]))

    def test_invalid_git_commit_is_rejected_before_report(self) -> None:
        self._write_manifest(self._manifest())
        report_path = self.root / "report.json"

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = verifier.main(
                [str(self.lock_path), "--git-commit", "main", "--report", str(report_path)]
            )

        self.assertEqual(result, 2)
        self.assertFalse(report_path.exists())

    def test_embeddinggemma_reproduction_is_cross_checked(self) -> None:
        source_sha = "1" * 64
        asset_record = self._record(self.asset)
        provenance = {
            "schema_version": "greppy.model-provenance.v1",
            "model_id": verifier.EMBEDDINGGEMMA_MODEL_ID,
            "release_ready": False,
            "conversion": {
                "source": {
                    "repository": "example/source",
                    "file_commit": "2" * 40,
                    "filename": "embeddinggemma-300M-F32.gguf",
                    "size": 123,
                    "sha256": source_sha,
                },
                "reproduction": {
                    "tool": "llama.cpp llama-quantize",
                    "revision": "3" * 40,
                    "architecture": "x86_64",
                    "command": verifier.EMBEDDINGGEMMA_REPRO_COMMAND,
                    "independent_runs": 2,
                    "bit_stable": True,
                    "byte_identical_to_bundled_asset": True,
                    "output_size": asset_record["size"],
                    "output_sha256": asset_record["sha256"],
                },
            },
            "bundled": {
                "gguf_size": asset_record["size"],
                "gguf_sha256": asset_record["sha256"],
            },
        }
        self.provenance.write_text(json.dumps(provenance), encoding="utf-8")
        manifest = self._manifest(release_ready=False, model_ready=False)
        manifest["models"][0]["id"] = verifier.EMBEDDINGGEMMA_MODEL_ID  # type: ignore[index]
        self._write_manifest(manifest)

        self.assertEqual(verifier.verify_lock(self.lock_path, self.root), [])

        provenance["conversion"]["reproduction"]["output_sha256"] = "4" * 64
        self.provenance.write_text(json.dumps(provenance), encoding="utf-8")
        self._write_manifest(self._manifest(release_ready=False, model_ready=False))
        manifest = json.loads(self.lock_path.read_text(encoding="utf-8"))
        manifest["models"][0]["id"] = verifier.EMBEDDINGGEMMA_MODEL_ID
        self._write_manifest(manifest)

        errors = verifier.verify_lock(self.lock_path, self.root)
        self.assertTrue(any("reproduction output does not match" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
