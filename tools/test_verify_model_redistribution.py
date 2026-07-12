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


if __name__ == "__main__":
    unittest.main()
