from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
import unittest

from tools.materialize_benchmark_corpus import (
    DEFAULT_LOCK,
    CorpusError,
    load_lock,
    materialize,
)


class MaterializeBenchmarkCorpusTests(unittest.TestCase):
    def test_repository_lock_is_valid_and_pins_all_release_gate_inputs(self) -> None:
        document = load_lock(DEFAULT_LOCK)
        self.assertEqual(document["repository"], "metric-space-ai/greppy")
        self.assertEqual(
            document["commit"], "95d359e2ff32f8a988f39b099aaf4a239814d7c4"
        )
        self.assertEqual(
            {entry["target"] for entry in document["files"]},
            {
                "bench/agent_efficiency/tasks_v2.json",
                "bench/agent_efficiency/realcorpus/candidates.json",
                "bench/summary_quality/cases_v1.json",
                "bench/agent_coding/tasks_v2.json",
            },
        )

    def fixture(self, data: bytes = b'{"tasks": []}\n') -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
        managed = tempfile.TemporaryDirectory(prefix="greppy-corpus-test-")
        self.addCleanup(managed.cleanup)
        temporary = pathlib.Path(managed.name)
        source = temporary / "source"
        root = temporary / "product"
        (source / "agent_efficiency").mkdir(parents=True)
        (source / "agent_efficiency" / "tasks_v2.json").write_bytes(data)
        lock = temporary / "lock.json"
        lock.write_text(
            json.dumps(
                {
                    "schema_version": "greppy.benchmark-corpus-lock.v1",
                    "repository": "metric-space-ai/greppy-bench",
                    "commit": "8" * 40,
                    "files": [
                        {
                            "source": "agent_efficiency/tasks_v2.json",
                            "target": "bench/agent_efficiency/tasks_v2.json",
                            "sha256": hashlib.sha256(data).hexdigest(),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return lock, source, root

    def test_materializes_verified_bytes(self) -> None:
        lock, source, root = self.fixture()
        written = materialize(lock, root, source_dir=source)
        self.assertEqual(written, [root / "bench/agent_efficiency/tasks_v2.json"])
        self.assertEqual(written[0].read_bytes(), b'{"tasks": []}\n')

    def test_digest_mismatch_does_not_replace_existing_target(self) -> None:
        lock, source, root = self.fixture()
        target = root / "bench/agent_efficiency/tasks_v2.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"preserve me")
        (source / "agent_efficiency/tasks_v2.json").write_bytes(b"tampered")
        with self.assertRaisesRegex(CorpusError, "digest mismatch"):
            materialize(lock, root, source_dir=source)
        self.assertEqual(target.read_bytes(), b"preserve me")

    def test_rejects_target_path_traversal(self) -> None:
        lock, source, root = self.fixture()
        document = json.loads(lock.read_text(encoding="utf-8"))
        document["files"][0]["target"] = "bench/../outside.json"
        lock.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(CorpusError, "non-traversing"):
            materialize(lock, root, source_dir=source)

    def test_rejects_windows_style_target_path_traversal(self) -> None:
        lock, source, root = self.fixture()
        document = json.loads(lock.read_text(encoding="utf-8"))
        document["files"][0]["target"] = "bench\\..\\outside.json"
        lock.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(CorpusError, "non-traversing"):
            materialize(lock, root, source_dir=source)


if __name__ == "__main__":
    unittest.main()
