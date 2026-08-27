from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools import verify_portable_cow_performance as verifier


SOURCE = "a" * 40


def evidence(os_name: str, arch: str) -> dict:
    return {
        "schema": "greppy.portable-cow-performance.v1",
        "source_commit": SOURCE,
        "source_tracked_worktree_dirty": False,
        "provider_binary": f"/provider/{os_name}",
        "provider_sha256": "b" * 64,
        "profile": "release",
        "os": os_name,
        "arch": arch,
        "hardware": "test hardware",
        "fixture_tracked_files": 300_000,
        "fixture_profile": {
            "schema": "greppy.portable-cow-fixture.v3",
            "tracked_files": 300_000,
        },
        "iterations": 25,
        "cold_prime": {"elapsed_ms": 5_000.0, "gate_ms": 120_000.0},
        "workspace_creation": {
            "end_to_end_p95_ms": 100.0,
            "end_to_end_p95_gate_ms": 500.0,
            "warmup": {
                "steady_state_p95_ms": 90.0,
                "steady_state_p95_gate_ms": 500.0,
            },
        },
        "space": {
            "untouched_physical_delta_bytes": 24_576,
            "untouched_gate_bytes": 1_048_576,
            "one_byte_write_physical_delta_bytes": 1_048_576,
            "one_byte_write_gate_bytes": 1_310_720,
            "one_byte_write_new_chunks": 1,
            "chunk_size": 1_048_576,
        },
        "parallel": {"workspaces": 50, "wall_ms": 1_000.0},
        "toolchains": [
            {"name": name, "release_gate": True, "overhead_percent": 5.0}
            for name in ("rust", "python", "node")
        ]
        + [
            {
                "name": "node-startup-diagnostic",
                "release_gate": False,
                "overhead_percent": 50.0,
            }
        ],
    }


class PortableCowPerformanceVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_set(self, values: list[dict]) -> list[Path]:
        paths = []
        for index, value in enumerate(values):
            path = self.root / f"evidence-{index}.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            paths.append(path)
        return paths

    def valid_values(self) -> list[dict]:
        return [
            evidence("linux", "x86_64"),
            evidence("macos", "aarch64"),
            evidence("windows", "x86_64"),
        ]

    def test_accepts_exact_complete_platform_set(self) -> None:
        result = verifier.verify(self.write_set(self.valid_values()), SOURCE)
        self.assertTrue(result["release_eligible"])
        self.assertEqual(result["source_commit"], SOURCE)
        self.assertEqual(len(result["platforms"]), 3)

    def test_rejects_missing_or_duplicate_platform(self) -> None:
        values = self.valid_values()
        with self.assertRaisesRegex(verifier.EvidenceError, "exactly three"):
            verifier.verify(self.write_set(values[:2]), SOURCE)
        values[2] = copy.deepcopy(values[0])
        with self.assertRaisesRegex(verifier.EvidenceError, "duplicate evidence"):
            verifier.verify(self.write_set(values), SOURCE)

    def test_rejects_cross_commit_or_dirty_evidence(self) -> None:
        values = self.valid_values()
        values[1]["source_commit"] = "c" * 40
        with self.assertRaisesRegex(verifier.EvidenceError, "source SHA mismatch"):
            verifier.verify(self.write_set(values), SOURCE)
        values = self.valid_values()
        values[2]["source_tracked_worktree_dirty"] = True
        with self.assertRaisesRegex(verifier.EvidenceError, "tracked-dirty"):
            verifier.verify(self.write_set(values), SOURCE)

    def test_rejects_relaxed_or_failed_latency_gates(self) -> None:
        values = self.valid_values()
        values[1]["workspace_creation"]["end_to_end_p95_gate_ms"] = 501.0
        with self.assertRaisesRegex(verifier.EvidenceError, "gate was relaxed"):
            verifier.verify(self.write_set(values), SOURCE)
        values = self.valid_values()
        values[1]["workspace_creation"]["end_to_end_p95_ms"] = 501.0
        with self.assertRaisesRegex(verifier.EvidenceError, "exceeds its gate"):
            verifier.verify(self.write_set(values), SOURCE)

    def test_rejects_relaxed_space_or_chunk_contract(self) -> None:
        values = self.valid_values()
        values[0]["space"]["untouched_gate_bytes"] = 1_048_577
        with self.assertRaisesRegex(verifier.EvidenceError, "gate was relaxed"):
            verifier.verify(self.write_set(values), SOURCE)
        values = self.valid_values()
        values[2]["space"]["one_byte_write_new_chunks"] = 2
        with self.assertRaisesRegex(verifier.EvidenceError, "exactly one chunk"):
            verifier.verify(self.write_set(values), SOURCE)

    def test_rejects_missing_or_slow_required_toolchain(self) -> None:
        values = self.valid_values()
        values[0]["toolchains"] = values[0]["toolchains"][1:]
        with self.assertRaisesRegex(verifier.EvidenceError, "toolchain set is incomplete"):
            verifier.verify(self.write_set(values), SOURCE)
        values = self.valid_values()
        values[2]["toolchains"][1]["overhead_percent"] = 20.01
        with self.assertRaisesRegex(verifier.EvidenceError, "overhead exceeds"):
            verifier.verify(self.write_set(values), SOURCE)

    def test_rejects_symlinked_evidence(self) -> None:
        paths = self.write_set(self.valid_values())
        link = self.root / "linked.json"
        link.symlink_to(paths[0])
        with self.assertRaisesRegex(verifier.EvidenceError, "not a regular file"):
            verifier.verify([link, paths[1], paths[2]], SOURCE)


if __name__ == "__main__":
    unittest.main()
