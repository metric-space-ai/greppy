#!/usr/bin/env python3
"""Unit tests for the publication-safe runtime footprint harness."""

from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bench import runtime_footprint


FAKE_GREPPY = r'''#!/usr/bin/env python3
import json
import os
import pathlib
import sys
import time

store = pathlib.Path(os.environ["GREPPY_STORE_DIR"])
store.mkdir(parents=True, exist_ok=True)
audit = os.environ.get("FAKE_AUDIT_PATH")
if audit:
    with open(audit, "a", encoding="utf-8") as handle:
        handle.write(str(store) + "\n")

args = sys.argv[1:]
command = args[0] if args else ""
if os.environ.get("FAKE_SLEEP_PHASE") == command:
    time.sleep(10)
if os.environ.get("FAKE_FAIL_PHASE") == command:
    print("TOP SECRET STDOUT")
    print("TOP SECRET STDERR", file=sys.stderr)
    raise SystemExit(7)

marker = store / "indexed"
if command == "doctor":
    indexed = marker.exists()
    daemon = {
        "endpoint": str(store / "TOP-SECRET.sock"),
        "protocol": "greppy-inference-v1",
        "state": "ready" if indexed else "stopped",
        "last_error": "TOP SECRET DAEMON ERROR",
    }
    value = {
        "command": "doctor",
        "status": "ok" if indexed else "no_index",
        "healthy": indexed,
        "store_exists": indexed,
        "root_path": "TOP SECRET ABSOLUTE REPO",
        "store_path": str(store),
        "store_bytes": 4096 if indexed else 0,
        "embedding_complete": indexed,
        "current_embedding_rows": 9 if indexed else 0,
        "fresh": indexed,
        "schema_version": 12 if indexed else None,
        "expected_schema_version": 12,
        "schema_current": indexed,
        "integrity_ok": indexed,
        "project_present": indexed,
        "graph_generation": 1 if indexed else None,
        "stats": {"files": 3, "nodes": 7, "edges": 5} if indexed else None,
        "incomplete_provider_count": 0 if indexed else None,
        "provider_failure_count": 0 if indexed else None,
        "git_tracked_files": 3,
        "vectors_missing_with_model": False,
        "inference": {
            "registry": {
                "version": 1,
                "preference": "auto",
                "explicit": False,
                "required_gpu_memory": 100,
                "selected_backend": "cpu",
                "selected_device_id": "cpu:0",
                "probes": [{
                    "backend": "cpu",
                    "backend_id": "fake-cpu-v1",
                    "build_info": "test",
                    "abi_version": 1,
                    "compiled": True,
                    "available": True,
                    "score": 100,
                    "reason": "TOP SECRET PROBE REASON",
                    "devices": [{
                        "backend": "cpu",
                        "id": "cpu:0",
                        "name": "Fake CPU",
                        "description": "Fake CPU",
                        "device_type": "cpu",
                        "memory_free": 1000,
                        "memory_total": 2000,
                        "compute_capability": None,
                        "metal_family": None,
                        "capabilities": ["test-simd"],
                        "rejection_reason": "TOP SECRET DEVICE REASON",
                    }],
                }],
            },
            "models": {
                "embedding": {
                    "model_id": "embedding-test",
                    "format": "gguf-q4k",
                    "embedded": True,
                    "model_sha256": "a" * 64,
                    "tokenizer_sha256": "b" * 64,
                    "model_bytes": 123,
                    "prompt_version": "embedding-v1",
                    "task_profile": "retrieval",
                    "last_error": "TOP SECRET MODEL ERROR",
                },
                "summary": {
                    "model_id": "summary-test",
                    "format": "gguf-q4-k-m-mtp",
                    "embedded": True,
                    "model_sha256": "c" * 64,
                    "tokenizer_sha256": "d" * 64,
                    "model_bytes": 456,
                    "prompt_version": "summary-v1",
                },
            },
            "daemons": {"embedding": daemon, "summary": daemon},
        },
    }
    print(json.dumps(value))
    raise SystemExit(0 if indexed else 1)

if command == "index":
    marker.write_text("ok", encoding="ascii")
    print("TOP SECRET INDEX STDOUT")
    raise SystemExit(0)

if command == "semantic-search":
    query = args[1]
    print(json.dumps({
        "schema_version": "greppy.semantic-search.v1",
        "command": "semantic-search",
        "status": "ok",
        "mode": "vector",
        "query": query,
        "project": "TOP SECRET PROJECT",
        "candidate_total": 2,
        "total_exact": 2,
        "retrieved": 2,
        "shown": 1,
        "omitted": 1,
        "unranked_candidates": 0,
        "truncated": True,
        "fresh": True,
        "expand_id": "TOP-SECRET-EXPAND",
        "hits": [{
            "file_path": "TOP/SECRET/source.rs",
            "source": "TOP SECRET SOURCE",
            "signature": "fn top_secret()",
            "summary": ["Useful purpose hint."],
        }],
    }))
    raise SystemExit(0)

if command == "brief":
    symbol = args[1]
    print(json.dumps({
        "schema_version": "greppy.brief.v1",
        "command": "brief",
        "status": "ok",
        "query": symbol,
        "project": "TOP SECRET PROJECT",
        "definitions": [{
            "source": "TOP SECRET SOURCE",
            "file_path": "TOP/SECRET/source.rs",
            "signature": "fn top_secret()",
            "summary": ["Useful purpose hint."],
        }],
        "callers": [{}],
        "references": [],
        "calls": [{}, {}],
        "expand_id": "TOP-SECRET-EXPAND",
    }))
    raise SystemExit(0)

if command == "cache" and len(args) > 1 and args[1] == "status":
    print(json.dumps({
        "data_root": str(store),
        "managed_bytes": 8000,
        "unmanaged_bytes": 0,
        "locked_bytes": 100,
        "quota_bytes": 9000,
        "low_water_bytes": 7000,
        "ttl_secs": 300,
        "entries": [{
            "kind": "workspace",
            "id": "TOP-SECRET-ID",
            "path": str(store / "TOP-SECRET"),
            "workspace_root": "TOP SECRET ABSOLUTE REPO",
            "bytes": 8000,
            "locked": True,
            "orphaned": False,
        }],
        "unmanaged": [str(store / "TOP-SECRET-UNMANAGED")],
    }))
    raise SystemExit(0)

raise SystemExit(9)
'''


class RuntimeFootprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / "secret-real-repository"
        self.repo.mkdir()
        (self.repo / ".git").mkdir()
        self.binary = self.root / "fake-greppy"
        self.binary.write_text(FAKE_GREPPY, encoding="utf-8")
        self.binary.chmod(self.binary.stat().st_mode | stat.S_IXUSR)
        self.output = self.root / "results" / "runtime.json"
        self.audit = self.root / "audit.txt"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _argv(self) -> list[str]:
        return [
            "--greppy",
            str(self.binary),
            "--repo",
            str(self.repo),
            "--semantic-query",
            "TOP SECRET QUERY",
            "--brief-symbol",
            "TopSecretSymbol",
            "--output",
            str(self.output),
            "--warm-repeats",
            "2",
            "--device",
            "cpu",
            "--timeout-seconds",
            "30",
        ]

    def test_redaction_timing_schema_and_cleanup(self) -> None:
        with mock.patch.dict(os.environ, {"FAKE_AUDIT_PATH": str(self.audit)}, clear=False):
            self.assertEqual(runtime_footprint.main(self._argv()), 0)
        value = json.loads(self.output.read_text(encoding="utf-8"))
        rendered = self.output.read_text(encoding="utf-8")

        self.assertEqual(value["schema_version"], runtime_footprint.SCHEMA_VERSION)
        self.assertEqual(value["configuration"]["warm_repeat_count"], 2)
        self.assertEqual(value["configuration"]["device"], "cpu")
        self.assertGreaterEqual(value["measurements"]["index"]["wall_time_ms"], 0)
        self.assertEqual(len(value["measurements"]["semantic_search"]["warm"]["samples_ms"]), 2)
        self.assertEqual(len(value["measurements"]["brief"]["warm"]["samples_ms"]), 2)
        self.assertEqual(value["measurements"]["cache"]["managed_bytes"], 8000)
        self.assertEqual(value["doctor"]["before"]["status"], "no_index")
        self.assertTrue(value["doctor"]["after"]["healthy"])
        self.assertEqual(value["doctor"]["after"]["inference"]["registry"]["selected_backend"], "cpu")
        self.assertEqual(value["measurements"]["semantic_search"]["result"]["hits_with_summary"], 1)

        for forbidden in (
            "TOP SECRET",
            "TopSecretSymbol",
            str(self.repo),
            "secret-real-repository",
            "TOP/SECRET/source.rs",
            "TOP-SECRET-EXPAND",
        ):
            self.assertNotIn(forbidden, rendered)
        self.assertTrue(
            all(len(record["argv_template_sha256"]) == 64 for record in value["commands"])
        )
        raw_sensitive_hash = runtime_footprint._argv_sha256(
            [str(self.binary), "brief", "TopSecretSymbol", "--root", str(self.repo)]
        )
        self.assertNotIn(raw_sensitive_hash, rendered)
        stores = {Path(line) for line in self.audit.read_text(encoding="utf-8").splitlines()}
        self.assertEqual(len(stores), 1)
        self.assertFalse(next(iter(stores)).exists())
        self.assertFalse(next(iter(stores)).is_relative_to(self.repo))

    def test_failed_command_is_redacted_and_store_is_cleaned(self) -> None:
        stderr = io.StringIO()
        with mock.patch.dict(
            os.environ,
            {"FAKE_AUDIT_PATH": str(self.audit), "FAKE_FAIL_PHASE": "semantic-search"},
            clear=False,
        ), contextlib.redirect_stderr(stderr):
            self.assertEqual(runtime_footprint.main(self._argv()), 1)
        self.assertFalse(self.output.exists())
        self.assertNotIn("TOP SECRET", stderr.getvalue())
        stores = {Path(line) for line in self.audit.read_text(encoding="utf-8").splitlines()}
        self.assertTrue(stores)
        self.assertTrue(all(not store.exists() for store in stores))

    def test_failure_preserves_existing_output_atomically(self) -> None:
        self.output.parent.mkdir(parents=True)
        self.output.write_text("sentinel\n", encoding="ascii")
        with mock.patch.dict(os.environ, {"FAKE_FAIL_PHASE": "brief"}, clear=False), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(runtime_footprint.main(self._argv()), 1)
        self.assertEqual(self.output.read_text(encoding="ascii"), "sentinel\n")
        self.assertEqual(list(self.output.parent.glob(".runtime.json.*.tmp")), [])

    def test_successful_output_is_atomic_and_private(self) -> None:
        self.assertEqual(runtime_footprint.main(self._argv()), 0)
        self.assertTrue(self.output.exists())
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o600)
        self.assertEqual(list(self.output.parent.glob(".runtime.json.*.tmp")), [])

    def test_timeout_kills_command_group_and_cleans_store(self) -> None:
        argv = self._argv()
        argv[argv.index("--timeout-seconds") + 1] = "3"
        with mock.patch.dict(
            os.environ,
            {"FAKE_AUDIT_PATH": str(self.audit), "FAKE_SLEEP_PHASE": "index"},
            clear=False,
        ), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(runtime_footprint.main(argv), 1)
        self.assertFalse(self.output.exists())
        stores = {Path(line) for line in self.audit.read_text(encoding="utf-8").splitlines()}
        self.assertTrue(stores)
        self.assertTrue(all(not store.exists() for store in stores))

    def test_argv_hash_is_unambiguous(self) -> None:
        self.assertNotEqual(
            runtime_footprint._argv_sha256(["ab", "c"]),
            runtime_footprint._argv_sha256(["a", "bc"]),
        )


if __name__ == "__main__":
    unittest.main()
