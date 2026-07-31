from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bench.agent_coding.v3 import preflight_gpu3 as preflight


class VersionTests(unittest.TestCase):
    def test_common_versions(self) -> None:
        self.assertEqual(preflight.parse_version("openjdk version \"17.0.15\""), (17, 0, 15, 0))
        self.assertEqual(preflight.parse_version("Apache Maven 3.9.11"), (3, 9, 11, 0))
        self.assertTrue(preflight.version_at_least((3, 9, 11), (3, 8, 6)))
        self.assertEqual(preflight.java_major("Java version: 17.0.15, vendor: Eclipse"), 17)

    def test_java_eight_spelling(self) -> None:
        self.assertEqual(preflight.java_major('java version "1.8.0_392"'), 8)


class StorageTests(unittest.TestCase):
    def test_same_device_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            base = Path(root)
            nvme = base / "nvme"
            nas = base / "nas"
            nvme.mkdir()
            nas.mkdir()
            config = {
                "nvme": {"root": str(nvme), "minimum_free_gib": 0.000001},
                "nas": {"root": str(nas), "minimum_free_gib": 0.000001},
            }
            path = base / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            result = preflight.check_storage(path, config)
            self.assertFalse(result["ready"])
            self.assertFalse(result["distinct_devices"])
            self.assertIn("storage_same_device", {row["code"] for row in result["failures"]})

    def test_missing_explicit_roots_is_machine_failure(self) -> None:
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {}, clear=True):
            path = Path(root) / "config.json"
            path.write_text("{}", encoding="utf-8")
            result = preflight.check_storage(path, {})
            self.assertFalse(result["ready"])
            self.assertEqual(result["failures"][0]["code"], "storage_config")


class ToolTests(unittest.TestCase):
    def test_java_home_is_preferred_without_hardcoded_path(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            java = Path(root) / "bin" / "java"
            java.parent.mkdir()
            java.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            java.chmod(0o755)
            env = {"JAVA_HOME": root, "PATH": ""}
            self.assertEqual(preflight.resolve_command("java", {}, env), str(java.resolve()))

    def test_all_registry_languages_require_working_tools(self) -> None:
        registry = {"primary_languages": list(preflight.LANGUAGE_TOOLS)}

        def fake_probe(name: str, executable: str, minimum: str | None, timeout: int, env: dict[str, str]) -> dict[str, object]:
            output = "tool 99.0.0"
            if name == "java":
                output = 'openjdk version "17.0.15"'
            elif name == "mvn":
                output = "Apache Maven 3.9.11\nJava version: 17.0.15, vendor: test"
            elif name == "greppy":
                output = "greppy 0.3.0"
            return {"name": name, "ready": True, "version_output": output}

        with mock.patch.object(preflight, "resolve_command", return_value="/fake/tool"), mock.patch.object(preflight, "probe_tool", side_effect=fake_probe):
            result = preflight.check_tools({"tools": {"required_java_major": 17}}, registry)
        self.assertTrue(result["ready"])
        self.assertEqual(set(result["language_coverage"]), set(preflight.LANGUAGE_TOOLS))
        self.assertTrue(all(row["ready"] for row in result["language_coverage"].values()))

    def test_container_mode_does_not_assume_host_language_tools(self) -> None:
        registry = {"primary_languages": list(preflight.LANGUAGE_TOOLS)}

        def fake_probe(name: str, executable: str, minimum: str | None, timeout: int, env: dict[str, str]) -> dict[str, object]:
            output = "tool 99.0.0" if name != "greppy" else "greppy 0.3.0"
            return {"name": name, "ready": True, "version_output": output}

        with mock.patch.object(preflight, "resolve_command", return_value="/fake/tool") as resolve, mock.patch.object(preflight, "probe_tool", side_effect=fake_probe):
            result = preflight.check_tools({"tools": {}}, registry, "container")
        self.assertTrue(result["ready"])
        self.assertEqual({call.args[0] for call in resolve.call_args_list}, set(preflight.CONTAINER_HOST_TOOLS))
        self.assertTrue(all(row["delegated_to"] for row in result["language_coverage"].values()))


class AdapterTests(unittest.TestCase):
    def _write_registry(self, root: Path) -> dict[str, object]:
        return {
            "repositories": [
                {"id": "repo-a", "toolchain_profile": "cargo"},
                {"id": "repo-b", "toolchain_profile": "pytest"},
            ]
        }

    def test_missing_adapter_cannot_drop_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "schema_version": preflight.ADAPTER_SCHEMA,
                "adapters": [{"repository_id": "repo-a", "status": "pending", "toolchain_profile": "cargo"}],
            }
            (root / "adapters.json").write_text(json.dumps(manifest), encoding="utf-8")
            config = {"adapter_manifest": "adapters.json"}
            result = preflight.check_adapters(root / "config.json", config, self._write_registry(root))
            self.assertFalse(result["ready"])
            self.assertEqual(result["expected"], 2)
            self.assertEqual(set(result["repositories"]), {"repo-a", "repo-b"})
            self.assertIn("adapter_missing", {row["code"] for row in result["failures"]})

    def test_all_ready_adapters_must_execute_exact_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for key, profile in (("repo-a", "cargo"), ("repo-b", "pytest")):
                commands = {}
                for role in ("probe", "metadata", "validation"):
                    commands[role] = [
                        sys.executable,
                        "-c",
                        f"import json; print(json.dumps({{'ready': True, 'repository_id': {key!r}, 'command_role': {role!r}, 'proof_sha256': {'a' * 64!r}}}))",
                    ]
                rows.append({
                    "repository_id": key,
                    "status": "ready",
                    "toolchain_profile": profile,
                    "proof_sha256": "a" * 64,
                    "image": "example.invalid/adapter@sha256:" + "b" * 64,
                    "image_id": "sha256:" + "c" * 64,
                    "commands": commands,
                })
            manifest = {"schema_version": preflight.ADAPTER_SCHEMA, "adapters": rows}
            (root / "adapters.json").write_text(json.dumps(manifest), encoding="utf-8")
            result = preflight.check_adapters(
                root / "config.json", {"adapter_manifest": "adapters.json"}, self._write_registry(root)
            )
            self.assertTrue(result["ready"])
            self.assertEqual(result["ready_count"], 2)

    def test_container_adapter_binds_image_and_tool_versions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proof = "a" * 64
            image_id = "sha256:" + "c" * 64
            image = "example.invalid/cargo@sha256:" + "b" * 64
            manifest = {
                "schema_version": preflight.ADAPTER_SCHEMA,
                "adapters": [{
                    "repository_id": "repo-a",
                    "status": "ready",
                    "toolchain_profile": "rust-cargo",
                    "proof_sha256": proof,
                    "image": image,
                    "image_id": image_id,
                    "commands": {
                        "probe": ["probe-adapter"],
                        "metadata": ["metadata-adapter"],
                        "validation": ["validation-adapter"],
                    },
                }],
            }
            (root / "adapters.json").write_text(json.dumps(manifest), encoding="utf-8")
            registry = {"repositories": [{"id": "repo-a", "toolchain_profile": "rust-cargo"}]}
            inspect = mock.Mock(returncode=0, stdout=json.dumps([{"Id": image_id}]), stderr="")
            def report(role: str, *, include_tools: bool = False) -> mock.Mock:
                payload = {
                    "ready": True,
                    "repository_id": "repo-a",
                    "command_role": role,
                    "proof_sha256": proof,
                }
                if include_tools:
                    payload.update({
                        "tools": {"cargo": "cargo 1.90.0", "rustc": "rustc 1.90.0"},
                        "agent_tools": {"rg": "ripgrep 14.1.0", "pi": "pi 0.80.2", "greppy": "greppy 0.3.0"},
                    })
                return mock.Mock(returncode=0, stdout=json.dumps(payload), stderr="")
            probe = report("probe", include_tools=True)
            metadata = report("metadata")
            validation = report("validation")
            with mock.patch.object(preflight, "resolve_command", return_value="/usr/bin/docker"), mock.patch.object(preflight.subprocess, "run", side_effect=[inspect, probe, metadata, validation]) as run:
                result = preflight.check_adapters(root / "config.json", {"adapter_manifest": "adapters.json"}, registry, "container")
            self.assertTrue(result["ready"])
            self.assertEqual(len(run.call_args_list), 4)
            self.assertIn("--network", run.call_args_list[1].args[0])
            self.assertEqual(run.call_args_list[1].args[0][run.call_args_list[1].args[0].index("--network") + 1], "none")


class ReportTests(unittest.TestCase):
    def test_invalid_config_still_returns_machine_readable_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")
            result = preflight.run_preflight(path)
            self.assertFalse(result["ready"])
            self.assertEqual(result["schema_version"], preflight.REPORT_SCHEMA)
            self.assertEqual(result["failures"][0]["code"], "configuration")


if __name__ == "__main__":
    unittest.main()
