from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from . import base
from . import build_manifest


def git(cwd: pathlib.Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


class LocalRepositoryFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="v3-adapter-test-")
        self.root = pathlib.Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.name", "Adapter Test")
        git(self.repo, "config", "user.email", "adapter@example.invalid")
        (self.repo / "value.txt").write_text("old\n", encoding="utf-8")
        (self.repo / "logic.py").write_text("VALUE = 'old'\n", encoding="utf-8")
        (self.repo / "test_existing.py").write_text(
            "import pathlib\nassert pathlib.Path('value.txt').exists()\n", encoding="utf-8"
        )
        git(self.repo, "add", ".")
        git(self.repo, "commit", "-qm", "parent")
        self.parent = git(self.repo, "rev-parse", "HEAD")
        (self.repo / "value.txt").write_text("new\n", encoding="utf-8")
        (self.repo / "logic.py").write_text("VALUE = 'new'\n", encoding="utf-8")
        (self.repo / "test_hidden.py").write_text(
            "import pathlib\nassert pathlib.Path('value.txt').read_text() == 'new\\n'\n",
            encoding="utf-8",
        )
        git(self.repo, "add", ".")
        git(self.repo, "commit", "-qm", "solution")
        self.solution = git(self.repo, "rev-parse", "HEAD")
        self.paths = git(
            self.repo, "diff", "--name-only", "--no-renames", self.parent, self.solution
        ).splitlines()
        self.config = {
            "schema_version": "greppy.agent-coding-v3.adapter-config.1",
            "repository_id": "fixture", "repository_url": "https://github.com/example/fixture",
            "primary_language": "python", "toolchain_profile": "python-pip",
            "test_globs": ["test_hidden.py"], "ignore_globs": [],
            "source_extensions": [".py", ".txt"],
            "f2p_command": [sys.executable, "{first_test}"],
            "p2p_command": [sys.executable, "test_existing.py"],
            "setup_commands": [],
            "post_patch_commands": [[sys.executable, "-c", "pass"]],
            "timeout_seconds": 20,
        }
        self.row = {
            "repository": "fixture", "repository_url": "https://github.com/example/fixture",
            "pr_number": 7, "issue_number": 8, "issue_url": "https://github.com/example/fixture/issues/8",
            "issue_title": "Change the value", "issue_body": "Expected new behavior.",
            "created_at": "2026-06-01T00:00:00Z", "merged_at": "2026-06-02T00:00:00Z",
            "solution_commit": self.solution, "parent_commit": self.parent,
            "authoritative_changed_paths": self.paths, "merge_strategy": "squash",
            "task_class": "reported_bugfix", "authoritative_metadata_sha256": "a" * 64,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()


class ValidationTests(LocalRepositoryFixture):
    def test_two_clean_offline_repetitions_prove_f2p_and_p2p(self) -> None:
        result = base.validate_candidate(
            self.row, self.repo, self.root / "scratch", 2, self.config, "sha256:" + "b" * 64
        )
        proof = result["validation"]
        self.assertEqual(proof["parent_plus_test"], "fail")
        self.assertEqual(proof["gold_plus_test"], "pass")
        self.assertEqual(proof["clean_room_repetitions"], 2)
        self.assertTrue(proof["offline"])
        self.assertEqual(proof["post_patch_commands"], [[sys.executable, "-c", "pass"]])
        self.assertEqual(result["changed_tests"], ["test_hidden.py"])
        self.assertEqual(set(result["changed_source"]), {"logic.py", "value.txt"})
        self.assertRegex(proof["logs_sha256"], r"^[0-9a-f]{64}$")

    def test_m1_mismatch_fails_closed(self) -> None:
        row = {**self.row, "parent_commit": "0" * 40}
        with self.assertRaisesRegex(base.AdapterError, "M\\^1 provenance"):
            base.validate_candidate(row, self.repo, self.root / "bad", 2, self.config, "sha256:" + "b" * 64)

    def test_post_patch_commands_reject_shell_strings(self) -> None:
        config = {**self.config, "post_patch_commands": ["python3 -c pass"]}
        with self.assertRaisesRegex(base.AdapterError, "post_patch_commands must be argv arrays"):
            base.validate_candidate(
                self.row, self.repo, self.root / "shell-string", 2, config,
                "sha256:" + "b" * 64,
            )


class FakeGitHub:
    def __init__(self, pr: dict): self.pr = pr
    def rest(self, path: str): return [{"number": 10, "created_at": "2026-06-01T00:00:00Z", "merged_at": "2026-06-02T00:00:00Z"}] if "page=1" in path else []
    def graphql(self, query: str, variables): return {"repository": {"pullRequest": self.pr}}


def github_pr(*, parents: list[str], merge_oid: str, commits: list[str]) -> dict:
    return {
        "number": 10, "createdAt": "2026-06-01T00:00:00Z", "mergedAt": "2026-06-02T00:00:00Z",
        "merged": True, "mergeCommit": {"oid": merge_oid, "parents": {"nodes": [{"oid": oid} for oid in parents]}},
        "commits": {"nodes": [{"commit": {"oid": oid}} for oid in commits]},
        "files": {"nodes": [{"path": "src/a.py"}, {"path": "tests/test_a.py"}]},
        "closingIssuesReferences": {"nodes": [{
            "number": 4, "title": "Fix behavior", "body": "Body", "url": "https://github.com/o/r/issues/4",
            "labels": {"nodes": [{"name": "bug"}]},
        }]},
    }


class MetadataTests(unittest.TestCase):
    NOW = dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)

    def test_merged_pr_with_linked_issue_emits_authoritative_metadata(self) -> None:
        pr = github_pr(parents=["1" * 40, "2" * 40], merge_oid="3" * 40, commits=["4" * 40])
        rows = base.harvest_metadata(
            client=FakeGitHub(pr), repository_id="repo", repository_url="https://github.com/o/r",
            created_after=self.NOW, merged_after=self.NOW,
            merged_before=dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc), target=1,
            config={"default_task_class": "reported_bugfix"},
        )
        self.assertEqual(rows[0]["issue_number"], 4)
        self.assertEqual(rows[0]["parent_commit"], "1" * 40)
        self.assertEqual(rows[0]["merge_strategy"], "merge")
        self.assertRegex(rows[0]["authoritative_metadata_sha256"], r"^[0-9a-f]{64}$")

    def test_multi_commit_rebase_is_rejected_as_ambiguous(self) -> None:
        merge_oid = "3" * 40
        pr = github_pr(parents=["2" * 40], merge_oid=merge_oid, commits=["1" * 40, merge_oid])
        with self.assertRaisesRegex(base.AdapterError, "yielded 0"):
            base.harvest_metadata(
                client=FakeGitHub(pr), repository_id="repo", repository_url="https://github.com/o/r",
                created_after=self.NOW, merged_after=self.NOW,
                merged_before=dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc), target=1,
                config={},
            )


class ProbeAndManifestTests(unittest.TestCase):
    def test_supported_productive_toolchain_families_are_real_profiles(self) -> None:
        self.assertTrue({
            "python-pip", "rust-cargo", "go-test", "java-maven", "java-gradle",
            "ts-pnpm", "javascript-node", "cpp-cmake", "ruby-bundler",
        } <= set(base.PROFILES))

    def test_configs_and_pending_manifest_cover_registry_exactly(self) -> None:
        adapter_root = pathlib.Path(__file__).parent
        registry = adapter_root.parent / "repository_registry.json"
        config_paths = sorted((adapter_root / "configs").glob("*.json"))
        loaded = build_manifest.validate_config_coverage(config_paths, registry)
        registry_ids = [row[1]["repository_id"] for row in loaded]
        manifest = json.loads((adapter_root / "smoke_manifest.json").read_text(encoding="utf-8"))
        manifest_ids = [row["repository_id"] for row in manifest["adapters"]]
        self.assertEqual(len(registry_ids), 24)
        self.assertEqual(set(manifest_ids), set(registry_ids))
        self.assertEqual(len(manifest_ids), len(set(manifest_ids)))
        self.assertTrue(all(row["status"] == "pending" and row.get("reason") for row in manifest["adapters"]))

    def test_config_coverage_rejects_one_missing_repository(self) -> None:
        adapter_root = pathlib.Path(__file__).parent
        registry = adapter_root.parent / "repository_registry.json"
        config_paths = sorted((adapter_root / "configs").glob("*.json"))
        with self.assertRaisesRegex(ValueError, "missing adapter configs"):
            build_manifest.validate_config_coverage(config_paths[:-1], registry)

    def test_preflight_executes_tools_and_binds_proof(self) -> None:
        with tempfile.TemporaryDirectory(prefix="adapter-probe-") as temporary:
            path = pathlib.Path(temporary) / "config.json"
            config = {
                "schema_version": "greppy.agent-coding-v3.adapter-config.1",
                "repository_id": "fixture", "repository_url": "https://github.com/o/r",
                "primary_language": "go", "toolchain_profile": "go-test",
            }
            path.write_text(json.dumps(config), encoding="utf-8")
            with mock.patch.object(base, "tool_version", side_effect=lambda name: f"{name} 1.2.3") as version:
                payload = base.preflight_payload("probe", path, config)
            self.assertTrue(payload["ready"])
            self.assertEqual(version.call_count, 4)
            self.assertEqual(payload["agent_tools"]["greppy"], "greppy 1.2.3")
            self.assertRegex(payload["proof_sha256"], r"^[0-9a-f]{64}$")

    def test_manifest_never_marks_unprobed_row_ready(self) -> None:
        config = pathlib.Path(__file__).parent / "configs" / "go-caddy.json"
        with mock.patch.object(build_manifest.subprocess, "run", return_value=mock.Mock(returncode=2, stdout="{}", stderr="missing tool")):
            row = build_manifest.build_row(
                config, image="example/adapter@sha256:" + "b" * 64,
                image_id="sha256:" + "c" * 64, runtime_config_root="/configs",
            )
        self.assertEqual(row["status"], "pending")
        self.assertIn("missing tool", row["reason"])

    def test_manifest_requires_and_binds_real_smoke_validation(self) -> None:
        config = pathlib.Path(__file__).parent / "configs" / "go-caddy.json"
        loaded = base.load_config(config)
        proof = base.proof_sha256(config, loaded)
        image_id = "sha256:" + "c" * 64
        report = json.dumps({
            "ready": True, "repository_id": "go-caddy", "proof_sha256": proof,
        })
        with tempfile.TemporaryDirectory(prefix="adapter-smoke-") as temporary:
            ledger = pathlib.Path(temporary) / "validated.jsonl"
            ledger.write_text(json.dumps({
                "repository": "go-caddy", "adapter_proof_sha256": proof,
                "changed_source": ["a.go"], "changed_tests": ["a_test.go"],
                "merge_provenance": {
                    "target_parent_verified": True, "merged_result_tree_verified": True,
                    "pr_delta_no_target_drift": True,
                },
                "validation": {
                    "parent_baseline": "pass", "parent_plus_test": "fail",
                    "gold_plus_test": "pass", "merged_plus_test": "pass",
                    "clean_room_repetitions": 2, "offline": True,
                    "runner_image_digest": image_id,
                },
            }) + "\n", encoding="utf-8")
            with mock.patch.object(
                build_manifest.subprocess, "run",
                return_value=mock.Mock(returncode=0, stdout=report, stderr=""),
            ):
                without = build_manifest.build_row(
                    config, image="example/adapter@sha256:" + "b" * 64,
                    image_id=image_id, runtime_config_root="/configs",
                )
                ready = build_manifest.build_row(
                    config, image="example/adapter@sha256:" + "b" * 64,
                    image_id=image_id, runtime_config_root="/configs", smoke_ledger=ledger,
                )
        self.assertEqual(without["status"], "pending")
        self.assertIn("smoke validation ledger", without["reason"])
        self.assertEqual(ready["status"], "ready")
        self.assertEqual(ready["commands"]["validation"][-2:], ["--runner-image-id", image_id])
        self.assertEqual(ready["commands"]["probe"][-1], "--preflight")


if __name__ == "__main__": unittest.main()
