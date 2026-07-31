from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from bench.agent_coding.v3.pipeline import (
    HarvestError,
    canonical_json,
    extract_patch,
    harvest,
    load_freeze,
    run_adapter_stage,
)
from bench.agent_coding.v3.runner import load_release
from bench.agent_coding.v3.storage import StorageError, StorageLayout, load_storage


COMMIT_ENV = {
    "GIT_AUTHOR_DATE": "2026-06-10T12:00:00+00:00",
    "GIT_COMMITTER_DATE": "2026-06-10T12:00:00+00:00",
}


def run(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    import os
    command_env = os.environ.copy()
    if env:
        command_env.update(env)
    return subprocess.run(
        list(args), cwd=repo, check=True, capture_output=True, text=True, env=command_env
    ).stdout.strip()


class PipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.layout = StorageLayout(root / "fast", root / "nas")
        self.layout.ensure()
        self.freeze_path = root / "freeze.json"
        self.freeze_path.write_text(json.dumps({
            "schema_version": "greppy.agent-coding-freeze.v1",
            "freeze_id": "sealed_test_2026q3",
            "frozen_at": "2026-07-31T12:00:00Z",
            "eligible_pr_created_after": "2026-05-01T00:00:00Z",
            "eligible_merged_after": "2026-05-01T00:00:00Z",
            "eligible_merged_before": "2026-07-15T23:59:59Z",
            "source_metadata_cutoff": "2026-07-20T00:00:00Z",
        }), encoding="utf-8")
        self.registry_path = root / "registry.json"
        self.registry_path.write_text(json.dumps({
            "schema_version": "greppy.agent-coding-v3.repository-registry.1",
            "registry_id": "test",
            "target_task_count": 2,
            "repository_count": 2,
            "tasks_per_repository": 1,
            "primary_languages": ["python"],
            "language_task_quota": 2,
            "selection_patterns": {"A": {"reported_bugfix": 1}},
            "toolchain_profiles": {"python-pip": {
                "gpu3_prerequisites": ["python3"], "runner_family": "pytest"
            }},
            "repositories": [
                {"id": "python-one", "url": "https://example.test/one", "primary_language": "python", "selection_pattern": "A", "toolchain_profile": "python-pip"},
                {"id": "python-two", "url": "https://example.test/two", "primary_language": "python", "selection_pattern": "A", "toolchain_profile": "python-pip"},
            ],
        }), encoding="utf-8")
        self.key_path = root / "selection.key"
        self.key_path.write_bytes(b"a private preregistered selection key" * 2)
        self.contract_path = root / "contract.json"
        self.contract_path.write_text(json.dumps({
            "schema_version": "greppy.agent-coding-v3.corpus-contract.1",
            "corpus": {"target_tasks": 2, "repositories": 2, "tasks_per_repository": 1, "languages": 1},
            "temporal_holdout": {
                "candidate_pr_created_at_or_after": "2026-05-01T00:00:00Z",
                "candidate_pr_merged_at_or_after": "2026-05-01T00:00:00Z",
                "candidate_pr_merged_at_or_before": "2026-07-15T23:59:59Z",
            },
            "repository_scale": {
                "minimum_eligible_source_files": 200,
                "minimum_eligible_source_loc": 25000,
            },
            "validation": {"minimum_candidate_pool_per_repo_class_slot": 2},
        }), encoding="utf-8")
        self.adapter_path = root / "adapters.json"
        self.adapter_path.write_text(json.dumps({
            "schema_version": "greppy.agent-coding-v3.adapter-manifest.1",
            "adapters": [
                {
                    "repository_id": key, "status": "ready", "toolchain_profile": "python-pip",
                    "image": "example.test/adapter@sha256:" + str(number) * 64,
                    "image_id": "sha256:" + str(number + 2) * 64,
                    "proof_sha256": str(number) * 64,
                    "commands": {
                        "probe": ["adapter-probe"],
                        "metadata": ["adapter-metadata"],
                        "validation": ["adapter-validation"],
                    },
                }
                for number, key in enumerate(("python-one", "python-two"), 4)
            ],
        }), encoding="utf-8")
        self.denylist_path = root / "denylist.json"
        self.denylist_path.write_text(json.dumps({
            "schema_version": "greppy.agent-coding-v3.denylist.1",
            "coverage": ["swe-bench", "prior-greppy"],
            "entries": [],
        }), encoding="utf-8")
        rows = []
        for number, key in enumerate(("python-one", "python-two"), 1):
            repo, parent, solutions = self.make_repo(key, number)
            parent_tree = run(repo, "git", "rev-parse", f"{parent}^{{tree}}")
            scale = {
                "measurement_revision": "v1", "parent_tree": parent_tree,
                "eligible_source_files": 250, "eligible_source_loc": 30000,
                "size_band": "medium", "excluded_vendor_generated_files": 0,
            }
            scale["measurement_sha256"] = hashlib.sha256(
                json.dumps(scale, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for variant_index, (variant, solution, sources, test_path) in enumerate(solutions, 1):
                paths = [*sources, test_path]
                test_patch = extract_patch(repo, parent, solution, [test_path])
                gold_patch = extract_patch(repo, parent, solution, sources)
                full_patch = extract_patch(repo, parent, solution, paths)
                identity = number * 10 + variant_index
                titles = {
                    11: "Lunar parser rejects inverted envelopes",
                    12: "Volcanic cache preserves cobalt markers",
                    21: "Orbital scheduler handles empty epochs",
                    22: "Quartz serializer normalizes broken frames",
                }
                rows.append({
                    "repository": key, "pr_number": 100 + identity,
                    "issue_number": 200 + identity,
                    "issue_url": f"https://example.test/issues/{200 + identity}",
                    "solution_commit": solution, "parent_commit": parent,
                    "merge_strategy": "squash",
                    "merge_provenance": {
                        "target_parent_verified": True, "merged_result_tree_verified": True,
                        "pr_delta_no_target_drift": True,
                        "authoritative_metadata_sha256": f"{identity % 10}" * 64,
                    },
                    "created_at": "2026-06-01T00:00:00Z",
                    "merged_at": "2026-06-10T12:00:00Z",
                    "issue_title": titles[identity],
                    "issue_body": f"The {variant} result violates a separately documented boundary.",
                    "task_class": "reported_bugfix", "changed_source": sources,
                    "changed_tests": [test_path], "repository_scale": scale,
                    "validation": {
                        "parent_baseline": "pass", "parent_plus_test": "fail",
                        "gold_plus_test": "pass", "merged_plus_test": "pass",
                        "clean_room_repetitions": 2, "offline": True,
                        "fail_to_pass": [f"{test_path}::test_{variant}"],
                        "pass_to_pass": [f"{test_path}::test_existing"],
                        "test_patch_sha256": hashlib.sha256(test_patch).hexdigest(),
                        "gold_patch_sha256": hashlib.sha256(gold_patch).hexdigest(),
                        "full_patch_sha256": hashlib.sha256(full_patch).hexdigest(),
                        "test_command": ["python3", "-m", "pytest", "-q"],
                        "setup_commands": [],
                        "post_patch_commands": [["python3", "-c", "pass"]],
                        "runner_image_digest": "sha256:" + "3" * 64,
                        "logs_sha256": f"{identity % 10}" * 64,
                        "validated_at": "2026-07-20T00:00:00Z",
                    },
                })
        self.metadata_stage_dir = root / "stages" / "metadata"
        self.validation_stage_dir = root / "stages" / "validate"
        self.metadata_stage_dir.mkdir(parents=True)
        self.validation_stage_dir.mkdir(parents=True)
        self.candidates_path = self.validation_stage_dir / "all.jsonl"
        self.candidates_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (self.metadata_stage_dir / "all.jsonl").write_text(
            "".join(json.dumps({"candidate": number}) + "\n" for number in range(36)),
            encoding="utf-8",
        )
        self.stage_manifest_paths = [
            self.metadata_stage_dir / "stage-manifest.json",
            self.validation_stage_dir / "stage-manifest.json",
        ]
        self.refresh_stage_manifests()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def harvest_release(self) -> Path:
        self.refresh_stage_manifests()
        return harvest(
            registry_path=self.registry_path, freeze_path=self.freeze_path,
            candidates_path=self.candidates_path, id_key_path=self.key_path,
            contract_path=self.contract_path, adapter_manifest_path=self.adapter_path,
            denylist_paths=[self.denylist_path],
            stage_manifest_paths=self.stage_manifest_paths, layout=self.layout,
        )

    def refresh_stage_manifests(self) -> None:
        adapters = json.loads(self.adapter_path.read_text())["adapters"]
        images = {
            row["repository_id"]: {"image": row["image"], "image_id": row["image_id"]}
            for row in sorted(adapters, key=lambda item: item["repository_id"])
        }
        adapter_hash = hashlib.sha256(self.adapter_path.read_bytes()).hexdigest()
        for stage, directory, network in (
            ("metadata", self.metadata_stage_dir, "bridge"),
            ("validate", self.validation_stage_dir, "none"),
        ):
            combined = directory / "all.jsonl"
            (directory / "stage-manifest.json").write_text(json.dumps({
                "schema_version": "greppy.agent-coding-v3.adapter-stage.1",
                "stage": stage, "freeze_id": "sealed_test_2026q3",
                "adapter_manifest_sha256": adapter_hash, "network": network,
                "images": images, "outputs": {},
                "combined_sha256": hashlib.sha256(combined.read_bytes()).hexdigest(),
            }), encoding="utf-8")

    def make_repo(self, key: str, number: int) -> tuple[Path, str, list[tuple[str, str, list[str], str]]]:
        repo = self.layout.mirrors / f"{key}.git"
        repo.mkdir()
        run(repo, "git", "init", "-q")
        run(repo, "git", "config", "user.email", "bench@example.test")
        run(repo, "git", "config", "user.name", "Bench")
        run(repo, "git", "remote", "add", "origin", f"https://example.test/{key.removeprefix('python-')}")
        (repo / "src").mkdir()
        (repo / "tests").mkdir()
        variants = ("lunar", "volcanic")
        repo_word = key.replace("python-", "")
        for variant in variants:
            for suffix in ("core", "edge"):
                (repo / f"src/{repo_word}_{variant}_{suffix}.py").write_text(
                    "".join(f"{repo_word}_{variant}_{suffix}_{line} = {line}\n" for line in range(20)),
                    encoding="utf-8",
                )
            (repo / f"tests/test_{repo_word}_{variant}.py").write_text(
                "def test_existing(): assert True\n", encoding="utf-8"
            )
        run(repo, "git", "add", ".")
        run(repo, "git", "commit", "-qm", "parent", env=COMMIT_ENV)
        parent = run(repo, "git", "rev-parse", "HEAD")
        solutions = []
        for variant_index, variant in enumerate(variants, 1):
            run(repo, "git", "checkout", "-q", parent)
            sources = [f"src/{repo_word}_{variant}_{suffix}.py" for suffix in ("core", "edge")]
            for suffix, source in zip(("core", "edge"), sources):
                (repo / source).write_text(
                    "".join(
                        f"fixed_{repo_word}_{variant}_{suffix}_{line} = {number * 100 + variant_index * 20 + line}\n"
                        for line in range(20)
                    ), encoding="utf-8",
                )
            test_path = f"tests/test_{repo_word}_{variant}.py"
            (repo / test_path).write_text(
                f"def test_existing(): assert True\ndef test_{variant}(): assert {number + variant_index} > 0\n",
                encoding="utf-8",
            )
            run(repo, "git", "add", ".")
            run(repo, "git", "commit", "-qm", f"merged {variant} solution", env=COMMIT_ENV)
            solutions.append((variant, run(repo, "git", "rev-parse", "HEAD"), sources, test_path))
        return repo, parent, solutions

    def test_seal_is_opaque_and_snapshot_has_only_parent_tree(self) -> None:
        output = self.harvest_release()
        public_text = (output / "public/taskbank.json").read_text(encoding="utf-8")
        public = json.loads(public_text)
        sealed = json.loads((output / "sealed/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(public["tasks"]), 2)
        self.assertFalse(public["execution_contract"]["apply_test_patch_before_agent"])
        self.assertNotIn("pr_number", public_text)
        self.assertNotIn("solution_commit", public_text)
        self.assertNotIn("tests/test_", public_text)
        self.assertNotIn("pytest", public_text)
        for private_task, public_task in zip(sealed["tasks"], public["tasks"]):
            self.assertNotIn(private_task["solution_commit"], public_text)
            self.assertRegex(public_task["id"], r"^task_[a-z2-7]{26}$")
            with tarfile.open(output / "public" / public_task["workspace"]["snapshot"]) as archive:
                names = archive.getnames()
                self.assertNotIn(".git", names)
                self.assertFalse(any(public_task["id"] in name for name in names))
                sample = next(name for name in names if name.startswith("src/") and name.endswith(".py"))
                self.assertNotIn(b"fixed_", archive.extractfile(sample).read())
            self.assertEqual(private_task["admission"]["production_path_count"], 2)
            self.assertEqual(private_task["admission"]["production_changed_lines"], 80)
            self.assertEqual(private_task["admission"]["total_path_count"], 3)
            evaluation = private_task["evaluation"]
            self.assertEqual(evaluation["post_patch_commands"], [["python3", "-c", "pass"]])
            self.assertEqual(
                private_task["hashes"]["evaluation_sha256"],
                hashlib.sha256(canonical_json(evaluation)).hexdigest(),
            )
        pairs = load_release(output / "public", output / "sealed")
        self.assertEqual(len(pairs), 2)
        commitments = sealed["commitments"]
        for field in (
            "corpus_contract_sha256", "canonical_candidate_ledger_sha256",
            "adapter_manifest_sha256", "toolchain_profiles_sha256", "freeze_manifest_sha256",
        ):
            self.assertRegex(commitments[field], r"^[0-9a-f]{64}$")

    def test_cutoff_is_enforced_before_writing_release(self) -> None:
        rows = [json.loads(line) for line in self.candidates_path.read_text().splitlines()]
        rows[0]["created_at"] = "2026-04-30T23:59:59Z"
        self.candidates_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(HarvestError, "creation cutoff"):
            self.harvest_release()
        self.assertFalse((self.layout.releases / "sealed_test_2026q3").exists())

    def test_seal_rejects_missing_or_shell_string_post_patch_commands(self) -> None:
        original = [json.loads(line) for line in self.candidates_path.read_text().splitlines()]
        for invalid in (None, ["python3 -c pass"]):
            rows = json.loads(json.dumps(original))
            if invalid is None:
                rows[0]["validation"].pop("post_patch_commands")
            else:
                rows[0]["validation"]["post_patch_commands"] = invalid
            self.candidates_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                HarvestError, "post_patch_commands"
            ):
                self.harvest_release()

    def test_every_repo_class_slot_requires_two_passing_candidates(self) -> None:
        rows = [json.loads(line) for line in self.candidates_path.read_text().splitlines()]
        self.candidates_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows if not (
                row["repository"] == "python-one" and row["pr_number"] == 112
            )), encoding="utf-8",
        )
        with self.assertRaisesRegex(HarvestError, "requires at least 2"):
            self.harvest_release()

    def test_repository_scale_floor_is_enforced(self) -> None:
        rows = [json.loads(line) for line in self.candidates_path.read_text().splitlines()]
        rows[0]["repository_scale"]["eligible_source_loc"] = 24999
        material = {
            key: value for key, value in rows[0]["repository_scale"].items()
            if key != "measurement_sha256"
        }
        rows[0]["repository_scale"]["measurement_sha256"] = hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.candidates_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(HarvestError, "source-LOC floor"):
            self.harvest_release()

    def test_exact_denylist_match_blocks_candidate(self) -> None:
        row = json.loads(self.candidates_path.read_text().splitlines()[0])
        self.denylist_path.write_text(json.dumps({
            "schema_version": "greppy.agent-coding-v3.denylist.1",
            "coverage": ["swe-bench", "prior-greppy"],
            "entries": [{
                "repository_url": "https://example.test/one",
                "solution_commit": row["solution_commit"],
            }],
        }))
        with self.assertRaisesRegex(HarvestError, "matches denylist"):
            self.harvest_release()

    def test_near_duplicate_requires_blinded_review_hook(self) -> None:
        rows = [json.loads(line) for line in self.candidates_path.read_text().splitlines()]
        rows[1]["issue_title"] = rows[0]["issue_title"]
        self.candidates_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(HarvestError, "near-duplicate"):
            self.harvest_release()

    def test_adapter_stages_run_only_in_pinned_images_with_isolated_mounts(self) -> None:
        adapters = json.loads(self.adapter_path.read_text())["adapters"]
        image_ids = {row["image"]: row["image_id"] for row in adapters}
        docker_runs: list[list[str]] = []

        def fake_docker(argv, **kwargs):
            command = list(argv)
            if command[1:3] == ["image", "inspect"]:
                return subprocess.CompletedProcess(
                    command, 0, json.dumps([{"Id": image_ids[command[3]]}]), ""
                )
            self.assertEqual(command[:2], ["fake-docker", "run"])
            docker_runs.append(command)
            mounts = {}
            for index, value in enumerate(command):
                if value != "--mount":
                    continue
                fields = dict(
                    field.split("=", 1) for field in command[index + 1].split(",")
                    if "=" in field
                )
                mounts[fields["dst"]] = Path(fields["src"])
            output_value = command[command.index("--output") + 1]
            host_output = mounts["/output"] / Path(output_value).name
            if "adapter-metadata" in command:
                rows = [
                    {
                        "candidate": number,
                        "authoritative_changed_paths": (
                            ["src/one.py", "src/two.py", "tests/test_one.py"]
                            if number < 18 else ["README.md"]
                        ),
                    }
                    for number in range(36)
                ]
            else:
                rows = [{"validated": True}, {"validated": True}]
            host_output.write_text("".join(json.dumps(row) + "\n" for row in rows))
            return subprocess.CompletedProcess(command, 0, "", "")

        with mock.patch(
            "bench.agent_coding.v3.pipeline.subprocess.run", side_effect=fake_docker
        ):
            metadata_env = Path(self.temp.name) / "github.env"
            metadata_env.write_text("GITHUB_TOKEN=fake-test-token\n")
            metadata = run_adapter_stage(
                stage="metadata", registry_path=self.registry_path,
                freeze_path=self.freeze_path, adapter_manifest_path=self.adapter_path,
                layout=self.layout, docker_binary="fake-docker",
                metadata_env_file=metadata_env,
            )
            validated = run_adapter_stage(
                stage="validate", registry_path=self.registry_path,
                freeze_path=self.freeze_path, adapter_manifest_path=self.adapter_path,
                layout=self.layout, docker_binary="fake-docker",
            )
        self.assertTrue(metadata.is_file())
        self.assertTrue(validated.is_file())
        metadata_runs = [argv for argv in docker_runs if "adapter-metadata" in argv]
        validation_runs = [argv for argv in docker_runs if "adapter-validation" in argv]
        self.assertEqual(len(metadata_runs), 2)
        self.assertEqual(len(validation_runs), 2)
        for argv in metadata_runs:
            self.assertEqual(argv[argv.index("--network") + 1], "bridge")
            mount_specs = [argv[index + 1] for index, value in enumerate(argv) if value == "--mount"]
            self.assertEqual(len(mount_specs), 1)
            self.assertIn("dst=/output", mount_specs[0])
            self.assertIn("--per-repo", argv)
            self.assertEqual(argv[argv.index("--per-repo") + 1], "36")
        for argv in validation_runs:
            self.assertEqual(argv[argv.index("--network") + 1], "none")
            mount_specs = [argv[index + 1] for index, value in enumerate(argv) if value == "--mount"]
            self.assertEqual(len(mount_specs), 4)
            self.assertTrue(any("dst=/input/mirror,readonly" in spec for spec in mount_specs))
            self.assertTrue(any("dst=/input/metadata.jsonl,readonly" in spec for spec in mount_specs))
            self.assertIn("--runner-image-id", argv)
            self.assertRegex(argv[argv.index("--runner-image-id") + 1], r"^sha256:[0-9a-f]{64}$")
        manifest = json.loads(
            (validated.parent / "stage-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["network"], "none")
        self.assertEqual(
            manifest["images"]["python-one"]["image_id"], image_ids[adapters[0]["image"]]
        )


class StorageTest(unittest.TestCase):
    def test_explicit_distinct_roots_required(self) -> None:
        with self.assertRaises(StorageError):
            load_storage(environ={}, create=False)
        with self.assertRaises(StorageError):
            load_storage(environ={
                "GREPPY_BENCH_NVME_ROOT": "/tmp/shared",
                "GREPPY_BENCH_NAS_ROOT": "/tmp/shared/nas",
            }, create=False)

    def test_freeze_rejects_naive_times(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "freeze.json"
            path.write_text(json.dumps({
                "schema_version": "greppy.agent-coding-freeze.v1",
                "freeze_id": "x1", "frozen_at": "2026-08-01",
                "eligible_pr_created_after": "2026-05-01T00:00:00Z",
                "eligible_merged_after": "2026-05-01T00:00:00Z",
                "eligible_merged_before": "2026-07-01T00:00:00Z",
                "source_metadata_cutoff": "2026-07-02T00:00:00Z",
            }))
            with self.assertRaisesRegex(HarvestError, "timezone"):
                load_freeze(path)


if __name__ == "__main__":
    unittest.main()
