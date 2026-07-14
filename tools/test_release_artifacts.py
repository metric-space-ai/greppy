from __future__ import annotations

import hashlib
import json
import pathlib
import tempfile
import unittest

import tools.release_artifacts as release


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]


class ReleaseArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)

    @staticmethod
    def digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def make_training_tree(self) -> tuple[pathlib.Path, pathlib.Path]:
        training = self.root / "training/qwen35"
        training.mkdir(parents=True)
        files = {
            "README.md": b"training evidence\n",
            "audit-report-2026-07-13.json.gz": b"published audit\n",
        }
        for name, data in files.items():
            (training / name).write_bytes(data)
        manifest = training / "MANIFEST.sha256"
        manifest.write_text(
            "".join(
                f"{self.digest(data)}  {name}\n" for name, data in sorted(files.items())
            ),
            encoding="ascii",
        )
        return training, manifest

    def test_training_archive_is_deterministic_and_self_verifying(self) -> None:
        _, manifest = self.make_training_tree()
        first = self.root / "first.tar.gz"
        second = self.root / "second.tar.gz"

        release.create_training_archive(self.root, manifest, first)
        release.create_training_archive(self.root, manifest, second)

        self.assertEqual(first.read_bytes(), second.read_bytes())
        release.verify_training_archive(first)

    def make_spdx_fixture(self) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
        dist = self.root / "dist"
        dist.mkdir()
        (dist / "greppy").write_bytes(b"binary")
        (dist / "README.md").write_text("read me\n", encoding="utf-8")
        cargo_lock = self.root / "Cargo.lock"
        cargo_lock.write_text(
            "version = 4\n\n"
            "[[package]]\n"
            'name = "dep"\n'
            'version = "1.2.3"\n'
            'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
            f'checksum = "{"1" * 64}"\n\n'
            "[[package]]\n"
            'name = "greppy"\n'
            'version = "0.2.0"\n',
            encoding="utf-8",
        )
        sbom = self.root / "greppy.spdx.json"
        sbom.write_text(
            json.dumps(
                {
                    "spdxVersion": "SPDX-2.3",
                    "dataLicense": "CC0-1.0",
                    "SPDXID": "SPDXRef-DOCUMENT",
                    "name": "syft-generated",
                    "documentNamespace": "https://example.invalid/spdx/test",
                    "creationInfo": {
                        "created": "2026-07-13T00:00:00Z",
                        "creators": ["Tool: syft-test"],
                    },
                    "packages": [],
                    "relationships": [],
                }
            ),
            encoding="utf-8",
        )
        return dist, cargo_lock, sbom

    def test_spdx_binds_package_files_and_complete_lock_graph(self) -> None:
        dist, cargo_lock, sbom = self.make_spdx_fixture()

        release.augment_spdx(sbom, dist, cargo_lock, "x86_64-unknown-linux-gnu")
        release.verify_spdx(sbom, dist, cargo_lock, "x86_64-unknown-linux-gnu")

        document = json.loads(sbom.read_text(encoding="utf-8"))
        lock_packages = [
            package
            for package in document["packages"]
            if package["SPDXID"].startswith("SPDXRef-CargoLock-")
        ]
        self.assertEqual(
            {package["name"] for package in lock_packages}, {"dep", "greppy"}
        )
        root = next(
            package
            for package in document["packages"]
            if package["SPDXID"] == release.RELEASE_PACKAGE_ID
        )
        self.assertIn("do not assert inclusion", root["comment"])

    def test_spdx_rejects_package_content_changed_after_generation(self) -> None:
        dist, cargo_lock, sbom = self.make_spdx_fixture()
        release.augment_spdx(sbom, dist, cargo_lock, "x86_64-unknown-linux-gnu")
        (dist / "greppy").write_bytes(b"tampered")

        with self.assertRaises(release.ReleaseArtifactError):
            release.verify_spdx(sbom, dist, cargo_lock, "x86_64-unknown-linux-gnu")

    def make_small_contract(self) -> pathlib.Path:
        contract = self.root / "contract.json"
        contract.write_text(
            json.dumps(
                {
                    "schema_version": release.CONTRACT_SCHEMA,
                    "repository": "metric-space-ai/greppy",
                    "assets": [
                        {"name": "payload.bin", "role": "package"},
                        {"name": "payload.bin.sha256", "role": "checksum-sidecar"},
                        {
                            "name": release.RELEASE_MANIFEST_NAME,
                            "role": "release-manifest",
                            "generated": True,
                        },
                        {
                            "name": release.AGGREGATE_CHECKSUM_NAME,
                            "role": "aggregate-checksums",
                            "generated": True,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return contract

    def test_staging_enforces_exact_asset_names_and_checksums(self) -> None:
        source = self.root / "source"
        source.mkdir()
        payload = source / "payload.bin"
        payload.write_bytes(b"payload")
        (source / "payload.bin.sha256").write_text(
            f"{release._sha256_file(payload)}  payload.bin\n", encoding="ascii"
        )
        output = self.root / "publish"
        contract = self.make_small_contract()

        release.stage_release(source, output, contract, "a" * 40, "v0.2.0")
        release.verify_staged_release(output, contract, "a" * 40, "v0.2.0")
        self.assertEqual(
            {path.name for path in output.iterdir()},
            {
                "payload.bin",
                "payload.bin.sha256",
                release.RELEASE_MANIFEST_NAME,
                release.AGGREGATE_CHECKSUM_NAME,
            },
        )

        (output / "unexpected.txt").write_text("unexpected", encoding="utf-8")
        with self.assertRaises(release.ReleaseArtifactError):
            release.verify_staged_release(output, contract, "a" * 40, "v0.2.0")

    def test_build_environment_record_is_bound_to_platform_commit_and_lock(
        self,
    ) -> None:
        cargo_lock = self.root / "Cargo.lock"
        cargo_lock.write_text("version = 4\n", encoding="utf-8")
        expected = release.BUILD_ENVIRONMENTS["build-environment-linux-x86_64.json"]
        record = self.root / "build-environment-linux-x86_64.json"
        record.write_text(
            json.dumps(
                {
                    "schema_version": release.BUILD_ENVIRONMENT_SCHEMA,
                    "git_commit": "b" * 40,
                    **expected,
                    "machine_arch": "x86_64",
                    "rustc": {
                        "host": expected["rust_host"],
                        "release": "1.95.0",
                        "commit-hash": "c" * 40,
                    },
                    "cargo": "cargo 1.95.0 (test)",
                    "cargo_lock_sha256": release._sha256_file(cargo_lock),
                    "github_run_id": "1234",
                    "github_run_attempt": "1",
                }
            ),
            encoding="utf-8",
        )

        release.verify_build_environment_record(record, expected, "b" * 40, cargo_lock)
        contents = json.loads(record.read_text(encoding="utf-8"))
        contents["rust_host"] = "wrong-target"
        record.write_text(json.dumps(contents), encoding="utf-8")
        with self.assertRaises(release.ReleaseArtifactError):
            release.verify_build_environment_record(
                record, expected, "b" * 40, cargo_lock
            )

    def test_repository_contract_lists_every_asset_exactly_once(self) -> None:
        contract = release.load_contract(
            REPOSITORY_ROOT / "tools/release_asset_contract.v1.json"
        )
        names = [asset["name"] for asset in contract["assets"]]

        self.assertEqual(len(names), 27)
        self.assertEqual(len(names), len(set(names)))
        self.assertIn(release.TRAINING_ARCHIVE_NAME, names)
        self.assertIn("build-environment-windows-x86_64.json", names)

    def test_release_workflow_keeps_hardening_gates(self) -> None:
        workflow = (REPOSITORY_ROOT / ".github/workflows/release.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("cargo build --locked --release", workflow)
        self.assertIn("record-build-environment", workflow)
        self.assertIn("create-training-archive", workflow)
        self.assertIn("augment-spdx", workflow)
        self.assertIn("stage-release", workflow)
        self.assertIn('gh release create "$GITHUB_REF_NAME"', workflow)
        self.assertNotIn("softprops/action-gh-release", workflow)
        self.assertNotIn("wc -l < release-assets/SHA256SUMS", workflow)
        self.assertGreaterEqual(workflow.count("--timeout-seconds 7200"), 2)
        self.assertIn('if [ "$device" = cpu ]', workflow)


if __name__ == "__main__":
    unittest.main()
