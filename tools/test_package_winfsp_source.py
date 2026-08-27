from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import package_winfsp_source as source_package


class WinFspSourceArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.repository = self.root / "repository"
        self.fork = self.repository / "third_party/winfsp-greppy"
        self.source.mkdir()
        self.fork.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", self.source], check=True)
        subprocess.run(
            ["git", "-C", self.source, "config", "user.email", "test@example.invalid"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", self.source, "config", "user.name", "Test"], check=True
        )
        (self.source / "source.c").write_text("old\n", encoding="utf-8")
        (self.source / "untouched.h").write_text("header\n", encoding="utf-8")
        subprocess.run(["git", "-C", self.source, "add", "."], check=True)
        subprocess.run(["git", "-C", self.source, "commit", "-qm", "base"], check=True)
        self.commit = (
            subprocess.run(
                ["git", "-C", self.source, "rev-parse", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            .stdout.strip()
        )
        self.patch = self.fork / "patches/change.patch"
        self.patch.parent.mkdir()
        self.patch.write_text(
            "diff --git a/source.c b/source.c\n"
            "--- a/source.c\n"
            "+++ b/source.c\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n",
            encoding="utf-8",
        )
        subprocess.run(
            ["git", "-C", self.source, "apply", self.patch], check=True
        )
        for relative in source_package.SUPPORT_FILES:
            path = self.repository / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(f"support:{relative}\n", encoding="utf-8")
        self.fork_result = {
            "repository": "https://github.com/winfsp/winfsp",
            "tag": "v2.1",
            "commit": self.commit,
            "tag_object": "1" * 40,
            "license": "GPL-3.0-only WITH WinFsp-FLOSS-exception",
            "modified_files": ["source.c"],
            "submodules": [],
            "patches": [
                {
                    "path": "patches/change.patch",
                    "sha256": hashlib.sha256(self.patch.read_bytes()).hexdigest(),
                    "modified_files": ["source.c"],
                }
            ],
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def create(self, output: Path) -> dict[str, object]:
        with mock.patch.object(
            source_package.verify_winfsp_fork,
            "verify",
            return_value=copy.deepcopy(self.fork_result),
        ):
            return source_package.create_archive(
                self.source, self.repository, self.fork, output
            )

    def test_archive_is_deterministic_complete_and_self_verifying(self) -> None:
        first = self.root / "first.tar.gz"
        second = self.root / "second.tar.gz"
        first_manifest = self.create(first)
        second_manifest = self.create(second)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(first_manifest, second_manifest)
        verified = source_package.verify_archive(first)
        self.assertEqual(verified["upstream"]["commit"], self.commit)
        self.assertIn(
            f"{source_package.ARCHIVE_ROOT}/upstream/source.c",
            verified["members"],
        )
        self.assertNotIn(
            f"{source_package.ARCHIVE_ROOT}/upstream/.git/HEAD",
            verified["members"],
        )

    def test_rejects_unbound_source_change(self) -> None:
        (self.source / "untouched.h").write_text("changed\n", encoding="utf-8")
        with self.assertRaisesRegex(source_package.SourceArchiveError, "file set mismatch"):
            self.create(self.root / "bad.tar.gz")

    def test_verifier_rejects_member_digest_mismatch(self) -> None:
        name = f"{source_package.ARCHIVE_ROOT}/upstream/source.c"
        manifest = {
            "schema": source_package.SCHEMA,
            "members": {
                name: {"kind": "file", "mode": 0o644, "sha256": "0" * 64}
            },
        }
        path = self.root / "tampered.tar.gz"
        with path.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as archive:
                    for member_name, data in (
                        (
                            source_package.MANIFEST_MEMBER,
                            (json.dumps(manifest) + "\n").encode("utf-8"),
                        ),
                        (name, b"new\n"),
                    ):
                        info = tarfile.TarInfo(member_name)
                        info.size = len(data)
                        info.mode = 0o644
                        archive.addfile(info, io.BytesIO(data))
        with self.assertRaisesRegex(source_package.SourceArchiveError, "digest mismatch"):
            source_package.verify_archive(path)


if __name__ == "__main__":
    unittest.main()
