#!/usr/bin/env python3
"""Fail closed unless the minimal Greppy WinFsp fork is exactly pinned."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any


EXPECTED_REPOSITORY = "https://github.com/winfsp/winfsp"
EXPECTED_TAG = "v2.1"
EXPECTED_COMMIT = "ddca7bd5481857a65ba552f643b8776fd070836f"
EXPECTED_TAG_OBJECT = "bcc52225ec7e6a9f5c889b5cdb8051adf41c4b91"
EXPECTED_LICENSE = "GPL-3.0-only WITH WinFsp-FLOSS-exception"
EXPECTED_FILES = {
    "inc/winfsp/fsctl.h",
    "inc/winfsp/winfsp.h",
    "src/dll/fsop.c",
    "src/dll/fuse/fuse_intf.c",
    "src/dll/fuse/fuse_loop.c",
    "src/sys/fileinfo.c",
    "src/sys/volinfo.c",
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ForkError(ValueError):
    """The fork metadata or patch cannot authorize a build."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ForkError(message)


def regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")


def safe_relative_path(value: Any, label: str) -> str:
    require(isinstance(value, str) and value, f"{label} must be a path")
    path = PurePosixPath(value)
    require(not path.is_absolute(), f"{label} must be relative")
    require(".." not in path.parts, f"{label} must not traverse parents")
    require("\\" not in value, f"{label} must use POSIX separators")
    return value


def patch_files(data: bytes, origin: Path) -> set[str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ForkError(f"{origin}: patch is not UTF-8") from error
    require("GIT binary patch" not in text, f"{origin}: binary patch is forbidden")
    old_paths: list[str] = []
    new_paths: list[str] = []
    for line in text.splitlines():
        if line.startswith("--- a/"):
            old_paths.append(line[6:])
        elif line.startswith("+++ b/"):
            new_paths.append(line[6:])
    require(old_paths, f"{origin}: patch contains no files")
    require(old_paths == new_paths, f"{origin}: add/delete/rename is forbidden")
    for index, value in enumerate(old_paths):
        safe_relative_path(value, f"{origin}: file {index}")
    require(len(old_paths) == len(set(old_paths)), f"{origin}: duplicate file section")
    return set(old_paths)


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "upstream.json"
    regular_file(manifest_path, str(manifest_path))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ForkError(f"cannot read {manifest_path}: {error}") from error
    require(isinstance(manifest, dict), "upstream manifest must be an object")
    require(manifest.get("schema") == "greppy.winfsp-transport-upstream.v1", "unsupported schema")
    require(manifest.get("repository") == EXPECTED_REPOSITORY, "upstream repository changed")
    require(manifest.get("tag") == EXPECTED_TAG, "upstream tag changed")
    require(manifest.get("commit") == EXPECTED_COMMIT, "upstream commit changed")
    require(manifest.get("tag_object") == EXPECTED_TAG_OBJECT, "upstream tag object changed")
    require(manifest.get("license") == EXPECTED_LICENSE, "license declaration changed")
    patches = manifest.get("patches")
    require(isinstance(patches, list) and len(patches) == 1, "exactly one patch is required")
    patch = patches[0]
    require(isinstance(patch, dict), "patch record must be an object")
    relative = safe_relative_path(patch.get("path"), "patch path")
    patch_path = root / relative
    regular_file(patch_path, str(patch_path))
    digest = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    expected_digest = patch.get("sha256")
    require(isinstance(expected_digest, str) and SHA256.fullmatch(expected_digest) is not None, "invalid patch SHA-256")
    require(digest == expected_digest, "patch SHA-256 mismatch")
    declared_files = patch.get("modified_files")
    require(isinstance(declared_files, list), "modified_files must be an array")
    declared = {
        safe_relative_path(value, f"modified_files[{index}]")
        for index, value in enumerate(declared_files)
    }
    require(len(declared) == len(declared_files), "modified_files contains duplicates")
    require(declared == EXPECTED_FILES, "modified_files is not the minimal approved set")
    actual = patch_files(patch_path.read_bytes(), patch_path)
    require(actual == declared, "patch file set differs from manifest")
    return {
        "schema": "greppy.winfsp-transport-verification.v1",
        "release_eligible_source": True,
        "repository": EXPECTED_REPOSITORY,
        "tag": EXPECTED_TAG,
        "commit": EXPECTED_COMMIT,
        "tag_object": EXPECTED_TAG_OBJECT,
        "license": EXPECTED_LICENSE,
        "patch_sha256": digest,
        "modified_files": sorted(actual),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path("third_party/winfsp-greppy"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(args.root)
        encoded = json.dumps(result, sort_keys=True, indent=2) + "\n"
        if args.output is not None:
            args.output.write_text(encoded, encoding="utf-8")
        else:
            sys.stdout.write(encoded)
    except ForkError as error:
        print(f"WinFsp fork verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
