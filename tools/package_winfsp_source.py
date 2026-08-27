#!/usr/bin/env python3
"""Create and verify Greppy's deterministic WinFsp corresponding-source archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

if __package__:
    from tools import verify_winfsp_fork
else:
    import verify_winfsp_fork


SCHEMA = "greppy.winfsp-corresponding-source.v1"
ARCHIVE_ROOT = "greppy-winfsp-source"
MANIFEST_MEMBER = f"{ARCHIVE_ROOT}/MANIFEST.json"
SUPPORT_FILES = (
    "third_party/winfsp-greppy/README.md",
    "third_party/winfsp-greppy/upstream.json",
    "third_party/winfsp-greppy/patches/0001-greppy-hardlink-transport.patch",
    "third_party/winfsp-greppy/patches/0002-greppy-product-identity.patch",
    "tools/build_winfsp_fork.ps1",
    "tools/package_winfsp_source.py",
    "tools/verify_winfsp_fork.py",
)


class SourceArchiveError(ValueError):
    """The source tree or corresponding-source archive is not release-safe."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SourceArchiveError(message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_git(source_root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(source_root), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", b"").decode("utf-8", "replace").strip()
        raise SourceArchiveError(
            f"git {' '.join(arguments)} failed for {source_root}: {detail or error}"
        ) from error
    return result.stdout


def safe_path(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    require(value != "", f"{label} is empty")
    require(not path.is_absolute(), f"{label} is absolute")
    require(".." not in path.parts, f"{label} traverses parents")
    require("\\" not in value, f"{label} uses a backslash")
    return path


def tracked_entries(
    source_root: Path, *, allow_gitlinks: bool = False
) -> list[tuple[str, str, str]]:
    raw = run_git(source_root, "ls-files", "--stage", "-z")
    entries: list[tuple[str, str, str]] = []
    for index, record in enumerate(raw.split(b"\0")):
        if not record:
            continue
        try:
            prefix, encoded_path = record.split(b"\t", 1)
            mode, object_id, stage = prefix.decode("ascii").split(" ")
            relative = encoded_path.decode("utf-8")
        except (ValueError, UnicodeError) as error:
            raise SourceArchiveError(f"malformed git index record {index}") from error
        require(stage == "0", f"unmerged git index entry: {relative}")
        allowed_modes = {"100644", "100755", "120000"}
        if allow_gitlinks:
            allowed_modes.add("160000")
        require(mode in allowed_modes, f"unsupported git mode {mode}: {relative}")
        safe_path(relative, f"tracked path {index}")
        entries.append((relative, mode, object_id))
    require(entries, "WinFsp source index is empty")
    require(
        len(entries) == len({path for path, _mode, _object_id in entries}),
        "duplicate tracked path",
    )
    return sorted(entries)


def source_bytes(source_root: Path, relative: str, mode: str) -> bytes:
    if mode == "120000":
        target = run_git(source_root, "show", f"HEAD:{relative}")
        require(target != b"", f"empty symlink target: {relative}")
        return target
    path = source_root / relative
    require(path.is_file() and not path.is_symlink(), f"source is not a regular file: {relative}")
    try:
        return path.read_bytes()
    except OSError as error:
        raise SourceArchiveError(f"cannot read source file {relative}: {error}") from error


def validate_patched_tree(source_root: Path, fork_result: dict[str, Any]) -> None:
    head = run_git(source_root, "rev-parse", "HEAD").decode("ascii").strip()
    require(head == fork_result["commit"], f"source commit mismatch: {head}")
    changed = {
        line
        for line in run_git(source_root, "diff", "--name-only", "--").decode("utf-8").splitlines()
        if line
    }
    expected = set(fork_result["modified_files"])
    require(
        changed == expected,
        f"patched source file set mismatch; missing={sorted(expected - changed)}, extra={sorted(changed - expected)}",
    )
    for patch in fork_result["patches"]:
        patch_path = Path(patch["path"])
        # Every exact patch must be visible in the supplied tree. The original
        # commit, patch bytes and total changed-file set are independently bound.
        try:
            subprocess.run(
                [
                    "git",
                    "-C",
                    os.fspath(source_root),
                    "apply",
                    "--ignore-space-change",
                    "--reverse",
                    "--check",
                    os.fspath(patch_path),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            detail = getattr(error, "stderr", b"").decode("utf-8", "replace").strip()
            raise SourceArchiveError(
                f"patched tree does not contain {patch['path']}: {detail or error}"
            ) from error


def validate_submodule(source_root: Path, record: dict[str, str]) -> Path:
    relative = record["path"]
    submodule_root = (source_root / relative).resolve()
    try:
        submodule_root.relative_to(source_root)
    except ValueError as error:
        raise SourceArchiveError(f"submodule escapes source root: {relative}") from error
    require(submodule_root.is_dir(), f"submodule is absent: {relative}")
    commit = run_git(submodule_root, "rev-parse", "HEAD").decode("ascii").strip()
    require(commit == record["commit"], f"submodule commit mismatch for {relative}: {commit}")
    repository = run_git(submodule_root, "remote", "get-url", "origin").decode("utf-8").strip()
    require(
        repository == record["repository"],
        f"submodule repository mismatch for {relative}: {repository}",
    )
    dirty = run_git(submodule_root, "status", "--porcelain", "--untracked-files=all")
    require(dirty == b"", f"submodule is dirty: {relative}")
    return submodule_root


def create_archive(
    source_root: Path,
    repository_root: Path,
    fork_root: Path,
    output: Path,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    repository_root = repository_root.resolve()
    fork_root = fork_root.resolve()
    fork_result = verify_winfsp_fork.verify(fork_root)
    # Resolve patch arguments against the Greppy repository for the reverse
    # presence check without mutating the patched source checkout.
    for patch_record in fork_result["patches"]:
        patch_record["path"] = os.fspath(fork_root / patch_record["path"])
    validate_patched_tree(source_root, fork_result)

    members: dict[str, tuple[str, int, bytes]] = {}
    submodules = {record["path"]: record for record in fork_result["submodules"]}
    seen_submodules: set[str] = set()
    for relative, git_mode, object_id in tracked_entries(source_root, allow_gitlinks=True):
        if git_mode == "160000":
            record = submodules.get(relative)
            require(record is not None, f"unbound submodule: {relative}")
            require(object_id == record["commit"], f"submodule gitlink mismatch: {relative}")
            seen_submodules.add(relative)
            submodule_root = validate_submodule(source_root, record)
            for child, child_mode, _child_object in tracked_entries(submodule_root):
                data = source_bytes(submodule_root, child, child_mode)
                kind = "symlink" if child_mode == "120000" else "file"
                mode = (
                    0o755
                    if child_mode == "100755"
                    else (0o777 if kind == "symlink" else 0o644)
                )
                members[f"{ARCHIVE_ROOT}/upstream/{relative}/{child}"] = (
                    kind,
                    mode,
                    data,
                )
            continue
        data = source_bytes(source_root, relative, git_mode)
        kind = "symlink" if git_mode == "120000" else "file"
        mode = 0o755 if git_mode == "100755" else (0o777 if kind == "symlink" else 0o644)
        members[f"{ARCHIVE_ROOT}/upstream/{relative}"] = (kind, mode, data)
    require(seen_submodules == set(submodules), "required submodule gitlink is absent")

    for relative in SUPPORT_FILES:
        safe_path(relative, "support path")
        path = repository_root / relative
        require(path.is_file() and not path.is_symlink(), f"support file is absent: {relative}")
        mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
        members[f"{ARCHIVE_ROOT}/greppy-build/{relative}"] = (
            "file",
            mode,
            path.read_bytes(),
        )

    manifest = {
        "schema": SCHEMA,
        "upstream": {
            "repository": fork_result["repository"],
            "tag": fork_result["tag"],
            "commit": fork_result["commit"],
            "tag_object": fork_result["tag_object"],
            "license": fork_result["license"],
        },
        "patches": [
            {
                "path": Path(patch["path"]).relative_to(fork_root).as_posix(),
                "sha256": patch["sha256"],
                "modified_files": patch["modified_files"],
            }
            for patch in fork_result["patches"]
        ],
        "submodules": list(fork_result["submodules"]),
        "members": {
            name: {"kind": kind, "mode": mode, "sha256": sha256(data)}
            for name, (kind, mode, data) in sorted(members.items())
        },
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    members[MANIFEST_MEMBER] = ("file", 0o644, manifest_bytes)

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                    for name, (kind, mode, data) in sorted(members.items()):
                        info = tarfile.TarInfo(name)
                        info.mode = mode
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        if kind == "symlink":
                            info.type = tarfile.SYMTYPE
                            info.linkname = data.decode("utf-8")
                            archive.addfile(info)
                        else:
                            info.size = len(data)
                            archive.addfile(info, io.BytesIO(data))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    verify_archive(output)
    return manifest


def verify_archive(path: Path) -> dict[str, Any]:
    try:
        with tarfile.open(path, "r:gz") as archive:
            archive_members = archive.getmembers()
            names = [member.name for member in archive_members]
            require(len(names) == len(set(names)), "archive contains duplicate members")
            by_name = {member.name: member for member in archive_members}
            manifest_member = by_name.get(MANIFEST_MEMBER)
            require(manifest_member is not None and manifest_member.isfile(), "source manifest is absent")
            handle = archive.extractfile(manifest_member)
            require(handle is not None, "cannot read source manifest")
            manifest = json.loads(handle.read())
            require(isinstance(manifest, dict) and manifest.get("schema") == SCHEMA, "invalid source schema")
            records = manifest.get("members")
            require(isinstance(records, dict), "source members must be an object")
            expected_names = set(records) | {MANIFEST_MEMBER}
            require(set(names) == expected_names, "archive member set differs from manifest")
            for name, record in records.items():
                safe_path(name, "archive member")
                require(name.startswith(f"{ARCHIVE_ROOT}/"), f"member escapes archive root: {name}")
                require(isinstance(record, dict), f"invalid member record: {name}")
                member = by_name[name]
                kind = record.get("kind")
                if kind == "symlink":
                    require(member.issym(), f"member is not a symlink: {name}")
                    data = member.linkname.encode("utf-8")
                else:
                    require(kind == "file" and member.isfile(), f"member is not a file: {name}")
                    member_handle = archive.extractfile(member)
                    require(member_handle is not None, f"cannot read member: {name}")
                    data = member_handle.read()
                require(member.mode == record.get("mode"), f"member mode mismatch: {name}")
                require(sha256(data) == record.get("sha256"), f"member digest mismatch: {name}")
            return manifest
    except (OSError, tarfile.TarError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceArchiveError(f"cannot verify source archive {path}: {error}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--source", type=Path, required=True)
    create.add_argument("--repository-root", type=Path, default=Path("."))
    create.add_argument("--fork-root", type=Path, default=Path("third_party/winfsp-greppy"))
    create.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("archive", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            result = create_archive(args.source, args.repository_root, args.fork_root, args.output)
        else:
            result = verify_archive(args.archive)
        print(json.dumps(result, sort_keys=True, indent=2))
    except (SourceArchiveError, verify_winfsp_fork.ForkError) as error:
        print(f"WinFsp source archive failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
