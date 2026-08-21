#!/usr/bin/env python3
"""Materialize digest-locked benchmark corpus files from public Git history."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import tempfile
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "bench" / "BENCHMARK_CORPUS.lock.json"
SCHEMA = "greppy.benchmark-corpus-lock.v1"
REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
SAFE_PATH = re.compile(r"^[A-Za-z0-9._/-]+$")


class CorpusError(ValueError):
    """The corpus lock or downloaded bytes violate the pinned contract."""


def _relative_path(value: object, field: str) -> pathlib.PurePosixPath:
    if not isinstance(value, str):
        raise CorpusError(f"{field} must be a string")
    path = pathlib.PurePosixPath(value)
    if (
        not value
        or not SAFE_PATH.fullmatch(value)
        or "\\" in value
        or path.is_absolute()
        or ".." in path.parts
    ):
        raise CorpusError(f"{field} must be a non-traversing relative path: {value!r}")
    return path


def load_lock(path: pathlib.Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorpusError(f"cannot read corpus lock {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != SCHEMA:
        raise CorpusError(f"corpus lock must use schema {SCHEMA}")
    repository = document.get("repository")
    commit = document.get("commit")
    files = document.get("files")
    if not isinstance(repository, str) or not REPOSITORY.fullmatch(repository):
        raise CorpusError("repository must be an owner/name GitHub repository")
    if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
        raise CorpusError("commit must be a lowercase 40-hex object id")
    if not isinstance(files, list) or not files:
        raise CorpusError("files must be a non-empty array")
    seen_targets: set[str] = set()
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise CorpusError(f"files[{index}] must be an object")
        _relative_path(entry.get("source"), f"files[{index}].source")
        target = _relative_path(entry.get("target"), f"files[{index}].target")
        if not target.parts or target.parts[0] != "bench":
            raise CorpusError(f"files[{index}].target must stay under bench/")
        target_text = target.as_posix()
        if target_text in seen_targets:
            raise CorpusError(f"duplicate target: {target_text}")
        seen_targets.add(target_text)
        digest = entry.get("sha256")
        if not isinstance(digest, str) or not SHA256.fullmatch(digest):
            raise CorpusError(f"files[{index}].sha256 must be lowercase 64-hex")
    return document


def _download(repository: str, commit: str, source: pathlib.PurePosixPath) -> bytes:
    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{source.as_posix()}"
    request = urllib.request.Request(url, headers={"User-Agent": "greppy-release-corpus/1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()
    except OSError as exc:
        raise CorpusError(f"cannot download {url}: {exc}") from exc


def materialize(
    lock_path: pathlib.Path,
    root: pathlib.Path,
    targets: set[str] | None = None,
    source_dir: pathlib.Path | None = None,
) -> list[pathlib.Path]:
    document = load_lock(lock_path)
    repository = str(document["repository"])
    commit = str(document["commit"])
    entries = document["files"]
    assert isinstance(entries, list)
    known_targets = {str(entry["target"]) for entry in entries}
    if targets is not None:
        unknown = targets - known_targets
        if unknown:
            raise CorpusError(f"target not present in corpus lock: {sorted(unknown)}")
    written: list[pathlib.Path] = []
    for entry in entries:
        assert isinstance(entry, dict)
        target_text = str(entry["target"])
        if targets is not None and target_text not in targets:
            continue
        source = _relative_path(entry["source"], "source")
        data = (
            (source_dir.joinpath(*source.parts)).read_bytes()
            if source_dir is not None
            else _download(repository, commit, source)
        )
        actual = hashlib.sha256(data).hexdigest()
        expected = str(entry["sha256"])
        if actual != expected:
            raise CorpusError(
                f"digest mismatch for {source}: expected {expected}, got {actual}"
            )
        target_rel = _relative_path(target_text, "target")
        target = root.joinpath(*target_rel.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except BaseException:
            pathlib.Path(temporary).unlink(missing_ok=True)
            raise
        written.append(target)
        print(f"materialized {target_text} sha256={actual}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=pathlib.Path, default=DEFAULT_LOCK)
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--source-dir", type=pathlib.Path)
    parser.add_argument("--target", action="append", dest="targets")
    args = parser.parse_args(argv)
    try:
        materialize(
            args.lock.resolve(),
            args.root.resolve(),
            set(args.targets) if args.targets else None,
            args.source_dir.resolve() if args.source_dir else None,
        )
    except (CorpusError, OSError) as exc:
        parser.exit(2, f"benchmark corpus: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
