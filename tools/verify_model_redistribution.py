#!/usr/bin/env python3
"""Verify the model redistribution lock file using only the standard library.

Version 1 has this shape::

    {
      "schema": "greppy.model-redistribution-lock",
      "version": 1,
      "release_ready": true,
      "models": [{
        "id": "model-id",
        "release_ready": true,
        "assets": [{"path": "...", "sha256": "...", "size": 123}],
        "license": [{"path": "...", "sha256": "...", "size": 123}],
        "provenance": [{"path": "...", "sha256": "...", "size": 123}],
        "modifications": [{"path": "...", "sha256": "...", "size": 123}]
      }]
    }

Every path is relative to the repository root. The release gate is deliberately
separate from integrity validation so draft manifests remain verifiable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
from typing import Any, Sequence


SCHEMA = "greppy.model-redistribution-lock"
VERSION = 1
DEFAULT_LOCK = Path(__file__).resolve().parents[1] / "licenses" / "MODEL-REDISTRIBUTION.lock.json"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FILE_SECTIONS = ("assets", "license", "provenance", "modifications")
_PROVENANCE_SCHEMAS = {
    "greppy.model-provenance.v1",
    "greppy.training-data-manifest.v1",
}


def _is_int(value: object) -> bool:
    """Return true for JSON integers but not booleans."""
    return isinstance(value, int) and not isinstance(value, bool)


def _safe_relative_path(value: object, label: str, errors: list[str]) -> Path | None:
    if not isinstance(value, str) or not value:
        errors.append(f"{label}.path must be a non-empty string")
        return None
    if "\\" in value:
        errors.append(f"{label}.path is unsafe: backslashes are not allowed: {value!r}")
        return None

    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or ".." in posix_path.parts
        or "." in posix_path.parts
    ):
        errors.append(f"{label}.path must be a normalized relative path without '.' or '..': {value!r}")
        return None
    return Path(*posix_path.parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(entry: object, label: str, root: Path, errors: list[str]) -> None:
    if not isinstance(entry, dict):
        errors.append(f"{label} must be an object")
        return

    relative_path = _safe_relative_path(entry.get("path"), label, errors)
    expected_sha = entry.get("sha256")
    expected_size = entry.get("size")
    valid_sha = isinstance(expected_sha, str) and _SHA256_RE.fullmatch(expected_sha) is not None
    valid_size = _is_int(expected_size) and expected_size >= 0

    if not valid_sha:
        errors.append(f"{label}.sha256 must be a 64-character lowercase hexadecimal digest")
    if not valid_size:
        errors.append(f"{label}.size must be a non-negative integer")
    if relative_path is None:
        return

    candidate = root / relative_path
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        errors.append(f"{label}: file is missing or inaccessible: {relative_path} ({exc})")
        return

    try:
        resolved.relative_to(root)
    except ValueError:
        errors.append(f"{label}.path escapes the repository root: {relative_path}")
        return
    if not resolved.is_file():
        errors.append(f"{label}: path is not a regular file: {relative_path}")
        return

    try:
        actual_size = resolved.stat().st_size
        actual_sha = _sha256(resolved)
    except OSError as exc:
        errors.append(f"{label}: cannot read {relative_path}: {exc}")
        return

    if valid_size and actual_size != expected_size:
        errors.append(f"{label}: size mismatch for {relative_path}: expected {expected_size}, got {actual_size}")
    if valid_sha and actual_sha != expected_sha:
        errors.append(f"{label}: SHA256 mismatch for {relative_path}: expected {expected_sha}, got {actual_sha}")


def _verify_file_section(
    model: dict[str, Any], section: str, model_label: str, root: Path, errors: list[str]
) -> None:
    entries = model.get(section)
    label = f"{model_label}.{section}"
    if not isinstance(entries, list) or not entries:
        errors.append(f"{label} must be a non-empty array of file records")
        return
    for index, entry in enumerate(entries):
        _verify_file(entry, f"{label}[{index}]", root, errors)


def _verify_release_provenance(entry: object, label: str, root: Path, errors: list[str]) -> None:
    if not isinstance(entry, dict):
        return
    relative_path = _safe_relative_path(entry.get("path"), label, errors)
    if relative_path is None:
        return
    if relative_path.suffix != ".json":
        errors.append(f"release gate: {label}.path must be a JSON provenance record")
        return

    candidate = root / relative_path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
        with resolved.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (FileNotFoundError, OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"release gate: cannot validate provenance {relative_path}: {exc}")
        return

    if not isinstance(document, dict):
        errors.append(f"release gate: provenance {relative_path} must contain a JSON object")
        return
    if document.get("schema_version") not in _PROVENANCE_SCHEMAS:
        errors.append(f"release gate: provenance {relative_path} has an unsupported schema_version")
    if document.get("release_ready") is not True:
        errors.append(f"release gate: provenance {relative_path} is not release_ready")


def verify_lock(lock_path: Path, root: Path, *, release: bool = False) -> list[str]:
    """Return all validation errors for *lock_path*, or an empty list."""
    errors: list[str] = []
    try:
        root = root.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        return [f"repository root is missing or inaccessible: {root} ({exc})"]
    if not root.is_dir():
        return [f"repository root is not a directory: {root}"]

    try:
        with lock_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except FileNotFoundError:
        return [f"lock file does not exist: {lock_path}"]
    except (OSError, UnicodeError) as exc:
        return [f"cannot read lock file {lock_path}: {exc}"]
    except json.JSONDecodeError as exc:
        return [f"invalid JSON in {lock_path}: line {exc.lineno}, column {exc.colno}: {exc.msg}"]

    if not isinstance(manifest, dict):
        return ["lock file root must be a JSON object"]

    if manifest.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")
    version = manifest.get("version")
    if not _is_int(version) or version != VERSION:
        errors.append(f"version must be integer {VERSION}")

    global_ready = manifest.get("release_ready")
    if not isinstance(global_ready, bool):
        errors.append("release_ready must be a boolean")
    elif release and not global_ready:
        errors.append("release gate: global release_ready is false")

    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        errors.append("models must be a non-empty array")
        return errors

    seen_ids: set[str] = set()
    for index, model in enumerate(models):
        model_label = f"models[{index}]"
        if not isinstance(model, dict):
            errors.append(f"{model_label} must be an object")
            continue

        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            errors.append(f"{model_label}.id must be a non-empty string")
        elif model_id in seen_ids:
            errors.append(f"{model_label}.id is duplicated: {model_id!r}")
        else:
            seen_ids.add(model_id)

        model_ready = model.get("release_ready")
        if not isinstance(model_ready, bool):
            errors.append(f"{model_label}.release_ready must be a boolean")
        elif release and not model_ready:
            errors.append(f"release gate: {model_label} ({model_id!r}) release_ready is false")

        for section in _FILE_SECTIONS:
            _verify_file_section(model, section, model_label, root, errors)
        if release and isinstance(model.get("provenance"), list):
            for provenance_index, entry in enumerate(model["provenance"]):
                _verify_release_provenance(
                    entry,
                    f"{model_label}.provenance[{provenance_index}]",
                    root,
                    errors,
                )

    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify model redistribution metadata and file integrity.")
    parser.add_argument(
        "lock_file",
        nargs="?",
        type=Path,
        default=DEFAULT_LOCK,
        help=f"lock file to verify (default: {DEFAULT_LOCK})",
    )
    parser.add_argument(
        "--root",
        type=Path,
        help="repository root used to resolve file records (default: parent of the lock file's directory)",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="also require global and per-model release_ready flags",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    lock_path = args.lock_file.resolve()
    root = args.root.resolve() if args.root is not None else lock_path.parent.parent
    errors = verify_lock(lock_path, root, release=args.release)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"model redistribution verification failed with {len(errors)} error(s)", file=sys.stderr)
        return 1

    mode = "release and integrity" if args.release else "integrity"
    print(f"model redistribution {mode} verification passed: {lock_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
