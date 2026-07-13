#!/usr/bin/env python3
"""Create and verify immutable Greppy release evidence."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from typing import Any


CONTRACT_SCHEMA = "greppy.release-asset-contract.v1"
RELEASE_MANIFEST_SCHEMA = "greppy.release-assets.v1"
BUILD_ENVIRONMENT_SCHEMA = "greppy.build-environment.v1"
RELEASE_PACKAGE_ID = "SPDXRef-Package-greppy-release"
RELEASE_MANIFEST_NAME = "RELEASE-ASSETS.json"
AGGREGATE_CHECKSUM_NAME = "SHA256SUMS"
TRAINING_ARCHIVE_NAME = "greppy-qwen35-training-evidence.tar.gz"
TRAINING_MANIFEST_MEMBER = "training/qwen35/MANIFEST.sha256"
TRAINING_REPORT_MEMBER = "training/qwen35/audit-report-2026-07-13.json.gz"
BUILD_ENVIRONMENTS = {
    "build-environment-macos-arm64.json": {
        "matrix_name": "macos-arm64",
        "build_features": "metal",
        "runner_os": "macOS",
        "runner_arch": "ARM64",
        "rust_host": "aarch64-apple-darwin",
    },
    "build-environment-linux-x86_64.json": {
        "matrix_name": "linux-x86_64",
        "build_features": "cuda",
        "runner_os": "Linux",
        "runner_arch": "X64",
        "rust_host": "x86_64-unknown-linux-gnu",
    },
    "build-environment-windows-x86_64.json": {
        "matrix_name": "windows-x86_64",
        "build_features": "cpu",
        "runner_os": "Windows",
        "runner_arch": "X64",
        "rust_host": "x86_64-pc-windows-msvc",
    },
}
SHA256_LINE = re.compile(r"^([0-9a-f]{64})  ([^/\\]+)$")
MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  ([^\\]+)$")
LOCK_FIELD = re.compile(r'^(name|version|source|checksum) = ("(?:[^"\\]|\\.)*")$')


class ReleaseArtifactError(RuntimeError):
    """A release artifact violates its declared contract."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha1_file(path: pathlib.Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseArtifactError(f"cannot read JSON {path}: {exc}") from exc


def _safe_relative_path(value: str) -> pathlib.PurePosixPath:
    path = pathlib.PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ReleaseArtifactError(f"unsafe relative path: {value!r}")
    return path


def _parse_training_manifest(data: bytes) -> dict[str, str]:
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ReleaseArtifactError("training manifest is not UTF-8") from exc
    entries: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        match = MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise ReleaseArtifactError(
                f"invalid training manifest line {line_number}: {line!r}"
            )
        digest, name = match.groups()
        path = _safe_relative_path(name)
        normalized = path.as_posix()
        if normalized in entries:
            raise ReleaseArtifactError(
                f"duplicate training manifest entry: {normalized}"
            )
        entries[normalized] = digest
    if not entries:
        raise ReleaseArtifactError("training manifest is empty")
    if "audit-report-2026-07-13.json.gz" not in entries:
        raise ReleaseArtifactError(
            "published Qwen audit report is absent from manifest"
        )
    return entries


def create_training_archive(
    repository_root: pathlib.Path,
    manifest_path: pathlib.Path,
    output_path: pathlib.Path,
) -> None:
    repository_root = repository_root.resolve()
    manifest_path = manifest_path.resolve()
    training_root = manifest_path.parent
    manifest_bytes = manifest_path.read_bytes()
    entries = _parse_training_manifest(manifest_bytes)

    sources: list[tuple[str, pathlib.Path, bytes | None]] = [
        (TRAINING_MANIFEST_MEMBER, manifest_path, manifest_bytes)
    ]
    for relative, expected_digest in sorted(entries.items()):
        source = (training_root / relative).resolve()
        try:
            source.relative_to(repository_root)
        except ValueError as exc:
            raise ReleaseArtifactError(
                f"training file escapes repository: {source}"
            ) from exc
        if source.is_symlink() or not source.is_file():
            raise ReleaseArtifactError(
                f"training evidence is not a regular file: {source}"
            )
        actual_digest = _sha256_file(source)
        if actual_digest != expected_digest:
            raise ReleaseArtifactError(
                f"training evidence digest mismatch for {relative}: "
                f"{actual_digest} != {expected_digest}"
            )
        sources.append((f"training/qwen35/{relative}", source, None))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary.open("wb") as raw_output:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_output, compresslevel=9, mtime=0
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.GNU_FORMAT
                ) as archive:
                    for member_name, source, cached_data in sorted(sources):
                        data = (
                            cached_data
                            if cached_data is not None
                            else source.read_bytes()
                        )
                        info = tarfile.TarInfo(member_name)
                        info.size = len(data)
                        info.mode = 0o755 if source.stat().st_mode & 0o111 else 0o644
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        archive.addfile(info, io.BytesIO(data))
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    verify_training_archive(output_path)


def verify_training_archive(path: pathlib.Path) -> None:
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise ReleaseArtifactError(
                    "training archive contains duplicate members"
                )
            if any(not member.isfile() for member in members):
                raise ReleaseArtifactError(
                    "training archive contains a non-regular member"
                )
            by_name = {member.name: member for member in members}
            manifest_member = by_name.get(TRAINING_MANIFEST_MEMBER)
            if manifest_member is None:
                raise ReleaseArtifactError(
                    "training archive is missing MANIFEST.sha256"
                )
            manifest_handle = archive.extractfile(manifest_member)
            if manifest_handle is None:
                raise ReleaseArtifactError("cannot read archived training manifest")
            entries = _parse_training_manifest(manifest_handle.read())
            expected_names = {TRAINING_MANIFEST_MEMBER}
            expected_names.update(f"training/qwen35/{name}" for name in entries)
            if set(names) != expected_names:
                missing = sorted(expected_names - set(names))
                extra = sorted(set(names) - expected_names)
                raise ReleaseArtifactError(
                    f"training archive member mismatch; missing={missing}, extra={extra}"
                )
            if TRAINING_REPORT_MEMBER not in by_name:
                raise ReleaseArtifactError(
                    "training archive is missing the published audit report"
                )
            for relative, expected_digest in entries.items():
                member = by_name[f"training/qwen35/{relative}"]
                handle = archive.extractfile(member)
                if handle is None:
                    raise ReleaseArtifactError(
                        f"cannot read training member {relative}"
                    )
                actual_digest = _sha256_bytes(handle.read())
                if actual_digest != expected_digest:
                    raise ReleaseArtifactError(
                        f"archived training digest mismatch for {relative}: "
                        f"{actual_digest} != {expected_digest}"
                    )
    except (OSError, tarfile.TarError) as exc:
        raise ReleaseArtifactError(
            f"cannot verify training archive {path}: {exc}"
        ) from exc


def _parse_cargo_lock(path: pathlib.Path) -> list[dict[str, str]]:
    packages: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if line == "[[package]]":
            if current is not None:
                packages.append(current)
            current = {}
            continue
        if current is None:
            continue
        match = LOCK_FIELD.fullmatch(line)
        if match is None:
            continue
        field, quoted = match.groups()
        try:
            current[field] = json.loads(quoted)
        except json.JSONDecodeError as exc:
            raise ReleaseArtifactError(
                f"invalid Cargo.lock string at line {line_number}"
            ) from exc
    if current is not None:
        packages.append(current)
    for package in packages:
        if not package.get("name") or not package.get("version"):
            raise ReleaseArtifactError("Cargo.lock package lacks name or version")
    if not packages:
        raise ReleaseArtifactError("Cargo.lock contains no packages")
    return packages


def _cargo_package_id(package: dict[str, str]) -> str:
    identity = json.dumps(
        [package["name"], package["version"], package.get("source", "workspace")],
        separators=(",", ":"),
    ).encode("utf-8")
    return f"SPDXRef-CargoLock-{hashlib.sha256(identity).hexdigest()[:24]}"


def _file_id(relative: str) -> str:
    return f"SPDXRef-ReleaseFile-{hashlib.sha256(relative.encode('utf-8')).hexdigest()[:24]}"


def _dist_files(dist: pathlib.Path) -> list[tuple[str, pathlib.Path]]:
    files: list[tuple[str, pathlib.Path]] = []
    for path in sorted(dist.rglob("*")):
        if path.is_symlink():
            raise ReleaseArtifactError(f"release dist contains a symlink: {path}")
        if path.is_file():
            files.append((path.relative_to(dist).as_posix(), path))
    if not files:
        raise ReleaseArtifactError("release dist is empty")
    return files


def _greppy_version(packages: list[dict[str, str]]) -> str:
    matches = [
        package["version"]
        for package in packages
        if package["name"] == "greppy" and "source" not in package
    ]
    if len(matches) != 1:
        raise ReleaseArtifactError(
            f"expected one workspace greppy package in Cargo.lock, found {len(matches)}"
        )
    return matches[0]


def _validate_spdx_base(document: dict[str, Any]) -> None:
    if document.get("spdxVersion") not in ("SPDX-2.2", "SPDX-2.3"):
        raise ReleaseArtifactError("SBOM must be SPDX 2.2 or 2.3 JSON")
    if document.get("SPDXID") != "SPDXRef-DOCUMENT":
        raise ReleaseArtifactError("SBOM document SPDXID is invalid")
    if document.get("dataLicense") != "CC0-1.0":
        raise ReleaseArtifactError("SBOM dataLicense must be CC0-1.0")
    if not str(document.get("documentNamespace", "")).startswith(
        ("https://", "http://")
    ):
        raise ReleaseArtifactError("SBOM documentNamespace is missing or invalid")
    creation = document.get("creationInfo")
    if not isinstance(creation, dict) or not creation.get("creators"):
        raise ReleaseArtifactError("SBOM creationInfo.creators is missing")


def augment_spdx(
    sbom_path: pathlib.Path,
    dist: pathlib.Path,
    cargo_lock: pathlib.Path,
    target: str,
) -> None:
    document = _load_json(sbom_path)
    if not isinstance(document, dict):
        raise ReleaseArtifactError("SBOM root must be an object")
    _validate_spdx_base(document)
    lock_packages = _parse_cargo_lock(cargo_lock)
    files = _dist_files(dist)

    generated_ids = {RELEASE_PACKAGE_ID}
    generated_ids.update(_cargo_package_id(package) for package in lock_packages)
    generated_ids.update(_file_id(relative) for relative, _ in files)
    document["packages"] = [
        package
        for package in document.get("packages", [])
        if package.get("SPDXID") not in generated_ids
    ]
    document["files"] = [
        file_entry
        for file_entry in document.get("files", [])
        if file_entry.get("SPDXID") not in generated_ids
    ]
    document["relationships"] = [
        relationship
        for relationship in document.get("relationships", [])
        if relationship.get("spdxElementId") not in generated_ids
        and relationship.get("relatedSpdxElement") not in generated_ids
    ]
    described = [
        identifier
        for identifier in document.get("documentDescribes", [])
        if identifier not in generated_ids
    ]
    described.append(RELEASE_PACKAGE_ID)
    document["documentDescribes"] = described

    file_sha1 = [_sha1_file(path) for _, path in files]
    verification_code = hashlib.sha1(
        "".join(sorted(file_sha1)).encode("ascii")
    ).hexdigest()
    document["packages"].append(
        {
            "name": "greppy",
            "SPDXID": RELEASE_PACKAGE_ID,
            "versionInfo": _greppy_version(lock_packages),
            "supplier": "Organization: metric-space-ai",
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": True,
            "packageVerificationCode": {
                "packageVerificationCodeValue": verification_code
            },
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "copyrightText": "NOASSERTION",
            "primaryPackagePurpose": "APPLICATION",
            "comment": (
                f"Greppy release package for {target}. File inventory is generated "
                "from dist. Cargo.lock entries describe the locked build graph and "
                "do not assert inclusion in the final binary."
            ),
        }
    )

    for relative, path in files:
        identifier = _file_id(relative)
        document["files"].append(
            {
                "fileName": f"./{relative}",
                "SPDXID": identifier,
                "checksums": [
                    {"algorithm": "SHA1", "checksumValue": _sha1_file(path)},
                    {"algorithm": "SHA256", "checksumValue": _sha256_file(path)},
                ],
                "licenseConcluded": "NOASSERTION",
                "copyrightText": "NOASSERTION",
            }
        )
        document["relationships"].append(
            {
                "spdxElementId": RELEASE_PACKAGE_ID,
                "relationshipType": "CONTAINS",
                "relatedSpdxElement": identifier,
            }
        )

    relationship_comment = (
        "Resolved by Cargo.lock; the package may be target-, feature-, build-, "
        "or development-specific."
    )
    for package in lock_packages:
        identifier = _cargo_package_id(package)
        source = package.get("source", "workspace")
        entry: dict[str, Any] = {
            "name": package["name"],
            "SPDXID": identifier,
            "versionInfo": package["version"],
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "licenseConcluded": "NOASSERTION",
            "licenseDeclared": "NOASSERTION",
            "copyrightText": "NOASSERTION",
            "comment": f"Cargo.lock source: {source}",
        }
        checksum = package.get("checksum")
        if checksum is not None:
            if not re.fullmatch(r"[0-9a-f]{64}", checksum):
                raise ReleaseArtifactError(
                    f"invalid Cargo.lock checksum for {package['name']} {package['version']}"
                )
            entry["checksums"] = [{"algorithm": "SHA256", "checksumValue": checksum}]
        document["packages"].append(entry)
        document["relationships"].append(
            {
                "spdxElementId": RELEASE_PACKAGE_ID,
                "relationshipType": "OTHER",
                "relatedSpdxElement": identifier,
                "comment": relationship_comment,
            }
        )

    _write_json(sbom_path, document)
    verify_spdx(sbom_path, dist, cargo_lock, target)


def _checksum_map(entry: dict[str, Any]) -> dict[str, str]:
    return {
        checksum.get("algorithm", ""): checksum.get("checksumValue", "")
        for checksum in entry.get("checksums", [])
        if isinstance(checksum, dict)
    }


def verify_spdx(
    sbom_path: pathlib.Path,
    dist: pathlib.Path,
    cargo_lock: pathlib.Path,
    target: str,
) -> None:
    document = _load_json(sbom_path)
    if not isinstance(document, dict):
        raise ReleaseArtifactError("SBOM root must be an object")
    _validate_spdx_base(document)
    lock_packages = _parse_cargo_lock(cargo_lock)
    dist_files = _dist_files(dist)

    packages = document.get("packages", [])
    files = document.get("files", [])
    relationships = document.get("relationships", [])
    if not all(isinstance(item, dict) for item in packages + files + relationships):
        raise ReleaseArtifactError(
            "SBOM packages, files, and relationships must be objects"
        )
    identifiers = [item.get("SPDXID") for item in packages + files]
    if any(not identifier for identifier in identifiers) or len(identifiers) != len(
        set(identifiers)
    ):
        raise ReleaseArtifactError(
            "SBOM contains missing or duplicate SPDX identifiers"
        )

    package_by_id = {package["SPDXID"]: package for package in packages}
    root = package_by_id.get(RELEASE_PACKAGE_ID)
    if root is None:
        raise ReleaseArtifactError("SBOM lacks Greppy release package identity")
    if root.get("name") != "greppy" or root.get("versionInfo") != _greppy_version(
        lock_packages
    ):
        raise ReleaseArtifactError("SBOM Greppy package name or version is incorrect")
    if target not in root.get("comment", "") or root.get("filesAnalyzed") is not True:
        raise ReleaseArtifactError(
            "SBOM Greppy target or file-analysis assertion is missing"
        )
    if RELEASE_PACKAGE_ID not in document.get("documentDescribes", []):
        raise ReleaseArtifactError("SBOM document does not describe the Greppy package")

    file_by_id = {entry["SPDXID"]: entry for entry in files}
    expected_file_ids = {_file_id(relative) for relative, _ in dist_files}
    contains = {
        relationship.get("relatedSpdxElement")
        for relationship in relationships
        if relationship.get("spdxElementId") == RELEASE_PACKAGE_ID
        and relationship.get("relationshipType") == "CONTAINS"
    }
    if contains != expected_file_ids:
        raise ReleaseArtifactError("SBOM package file relationships do not match dist")
    file_sha1: list[str] = []
    for relative, path in dist_files:
        entry = file_by_id.get(_file_id(relative))
        if entry is None or entry.get("fileName") != f"./{relative}":
            raise ReleaseArtifactError(f"SBOM lacks release file {relative}")
        checksums = _checksum_map(entry)
        expected_sha1 = _sha1_file(path)
        file_sha1.append(expected_sha1)
        if checksums.get("SHA1") != expected_sha1:
            raise ReleaseArtifactError(f"SBOM SHA1 mismatch for {relative}")
        if checksums.get("SHA256") != _sha256_file(path):
            raise ReleaseArtifactError(f"SBOM SHA256 mismatch for {relative}")
    expected_code = hashlib.sha1("".join(sorted(file_sha1)).encode("ascii")).hexdigest()
    actual_code = root.get("packageVerificationCode", {}).get(
        "packageVerificationCodeValue"
    )
    if actual_code != expected_code:
        raise ReleaseArtifactError("SBOM package verification code is incorrect")

    locked_relationships = {
        relationship.get("relatedSpdxElement"): relationship
        for relationship in relationships
        if relationship.get("spdxElementId") == RELEASE_PACKAGE_ID
        and relationship.get("relationshipType") == "OTHER"
    }
    for package in lock_packages:
        identifier = _cargo_package_id(package)
        entry = package_by_id.get(identifier)
        if entry is None:
            raise ReleaseArtifactError(
                f"SBOM lacks Cargo.lock entry {package['name']} {package['version']}"
            )
        if (
            entry.get("name") != package["name"]
            or entry.get("versionInfo") != package["version"]
        ):
            raise ReleaseArtifactError(
                f"SBOM Cargo.lock identity mismatch for {identifier}"
            )
        expected_source = package.get("source", "workspace")
        if entry.get("comment") != f"Cargo.lock source: {expected_source}":
            raise ReleaseArtifactError(
                f"SBOM Cargo.lock source mismatch for {identifier}"
            )
        checksum = package.get("checksum")
        if checksum is not None and _checksum_map(entry).get("SHA256") != checksum:
            raise ReleaseArtifactError(
                f"SBOM Cargo.lock checksum mismatch for {identifier}"
            )
        relationship = locked_relationships.get(identifier)
        if relationship is None or "do not assert" not in root.get("comment", ""):
            raise ReleaseArtifactError(
                f"SBOM Cargo.lock qualification missing for {identifier}"
            )


def record_build_environment(
    output: pathlib.Path,
    expected_os: str,
    expected_arch: str,
    expected_host: str,
    matrix_name: str,
    features: str,
    git_commit: str,
    cargo_lock: pathlib.Path,
) -> None:
    runner_os = os.environ.get("RUNNER_OS", "")
    runner_arch = os.environ.get("RUNNER_ARCH", "")
    if runner_os != expected_os:
        raise ReleaseArtifactError(
            f"runner OS mismatch: {runner_os!r} != {expected_os!r}"
        )
    if runner_arch != expected_arch:
        raise ReleaseArtifactError(
            f"runner architecture mismatch: {runner_arch!r} != {expected_arch!r}"
        )
    machine = platform.machine()
    accepted_machine = {
        "ARM64": {"arm64", "aarch64", "ARM64"},
        "X64": {"x86_64", "AMD64", "x64"},
    }.get(expected_arch, {expected_arch})
    if machine not in accepted_machine:
        raise ReleaseArtifactError(
            f"host machine architecture mismatch: {machine!r} not in {sorted(accepted_machine)}"
        )
    rustc_verbose = subprocess.run(
        ["rustc", "-vV"], check=True, text=True, capture_output=True
    ).stdout.strip()
    rust_fields = dict(
        line.split(": ", 1) for line in rustc_verbose.splitlines() if ": " in line
    )
    host = rust_fields.get("host")
    if host != expected_host:
        raise ReleaseArtifactError(f"Rust host mismatch: {host!r} != {expected_host!r}")
    cargo_version = subprocess.run(
        ["cargo", "--version"], check=True, text=True, capture_output=True
    ).stdout.strip()
    _write_json(
        output,
        {
            "schema_version": BUILD_ENVIRONMENT_SCHEMA,
            "git_commit": git_commit,
            "matrix_name": matrix_name,
            "build_features": features,
            "runner_os": runner_os,
            "runner_arch": runner_arch,
            "machine_arch": machine,
            "rust_host": host,
            "rustc": rust_fields,
            "cargo": cargo_version,
            "cargo_lock_sha256": _sha256_file(cargo_lock),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "runner_image_os": os.environ.get("ImageOS"),
            "runner_image_version": os.environ.get("ImageVersion"),
        },
    )


def verify_build_environment_record(
    path: pathlib.Path,
    expected: dict[str, str],
    git_commit: str,
    cargo_lock: pathlib.Path,
) -> None:
    record = _load_json(path)
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != BUILD_ENVIRONMENT_SCHEMA
    ):
        raise ReleaseArtifactError(f"invalid build-environment schema in {path.name}")
    if record.get("git_commit") != git_commit:
        raise ReleaseArtifactError(f"build-environment commit mismatch in {path.name}")
    for field, value in expected.items():
        if record.get(field) != value:
            raise ReleaseArtifactError(
                f"build-environment {field} mismatch in {path.name}: "
                f"{record.get(field)!r} != {value!r}"
            )
    rustc = record.get("rustc")
    if not isinstance(rustc, dict) or rustc.get("host") != expected["rust_host"]:
        raise ReleaseArtifactError(
            f"build-environment rustc host mismatch in {path.name}"
        )
    if not rustc.get("release") or not rustc.get("commit-hash"):
        raise ReleaseArtifactError(
            f"build-environment rustc identity is incomplete in {path.name}"
        )
    if not str(record.get("cargo", "")).startswith("cargo "):
        raise ReleaseArtifactError(
            f"build-environment Cargo identity is missing in {path.name}"
        )
    if record.get("cargo_lock_sha256") != _sha256_file(cargo_lock):
        raise ReleaseArtifactError(
            f"build-environment Cargo.lock mismatch in {path.name}"
        )
    if not record.get("github_run_id") or not record.get("github_run_attempt"):
        raise ReleaseArtifactError(
            f"build-environment run identity is missing in {path.name}"
        )


def load_contract(path: pathlib.Path) -> dict[str, Any]:
    contract = _load_json(path)
    if (
        not isinstance(contract, dict)
        or contract.get("schema_version") != CONTRACT_SCHEMA
    ):
        raise ReleaseArtifactError("release asset contract schema is invalid")
    if contract.get("repository") != "metric-space-ai/greppy":
        raise ReleaseArtifactError("release asset contract repository is invalid")
    assets = contract.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ReleaseArtifactError("release asset contract has no assets")
    names: list[str] = []
    for asset in assets:
        if (
            not isinstance(asset, dict)
            or not asset.get("name")
            or not asset.get("role")
        ):
            raise ReleaseArtifactError("release asset contract entry is invalid")
        name = asset["name"]
        if pathlib.PurePath(name).name != name:
            raise ReleaseArtifactError(f"release asset name is not a basename: {name}")
        names.append(name)
    if len(names) != len(set(names)):
        raise ReleaseArtifactError("release asset contract contains duplicate names")
    generated = {asset["name"] for asset in assets if asset.get("generated") is True}
    if generated != {RELEASE_MANIFEST_NAME, AGGREGATE_CHECKSUM_NAME}:
        raise ReleaseArtifactError(
            f"unexpected generated release assets: {sorted(generated)}"
        )
    return contract


def _find_source_asset(source_root: pathlib.Path, name: str) -> pathlib.Path:
    matches = [path for path in source_root.rglob(name) if path.is_file()]
    if len(matches) != 1:
        raise ReleaseArtifactError(
            f"expected exactly one source asset named {name}, found {len(matches)}"
        )
    if matches[0].is_symlink():
        raise ReleaseArtifactError(f"release source asset is a symlink: {matches[0]}")
    return matches[0]


def _parse_checksum_file(path: pathlib.Path) -> tuple[str, str]:
    try:
        line = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise ReleaseArtifactError(f"cannot read checksum file {path}: {exc}") from exc
    match = SHA256_LINE.fullmatch(line)
    if match is None:
        raise ReleaseArtifactError(f"invalid checksum line in {path.name}: {line!r}")
    return match.group(1), match.group(2)


def stage_release(
    source_root: pathlib.Path,
    output_root: pathlib.Path,
    contract_path: pathlib.Path,
    git_commit: str,
    tag: str,
    cargo_lock: pathlib.Path | None = None,
) -> None:
    contract = load_contract(contract_path)
    if output_root.exists() and any(output_root.iterdir()):
        raise ReleaseArtifactError(
            f"release staging directory is not empty: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    assets = contract["assets"]
    for asset in assets:
        if asset.get("generated") is True:
            continue
        source = _find_source_asset(source_root, asset["name"])
        shutil.copyfile(source, output_root / asset["name"])

    manifest_assets: list[dict[str, Any]] = []
    for asset in assets:
        destination = output_root / asset["name"]
        generated = asset.get("generated") is True
        manifest_assets.append(
            {
                "name": asset["name"],
                "role": asset["role"],
                "sha256": None if generated else _sha256_file(destination),
                "size_bytes": None if generated else destination.stat().st_size,
            }
        )
    _write_json(
        output_root / RELEASE_MANIFEST_NAME,
        {
            "schema_version": RELEASE_MANIFEST_SCHEMA,
            "repository": contract["repository"],
            "git_commit": git_commit,
            "tag": tag,
            "assets": manifest_assets,
        },
    )

    checksummed_names = sorted(
        asset["name"] for asset in assets if asset["name"] != AGGREGATE_CHECKSUM_NAME
    )
    checksum_lines = [
        f"{_sha256_file(output_root / name)}  {name}" for name in checksummed_names
    ]
    (output_root / AGGREGATE_CHECKSUM_NAME).write_text(
        "\n".join(checksum_lines) + "\n", encoding="ascii"
    )
    verify_staged_release(
        output_root, contract_path, git_commit, tag, cargo_lock=cargo_lock
    )


def verify_staged_release(
    output_root: pathlib.Path,
    contract_path: pathlib.Path,
    git_commit: str,
    tag: str,
    cargo_lock: pathlib.Path | None = None,
) -> None:
    contract = load_contract(contract_path)
    expected_names = {asset["name"] for asset in contract["assets"]}
    actual_names = {path.name for path in output_root.iterdir() if path.is_file()}
    directory_names = {path.name for path in output_root.iterdir() if path.is_dir()}
    if directory_names or actual_names != expected_names:
        raise ReleaseArtifactError(
            f"staged asset mismatch; missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}, dirs={sorted(directory_names)}"
        )

    manifest = _load_json(output_root / RELEASE_MANIFEST_NAME)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != RELEASE_MANIFEST_SCHEMA
    ):
        raise ReleaseArtifactError("release manifest schema is invalid")
    if manifest.get("repository") != contract["repository"]:
        raise ReleaseArtifactError("release manifest repository is invalid")
    if manifest.get("git_commit") != git_commit or manifest.get("tag") != tag:
        raise ReleaseArtifactError("release manifest subject does not match release")
    expected_order = [asset["name"] for asset in contract["assets"]]
    manifest_assets = manifest.get("assets")
    if (
        not isinstance(manifest_assets, list)
        or [asset.get("name") for asset in manifest_assets] != expected_order
    ):
        raise ReleaseArtifactError(
            "release manifest asset names or order differ from contract"
        )
    contract_by_name = {asset["name"]: asset for asset in contract["assets"]}
    for asset in manifest_assets:
        name = asset["name"]
        if asset.get("role") != contract_by_name[name]["role"]:
            raise ReleaseArtifactError(f"release manifest role mismatch for {name}")
        if contract_by_name[name].get("generated") is True:
            if asset.get("sha256") is not None or asset.get("size_bytes") is not None:
                raise ReleaseArtifactError(
                    f"generated asset metadata must be null for {name}"
                )
        else:
            path = output_root / name
            if asset.get("sha256") != _sha256_file(path):
                raise ReleaseArtifactError(
                    f"release manifest digest mismatch for {name}"
                )
            if asset.get("size_bytes") != path.stat().st_size:
                raise ReleaseArtifactError(f"release manifest size mismatch for {name}")

    checksum_entries: dict[str, str] = {}
    for line_number, line in enumerate(
        (output_root / AGGREGATE_CHECKSUM_NAME)
        .read_text(encoding="ascii")
        .splitlines(),
        1,
    ):
        match = SHA256_LINE.fullmatch(line)
        if match is None:
            raise ReleaseArtifactError(
                f"invalid SHA256SUMS line {line_number}: {line!r}"
            )
        digest, name = match.groups()
        if name in checksum_entries:
            raise ReleaseArtifactError(f"duplicate SHA256SUMS entry: {name}")
        checksum_entries[name] = digest
    expected_checksums = expected_names - {AGGREGATE_CHECKSUM_NAME}
    if set(checksum_entries) != expected_checksums:
        raise ReleaseArtifactError(
            "SHA256SUMS names differ from release asset contract"
        )
    for name, digest in checksum_entries.items():
        if digest != _sha256_file(output_root / name):
            raise ReleaseArtifactError(f"SHA256SUMS digest mismatch for {name}")

    for sidecar in sorted(output_root.glob("*.sha256")):
        digest, target = _parse_checksum_file(sidecar)
        if target not in expected_names or not (output_root / target).is_file():
            raise ReleaseArtifactError(f"checksum sidecar target is invalid: {target}")
        if digest != _sha256_file(output_root / target):
            raise ReleaseArtifactError(f"checksum sidecar mismatch for {target}")
    if TRAINING_ARCHIVE_NAME in expected_names:
        verify_training_archive(output_root / TRAINING_ARCHIVE_NAME)
    expected_build_records = expected_names.intersection(BUILD_ENVIRONMENTS)
    if expected_build_records:
        if cargo_lock is None:
            raise ReleaseArtifactError(
                "Cargo.lock is required to verify build environments"
            )
        for name in sorted(expected_build_records):
            verify_build_environment_record(
                output_root / name,
                BUILD_ENVIRONMENTS[name],
                git_commit,
                cargo_lock,
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    training = subparsers.add_parser("create-training-archive")
    training.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."))
    training.add_argument("--manifest", type=pathlib.Path, required=True)
    training.add_argument("--output", type=pathlib.Path, required=True)

    build = subparsers.add_parser("record-build-environment")
    build.add_argument("--output", type=pathlib.Path, required=True)
    build.add_argument("--expected-os", required=True)
    build.add_argument("--expected-arch", required=True)
    build.add_argument("--expected-host", required=True)
    build.add_argument("--matrix-name", required=True)
    build.add_argument("--features", required=True)
    build.add_argument("--git-commit", required=True)
    build.add_argument(
        "--cargo-lock", type=pathlib.Path, default=pathlib.Path("Cargo.lock")
    )

    augment = subparsers.add_parser("augment-spdx")
    augment.add_argument("--sbom", type=pathlib.Path, required=True)
    augment.add_argument("--dist", type=pathlib.Path, required=True)
    augment.add_argument(
        "--cargo-lock", type=pathlib.Path, default=pathlib.Path("Cargo.lock")
    )
    augment.add_argument("--target", required=True)

    verify = subparsers.add_parser("verify-spdx")
    verify.add_argument("--sbom", type=pathlib.Path, required=True)
    verify.add_argument("--dist", type=pathlib.Path, required=True)
    verify.add_argument(
        "--cargo-lock", type=pathlib.Path, default=pathlib.Path("Cargo.lock")
    )
    verify.add_argument("--target", required=True)

    stage = subparsers.add_parser("stage-release")
    stage.add_argument("--source", type=pathlib.Path, required=True)
    stage.add_argument("--output", type=pathlib.Path, required=True)
    stage.add_argument("--contract", type=pathlib.Path, required=True)
    stage.add_argument("--git-commit", required=True)
    stage.add_argument("--tag", required=True)
    stage.add_argument(
        "--cargo-lock", type=pathlib.Path, default=pathlib.Path("Cargo.lock")
    )

    staged = subparsers.add_parser("verify-staged-release")
    staged.add_argument("--output", type=pathlib.Path, required=True)
    staged.add_argument("--contract", type=pathlib.Path, required=True)
    staged.add_argument("--git-commit", required=True)
    staged.add_argument("--tag", required=True)
    staged.add_argument(
        "--cargo-lock", type=pathlib.Path, default=pathlib.Path("Cargo.lock")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create-training-archive":
            create_training_archive(args.root, args.manifest, args.output)
        elif args.command == "record-build-environment":
            record_build_environment(
                args.output,
                args.expected_os,
                args.expected_arch,
                args.expected_host,
                args.matrix_name,
                args.features,
                args.git_commit,
                args.cargo_lock,
            )
        elif args.command == "augment-spdx":
            augment_spdx(args.sbom, args.dist, args.cargo_lock, args.target)
        elif args.command == "verify-spdx":
            verify_spdx(args.sbom, args.dist, args.cargo_lock, args.target)
        elif args.command == "stage-release":
            stage_release(
                args.source,
                args.output,
                args.contract,
                args.git_commit,
                args.tag,
                args.cargo_lock,
            )
        elif args.command == "verify-staged-release":
            verify_staged_release(
                args.output,
                args.contract,
                args.git_commit,
                args.tag,
                args.cargo_lock,
            )
        else:
            raise AssertionError(f"unhandled command: {args.command}")
    except (OSError, subprocess.CalledProcessError, ReleaseArtifactError) as exc:
        print(f"release artifact verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
