#!/usr/bin/env python3
"""Verify package hashes, JSON syntax, and high-confidence secret signatures."""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AUDIT_SUMMARY = ROOT / "audit-summary-2026-07-13.json"
AUDIT_REPORT = ROOT / "audit-report-2026-07-13.json.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[tuple[str, Path]]:
    entries = []
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\0]+)", line)
        if not match:
            raise ValueError(f"invalid manifest row {path.name}:{line_number}")
        relative = Path(match.group(2))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe manifest path {relative}")
        entries.append((match.group(1), ROOT / relative))
    return entries


def verify_manifest(name: str) -> set[Path]:
    manifest = ROOT / name
    checked = set()
    for expected, path in read_manifest(manifest):
        if not path.is_file():
            raise ValueError(f"missing file listed by {name}: {path.relative_to(ROOT)}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(
                f"hash mismatch for {path.relative_to(ROOT)}: expected {expected}, got {actual}"
            )
        checked.add(path.resolve())
    return checked


def scan_secrets(files: list[Path]) -> None:
    signatures = {
        "private key": re.compile(b"-----BEGIN " + b"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        "AWS access key": re.compile(b"AK" + b"IA[0-9A-Z]{16}"),
        "GitHub token": re.compile(b"gh" + b"[pousr]_[A-Za-z0-9]{30,}"),
        "OpenAI-style key": re.compile(b"sk" + b"-[A-Za-z0-9_-]{24,}"),
        "Google API key": re.compile(b"AI" + b"za[0-9A-Za-z_-]{35}"),
        "Slack token": re.compile(b"xo" + b"[xbaprs]-[A-Za-z0-9-]{20,}"),
    }
    for path in files:
        data = path.read_bytes()
        for label, signature in signatures.items():
            if signature.search(data):
                raise ValueError(f"possible {label} in {path.relative_to(ROOT)}")


def verify_published_audit() -> None:
    summary = json.loads(AUDIT_SUMMARY.read_text(encoding="utf-8"))
    publication = summary.get("published_full_report")
    if summary.get("full_report_published") is not True or not isinstance(publication, dict):
        raise ValueError("published full audit declaration is missing")
    if publication.get("path") != "training/qwen35/audit-report-2026-07-13.json.gz":
        raise ValueError("published full audit path is not canonical")

    compressed = AUDIT_REPORT.read_bytes()
    if len(compressed) != publication.get("compressed_size"):
        raise ValueError("published full audit compressed size mismatch")
    if hashlib.sha256(compressed).hexdigest() != publication.get("compressed_sha256"):
        raise ValueError("published full audit compressed digest mismatch")
    try:
        raw = gzip.decompress(compressed)
    except OSError as exc:
        raise ValueError(f"published full audit is not valid gzip: {exc}") from exc
    if len(raw) != publication.get("uncompressed_size"):
        raise ValueError("published full audit uncompressed size mismatch")
    raw_digest = hashlib.sha256(raw).hexdigest()
    if raw_digest != publication.get("uncompressed_sha256"):
        raise ValueError("published full audit uncompressed digest mismatch")
    if raw_digest != summary.get("full_private_report_sha256"):
        raise ValueError("published full audit does not match the private audit anchor")

    try:
        report = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"published full audit JSON is invalid: {exc}") from exc
    expected_keys = {
        "aggregate_sha256",
        "disclosure",
        "holdout_denylist",
        "language_histogram",
        "license_histogram",
        "prompt_version",
        "repository_histogram",
        "row_histogram",
        "schema_version",
    }
    if set(report) != expected_keys:
        raise ValueError("published full audit contains unexpected fields")
    if report.get("schema_version") != "greppy.qwen35-sft-audit.v1":
        raise ValueError("published full audit schema mismatch")
    if report.get("disclosure") != "No source text or generated labels are included in this report.":
        raise ValueError("published full audit disclosure mismatch")
    if report.get("aggregate_sha256") != summary.get("aggregate_row_sha256"):
        raise ValueError("published full audit aggregate digest mismatch")
    if report.get("prompt_version") != summary.get("prompt_version"):
        raise ValueError("published full audit prompt version mismatch")
    if report.get("holdout_denylist") != summary.get("summary_quality_holdout"):
        raise ValueError("published full audit holdout result mismatch")
    if report.get("language_histogram") != summary.get("languages"):
        raise ValueError("published full audit language histogram mismatch")

    repositories = report.get("repository_histogram")
    licenses = report.get("license_histogram")
    rows = report.get("row_histogram")
    if not isinstance(repositories, dict) or len(repositories) != summary.get("repository_count"):
        raise ValueError("published full audit repository histogram mismatch")
    if not isinstance(licenses, dict) or len(licenses) != summary.get("license_combination_count"):
        raise ValueError("published full audit license histogram mismatch")
    if not isinstance(rows, dict):
        raise ValueError("published full audit row histogram is missing")
    for field, expected in summary.get("rows", {}).items():
        if rows.get(field) != expected:
            raise ValueError(f"published full audit row count mismatch: {field}")
    for label, histogram in (("repository", repositories), ("license", licenses)):
        if any(
            not isinstance(key, str)
            or not key
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            for key, count in histogram.items()
        ):
            raise ValueError(f"published full audit {label} histogram is invalid")


def main() -> int:
    try:
        source_scripts = verify_manifest("SOURCE_SCRIPTS.sha256")
        package_files = verify_manifest("MANIFEST.sha256")
        expected_scripts = {path.resolve() for path in (ROOT / "scripts").iterdir() if path.is_file()}
        if source_scripts != expected_scripts:
            raise ValueError("SOURCE_SCRIPTS.sha256 does not cover exactly scripts/*")

        actual_files = {
            path.resolve()
            for path in ROOT.rglob("*")
            if path.is_file()
            and path.name != "MANIFEST.sha256"
            and "__pycache__" not in path.parts
        }
        if package_files != actual_files:
            missing = sorted(str(path.relative_to(ROOT)) for path in actual_files - package_files)
            extra = sorted(str(path.relative_to(ROOT)) for path in package_files - actual_files)
            raise ValueError(f"MANIFEST.sha256 coverage mismatch: missing={missing}, extra={extra}")

        for name in ("environment.lock.json", "provenance.json"):
            json.loads((ROOT / name).read_text(encoding="utf-8"))
        verify_published_audit()
        scan_secrets(sorted(actual_files | {(ROOT / "MANIFEST.sha256").resolve()}))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        print(f"package verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"verified {len(actual_files) + 1} package files; secret scan clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
