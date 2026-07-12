#!/usr/bin/env python3
"""Audit private Qwen3.5 raw rows against private SFT JSONL without disclosure."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


PROMPT_VERSION = "qwen35-brief-v3"
PROMPT_PREFIX = "<|im_start|>user\nbrief:\n"
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
COMPLETION_SUFFIX = "<|im_end|>"
DEFAULT_DENYLIST = Path(__file__).with_name("summary_quality_holdout_repos.txt")

RAW_CORE_FIELDS = {
    "lang",
    "repo",
    "path",
    "source",
    "license",
    "hexsha",
    "docstring_stripped",
}
RAW_ACCEPTED_FIELDS = RAW_CORE_FIELDS | {"summary", "tokens"}
RAW_DROPPED_FIELDS = RAW_CORE_FIELDS | {"summary", "dropped", "last_errors"}
SFT_FIELDS = {"prompt", "completion", "lang", "repo"}


class AuditError(ValueError):
    """A deterministic audit failure that is safe to print."""


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _location(path: Path, line_number: int) -> str:
    return f"{path.name}:{line_number}"


def _jsonl(paths: Iterable[Path]):
    for path in sorted(paths, key=lambda item: item.as_posix()):
        try:
            handle = path.open("r", encoding="utf-8", newline="")
        except OSError as exc:
            raise AuditError(f"cannot read {path}: {exc}") from exc
        with handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise AuditError(f"unexpected blank row at {_location(path, line_number)}")
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AuditError(
                        f"invalid JSON at {_location(path, line_number)}: {exc.msg}"
                    ) from exc
                if not isinstance(row, dict):
                    raise AuditError(f"row is not an object at {_location(path, line_number)}")
                yield path, line_number, row


def _nonempty_string(value, field: str, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuditError(f"{field} must be a non-empty string at {where}")
    return value


def _license_label(value, where: str) -> str:
    if isinstance(value, str):
        if not value.strip():
            raise AuditError(f"missing license at {where}")
        return value.strip()
    if isinstance(value, list):
        if not value:
            raise AuditError(f"missing license at {where}")
        normalized = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise AuditError(f"invalid license metadata at {where}")
            normalized.append(item.strip())
        if len(set(normalized)) != len(normalized):
            raise AuditError(f"duplicate license metadata at {where}")
        return json.dumps(sorted(normalized), ensure_ascii=True, separators=(",", ":"))
    raise AuditError(f"missing license at {where}")


def canonical_repo(value: str) -> str:
    """Normalize common GitHub identity spellings to lowercase owner/repo."""
    repo = value.strip()
    if "://" in repo:
        parsed = urlparse(repo)
        if parsed.hostname and parsed.hostname.lower() == "github.com":
            repo = parsed.path
    elif repo.lower().startswith("github.com/"):
        repo = repo[len("github.com/") :]
    repo = repo.strip("/").lower()
    if repo.endswith(".git"):
        repo = repo[:-4]
    return repo


def _load_denylist(paths: Iterable[Path]) -> set[str]:
    denied: set[str] = set()
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise AuditError(f"cannot read denylist {path}: {exc}") from exc
        for line_number, line in enumerate(lines, 1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            repo = canonical_repo(value)
            if len(repo.split("/")) != 2:
                raise AuditError(f"invalid repository at {_location(path, line_number)}")
            if repo in denied:
                raise AuditError(f"duplicate denylist repository {repo}")
            denied.add(repo)
    return denied


def _validate_raw(row: dict, where: str) -> tuple[bool, str]:
    fields = set(row)
    if fields == RAW_ACCEPTED_FIELDS:
        accepted = True
    elif fields == RAW_DROPPED_FIELDS:
        accepted = False
    else:
        expected = [sorted(RAW_ACCEPTED_FIELDS), sorted(RAW_DROPPED_FIELDS)]
        raise AuditError(f"unexpected raw schema at {where}; expected one of {expected}")

    for field in ("lang", "repo", "path", "source", "hexsha"):
        _nonempty_string(row[field], field, where)
    if not isinstance(row["docstring_stripped"], bool):
        raise AuditError(f"docstring_stripped must be boolean at {where}")
    license_label = _license_label(row["license"], where)

    if accepted:
        summary = row["summary"]
        if not isinstance(summary, list) or not 1 <= len(summary) <= 2:
            raise AuditError(f"summary must contain one or two lines at {where}")
        for item in summary:
            _nonempty_string(item, "summary item", where)
            if "\n" in item or "\r" in item:
                raise AuditError(f"summary item must be one line at {where}")
        if row["tokens"] is not None and (
            not isinstance(row["tokens"], int) or isinstance(row["tokens"], bool)
        ):
            raise AuditError(f"tokens must be an integer or null at {where}")
    else:
        if row["summary"] is not None or row["dropped"] is not True:
            raise AuditError(f"invalid dropped-row state at {where}")
        errors = row["last_errors"]
        if not isinstance(errors, list) or not errors or not all(
            isinstance(item, str) and item for item in errors
        ):
            raise AuditError(f"last_errors must be a non-empty string list at {where}")
    return accepted, license_label


def _validate_sft(row: dict, where: str) -> None:
    if set(row) != SFT_FIELDS:
        raise AuditError(f"unexpected SFT schema at {where}; expected {sorted(SFT_FIELDS)}")
    for field in sorted(SFT_FIELDS):
        _nonempty_string(row[field], field, where)


def audit(
    raw_paths: Iterable[Path],
    sft_paths: Iterable[Path],
    extra_denylist_paths: Iterable[Path] = (),
) -> dict:
    raw_paths = tuple(Path(path) for path in raw_paths)
    sft_paths = tuple(Path(path) for path in sft_paths)
    if not raw_paths or not sft_paths:
        raise AuditError("at least one raw and one SFT JSONL file are required")
    if len(set(raw_paths)) != len(raw_paths) or len(set(sft_paths)) != len(sft_paths):
        raise AuditError("duplicate input path")

    deny_paths = (DEFAULT_DENYLIST, *(Path(path) for path in extra_denylist_paths))
    denied_repos = _load_denylist(deny_paths)

    prompt_index: dict[str, list[tuple[dict, str, str]]] = {}
    source_by_digest: dict[str, str] = {}
    raw_total = accepted_raw = dropped_raw = raw_denylist_matches = 0
    duplicate_raw_sources = 0

    for path, line_number, row in _jsonl(raw_paths):
        where = _location(path, line_number)
        accepted, license_label = _validate_raw(row, where)
        raw_total += 1
        source = row["source"]
        source_digest = _sha256(source)
        prior = source_by_digest.get(source_digest)
        if prior is not None and prior != source:
            raise AuditError(f"SHA-256 source collision at {where}")
        source_by_digest[source_digest] = source
        if canonical_repo(row["repo"]) in denied_repos:
            raw_denylist_matches += 1
        if not accepted:
            dropped_raw += 1
            continue
        accepted_raw += 1
        expected_prompt = PROMPT_PREFIX + source.strip() + PROMPT_SUFFIX
        candidates = prompt_index.setdefault(expected_prompt, [])
        if candidates:
            if any(candidate[0]["source"] != source for candidate in candidates):
                raise AuditError(f"normalized prompt collision at {where}")
            duplicate_raw_sources += 1
        candidates.append((row, license_label, source_digest))

    seen_prompts: set[str] = set()
    selected_prompts: set[str] = set()
    repo_histogram: Counter[str] = Counter()
    language_histogram: Counter[str] = Counter()
    license_histogram: Counter[str] = Counter()
    aggregate = hashlib.sha256()
    sft_total = 0

    for path, line_number, row in _jsonl(sft_paths):
        where = _location(path, line_number)
        _validate_sft(row, where)
        sft_total += 1
        prompt = row["prompt"]
        if prompt in seen_prompts:
            raise AuditError(f"duplicate SFT prompt at {where}")
        seen_prompts.add(prompt)
        candidates = prompt_index.get(prompt)
        if candidates is None:
            raise AuditError(f"unmapped SFT prompt at {where}")
        matched = [
            candidate
            for candidate in candidates
            if row["lang"] == candidate[0]["lang"]
            and row["repo"] == candidate[0]["repo"]
        ]
        if len(matched) != 1:
            raise AuditError(f"SFT provenance is ambiguous at {where}")
        raw, license_label, source_digest = matched[0]
        selected_prompts.add(prompt)

        expected_completion = "\n".join(raw["summary"]) + COMPLETION_SUFFIX
        if row["completion"] != expected_completion:
            raise AuditError(f"SFT label mismatch at {where}")
        canonical = canonical_repo(raw["repo"])
        if canonical in denied_repos:
            raise AuditError(f"selected row uses denied holdout repository {canonical} at {where}")

        repo_histogram[raw["repo"]] += 1
        language_histogram[raw["lang"]] += 1
        license_histogram[license_label] += 1
        digest_row = {
            "completion_sha256": _sha256(row["completion"]),
            "language": raw["lang"],
            "license": license_label,
            "repo": raw["repo"],
            "row": sft_total,
            "source_sha256": source_digest,
        }
        encoded = json.dumps(
            digest_row, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        aggregate.update(encoded + b"\n")

    missing = set(prompt_index).difference(selected_prompts)
    if missing:
        raise AuditError(f"{len(missing)} accepted raw rows are absent from the SFT input")

    return {
        "schema_version": "greppy.qwen35-sft-audit.v1",
        "prompt_version": PROMPT_VERSION,
        "row_histogram": {
            "accepted_raw": accepted_raw,
            "distinct_accepted_prompts": len(prompt_index),
            "duplicate_raw_sources": duplicate_raw_sources,
            "dropped_raw": dropped_raw,
            "mapped_sft": sft_total,
            "raw_total": raw_total,
            "sft_total": sft_total,
        },
        "repository_histogram": dict(sorted(repo_histogram.items())),
        "language_histogram": dict(sorted(language_histogram.items())),
        "license_histogram": dict(sorted(license_histogram.items())),
        "holdout_denylist": {
            "repositories": sorted(denied_repos),
            "raw_matches": raw_denylist_matches,
            "selected_matches": 0,
        },
        "aggregate_sha256": aggregate.hexdigest(),
        "disclosure": "No source text or generated labels are included in this report.",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", nargs="+", required=True, type=Path)
    parser.add_argument("--sft", nargs="+", required=True, type=Path)
    parser.add_argument(
        "--denylist",
        action="append",
        default=[],
        type=Path,
        help="additional newline-delimited owner/repo denylist (default list is always active)",
    )
    parser.add_argument("--output", type=Path, help="write the report to this JSON file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = audit(args.raw, args.sft, args.denylist)
    except AuditError as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
