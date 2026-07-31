#!/usr/bin/env python3
"""Orchestrate and seal a quota-balanced v3 taskbank from merged PRs.

Repository adapters harvest authoritative PR/issue metadata and execute the
repository-native clean-room proof.  The sealing stage binds that proof to
freshly extracted patches, applies preregistered quotas, and emits source-only
parent snapshots.  The orchestrator never clones repositories itself.
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import fnmatch
import hashlib
import hmac
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

try:
    from . import SCHEMA_VERSION
    from .storage import StorageError, StorageLayout, load_storage
except ImportError:  # direct script execution
    from __init__ import SCHEMA_VERSION
    from storage import StorageError, StorageLayout, load_storage


HEX_OID = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
OPAQUE_ID = re.compile(r"^task_[a-z2-7]{26}$")
SAFE_KEY = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
UTC = dt.timezone.utc
HERE = Path(__file__).resolve().parent
LEGACY_DENYLISTS = (
    HERE.parent / "tasks_v2.json",
    HERE.parent / "harvest_candidates_v2.jsonl",
    HERE.parent / "harvest_candidates_v2_serious.jsonl",
)

# These are the same conservative source/test split rules used by
# swe_bench_adapter.py, extended to the v3 registry's languages.  A path that
# mixes production and tests cannot be a hidden-test task and is rejected by
# the declared path-set equality checks.
DEFAULT_TEST_GLOBS: Mapping[str, tuple[str, ...]] = {
    "python": ("tests/**", "test/**", "**/tests/**", "**/test_*.py", "**/*_test.py", "**/conftest.py"),
    "rust": ("tests/**", "**/tests/**"),
    "go": ("*_test.go", "**/*_test.go"),
    "java": ("**/src/test/**", "**/*Test.java", "**/*Tests.java"),
    "typescript": (
        "tests/**", "test/**", "**/tests/**", "**/*.test.ts", "**/*.test.tsx",
        "**/*.spec.ts", "**/*.spec.tsx", "**/*.test.js", "**/*.spec.js",
    ),
    "javascript": (
        "tests/**", "test/**", "**/tests/**", "**/*.test.js", "**/*.test.mjs",
        "**/*.spec.js", "**/*.spec.mjs",
    ),
    "cpp": ("test/**", "tests/**", "**/test/**", "**/tests/**", "**/*_test.cc", "**/*_test.cpp"),
    "ruby": ("test/**", "spec/**", "**/test/**", "**/spec/**", "**/*_test.rb", "**/*_spec.rb"),
}
DEFAULT_IGNORE_GLOBS = (
    "docs/**", "doc/**", "**/docs/**", "**/doc/**", ".github/**", "**/.github/**",
    "CHANGELOG*", "CHANGES*", "HISTORY*", "NEWS*", "**/CHANGELOG*", "**/CHANGES*",
    "**/HISTORY*", "**/NEWS*", "*.lock", "**/*.lock", "package-lock.json",
    "pnpm-lock.yaml", "yarn.lock", "Cargo.lock", "**/package-lock.json",
    "**/pnpm-lock.yaml", "**/yarn.lock", "**/Cargo.lock",
)


class HarvestError(ValueError):
    """An input cannot safely enter the sealed taskbank."""


@dataclass(frozen=True)
class Freeze:
    freeze_id: str
    frozen_at: dt.datetime
    eligible_created_after: dt.datetime
    eligible_after: dt.datetime
    eligible_before: dt.datetime
    source_metadata_cutoff: dt.datetime


@dataclass(frozen=True)
class RepoRule:
    key: str
    url: str
    language: str
    mirror: str
    minimum: int
    quota: int
    test_globs: tuple[str, ...]
    ignore_globs: tuple[str, ...]
    task_class_slots: Mapping[str, int]
    toolchain: str
    allow_submodules: bool = False


@dataclass(frozen=True)
class Quotas:
    target_tasks: int
    min_repositories: int
    max_per_language: Mapping[str, int]
    task_class_quotas: Mapping[str, int]


@dataclass(frozen=True)
class AdmissionRules:
    minimum_source_files: int
    minimum_source_loc: int
    minimum_band_counts: Mapping[str, int]
    minimum_production_paths: int = 2
    maximum_production_paths: int = 20
    minimum_production_changed_lines: int = 40
    maximum_production_changed_lines: int = 1200
    maximum_total_paths: int = 30
    minimum_candidates_per_slot: int = 2


@dataclass
class PreparedTask:
    task_id: str
    repo: RepoRule
    row: dict[str, Any]
    parent: str
    solution: str
    merged_at: dt.datetime
    changed_paths: list[str]
    test_paths: list[str]
    source_paths: list[str]
    test_patch: bytes
    gold_patch: bytes
    full_patch: bytes
    test_sha256: str
    gold_sha256: str
    full_sha256: str
    selection_score: str
    production_changed_lines: int
    ignored_paths: list[str]
    repository_scale: dict[str, Any]
    candidate_commitment: str


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_time(value: Any, field: str) -> dt.datetime:
    if not isinstance(value, str) or not value.strip():
        raise HarvestError(f"{field} must be an ISO-8601 timestamp")
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError as exc:
        raise HarvestError(f"{field} is not ISO-8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise HarvestError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def format_time(value: dt.datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarvestError(f"cannot read JSON from {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise HarvestError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise HarvestError(f"{path}:{line_number}: expected an object")
                rows.append(value)
    except OSError as exc:
        raise HarvestError(f"cannot read {path}: {exc}") from exc
    return rows


def load_freeze(path: Path) -> Freeze:
    document = load_json(path)
    if not isinstance(document, dict):
        raise HarvestError("freeze document must be an object")
    if document.get("schema_version") != "greppy.agent-coding-freeze.v1":
        raise HarvestError("unsupported freeze schema_version")
    freeze_id = document.get("freeze_id")
    if not isinstance(freeze_id, str) or SAFE_KEY.fullmatch(freeze_id) is None:
        raise HarvestError("freeze_id must be a safe opaque path component")
    freeze = Freeze(
        freeze_id=freeze_id,
        frozen_at=parse_time(document.get("frozen_at"), "frozen_at"),
        eligible_created_after=parse_time(
            document.get("eligible_pr_created_after"), "eligible_pr_created_after"
        ),
        eligible_after=parse_time(document.get("eligible_merged_after"), "eligible_merged_after"),
        eligible_before=parse_time(document.get("eligible_merged_before"), "eligible_merged_before"),
        source_metadata_cutoff=parse_time(
            document.get("source_metadata_cutoff"), "source_metadata_cutoff"
        ),
    )
    if not (
        freeze.eligible_created_after <= freeze.eligible_after <= freeze.eligible_before
        <= freeze.source_metadata_cutoff <= freeze.frozen_at
    ):
        raise HarvestError(
            "freeze times must satisfy created_after <= merged_after <= merged_before <= "
            "source_metadata_cutoff <= frozen_at"
        )
    return freeze


def load_contract(path: Path, freeze: Freeze) -> tuple[dict[str, Any], AdmissionRules, bytes]:
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HarvestError(f"invalid corpus contract JSON: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != "greppy.agent-coding-v3.corpus-contract.1":
        raise HarvestError("unsupported corpus contract schema_version")
    temporal = document.get("temporal_holdout")
    scale = document.get("repository_scale")
    validation = document.get("validation")
    if not all(isinstance(value, dict) for value in (temporal, scale, validation)):
        raise HarvestError("corpus contract lacks temporal, scale, or validation sections")
    expected_times = {
        "candidate_pr_created_at_or_after": freeze.eligible_created_after,
        "candidate_pr_merged_at_or_after": freeze.eligible_after,
        "candidate_pr_merged_at_or_before": freeze.eligible_before,
    }
    for field, expected in expected_times.items():
        if parse_time(temporal.get(field), f"temporal_holdout.{field}") != expected:
            raise HarvestError(f"freeze does not match corpus contract field {field}")
    rules = AdmissionRules(
        minimum_source_files=_positive_int(
            scale.get("minimum_eligible_source_files"), "repository_scale.minimum_eligible_source_files"
        ),
        minimum_source_loc=_positive_int(
            scale.get("minimum_eligible_source_loc"), "repository_scale.minimum_eligible_source_loc"
        ),
        minimum_band_counts={
            str(key): _positive_int(value, f"repository_scale.minimum_band_counts.{key}")
            for key, value in (scale.get("minimum_band_counts") or {}).items()
        },
        minimum_candidates_per_slot=_positive_int(
            validation.get("minimum_candidate_pool_per_repo_class_slot"),
            "validation.minimum_candidate_pool_per_repo_class_slot",
        ),
    )
    return document, rules, raw


def _positive_int(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < (0 if allow_zero else 1):
        raise HarvestError(f"{field} must be a {'non-negative' if allow_zero else 'positive'} integer")
    return value


def load_registry(path: Path) -> tuple[dict[str, RepoRule], Quotas, bytes]:
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HarvestError(f"invalid registry JSON: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != "greppy.agent-coding-v3.repository-registry.1":
        raise HarvestError("unsupported registry schema_version")
    rows = document.get("repositories")
    patterns = document.get("selection_patterns")
    languages = document.get("primary_languages")
    if not isinstance(rows, list) or not rows or not isinstance(patterns, dict):
        raise HarvestError("registry needs selection_patterns and repositories")
    if not isinstance(languages, list) or not all(isinstance(value, str) for value in languages):
        raise HarvestError("registry primary_languages must be a string array")
    target = _positive_int(document.get("target_task_count"), "target_task_count")
    repository_count = _positive_int(document.get("repository_count"), "repository_count")
    per_repo = _positive_int(document.get("tasks_per_repository"), "tasks_per_repository")
    per_language = _positive_int(document.get("language_task_quota"), "language_task_quota")
    if repository_count != len(rows) or target != repository_count * per_repo:
        raise HarvestError("registry task/repository counts are not internally consistent")
    quotas = Quotas(
        target_tasks=target,
        min_repositories=repository_count,
        max_per_language={str(language): per_language for language in languages},
        task_class_quotas={},
    )
    rules: dict[str, RepoRule] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise HarvestError(f"repositories[{index}] must be an object")
        key = row.get("id")
        if not isinstance(key, str) or SAFE_KEY.fullmatch(key) is None or key in rules:
            raise HarvestError(f"repositories[{index}].key is invalid or duplicated")
        values = (row.get("url"), row.get("primary_language"), row.get("toolchain_profile"))
        if not all(isinstance(value, str) and value for value in values):
            raise HarvestError(f"repository {key} needs url, primary_language and toolchain_profile")
        mirror = f"{key}.git"
        if PurePosixPath(mirror).is_absolute() or ".." in PurePosixPath(mirror).parts:
            raise HarvestError(f"repository {key} mirror must be relative to the NVMe mirror root")
        language = str(row["primary_language"])
        if language not in DEFAULT_TEST_GLOBS:
            raise HarvestError(f"repository {key} has no path classifier for language {language}")
        pattern_name = row.get("selection_pattern")
        class_slots = patterns.get(pattern_name) if isinstance(pattern_name, str) else None
        if not isinstance(class_slots, dict) or sum(class_slots.values()) != per_repo:
            raise HarvestError(f"repository {key} has an invalid selection pattern")
        rules[key] = RepoRule(
            key=key,
            url=str(row["url"]).removesuffix(".git").rstrip("/"),
            language=language,
            mirror=mirror,
            minimum=0,
            quota=per_repo,
            test_globs=DEFAULT_TEST_GLOBS[language],
            ignore_globs=DEFAULT_IGNORE_GLOBS,
            task_class_slots={
                str(name): _positive_int(
                    count, f"repository {key} task_class_slots.{name}", allow_zero=True
                )
                for name, count in class_slots.items()
            },
            toolchain=str(row["toolchain_profile"]),
            allow_submodules=bool(row.get("allow_submodules", False)),
        )
    declared_slots = Counter()
    for rule in rules.values():
        declared_slots.update(rule.task_class_slots)
        if rule.task_class_slots and sum(rule.task_class_slots.values()) != rule.quota:
            raise HarvestError(f"repository {rule.key} class slots do not sum to its quota")
    quotas = Quotas(
        target_tasks=quotas.target_tasks,
        min_repositories=quotas.min_repositories,
        max_per_language=quotas.max_per_language,
        task_class_quotas=dict(declared_slots),
    )
    return rules, quotas, raw


def git(repo: Path, args: Sequence[str], *, text: bool = False) -> bytes | str:
    proc = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=text,
        errors="replace" if text else None,
    )
    if proc.returncode:
        stderr = proc.stderr if text else proc.stderr.decode("utf-8", "replace")
        raise HarvestError(f"git {' '.join(args)} failed in {repo}: {stderr.strip()[-600:]}")
    return proc.stdout


def normalize_url(value: str) -> str:
    value = value.strip().removesuffix(".git").rstrip("/")
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    return value


def verify_mirror(path: Path, rule: RepoRule) -> None:
    if not path.exists():
        raise HarvestError(f"local mirror is missing for {rule.key}: {path}")
    origin = str(git(path, ["remote", "get-url", "origin"], text=True)).strip()
    if normalize_url(origin) != normalize_url(rule.url):
        raise HarvestError(f"origin mismatch for {rule.key}: {origin!r}")


def changed_paths(repo: Path, parent: str, solution: str) -> list[str]:
    output = str(git(repo, ["diff", "--name-only", "--no-renames", parent, solution], text=True))
    paths = [line for line in output.splitlines() if line]
    if not paths:
        raise HarvestError("solution commit has no changed paths")
    if any(path.startswith("/") or ".." in PurePosixPath(path).parts for path in paths):
        raise HarvestError("commit contains an unsafe path")
    return paths


VENDOR_GENERATED_PARTS = {
    "vendor", "vendors", "third_party", "third-party", "node_modules",
    "target", "dist", "build", "generated", "gen", "external", "externals",
}


def is_vendor_or_generated(path: str) -> bool:
    pure = PurePosixPath(path)
    lowered = {part.lower() for part in pure.parts[:-1]}
    name = pure.name.lower()
    return bool(
        lowered & VENDOR_GENERATED_PARTS
        or ".generated." in name or name.endswith((".min.js", ".min.css", "_pb.go", ".pb.cc", ".pb.h"))
    )


def production_changed_lines(
    repo: Path, parent: str, solution: str, paths: Sequence[str]
) -> int:
    output = str(git(
        repo, ["diff", "--numstat", "--no-renames", parent, solution, "--", *paths], text=True
    ))
    total = 0
    observed: set[str] = set()
    for line in output.splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3 or parts[0] == "-" or parts[1] == "-":
            raise HarvestError("production diff contains binary or malformed numstat data")
        total += int(parts[0]) + int(parts[1])
        observed.add(parts[2])
    if observed != set(paths):
        raise HarvestError("production numstat does not cover the declared source path set")
    return total


def validate_repository_scale(
    row: Mapping[str, Any], repo: Path, parent: str, rules: AdmissionRules
) -> dict[str, Any]:
    scale = row.get("repository_scale")
    if not isinstance(scale, dict):
        raise HarvestError("candidate lacks parent-bound repository_scale evidence")
    parent_tree = str(git(repo, ["rev-parse", f"{parent}^{{tree}}"], text=True)).strip().lower()
    required = {
        "measurement_revision": "v1",
        "parent_tree": parent_tree,
    }
    for field, expected in required.items():
        if scale.get(field) != expected:
            raise HarvestError(f"repository_scale.{field} is not bound to the parent tree")
    files = scale.get("eligible_source_files")
    loc = scale.get("eligible_source_loc")
    if not isinstance(files, int) or isinstance(files, bool) or files < rules.minimum_source_files:
        raise HarvestError("repository is below the eligible source-file floor")
    if not isinstance(loc, int) or isinstance(loc, bool) or loc < rules.minimum_source_loc:
        raise HarvestError("repository is below the eligible source-LOC floor")
    band = scale.get("size_band")
    if band not in ("medium", "large", "very_large"):
        raise HarvestError("repository_scale.size_band is invalid")
    band_matches = (
        (band == "medium" and 25_000 <= loc < 100_000)
        or (band == "large" and 100_000 <= loc < 500_000)
        or (band == "very_large" and loc >= 500_000)
    )
    if not band_matches:
        raise HarvestError("repository_scale.size_band does not match eligible source LOC")
    measurement_hash = scale.get("measurement_sha256")
    material = {key: value for key, value in scale.items() if key != "measurement_sha256"}
    if measurement_hash != sha256(canonical_json(material)):
        raise HarvestError("repository scale measurement hash mismatch")
    return dict(scale)


def extract_patch(repo: Path, parent: str, solution: str, paths: Sequence[str]) -> bytes:
    if not paths:
        return b""
    return bytes(git(repo, [
        "diff", "--binary", "--full-index", "--no-renames", parent, solution, "--", *paths
    ]))


def opaque_id(key: bytes, freeze: Freeze, repo_key: str, pr_number: int, solution: str) -> str:
    identity = canonical_json({
        "freeze": freeze.freeze_id,
        "repo": repo_key,
        "pr": pr_number,
        "solution": solution,
    })
    digest = hmac.new(key, b"opaque-id-v3\0" + identity, hashlib.sha256).digest()[:16]
    import base64
    return "task_" + base64.b32encode(digest).decode("ascii").lower().rstrip("=")


def candidate_identity(repo_key: str, pr_number: int, parent: str, solution: str) -> bytes:
    return canonical_json({
        "repository": repo_key,
        "pull_request": pr_number,
        "parent_commit": parent,
        "merged_result": solution,
    })


def require_row(row: Mapping[str, Any], field: str, expected: type) -> Any:
    value = row.get(field)
    if not isinstance(value, expected) or isinstance(value, bool):
        raise HarvestError(f"candidate {field} must be {expected.__name__}")
    return value


def validate_proof(row: Mapping[str, Any], hashes: Mapping[str, str]) -> None:
    proof = row.get("validation")
    if not isinstance(proof, dict):
        raise HarvestError("candidate lacks validation proof")
    required_states = {
        "parent_baseline": "pass",
        "parent_plus_test": "fail",
        "gold_plus_test": "pass",
        "merged_plus_test": "pass",
        "clean_room_repetitions": 2,
        "offline": True,
    }
    for field, expected in required_states.items():
        if proof.get(field) != expected:
            raise HarvestError(f"validation proof {field} must be {expected!r}")
    fail_to_pass = proof.get("fail_to_pass")
    pass_to_pass = proof.get("pass_to_pass")
    if not isinstance(fail_to_pass, list) or not fail_to_pass or not all(
        isinstance(test, str) and test for test in fail_to_pass
    ):
        raise HarvestError("validation proof needs a non-empty FAIL_TO_PASS set")
    if not isinstance(pass_to_pass, list) or not pass_to_pass or not all(
        isinstance(test, str) and test for test in pass_to_pass
    ):
        raise HarvestError("validation proof needs a non-empty PASS_TO_PASS regression set")
    for field in ("test_patch_sha256", "gold_patch_sha256", "full_patch_sha256"):
        expected = proof.get(field)
        if not isinstance(expected, str) or expected != hashes[field]:
            raise HarvestError(f"validation proof {field} does not bind the extracted patch")
    command = proof.get("test_command")
    if not isinstance(command, list) or not command or not all(isinstance(arg, str) and arg for arg in command):
        raise HarvestError("validation proof needs an argv-array test_command")
    for field in ("setup_commands", "post_patch_commands"):
        commands = proof.get(field)
        if not isinstance(commands, list) or not all(
            isinstance(argv, list) and argv
            and all(isinstance(arg, str) and arg for arg in argv)
            for argv in commands
        ):
            raise HarvestError(f"validation proof {field} must be argv arrays")
    for field in ("fail_to_pass_commands", "pass_to_pass_commands"):
        commands = proof.get(field)
        if commands is not None and (
            not isinstance(commands, list) or not commands
            or not all(
                isinstance(argv, list) and argv
                and all(isinstance(arg, str) and arg for arg in argv)
                for argv in commands
            )
        ):
            raise HarvestError(f"validation proof {field} must be non-empty argv arrays")
    runner_digest = proof.get("runner_image_digest")
    if not isinstance(runner_digest, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", runner_digest) is None:
        raise HarvestError("validation proof needs a sha256 runner_image_digest")
    logs_hash = proof.get("logs_sha256")
    if not isinstance(logs_hash, str) or re.fullmatch(r"[0-9a-f]{64}", logs_hash) is None:
        raise HarvestError("validation proof needs logs_sha256")
    validated_at = parse_time(proof.get("validated_at"), "validation.validated_at")
    if validated_at > dt.datetime.now(UTC) + dt.timedelta(minutes=5):
        raise HarvestError("validation proof is dated in the future")


def prepare_candidate(
    row: dict[str, Any], rule: RepoRule, repo: Path, freeze: Freeze, id_key: bytes,
    admission: AdmissionRules,
) -> PreparedTask:
    pr_number = _positive_int(row.get("pr_number"), "candidate pr_number")
    solution = str(require_row(row, "solution_commit", str)).lower()
    if HEX_OID.fullmatch(solution) is None:
        raise HarvestError("solution_commit must be a full Git object id")
    merged_at = parse_time(row.get("merged_at"), "candidate merged_at")
    created_at = parse_time(row.get("created_at"), "candidate created_at")
    if created_at < freeze.eligible_created_after:
        raise HarvestError("candidate PR predates the frozen creation cutoff")
    if created_at > merged_at:
        raise HarvestError("candidate PR was created after it merged")
    if not freeze.eligible_after <= merged_at <= freeze.eligible_before:
        raise HarvestError("candidate falls outside the frozen merge window")
    issue_number = _positive_int(row.get("issue_number"), "candidate issue_number")
    issue_url = str(require_row(row, "issue_url", str))
    if not issue_url.startswith("https://") or "/pull/" in issue_url:
        raise HarvestError("issue_url must identify the linked issue, not a pull request")
    issue_title = str(require_row(row, "issue_title", str))
    issue_body = str(require_row(row, "issue_body", str))
    user_task = issue_title + ("\n\n" + issue_body if issue_body else "")
    task_type = str(require_row(row, "task_class", str)).strip()
    if not issue_title or not task_type:
        raise HarvestError("issue_title and task_class must be non-empty")
    if task_type not in rule.task_class_slots or rule.task_class_slots[task_type] == 0:
        raise HarvestError(f"task_class {task_type!r} has no slot for repository {rule.key}")
    if solution[:8].lower() in user_task.lower():
        raise HarvestError("user_task leaks the solution object id")
    if re.search(rf"(?:/pull/|\bPR\s*#?)\s*{pr_number}\b", user_task, re.IGNORECASE):
        raise HarvestError("user_task links the solution pull request")

    resolved = str(git(repo, ["rev-parse", f"{solution}^{{commit}}"], text=True)).strip().lower()
    if resolved != solution:
        raise HarvestError("solution_commit does not resolve exactly")
    parent = str(git(repo, ["rev-parse", f"{solution}^1"], text=True)).strip().lower()
    declared_parent = row.get("parent_commit")
    if not isinstance(declared_parent, str) or declared_parent.lower() != parent:
        raise HarvestError("authoritative parent_commit must equal the merged result's first parent")
    strategy = row.get("merge_strategy")
    if strategy not in ("merge", "squash", "rebase"):
        raise HarvestError("merge_strategy must be merge, squash, or rebase")
    provenance = row.get("merge_provenance")
    if not isinstance(provenance, dict) or any(
        provenance.get(field) is not True
        for field in ("target_parent_verified", "merged_result_tree_verified", "pr_delta_no_target_drift")
    ):
        raise HarvestError("merge provenance is incomplete or ambiguous")
    metadata_hash = provenance.get("authoritative_metadata_sha256")
    if not isinstance(metadata_hash, str) or re.fullmatch(r"[0-9a-f]{64}", metadata_hash) is None:
        raise HarvestError("merge provenance lacks authoritative metadata hash")
    commit_time = parse_time(
        str(git(repo, ["show", "-s", "--format=%cI", solution], text=True)).strip(),
        "solution commit time",
    )

    if commit_time > freeze.source_metadata_cutoff:
        raise HarvestError("solution commit is newer than the frozen source metadata cutoff")

    paths = changed_paths(repo, parent, solution)
    if len(paths) > admission.maximum_total_paths:
        raise HarvestError(f"full PR changes {len(paths)} paths; maximum is {admission.maximum_total_paths}")
    ignored = {
        path for path in paths
        if any(fnmatch.fnmatchcase(path, pattern) for pattern in rule.ignore_globs)
        or is_vendor_or_generated(path)
    }
    tests = [
        path for path in paths if path not in ignored
        and any(fnmatch.fnmatchcase(path, pattern) for pattern in rule.test_globs)
    ]
    sources = [path for path in paths if path not in ignored and path not in set(tests)]
    declared_tests = row.get("changed_tests")
    declared_sources = row.get("changed_source")
    if declared_tests is not None and declared_tests != tests:
        raise HarvestError("changed_tests does not exactly match registry classification")
    if declared_sources is not None and declared_sources != sources:
        raise HarvestError("changed_source does not exactly match registry classification")
    if not tests or not sources:
        raise HarvestError("candidate must change both classified tests and implementation/source files")
    if not admission.minimum_production_paths <= len(sources) <= admission.maximum_production_paths:
        raise HarvestError(
            f"candidate changes {len(sources)} production paths; required "
            f"{admission.minimum_production_paths}..{admission.maximum_production_paths}"
        )
    if len(ignored) * 2 >= len(paths):
        raise HarvestError("ignored, generated, or vendored paths make up at least half the PR")
    changed_line_count = production_changed_lines(repo, parent, solution, sources)
    if not (
        admission.minimum_production_changed_lines
        <= changed_line_count <= admission.maximum_production_changed_lines
    ):
        raise HarvestError(
            f"production diff changes {changed_line_count} lines; required "
            f"{admission.minimum_production_changed_lines}..{admission.maximum_production_changed_lines}"
        )
    scale = validate_repository_scale(row, repo, parent, admission)
    test_patch = extract_patch(repo, parent, solution, tests)
    gold_patch = extract_patch(repo, parent, solution, sources)
    full_patch = extract_patch(repo, parent, solution, paths)
    hashes = {
        "test_patch_sha256": sha256(test_patch),
        "gold_patch_sha256": sha256(gold_patch),
        "full_patch_sha256": sha256(full_patch),
    }
    validate_proof(row, hashes)
    task_id = opaque_id(id_key, freeze, rule.key, pr_number, solution)
    identity = candidate_identity(rule.key, pr_number, parent, solution)
    commitment = sha256(identity)
    score = hmac.new(id_key, b"selection-v3\0" + identity, hashlib.sha256).hexdigest()
    row = dict(row)
    row["user_task"] = user_task
    row["task_class"] = task_type
    row["issue_number"] = issue_number
    row["issue_url"] = issue_url
    return PreparedTask(
        task_id=task_id, repo=rule, row=row, parent=parent, solution=solution,
        merged_at=merged_at, changed_paths=paths, test_paths=tests, source_paths=sources,
        test_patch=test_patch, gold_patch=gold_patch, full_patch=full_patch,
        test_sha256=hashes["test_patch_sha256"], gold_sha256=hashes["gold_patch_sha256"],
        full_sha256=hashes["full_patch_sha256"], selection_score=score,
        production_changed_lines=changed_line_count, ignored_paths=sorted(ignored),
        repository_scale=scale, candidate_commitment=commitment,
    )


def _word_tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]{2,}", value.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _has_near_duplicate_review(task: PreparedTask, peer: PreparedTask) -> bool:
    reviews = task.row.get("near_duplicate_reviews")
    if not isinstance(reviews, list):
        return False
    for review in reviews:
        if not isinstance(review, dict) or review.get("peer_commitment") != peer.candidate_commitment:
            continue
        if review.get("decision") not in ("retain", "distinct"):
            continue
        if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
            continue
        try:
            parse_time(review.get("reviewed_at"), "near_duplicate_review.reviewed_at")
        except HarvestError:
            continue
        return True
    return False


def audit_near_duplicates(tasks: Sequence[PreparedTask]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for index, left in enumerate(tasks):
        for right in tasks[index + 1:]:
            title_similarity = difflib.SequenceMatcher(
                None, str(left.row["issue_title"]).lower(), str(right.row["issue_title"]).lower()
            ).ratio()
            path_similarity = _jaccard(set(left.source_paths), set(right.source_paths))
            diff_similarity = _jaccard(
                _word_tokens(left.gold_patch.decode("utf-8", "replace")),
                _word_tokens(right.gold_patch.decode("utf-8", "replace")),
            )
            maximum = max(title_similarity, path_similarity, diff_similarity)
            if maximum <= 0.80:
                continue
            if not (
                _has_near_duplicate_review(left, right)
                and _has_near_duplicate_review(right, left)
            ):
                raise HarvestError(
                    "near-duplicate candidates above 0.80 lack reciprocal blinded review: "
                    f"{left.candidate_commitment} vs {right.candidate_commitment}"
                )
            findings.append({
                "left": left.candidate_commitment,
                "right": right.candidate_commitment,
                "title_similarity": round(title_similarity, 6),
                "path_jaccard": round(path_similarity, 6),
                "production_diff_token_jaccard": round(diff_similarity, 6),
                "reviewed": True,
            })
    return findings


def require_candidate_pools(
    tasks: Sequence[PreparedTask], rules: Mapping[str, RepoRule], minimum_per_slot: int
) -> None:
    counts = Counter((task.repo.key, str(task.row["task_class"])) for task in tasks)
    bands_by_repo: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        bands_by_repo[task.repo.key].add(str(task.repository_scale["size_band"]))
    inconsistent = sorted(key for key, bands in bands_by_repo.items() if len(bands) != 1)
    if inconsistent:
        raise HarvestError(f"repository scale band varies across candidate parents: {inconsistent}")
    for key, rule in rules.items():
        for task_class, slots in rule.task_class_slots.items():
            if slots <= 0:
                continue
            required = slots * minimum_per_slot
            if counts[(key, task_class)] < required:
                raise HarvestError(
                    f"candidate pool {key}/{task_class} has {counts[(key, task_class)]} passing "
                    f"candidates; requires at least {required} for {slots} slot(s)"
                )


def enforce_band_distribution(selected: Sequence[PreparedTask], admission: AdmissionRules) -> None:
    repo_bands: dict[str, str] = {}
    for task in selected:
        repo_bands[task.repo.key] = str(task.repository_scale["size_band"])
    observed = Counter(repo_bands.values())
    for band, minimum in admission.minimum_band_counts.items():
        if observed[band] < minimum:
            raise HarvestError(
                f"selected corpus has {observed[band]} {band} repositories; requires at least {minimum}"
            )


def load_denylists(paths: Sequence[Path]) -> tuple[dict[str, set[Any]], list[dict[str, Any]]]:
    if not paths:
        raise HarvestError(
            "at least one explicit denylist is required for SWE-bench and prior Greppy corpus coverage"
        )
    denied: dict[str, set[Any]] = {
        "solutions": set(), "solution_prefixes": set(), "gold_hashes": set(),
        "test_hashes": set(), "issue_urls": set(),
    }
    commitments: list[dict[str, Any]] = []
    coverage: set[str] = set()

    def add_entry(entry: Mapping[str, Any]) -> None:
        url = entry.get("repository_url") or entry.get("repo")
        normalized = normalize_url(url) if isinstance(url, str) else None
        solution = entry.get("solution_commit") or entry.get("commit") or entry.get("merge_commit")
        if normalized and isinstance(solution, str) and re.fullmatch(r"[0-9a-fA-F]{40,64}", solution):
            denied["solutions"].add((normalized, solution.lower()))
        prefix = entry.get("solution_prefix")
        if normalized and isinstance(prefix, str) and re.fullmatch(r"[0-9a-fA-F]{8,64}", prefix):
            denied["solution_prefixes"].add((normalized, prefix.lower()))
        for field, bucket in (
            ("gold_patch_sha256", "gold_hashes"), ("test_patch_sha256", "test_hashes")
        ):
            value = entry.get(field)
            if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
                denied[bucket].add(value)
        issue_url = entry.get("issue_url")
        if isinstance(issue_url, str):
            denied["issue_urls"].add(issue_url)

    for path in [*LEGACY_DENYLISTS, *paths]:
        if not path.exists():
            if path in LEGACY_DENYLISTS:
                continue
            raise HarvestError(f"denylist is missing: {path}")
        raw = path.read_bytes()
        commitments.append({"path_label": path.name, "sha256": sha256(raw), "bytes": len(raw)})
        if path.suffix == ".jsonl":
            for row in load_jsonl(path):
                add_entry(row)
            continue
        document = load_json(path)
        if isinstance(document, dict) and document.get("schema_version") == "greppy.agent-coding-v3.denylist.1":
            declared = document.get("coverage")
            entries = document.get("entries")
            if not isinstance(declared, list) or not all(isinstance(item, str) for item in declared):
                raise HarvestError(f"denylist {path} has invalid coverage")
            if not isinstance(entries, list) or not all(isinstance(item, dict) for item in entries):
                raise HarvestError(f"denylist {path} has invalid entries")
            coverage.update(declared)
            for entry in entries:
                add_entry(entry)
        elif isinstance(document, dict) and isinstance(document.get("tasks"), list):
            for task in document["tasks"]:
                if not isinstance(task, dict):
                    continue
                repository = task.get("repository") or {}
                task_id = task.get("id")
                entry: dict[str, Any] = {"repository_url": repository.get("url")}
                if isinstance(task_id, str):
                    suffix = task_id.rsplit("-", 1)[-1]
                    if re.fullmatch(r"[0-9a-fA-F]{8,64}", suffix):
                        entry["solution_prefix"] = suffix
                add_entry(entry)
        else:
            raise HarvestError(f"unsupported denylist format: {path}")
    required_coverage = {"swe-bench", "prior-greppy"}
    if not required_coverage <= coverage:
        raise HarvestError(
            f"explicit denylists lack coverage for {sorted(required_coverage - coverage)}"
        )
    return denied, commitments


def enforce_denylists(tasks: Sequence[PreparedTask], denied: Mapping[str, set[Any]]) -> None:
    for task in tasks:
        repo_url = normalize_url(task.repo.url)
        reasons: list[str] = []
        if (repo_url, task.solution) in denied["solutions"]:
            reasons.append("solution commit")
        if any(
            repo_url == denied_url and task.solution.startswith(prefix)
            for denied_url, prefix in denied["solution_prefixes"]
        ):
            reasons.append("solution prefix")
        if task.gold_sha256 in denied["gold_hashes"]:
            reasons.append("gold patch")
        if task.test_sha256 in denied["test_hashes"]:
            reasons.append("test patch")
        if task.row["issue_url"] in denied["issue_urls"]:
            reasons.append("issue URL")
        if reasons:
            raise HarvestError(
                f"candidate {task.candidate_commitment} matches denylist by {', '.join(reasons)}"
            )


def validate_stage_manifests(
    paths: Sequence[Path], freeze: Freeze, adapter_manifest_path: Path,
    adapters: Mapping[str, Mapping[str, Any]], candidates_path: Path,
) -> list[dict[str, Any]]:
    if len(paths) != 2:
        raise HarvestError("seal requires exactly metadata and validation stage manifests")
    expected_adapter_hash = sha256(adapter_manifest_path.read_bytes())
    by_stage: dict[str, tuple[Path, dict[str, Any]]] = {}
    expected_images = {
        key: {"image": adapter["image"], "image_id": adapter["image_id"]}
        for key, adapter in sorted(adapters.items())
    }
    for path in paths:
        document = load_json(path)
        if not isinstance(document, dict) or document.get("schema_version") != "greppy.agent-coding-v3.adapter-stage.1":
            raise HarvestError(f"invalid adapter stage manifest: {path}")
        stage = document.get("stage")
        if stage not in ("metadata", "validate") or stage in by_stage:
            raise HarvestError("stage manifests must contain metadata and validate exactly once")
        if (
            document.get("freeze_id") != freeze.freeze_id
            or document.get("adapter_manifest_sha256") != expected_adapter_hash
            or document.get("images") != expected_images
            or document.get("network") != ("bridge" if stage == "metadata" else "none")
        ):
            raise HarvestError(f"adapter {stage} stage manifest commitments differ from seal inputs")
        combined = path.parent / "all.jsonl"
        if not combined.is_file() or sha256(combined.read_bytes()) != document.get("combined_sha256"):
            raise HarvestError(f"adapter {stage} combined ledger hash mismatch")
        if stage == "validate" and combined.resolve() != candidates_path.resolve():
            raise HarvestError("seal candidates must be the exact validation-stage combined ledger")
        by_stage[str(stage)] = (path, document)
    if set(by_stage) != {"metadata", "validate"}:
        raise HarvestError("both metadata and validation stage manifests are required")
    return [
        {
            "stage": stage,
            "manifest_sha256": sha256(path.read_bytes()),
            "combined_sha256": document["combined_sha256"],
            "network": document["network"],
            "images": document["images"],
        }
        for stage, (path, document) in sorted(by_stage.items())
    ]


def select_tasks(tasks: Sequence[PreparedTask], rules: Mapping[str, RepoRule], quotas: Quotas) -> list[PreparedTask]:
    pools: dict[str, list[PreparedTask]] = defaultdict(list)
    seen_sources: set[tuple[str, str]] = set()
    seen_ids: set[str] = set()
    seen_gold: set[str] = set()
    seen_tests: set[str] = set()
    seen_issue_states: set[tuple[str, str, str]] = set()
    for task in tasks:
        source = (task.repo.key, task.solution)
        issue_state = (task.repo.key, task.parent, str(task.row["issue_url"]))
        if (
            source in seen_sources or task.task_id in seen_ids
            or task.gold_sha256 in seen_gold or task.test_sha256 in seen_tests
            or issue_state in seen_issue_states
        ):
            raise HarvestError("duplicate candidate source, patch, issue state, or opaque id")
        seen_sources.add(source)
        seen_ids.add(task.task_id)
        seen_gold.add(task.gold_sha256)
        seen_tests.add(task.test_sha256)
        seen_issue_states.add(issue_state)
        pools[task.repo.key].append(task)
    for pool in pools.values():
        pool.sort(key=lambda task: task.selection_score)

    selected: list[PreparedTask] = []
    repo_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()

    def can_take(task: PreparedTask) -> bool:
        task_type = str(task.row["task_class"])
        language_cap = quotas.max_per_language.get(task.repo.language)
        type_cap = task.repo.task_class_slots.get(task_type) if task.repo.task_class_slots else None
        return (
            repo_counts[task.repo.key] < task.repo.quota
            and (language_cap is None or language_counts[task.repo.language] < language_cap)
            and (
                type_cap is None
                or sum(
                    1 for selected_task in selected
                    if selected_task.repo.key == task.repo.key
                    and selected_task.row["task_class"] == task_type
                ) < type_cap
            )
        )

    def take(task: PreparedTask) -> None:
        selected.append(task)
        repo_counts[task.repo.key] += 1
        language_counts[task.repo.language] += 1
        type_counts[str(task.row["task_class"])] += 1

    # First honor declared repository floors.  This prevents a high-yield repo
    # from replacing harder-to-validate repositories.
    for key in sorted(rules):
        rule = rules[key]
        if len(pools.get(key, [])) < rule.minimum:
            raise HarvestError(f"repository {key} has fewer eligible tasks than its minimum")
        for task in pools.get(key, [])[: rule.minimum]:
            if not can_take(task):
                raise HarvestError(f"global caps conflict with repository minimum for {key}")
            take(task)

    positions = {key: rules[key].minimum for key in rules}
    while len(selected) < quotas.target_tasks:
        progressed = False
        for key in sorted(rules):
            pool = pools.get(key, [])
            pos = positions[key]
            while pos < len(pool):
                task = pool[pos]
                pos += 1
                positions[key] = pos
                if can_take(task):
                    take(task)
                    progressed = True
                    break
            if len(selected) >= quotas.target_tasks:
                break
        if not progressed:
            break

    if len(selected) != quotas.target_tasks:
        raise HarvestError(f"quotas selected {len(selected)} tasks, need {quotas.target_tasks}")
    represented = {task.repo.key for task in selected}
    if len(represented) < quotas.min_repositories:
        raise HarvestError(
            f"only {len(represented)} repositories represented, need {quotas.min_repositories}"
        )
    for key, rule in rules.items():
        actual = Counter(
            str(task.row["task_class"]) for task in selected if task.repo.key == key
        )
        expected = Counter({name: count for name, count in rule.task_class_slots.items() if count})
        if actual != expected:
            raise HarvestError(f"selected class slots for {key} are {dict(actual)}, expected {dict(expected)}")
    if language_counts != Counter(quotas.max_per_language):
        raise HarvestError(
            f"selected language quotas are {dict(language_counts)}, expected {dict(quotas.max_per_language)}"
        )
    if type_counts != Counter(quotas.task_class_quotas):
        raise HarvestError(
            f"selected class quotas are {dict(type_counts)}, expected {dict(quotas.task_class_quotas)}"
        )
    return sorted(selected, key=lambda task: task.task_id)


def tree_entries(repo: Path, parent: str) -> list[tuple[int, str, str, str]]:
    raw = bytes(git(repo, ["ls-tree", "-rz", "--full-tree", parent]))
    entries: list[tuple[int, str, str, str]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode_text, kind, oid = metadata.decode("ascii").split(" ")
        path = raw_path.decode("utf-8", "surrogateescape")
        entries.append((int(mode_text, 8), kind, oid, path))
    return sorted(entries, key=lambda item: item[3])


def write_parent_snapshot(repo: Path, parent: str, output: Path, *, allow_submodules: bool) -> str:
    """Write a deterministic source-only tar; no Git objects, refs, or remotes."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with tarfile.open(temporary, "w", format=tarfile.PAX_FORMAT) as archive:
            directories: set[str] = set()
            for mode, kind, oid, path in tree_entries(repo, parent):
                if kind == "commit":
                    if allow_submodules:
                        continue
                    raise HarvestError(f"parent snapshot contains submodule {path}")
                if kind != "blob":
                    raise HarvestError(f"unsupported Git tree entry {kind} at {path}")
                parts = PurePosixPath(path).parts[:-1]
                current = ""
                for part in parts:
                    current = f"{current}/{part}".strip("/")
                    if current not in directories:
                        info = tarfile.TarInfo(current + "/")
                        info.type = tarfile.DIRTYPE
                        info.mode = 0o755
                        info.mtime = info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        archive.addfile(info)
                        directories.add(current)
                data = bytes(git(repo, ["cat-file", "blob", oid]))
                info = tarfile.TarInfo(path)
                info.mtime = info.uid = info.gid = 0
                info.uname = info.gname = ""
                if mode == 0o120000:
                    link_target = data.decode("utf-8", "surrogateescape")
                    combined = PurePosixPath(os.path.normpath(str(PurePosixPath(path).parent / link_target)))
                    if PurePosixPath(link_target).is_absolute() or ".." in combined.parts:
                        raise HarvestError(f"parent snapshot contains unsafe symlink {path}")
                    info.type = tarfile.SYMTYPE
                    info.linkname = link_target
                    info.mode = 0o777
                    archive.addfile(info)
                else:
                    info.mode = 0o755 if mode & 0o111 else 0o644
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
        digest = sha256(temporary.read_bytes())
        os.replace(temporary, output)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def build_release(
    selected: Sequence[PreparedTask], freeze: Freeze, layout: StorageLayout,
    registry_bytes: bytes, key_fingerprint: str, commitments: Mapping[str, Any],
    near_duplicate_findings: Sequence[Mapping[str, Any]],
) -> Path:
    target = layout.releases / freeze.freeze_id
    if target.exists():
        raise HarvestError(f"frozen release already exists: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{freeze.freeze_id}.", dir=layout.releases))
    try:
        public_tasks: list[dict[str, Any]] = []
        sealed_tasks: list[dict[str, Any]] = []
        for task in selected:
            snapshot_rel = f"snapshots/{task.task_id}.tar"
            snapshot_path = staging / "public" / snapshot_rel
            mirror = layout.mirrors / task.repo.mirror
            snapshot_hash = write_parent_snapshot(
                mirror, task.parent, snapshot_path, allow_submodules=task.repo.allow_submodules
            )
            patch_dir = staging / "sealed" / "patches"
            patch_dir.mkdir(parents=True, exist_ok=True)
            (patch_dir / f"{task.task_id}.test.patch").write_bytes(task.test_patch)
            (patch_dir / f"{task.task_id}.gold.patch").write_bytes(task.gold_patch)
            public_tasks.append({
                "id": task.task_id,
                "language": task.repo.language,
                "task_class": task.row["task_class"],
                "user_task": task.row["user_task"],
                "workspace": {
                    "snapshot": snapshot_rel,
                    "snapshot_sha256": snapshot_hash,
                    "git_history": "import snapshot into a fresh repository with exactly one commit",
                    "network": "disabled",
                },
            })
            proof = task.row["validation"]
            evaluation = {
                "test_command": proof["test_command"],
                "setup_commands": proof["setup_commands"],
                "post_patch_commands": proof["post_patch_commands"],
                "timeout_seconds": proof.get("timeout_seconds", 1800),
                **(
                    {"fail_to_pass_commands": proof["fail_to_pass_commands"]}
                    if "fail_to_pass_commands" in proof else {}
                ),
                **(
                    {"pass_to_pass_commands": proof["pass_to_pass_commands"]}
                    if "pass_to_pass_commands" in proof else {}
                ),
            }
            sealed_tasks.append({
                "id": task.task_id,
                "repository": {"key": task.repo.key, "url": task.repo.url},
                "pr_number": task.row["pr_number"],
                "parent_commit": task.parent,
                "solution_commit": task.solution,
                "merged_at": format_time(task.merged_at),
                "candidate_commitment": task.candidate_commitment,
                "changed_paths": task.changed_paths,
                "test_paths": task.test_paths,
                "source_paths": task.source_paths,
                "admission": {
                    "production_changed_lines": task.production_changed_lines,
                    "production_path_count": len(task.source_paths),
                    "total_path_count": len(task.changed_paths),
                    "ignored_generated_vendor_paths": task.ignored_paths,
                    "repository_scale": task.repository_scale,
                },
                "artifacts": {
                    "test_patch": f"patches/{task.task_id}.test.patch",
                    "gold_patch": f"patches/{task.task_id}.gold.patch",
                },
                "hashes": {
                    "test_patch_sha256": task.test_sha256,
                    "gold_patch_sha256": task.gold_sha256,
                    "full_patch_sha256": task.full_sha256,
                    "evaluation_sha256": sha256(canonical_json(evaluation)),
                },
                "issue_number": task.row["issue_number"],
                "evaluation": evaluation,
                "validation_evidence": {
                    "parent_plus_test": proof["parent_plus_test"],
                    "parent_baseline": proof["parent_baseline"],
                    "gold_plus_test": proof["gold_plus_test"],
                    "merged_plus_test": proof["merged_plus_test"],
                    "clean_room_repetitions": proof["clean_room_repetitions"],
                    "offline": proof["offline"],
                    "fail_to_pass": proof["fail_to_pass"],
                    "pass_to_pass": proof["pass_to_pass"],
                    "runner_image_digest": proof["runner_image_digest"],
                    "logs_sha256": proof["logs_sha256"],
                    "validated_at": proof["validated_at"],
                },
            })

        public_doc = {
            "schema_version": SCHEMA_VERSION,
            "freeze": {
                "id": freeze.freeze_id,
                "frozen_at": format_time(freeze.frozen_at),
                "eligible_merged_after": format_time(freeze.eligible_after),
                "eligible_merged_before": format_time(freeze.eligible_before),
            },
            "execution_contract": {
                "controller_mount": "public directory read-only; this mount is not inherited by the agent",
                "agent_mount": "fresh extracted parent workspace only",
                "snapshot_contains_git_metadata": False,
                "import_snapshot_as_single_commit_before_agent": True,
                "apply_test_patch_before_agent": False,
                "apply_test_patch_only_in_fresh_grading_workspace_after_agent": True,
                "expose_patch_files_to_agent": False,
                "expose_task_id_in_agent_prompt_path_or_environment": False,
                "network": "disabled",
            },
            "tasks": public_tasks,
        }
        sealed_doc = {
            "schema_version": "greppy.agent-coding-sealed.v3",
            "freeze_id": freeze.freeze_id,
            "id_key_sha256": key_fingerprint,
            "registry_sha256": sha256(registry_bytes),
            "commitments": dict(commitments),
            "tasks": sealed_tasks,
        }
        atomic_json(staging / "public" / "taskbank.json", public_doc)
        atomic_json(staging / "sealed" / "manifest.json", sealed_doc)
        evidence = {
            "schema_version": "greppy.agent-coding-harvest-evidence.v3",
            "freeze_id": freeze.freeze_id,
            "created_at": format_time(dt.datetime.now(UTC)),
            "task_count": len(selected),
            "repository_counts": dict(sorted(Counter(t.repo.key for t in selected).items())),
            "language_counts": dict(sorted(Counter(t.repo.language for t in selected).items())),
            "task_class_counts": dict(sorted(Counter(str(t.row["task_class"]) for t in selected).items())),
            "public_taskbank_sha256": sha256((staging / "public" / "taskbank.json").read_bytes()),
            "sealed_manifest_sha256": sha256((staging / "sealed" / "manifest.json").read_bytes()),
            "commitments": dict(commitments),
            "near_duplicate_review_findings": list(near_duplicate_findings),
        }
        evidence["artifact_inventory"] = [
            {
                "path": path.relative_to(staging).as_posix(),
                "sha256": sha256(path.read_bytes()),
                "bytes": path.stat().st_size,
            }
            for path in sorted(staging.rglob("*")) if path.is_file()
        ]
        atomic_json(staging / "evidence" / "harvest.json", evidence)
        for path in sorted(staging.rglob("*"), reverse=True):
            path.chmod(0o444 if path.is_file() else 0o555)
        os.replace(staging, target)
        target.chmod(0o555)
        directory_fd = os.open(layout.releases, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return target
    except Exception:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise


def read_id_key(path: Path) -> bytes:
    try:
        key = path.read_bytes().strip()
    except OSError as exc:
        raise HarvestError(f"cannot read id key {path}: {exc}") from exc
    if len(key) < 32:
        raise HarvestError("id key must contain at least 32 bytes")
    return key


def load_adapter_manifest(path: Path) -> dict[str, dict[str, Any]]:
    document = load_json(path)
    if not isinstance(document, dict) or document.get("schema_version") != "greppy.agent-coding-v3.adapter-manifest.1":
        raise HarvestError("unsupported adapter manifest schema_version")
    rows = document.get("adapters")
    if not isinstance(rows, list):
        raise HarvestError("adapter manifest needs an adapters array")
    adapters: dict[str, dict[str, Any]] = {}
    for adapter in rows:
        if not isinstance(adapter, dict):
            raise HarvestError("adapter rows must be objects")
        key = adapter.get("repository_id")
        if not isinstance(key, str) or not key or key in adapters:
            raise HarvestError("adapter repository_id is missing or duplicated")
        if adapter.get("status") != "ready":
            continue
        proof = adapter.get("proof_sha256")
        if not isinstance(proof, str) or re.fullmatch(r"[0-9a-f]{64}", proof) is None:
            raise HarvestError(f"ready adapter {key} lacks proof_sha256")
        image = adapter.get("image")
        image_id = adapter.get("image_id")
        if not isinstance(image, str) or re.fullmatch(r".+@sha256:[0-9a-f]{64}", image) is None:
            raise HarvestError(f"ready adapter {key} lacks digest-pinned image")
        if not isinstance(image_id, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
            raise HarvestError(f"ready adapter {key} lacks local image_id")
        commands = adapter.get("commands")
        if not isinstance(commands, dict) or set(commands) != {"probe", "metadata", "validation"}:
            raise HarvestError(f"ready adapter {key} needs exact command roles")
        for field in ("probe", "metadata", "validation"):
            command = commands.get(field)
            if not isinstance(command, list) or not command or not all(
                isinstance(arg, str) and arg for arg in command
            ):
                raise HarvestError(f"adapter {key} needs argv-array {field}")
        adapters[key] = adapter
    return adapters


def inspect_adapter_image(
    docker_binary: str, adapter: Mapping[str, Any], *, cwd: Path
) -> str:
    proc = subprocess.run(
        [docker_binary, "image", "inspect", str(adapter["image"])],
        cwd=cwd, capture_output=True, text=True, errors="replace", timeout=60,
    )
    try:
        payload = json.loads(proc.stdout) if proc.returncode == 0 else None
    except json.JSONDecodeError as exc:
        raise HarvestError(f"Docker returned invalid image inspection JSON: {exc}") from exc
    expected = adapter["image_id"]
    if not isinstance(payload, list) or len(payload) != 1 or payload[0].get("Id") != expected:
        raise HarvestError(
            f"adapter image {adapter['image']} is absent or resolves to a different image ID"
        )
    return str(expected)


def adapter_container_argv(
    docker_binary: str, adapter: Mapping[str, Any], role: str, *,
    network: str, mounts: Sequence[tuple[Path, str, bool]], arguments: Sequence[str],
    environment_file: Path | None = None,
) -> list[str]:
    argv = [
        docker_binary, "run", "--rm", "--network", network, "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "512", "--user", f"{os.getuid()}:{os.getgid()}",
        "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=2g",
    ]
    if environment_file is not None:
        environment_file = environment_file.resolve()
        if not environment_file.is_file():
            raise HarvestError(f"adapter environment file is missing: {environment_file}")
        argv.extend(("--env-file", str(environment_file)))
    for source, destination, readonly in mounts:
        source = source.resolve()
        if not source.exists():
            raise HarvestError(f"adapter mount source is missing: {source}")
        spec = f"type=bind,src={source},dst={destination}"
        if readonly:
            spec += ",readonly"
        argv.extend(("--mount", spec))
    argv.extend((str(adapter["image"]), *adapter["commands"][role], *arguments))
    return argv


def preflight(
    registry_path: Path, layout: StorageLayout, adapter_manifest_path: Path | None
) -> dict[str, Any]:
    rules, _, _ = load_registry(registry_path)
    registry = load_json(registry_path)
    adapters = load_adapter_manifest(adapter_manifest_path) if adapter_manifest_path else {}
    docker_binary = shutil.which("docker")
    greppy_binary = shutil.which("greppy")
    missing_adapters: list[str] = []
    for key, rule in sorted(rules.items()):
        adapter = adapters.get(key)
        if (
            docker_binary is None or greppy_binary is None or adapter is None
            or adapter.get("toolchain_profile") != rule.toolchain
        ):
            missing_adapters.append(key)
            continue
        try:
            inspect_adapter_image(docker_binary, adapter, cwd=adapter_manifest_path.parent)
            roles_ready = True
            for role in ("probe", "metadata", "validation"):
                argv = adapter_container_argv(
                    docker_binary, adapter, role, network="none",
                    mounts=((Path(greppy_binary), "/tools/greppy", True),),
                    # The adapter probe contract checks the exact 0.3.0 binary
                    # at the same in-container path used by gpu3 preflight.
                    arguments=("--preflight",),
                )
                proc = subprocess.run(
                    argv, cwd=adapter_manifest_path.parent, capture_output=True,
                    text=True, errors="replace", timeout=60,
                )
                payload = json.loads(proc.stdout) if proc.stdout.strip() else None
                roles_ready = roles_ready and bool(
                    proc.returncode == 0 and isinstance(payload, dict)
                    and payload.get("ready") is True
                    and payload.get("repository_id") == key
                    and payload.get("command_role") == role
                    and payload.get("proof_sha256") == adapter.get("proof_sha256")
                )
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError, HarvestError):
            roles_ready = False
        if not roles_ready:
            missing_adapters.append(key)
    mirror_missing = [key for key, rule in sorted(rules.items()) if not (layout.mirrors / rule.mirror).exists()]
    result = {
        "schema_version": "greppy.agent-coding-v3.preflight.1",
        "ok": docker_binary is not None and greppy_binary is not None and not missing_adapters and not mirror_missing,
        "storage": {
            "nvme_device": os.stat(layout.nvme_root).st_dev,
            "nas_device": os.stat(layout.nas_root).st_dev,
            "nvme_free_bytes": shutil.disk_usage(layout.nvme_root).free,
            "nas_free_bytes": shutil.disk_usage(layout.nas_root).free,
        },
        "docker_binary": docker_binary,
        "greppy_binary": greppy_binary,
        "missing_adapters": missing_adapters,
        "missing_mirrors": mirror_missing,
    }
    if result["storage"]["nvme_device"] == result["storage"]["nas_device"]:
        result["ok"] = False
        result["storage"]["error"] = "NVMe and NAS roots resolve to the same device"
    return result


def run_adapter_stage(
    *, stage: str, registry_path: Path, freeze_path: Path, adapter_manifest_path: Path,
    layout: StorageLayout, docker_binary: str | None = None,
    metadata_env_file: Path | None = None,
) -> Path:
    """Run registered per-repository metadata or clean-room validation adapters.

    Adapters are argv arrays, never shell strings.  They own repository-specific
    GitHub/API and test-runner details; this orchestrator owns cutoffs, storage,
    per-repository isolation and crash-safe combined ledgers.
    """
    if stage not in ("metadata", "validate"):
        raise HarvestError(f"unsupported adapter stage {stage}")
    rules, _, _ = load_registry(registry_path)
    freeze = load_freeze(freeze_path)
    adapters = load_adapter_manifest(adapter_manifest_path)
    missing = sorted(set(rules) - set(adapters))
    if missing:
        raise HarvestError(f"no executable v3 adapter for repositories: {missing}")
    docker_binary = docker_binary or shutil.which("docker")
    if docker_binary is None:
        raise HarvestError("Docker is required; adapter stages never execute on the host")
    stage_root = layout.scratch / freeze.freeze_id / stage
    stage_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    stage_images: dict[str, dict[str, str]] = {}
    stage_stats: dict[str, dict[str, int]] = {}
    for key in sorted(rules):
        rule = rules[key]
        adapter = adapters[key]
        observed_image_id = inspect_adapter_image(
            docker_binary, adapter, cwd=adapter_manifest_path.parent
        )
        stage_images[key] = {"image": str(adapter["image"]), "image_id": observed_image_id}
        output = stage_root / f"{key}.jsonl"
        io_dir = stage_root / ".adapter-io" / key
        io_dir.mkdir(parents=True, exist_ok=True)
        temp = io_dir / f"{key}.jsonl.partial"
        temp.unlink(missing_ok=True)
        if stage == "metadata":
            if metadata_env_file is None:
                raise HarvestError(
                    "metadata stage requires an explicit Docker --env-file for authenticated API harvest"
                )
            arguments = [
                "--repository-id", key,
                "--repository-url", rule.url,
                "--created-after", format_time(freeze.eligible_created_after),
                "--merged-after", format_time(freeze.eligible_after),
                "--merged-before", format_time(freeze.eligible_before),
                "--per-repo", "36",
                "--output", f"/output/{temp.name}",
            ]
            argv = adapter_container_argv(
                docker_binary, adapter, "metadata", network="bridge",
                mounts=((io_dir, "/output", False),), arguments=arguments,
                environment_file=metadata_env_file,
            )
        else:
            metadata = layout.scratch / freeze.freeze_id / "metadata" / f"{key}.jsonl"
            if not metadata.exists():
                raise HarvestError(f"metadata stage output is missing for {key}")
            scratch = layout.worktrees / freeze.freeze_id / key
            scratch.mkdir(parents=True, exist_ok=True)
            mirror = layout.mirrors / rule.mirror
            arguments = [
                "--repository-id", key,
                "--mirror", "/input/mirror",
                "--metadata", "/input/metadata.jsonl",
                "--scratch", "/scratch",
                "--repetitions", "2",
                "--offline",
                "--runner-image-id", observed_image_id,
                "--output", f"/output/{temp.name}",
            ]
            argv = adapter_container_argv(
                docker_binary, adapter, "validation", network="none",
                mounts=(
                    (mirror, "/input/mirror", True),
                    (metadata, "/input/metadata.jsonl", True),
                    (scratch, "/scratch", False),
                    (io_dir, "/output", False),
                ),
                arguments=arguments,
            )
        proc = subprocess.run(
            argv, cwd=adapter_manifest_path.parent, capture_output=True, text=True,
            errors="replace", timeout=900 if stage == "metadata" else 7200,
        )
        if proc.returncode:
            raise HarvestError(
                f"{stage} adapter failed for {key} with {proc.returncode}: {proc.stderr[-600:]}"
            )
        if not temp.exists():
            raise HarvestError(f"{stage} adapter for {key} did not produce its ledger")
        # Parse before publish so a truncated adapter output never becomes input
        # to the next stage.
        rows = load_jsonl(temp)
        if stage == "metadata":
            def structurally_eligible(row: Mapping[str, Any]) -> bool:
                paths = row.get("authoritative_changed_paths")
                if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
                    return False
                ignored = {
                    path for path in paths
                    if is_vendor_or_generated(path)
                    or any(fnmatch.fnmatchcase(path, pattern) for pattern in rule.ignore_globs)
                }
                tests = {
                    path for path in paths if path not in ignored
                    and any(fnmatch.fnmatchcase(path, pattern) for pattern in rule.test_globs)
                }
                sources = set(paths) - ignored - tests
                return bool(
                    len(paths) <= 30 and 2 <= len(sources) <= 20 and tests
                    and len(ignored) * 2 < len(paths)
                )

            eligible = sum(structurally_eligible(row) for row in rows)
            if len(rows) < 36 or eligible < 18:
                raise HarvestError(
                    f"metadata adapter for {key} produced {len(rows)} candidates / "
                    f"{eligible} structurally eligible; requires at least 36 / 18"
                )
            stage_stats[key] = {"rows": len(rows), "structurally_eligible": eligible}
        else:
            stage_stats[key] = {"rows": len(rows)}
        os.replace(temp, output)
        io_dir.rmdir()
        outputs.append(output)
    combined = stage_root / "all.jsonl"
    with tempfile.NamedTemporaryFile("wb", dir=stage_root, delete=False) as handle:
        temporary = Path(handle.name)
        for output in outputs:
            data = output.read_bytes()
            handle.write(data)
            if data and not data.endswith(b"\n"):
                handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, combined)
    atomic_json(stage_root / "stage-manifest.json", {
        "schema_version": "greppy.agent-coding-v3.adapter-stage.1",
        "stage": stage,
        "freeze_id": freeze.freeze_id,
        "adapter_manifest_sha256": sha256(adapter_manifest_path.read_bytes()),
        "network": "bridge" if stage == "metadata" else "none",
        "images": stage_images,
        "outputs": {
            output.name: {
                "sha256": sha256(output.read_bytes()),
                **stage_stats[output.stem],
            } for output in outputs
        },
        "combined_sha256": sha256(combined.read_bytes()),
    })
    return combined


def harvest(
    *, registry_path: Path, freeze_path: Path, candidates_path: Path,
    id_key_path: Path, contract_path: Path, adapter_manifest_path: Path,
    denylist_paths: Sequence[Path], stage_manifest_paths: Sequence[Path],
    layout: StorageLayout,
) -> Path:
    rules, quotas, registry_bytes = load_registry(registry_path)
    freeze = load_freeze(freeze_path)
    contract, admission, contract_bytes = load_contract(contract_path, freeze)
    corpus = contract.get("corpus") or {}
    if (
        corpus.get("target_tasks") != quotas.target_tasks
        or corpus.get("repositories") != len(rules)
        or corpus.get("tasks_per_repository") != next(iter(rules.values())).quota
        or corpus.get("languages") != len(quotas.max_per_language)
    ):
        raise HarvestError("corpus contract and repository registry quotas differ")
    id_key = read_id_key(id_key_path)
    adapters = load_adapter_manifest(adapter_manifest_path)
    if set(adapters) != set(rules):
        raise HarvestError("ready adapter manifest does not cover the exact frozen registry")
    for key, rule in rules.items():
        if adapters[key].get("toolchain_profile") != rule.toolchain:
            raise HarvestError(f"adapter toolchain profile mismatch for {key}")
    stage_commitments = validate_stage_manifests(
        stage_manifest_paths, freeze, adapter_manifest_path, adapters, candidates_path
    )
    denied, denylist_commitments = load_denylists(denylist_paths)
    candidate_rows = load_jsonl(candidates_path)
    prepared: list[PreparedTask] = []
    verified: set[str] = set()
    for index, row in enumerate(candidate_rows, 1):
        key = row.get("repository")
        if key not in rules:
            raise HarvestError(f"candidate {index} references unregistered repository {key!r}")
        rule = rules[str(key)]
        mirror = layout.mirrors / rule.mirror
        if key not in verified:
            verify_mirror(mirror, rule)
            verified.add(str(key))
        try:
            prepared.append(prepare_candidate(row, rule, mirror, freeze, id_key, admission))
        except HarvestError as exc:
            raise HarvestError(f"candidate {index} ({key}): {exc}") from exc
    enforce_denylists(prepared, denied)
    require_candidate_pools(prepared, rules, admission.minimum_candidates_per_slot)
    near_findings = audit_near_duplicates(prepared)
    selected = select_tasks(prepared, rules, quotas)
    enforce_band_distribution(selected, admission)
    canonical_rows = b"".join(
        canonical_json(row) + b"\n"
        for row in sorted(candidate_rows, key=lambda value: canonical_json(value))
    )
    registry_document = json.loads(registry_bytes)
    adapter_bytes = adapter_manifest_path.read_bytes()
    freeze_bytes = freeze_path.read_bytes()
    commitments = {
        "corpus_contract_sha256": sha256(contract_bytes),
        "repository_registry_sha256": sha256(registry_bytes),
        "canonical_candidate_ledger_sha256": sha256(canonical_rows),
        "adapter_manifest_sha256": sha256(adapter_bytes),
        "freeze_manifest_sha256": sha256(freeze_bytes),
        "selection_secret_sha256": sha256(id_key),
        "selection_algorithm": "HMAC-SHA256(selection_secret, canonical_candidate_identity)",
        "opaque_id_algorithm": "first 128 bits of domain-separated HMAC-SHA256",
        "toolchain_profiles_sha256": sha256(canonical_json(registry_document["toolchain_profiles"])),
        "adapter_proof_sha256": {
            key: adapters[key]["proof_sha256"] for key in sorted(adapters)
        },
        "adapter_images": {
            key: {"image": adapters[key]["image"], "image_id": adapters[key]["image_id"]}
            for key in sorted(adapters)
        },
        "runner_image_digests": sorted({
            str(task.row["validation"]["runner_image_digest"]) for task in prepared
        }),
        "denylist_inputs": denylist_commitments,
        "adapter_stage_runs": stage_commitments,
    }
    return build_release(
        selected, freeze, layout, registry_bytes, sha256(id_key), commitments, near_findings
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--registry", type=Path, required=True)
    preflight_parser.add_argument("--adapter-manifest", type=Path)
    preflight_parser.add_argument("--storage-config", type=Path)
    for name in ("metadata", "validate"):
        stage_parser = subparsers.add_parser(name)
        stage_parser.add_argument("--registry", type=Path, required=True)
        stage_parser.add_argument("--freeze", type=Path, required=True)
        stage_parser.add_argument("--adapter-manifest", type=Path, required=True)
        stage_parser.add_argument("--storage-config", type=Path)
        if name == "metadata":
            stage_parser.add_argument(
                "--metadata-env-file", type=Path, required=True,
                help="Docker env-file containing the GitHub API credential; its contents are never logged",
            )
    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--registry", type=Path, required=True)
    seal_parser.add_argument("--freeze", type=Path, required=True)
    seal_parser.add_argument("--candidates", type=Path, required=True)
    seal_parser.add_argument("--id-key-file", type=Path, required=True)
    seal_parser.add_argument("--contract", type=Path, required=True)
    seal_parser.add_argument("--adapter-manifest", type=Path, required=True)
    seal_parser.add_argument(
        "--denylist", type=Path, action="append", required=True,
        help="Repeatable denylist; combined coverage must include swe-bench and prior-greppy",
    )
    seal_parser.add_argument(
        "--stage-manifest", type=Path, action="append", required=True,
        help="Repeat exactly twice for the metadata and validation Docker stage manifests",
    )
    seal_parser.add_argument("--storage-config", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        layout = load_storage(args.storage_config)
        if args.stage == "preflight":
            result = preflight(args.registry, layout, args.adapter_manifest)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["ok"] else 2
        if args.stage in ("metadata", "validate"):
            output = run_adapter_stage(
                stage=args.stage, registry_path=args.registry, freeze_path=args.freeze,
                adapter_manifest_path=args.adapter_manifest, layout=layout,
                metadata_env_file=getattr(args, "metadata_env_file", None),
            )
        else:
            output = harvest(
                registry_path=args.registry, freeze_path=args.freeze,
                candidates_path=args.candidates, id_key_path=args.id_key_file,
                contract_path=args.contract, adapter_manifest_path=args.adapter_manifest,
                denylist_paths=args.denylist, stage_manifest_paths=args.stage_manifest,
                layout=layout,
            )
    except (HarvestError, StorageError) as exc:
        print(f"v3 harvest failed: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
