"""Shared, fail-closed implementation for V3 repository adapters."""

from __future__ import annotations

import contextlib
import datetime as dt
import fnmatch
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from . import ADAPTER_VERSION


UTC = dt.timezone.utc
HEX_OID = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
PROFILES: dict[str, dict[str, Any]] = {
    "python-pip": {
        "tools": ["python3"],
        "f2p": ["python3", "-m", "pytest", "-q", "{test_paths}"],
        "p2p": ["python3", "-m", "pytest", "-q"],
        "env": {"PIP_NO_INDEX": "1"},
    },
    "rust-cargo": {
        "tools": ["cargo", "rustc"], "f2p": ["cargo", "test", "--workspace", "--offline"],
        "p2p": ["cargo", "test", "--workspace", "--offline"], "env": {"CARGO_NET_OFFLINE": "true"},
    },
    "go-test": {
        "tools": ["go"], "f2p": ["go", "test", "./..."], "p2p": ["go", "test", "./..."],
        "env": {"GOPROXY": "off", "GOSUMDB": "off"},
    },
    "java-maven": {
        "tools": ["java", "mvn"], "f2p": ["mvn", "-o", "test"], "p2p": ["mvn", "-o", "test"],
        "env": {"MAVEN_OPTS": "-Djava.net.useSystemProxies=false"},
    },
    "java-gradle": {
        "tools": ["java"], "f2p": ["./gradlew", "--offline", "test"],
        "p2p": ["./gradlew", "--offline", "test"], "env": {},
    },
    "ts-pnpm": {
        "tools": ["node", "pnpm"], "f2p": ["pnpm", "--offline", "test", "--", "{test_paths}"],
        "p2p": ["pnpm", "--offline", "test"], "env": {"CI": "1", "npm_config_offline": "true"},
    },
    "javascript-node": {
        "tools": ["node", "npm"], "f2p": ["npm", "test", "--", "{test_paths}"],
        "p2p": ["npm", "test"], "env": {"CI": "1", "npm_config_offline": "true"},
    },
    "cpp-cmake": {
        "tools": ["cmake", "ninja", "c++"],
        "f2p": ["ctest", "--test-dir", "build", "--output-on-failure"],
        "p2p": ["ctest", "--test-dir", "build", "--output-on-failure"],
        "env": {"CMAKE_BUILD_PARALLEL_LEVEL": "2"},
    },
    "ruby-bundler": {
        "tools": ["ruby", "bundle"],
        "f2p": ["bundle", "exec", "rspec", "{test_paths}"],
        "p2p": ["bundle", "exec", "rspec"],
        "env": {"BUNDLE_FROZEN": "true", "BUNDLE_DISABLE_VERSION_CHECK": "true"},
    },
}


class AdapterError(RuntimeError):
    """Adapter input or mechanical proof is invalid."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_config(path: pathlib.Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"cannot load adapter config: {exc}") from exc
    if not isinstance(config, dict) or config.get("schema_version") != "greppy.agent-coding-v3.adapter-config.1":
        raise AdapterError("unsupported adapter config")
    profile = config.get("toolchain_profile")
    if profile not in PROFILES:
        raise AdapterError(f"unsupported toolchain profile: {profile!r}")
    for field in ("repository_id", "repository_url", "primary_language"):
        if not isinstance(config.get(field), str) or not config[field]:
            raise AdapterError(f"config needs {field}")
    for field in ("test_globs", "source_extensions"):
        value = config.get(field)
        if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
            raise AdapterError(f"config needs non-empty {field}")
    ignored = config.get("ignore_globs", [])
    if not isinstance(ignored, list) or not all(isinstance(item, str) and item for item in ignored):
        raise AdapterError("config ignore_globs must be strings")
    for field in ("setup_commands", "post_patch_commands"):
        commands = config.get(field, [])
        if not isinstance(commands, list) or not all(
            isinstance(command, list) and command
            and all(isinstance(part, str) and part for part in command)
            for command in commands
        ):
            raise AdapterError(f"config {field} must be argv arrays")
    return config


def proof_sha256(config_path: pathlib.Path, config: Mapping[str, Any]) -> str:
    module_root = pathlib.Path(__file__).parent
    material = {
        "adapter_version": ADAPTER_VERSION,
        "config_sha256": sha256(config_path.read_bytes()),
        "implementation_sha256": {
            name: sha256((module_root / name).read_bytes())
            for name in ("__init__.py", "base.py", "cli.py")
        },
    }
    return sha256(canonical_json(material))


def parse_time(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise AdapterError("timestamp lacks timezone")
    return parsed.astimezone(UTC)


def format_time(value: dt.datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def git(repo: pathlib.Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    proc = subprocess.run(
        ["git", *args], cwd=repo, input=input_bytes, capture_output=True,
    )
    if proc.returncode:
        raise AdapterError(f"git {shlex.join(args)} failed: {proc.stderr.decode('utf-8', 'replace')[-500:]}")
    return proc.stdout


def git_text(repo: pathlib.Path, *args: str) -> str:
    return git(repo, *args).decode("utf-8", "replace").strip()


def tool_version(tool: str) -> str:
    executable = "/tools/greppy" if tool == "greppy" and pathlib.Path("/tools/greppy").exists() else shutil.which(tool)
    if not executable:
        raise AdapterError(f"required tool is missing: {tool}")
    candidates = ([executable, "--version"], [executable, "version"])
    for argv in candidates:
        proc = subprocess.run(argv, capture_output=True, text=True, errors="replace", timeout=20)
        if proc.returncode == 0 and (proc.stdout + proc.stderr).strip():
            return " | ".join((proc.stdout + proc.stderr).strip().splitlines())[:500]
    raise AdapterError(f"version probe failed: {tool}")


def preflight_payload(role: str, config_path: pathlib.Path, config: Mapping[str, Any]) -> dict[str, Any]:
    profile = PROFILES[str(config["toolchain_profile"])]
    proof = proof_sha256(config_path, config)
    tools: dict[str, str] = {}
    agent_tools: dict[str, str] = {}
    try:
        if role == "probe":
            tools = {name: tool_version(name) for name in profile["tools"]}
            agent_tools = {name: tool_version(name) for name in ("rg", "pi", "greppy")}
        else:
            # Metadata and validation preflights prove their local parser/runner
            # dependencies without performing network or claiming task success.
            json.loads(canonical_json({"role": role}))
        ready, reason = True, None
    except (AdapterError, OSError, subprocess.SubprocessError) as exc:
        ready, reason = False, str(exc)
    payload = {
        "ready": ready, "repository_id": config["repository_id"],
        "command_role": role, "proof_sha256": proof,
    }
    if tools:
        payload["tools"] = tools
        payload["agent_tools"] = agent_tools
    if reason:
        payload["reason"] = reason
    return payload


class GitHubClient:
    def __init__(self, token: str) -> None:
        if not token:
            raise AdapterError("GITHUB_TOKEN is required for authoritative metadata")
        self.token = token

    def _request(self, url: str, *, body: dict[str, Any] | None = None) -> Any:
        data = canonical_json(body) if body is not None else None
        request = urllib.request.Request(
            url, data=data,
            headers={
                "Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                "Content-Type": "application/json", "X-GitHub-Api-Version": "2022-11-28",
            }, method="POST" if body is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            raise AdapterError(f"GitHub API request failed: {exc}") from exc

    def rest(self, path: str) -> Any:
        return self._request("https://api.github.com" + path)

    def graphql(self, query: str, variables: Mapping[str, Any]) -> Any:
        payload = self._request("https://api.github.com/graphql", body={"query": query, "variables": dict(variables)})
        if not isinstance(payload, dict) or payload.get("errors"):
            raise AdapterError(f"GitHub GraphQL error: {(payload or {}).get('errors')}")
        return payload["data"]


def github_repo_parts(url: str) -> tuple[str, str]:
    match = re.fullmatch(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?", url)
    if not match:
        raise AdapterError("repository URL must be canonical GitHub HTTPS")
    return match.group(1), match.group(2)


PR_QUERY = """
query($owner:String!,$name:String!,$number:Int!){
 repository(owner:$owner,name:$name){pullRequest(number:$number){
  number createdAt mergedAt merged mergeCommit{oid parents(first:2){nodes{oid}}}
  commits(first:100){nodes{commit{oid}}} files(first:100){nodes{path}}
  closingIssuesReferences(first:10){nodes{number title body url labels(first:20){nodes{name}}}}
 }}
}"""


def classify_task(labels: Sequence[str], config: Mapping[str, Any]) -> str:
    mapping = config.get("label_task_classes", {})
    if isinstance(mapping, dict):
        for label in labels:
            if label.lower() in mapping:
                return str(mapping[label.lower()])
    lowered = " ".join(labels).lower()
    if "bug" in lowered:
        return "reported_bugfix"
    if any(marker in lowered for marker in ("migration", "deprecat", "api change", "config")):
        return "api_or_config_migration"
    if any(marker in lowered for marker in ("refactor", "cleanup", "tech debt")):
        return "behavior_preserving_refactor"
    if any(marker in lowered for marker in ("robust", "reliab", "security", "flaky", "test")):
        return "robustness_validation"
    if any(marker in lowered for marker in ("performance", "concurrency", "cross-cutting")):
        return "cross_cutting_behavior"
    if "feature" in lowered or "enhancement" in lowered:
        return "feature_implementation"
    return str(config.get("default_task_class", "reported_bugfix"))


def harvest_metadata(
    *, client: GitHubClient, repository_id: str, repository_url: str,
    merged_after: dt.datetime, merged_before: dt.datetime,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Harvest every merged PR in the window, including technical exclusions."""
    owner, name = github_repo_parts(repository_url)
    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        pulls = client.rest(
            f"/repos/{owner}/{name}/pulls?state=closed&sort=updated&direction=desc&per_page=100&page={page}"
        )
        if not isinstance(pulls, list):
            raise AdapterError("GitHub pulls response is not an array")
        if not pulls:
            break
        for summary in pulls:
            if not isinstance(summary, dict) or not summary.get("merged_at"):
                continue
            created = parse_time(str(summary["created_at"]))
            merged = parse_time(str(summary["merged_at"]))
            if not merged_after <= merged <= merged_before:
                continue
            number = int(summary["number"])
            data = client.graphql(PR_QUERY, {"owner": owner, "name": name, "number": number})
            pr = ((data.get("repository") or {}).get("pullRequest") or {})
            issues = ((pr.get("closingIssuesReferences") or {}).get("nodes") or [])
            merge = pr.get("mergeCommit") or {}
            parents = [node.get("oid") for node in ((merge.get("parents") or {}).get("nodes") or [])]
            paths = [node.get("path") for node in ((pr.get("files") or {}).get("nodes") or [])]
            commit_oids = [
                ((node.get("commit") or {}).get("oid"))
                for node in ((pr.get("commits") or {}).get("nodes") or [])
            ]
            issue = issues[0] if len(issues) == 1 and isinstance(issues[0], dict) else {}
            labels = [node.get("name", "") for node in ((issue.get("labels") or {}).get("nodes") or [])]
            strategy = (
                "merge" if len(parents) > 1
                else "rebase" if merge.get("oid") in commit_oids
                else "squash"
            )
            exclusion_reason = None
            if not pr.get("merged") or len(issues) != 1:
                exclusion_reason = "not_merged_or_linked_issue"
            elif not merge.get("oid") or not parents or strategy == "rebase":
                exclusion_reason = "unreconstructible_parent_or_merge"
            authoritative = {
                "pr": number, "created_at": pr.get("createdAt"), "merged_at": pr.get("mergedAt"),
                "solution": merge.get("oid"), "parents": parents, "paths": paths,
                "issue": {
                    "number": issue.get("number"), "title": issue.get("title"),
                    "body": issue.get("body"), "url": issue.get("url"),
                },
            }
            row = {
                "repository": repository_id, "repository_url": repository_url,
                "candidate_id": f"pull-request:{number}", "pr_number": number,
                "issue_number": issue.get("number"), "issue_url": issue.get("url"),
                "issue_title": issue.get("title") or "", "issue_body": issue.get("body") or "",
                "created_at": pr.get("createdAt") or summary.get("created_at"),
                "merged_at": pr.get("mergedAt") or summary.get("merged_at"),
                "solution_commit": str(merge.get("oid") or "").lower(),
                "parent_commit": str(parents[0] if parents else "").lower(),
                "authoritative_changed_paths": paths, "merge_strategy": strategy,
                "task_class": classify_task(labels, config),
                "authoritative_metadata_sha256": sha256(canonical_json(authoritative)),
            }
            if exclusion_reason:
                row["exclusion_reason"] = exclusion_reason
                row["exclusion_cause"] = "authoritative GitHub metadata"
            rows.append(row)
        if len(pulls) < 100:
            break
        page += 1
    return rows


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise AdapterError(f"{path}:{number}: expected object")
        rows.append(value)
    return rows


def atomic_jsonl(path: pathlib.Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        with contextlib.suppress(FileNotFoundError): os.unlink(name)


def matches_any(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def split_paths(paths: Sequence[str], config: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    ignored = config.get("ignore_globs", ["**/vendor/**", "**/generated/**", "**/node_modules/**", "**/*.lock"])
    tests = config.get("test_globs", [])
    kept = [path for path in paths if not matches_any(path, ignored)]
    test_paths = [path for path in kept if matches_any(path, tests)]
    return [path for path in kept if path not in set(test_paths)], test_paths


def expand_command(template: Sequence[str], test_paths: Sequence[str]) -> list[str]:
    result: list[str] = []
    for item in template:
        if item == "{test_paths}": result.extend(test_paths)
        elif item == "{first_test}": result.append(test_paths[0])
        else: result.append(item)
    return result


def commands_for(config: Mapping[str, Any], test_paths: Sequence[str]) -> tuple[list[str], list[str]]:
    profile = PROFILES[str(config["toolchain_profile"])]
    f2p = config.get("f2p_command", profile["f2p"])
    p2p = config.get("p2p_command", profile["p2p"])
    if not all(isinstance(value, list) and value and all(isinstance(part, str) for part in value) for value in (f2p, p2p)):
        raise AdapterError("test commands must be non-empty argv arrays")
    return expand_command(f2p, test_paths), expand_command(p2p, test_paths)


def offline_environment(config: Mapping[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(PROFILES[str(config["toolchain_profile"])]["env"])
    env.update({str(key): str(value) for key, value in (config.get("offline_env") or {}).items()})
    env.pop("GITHUB_TOKEN", None)
    return env


def run_logged(argv: Sequence[str], cwd: pathlib.Path, env: Mapping[str, str], timeout: int) -> dict[str, Any]:
    started = dt.datetime.now(UTC)
    try:
        proc = subprocess.run(argv, cwd=cwd, env=dict(env), capture_output=True, timeout=timeout)
        return {
            "argv": list(argv), "returncode": proc.returncode, "timed_out": False,
            "output_sha256": sha256(proc.stdout + proc.stderr), "started_at": format_time(started),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": list(argv), "returncode": None, "timed_out": True,
            "output_sha256": sha256((exc.stdout or b"") + (exc.stderr or b"")), "started_at": format_time(started),
        }


def command_list(value: Any, field: str) -> list[list[str]]:
    if not isinstance(value, list) or not all(
        isinstance(command, list) and command
        and all(isinstance(part, str) and part for part in command)
        for command in value
    ):
        raise AdapterError(f"{field} must be argv arrays")
    return value


@contextlib.contextmanager
def worktree(mirror: pathlib.Path, commit: str, parent: pathlib.Path, label: str) -> Iterator[pathlib.Path]:
    path = parent / label
    path.parent.mkdir(parents=True, exist_ok=True)
    git(mirror, "worktree", "add", "--detach", str(path), commit)
    try: yield path
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(path)], cwd=mirror, capture_output=True)
        shutil.rmtree(path, ignore_errors=True)


def apply_patch(workspace: pathlib.Path, patch: bytes) -> None:
    git(workspace, "apply", "--binary", "--check", "-", input_bytes=patch)
    git(workspace, "apply", "--binary", "-", input_bytes=patch)


def extract_patch(mirror: pathlib.Path, parent: str, solution: str, paths: Sequence[str]) -> bytes:
    return git(mirror, "diff", "--binary", "--full-index", "--no-renames", parent, solution, "--", *paths)


def repository_scale(mirror: pathlib.Path, parent: str, config: Mapping[str, Any]) -> dict[str, Any]:
    extensions = tuple(config.get("source_extensions", []))
    paths = git_text(mirror, "ls-tree", "-r", "--name-only", parent).splitlines()
    eligible = [path for path in paths if (not extensions or path.endswith(extensions)) and not matches_any(path, config.get("ignore_globs", []))]
    loc = sum(len(git(mirror, "show", f"{parent}:{path}").splitlines()) for path in eligible)
    tree = git_text(mirror, "rev-parse", f"{parent}^{{tree}}")
    band = "very_large" if loc >= 500_000 else "large" if loc >= 100_000 else "medium"
    material = {
        "measurement_revision": "v1", "parent_tree": tree,
        "eligible_source_files": len(eligible), "eligible_source_loc": loc, "size_band": band,
    }
    return {**material, "measurement_sha256": sha256(canonical_json(material))}


def validate_candidate(
    row: Mapping[str, Any], mirror: pathlib.Path, scratch: pathlib.Path,
    repetitions: int, config: Mapping[str, Any], runner_image_digest: str,
    adapter_proof_sha256: str | None = None,
) -> dict[str, Any]:
    solution = str(row.get("solution_commit", "")).lower()
    if not HEX_OID.fullmatch(solution): raise AdapterError("solution commit is invalid")
    resolved = git_text(mirror, "rev-parse", f"{solution}^{{commit}}")
    parent = git_text(mirror, "rev-parse", f"{solution}^1")
    if resolved != solution or parent != str(row.get("parent_commit", "")).lower():
        raise AdapterError("local M^1 provenance differs from authoritative metadata")
    paths = git_text(mirror, "diff", "--name-only", "--no-renames", parent, solution).splitlines()
    if paths != row.get("authoritative_changed_paths"):
        raise AdapterError("local changed paths differ from authoritative PR metadata")
    source_paths, test_paths = split_paths(paths, config)
    if not source_paths:
        raise AdapterError("candidate has no observable source or runtime-configuration fix")
    if not test_paths:
        raise AdapterError("candidate has no derivable independent behavior tests")
    test_patch = extract_patch(mirror, parent, solution, test_paths)
    gold_patch = extract_patch(mirror, parent, solution, source_paths)
    full_patch = extract_patch(mirror, parent, solution, paths)
    f2p, p2p = commands_for(config, test_paths)
    env = offline_environment(config)
    timeout = int(config.get("timeout_seconds", 1800))
    setup = command_list(config.get("setup_commands", []), "setup_commands")
    post_patch = command_list(config.get("post_patch_commands", []), "post_patch_commands")
    logs: list[dict[str, Any]] = []
    failure_mode = "test"
    for repetition in range(repetitions):
        with worktree(mirror, parent, scratch, f"rep-{repetition}-parent") as parent_tree:
            for command in setup:
                result = run_logged(command, parent_tree, env, timeout); logs.append(result)
                if result["returncode"] != 0: raise AdapterError("offline setup failed")
            baseline = run_logged(p2p, parent_tree, env, timeout); logs.append(baseline)
            if baseline["returncode"] != 0: raise AdapterError("parent PASS_TO_PASS baseline failed")
            apply_patch(parent_tree, test_patch)
            parent_build_failed = False
            for command in post_patch:
                result = run_logged(command, parent_tree, env, timeout); logs.append(result)
                if result["returncode"] != 0:
                    parent_build_failed = True
                    failure_mode = "build"
                    break
            if not parent_build_failed:
                failure = run_logged(f2p, parent_tree, env, timeout); logs.append(failure)
                if failure["returncode"] in (None, 0):
                    raise AdapterError("parent plus hidden tests did not fail")
        with worktree(mirror, parent, scratch, f"rep-{repetition}-gold") as gold_tree:
            for command in setup:
                result = run_logged(command, gold_tree, env, timeout); logs.append(result)
                if result["returncode"] != 0: raise AdapterError("offline setup failed")
            apply_patch(gold_tree, full_patch)
            for command in post_patch:
                result = run_logged(command, gold_tree, env, timeout); logs.append(result)
                if result["returncode"] != 0: raise AdapterError("offline post-patch setup failed")
            f2p_pass = run_logged(f2p, gold_tree, env, timeout); logs.append(f2p_pass)
            p2p_pass = run_logged(p2p, gold_tree, env, timeout); logs.append(p2p_pass)
            if f2p_pass["returncode"] != 0 or p2p_pass["returncode"] != 0:
                raise AdapterError("gold F2P/P2P proof failed")
    hashes = {
        "test_patch_sha256": sha256(test_patch), "gold_patch_sha256": sha256(gold_patch),
        "full_patch_sha256": sha256(full_patch),
    }
    metadata_hash = str(row.get("authoritative_metadata_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", metadata_hash): raise AdapterError("metadata hash is missing")
    return {
        **dict(row), "validation_outcome": "passed",
        "changed_source": source_paths, "changed_tests": test_paths,
        **({"adapter_proof_sha256": adapter_proof_sha256} if adapter_proof_sha256 else {}),
        "repository_scale": repository_scale(mirror, parent, config),
        "merge_provenance": {
            "target_parent_verified": True, "merged_result_tree_verified": True,
            "pr_delta_no_target_drift": True, "authoritative_metadata_sha256": metadata_hash,
        },
        "validation": {
            "parent_baseline": "pass", "parent_plus_test": "fail", "failure_mode": failure_mode,
            "gold_plus_test": "pass", "merged_plus_test": "pass",
            "clean_room_repetitions": repetitions, "offline": True,
            "fail_to_pass": test_paths, "pass_to_pass": [shlex.join(p2p)],
            "test_command": f2p, "fail_to_pass_commands": [f2p], "pass_to_pass_commands": [p2p],
            "setup_commands": setup, "post_patch_commands": post_patch, "timeout_seconds": timeout,
            **hashes, "runner_image_digest": runner_image_digest,
            "logs_sha256": sha256(canonical_json(logs)), "validated_at": format_time(dt.datetime.now(UTC)),
        },
    }


def validate_candidate_for_ledger(
    row: Mapping[str, Any], mirror: pathlib.Path, scratch: pathlib.Path,
    repetitions: int, config: Mapping[str, Any], runner_image_digest: str,
    adapter_proof_sha256: str | None = None,
) -> dict[str, Any]:
    """Return exactly one auditable row even when technical validation excludes it."""
    if row.get("exclusion_reason"):
        return {**dict(row), "validation_outcome": "not_run"}
    try:
        return validate_candidate(
            row, mirror, scratch, repetitions, config, runner_image_digest,
            adapter_proof_sha256,
        )
    except AdapterError as exc:
        message = str(exc)
        if "provenance" in message or "changed paths differ" in message or "solution commit" in message:
            reason = "unreconstructible_parent_or_merge"
        elif "no observable" in message:
            reason = "no_observable_code_or_config_fix"
        elif "no derivable" in message:
            reason = "no_derivable_independent_behavior_tests"
        elif "parent plus hidden tests" in message:
            reason = "parent_hidden_wrong_failure"
        elif "PASS_TO_PASS" in message or "gold F2P/P2P" in message:
            reason = "gold_or_pass_to_pass_not_green"
        else:
            reason = "registered_budget_inexecutable"
        if "greppy" in message.lower():
            raise AdapterError("technical exclusion causes may not be Greppy-specific") from exc
        return {
            **dict(row), "exclusion_reason": reason, "exclusion_cause": message,
            "validation_outcome": "failed",
        }
