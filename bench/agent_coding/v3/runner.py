#!/usr/bin/env python3
"""Leak-isolated execution and trusted grading for the sealed V3 taskbank."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import ipaddress
import json
import os
import pathlib
import random
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

try:
    from . import SCHEMA_VERSION
    from .storage import StorageError, StorageLayout, load_storage
except ImportError:  # direct script execution
    from __init__ import SCHEMA_VERSION
    from storage import StorageError, StorageLayout, load_storage


SEALED_SCHEMA = "greppy.agent-coding-sealed.v3"
ARMS = ("control", "treatment")
ARM_TOOLS = {arm: "bash,read,edit,write" for arm in ARMS}
PROVIDER_PRICE_USD_PER_MILLION = {
    "uncached_input_tokens": 0.30,
    "output_tokens": 1.20,
    "cache_read_tokens": 0.06,
    "cache_write_tokens": 0.00,
}
V3_COST_RATIO_MAX = 0.80
V3_CORRECTNESS_MARGIN = -0.05
V3_MIN_GREPPY_ADOPTION = 0.80
PREFLIGHT_REPORT_SCHEMA = "greppy.agent-coding-v3.gpu3-preflight.1"
PREFLIGHT_ATTESTATION_SCHEMA = "greppy.agent-coding-v3.gpu3-preflight-attestation.1"
SMOKE_EVIDENCE_SCHEMA = "greppy.agent-coding-v3.smoke-evidence.1"
GATE_CONTRACT = {
    "correctness_margin": V3_CORRECTNESS_MARGIN,
    "cost_ratio_max": V3_COST_RATIO_MAX,
    "greppy_adoption_min": V3_MIN_GREPPY_ADOPTION,
    "correctness_bootstrap_samples": 10_000,
    "cost_bootstrap_samples": 10_000,
    "transactionality_gate": False,
    "credential_isolation_required_for_full_run": True,
}
CREDENTIAL_BOUNDARY_STATUS = "unresolved_agent_process_scope"
SHARED_SYSTEM_PROMPT = (
    "You are a coding agent in an unfamiliar repository. Implement the user's task "
    "directly in the current worktree, keep the change focused, and do not commit, "
    "change revisions, inspect environment variables, or access paths outside the "
    "worktree. The normal shell includes rg; use the available read, edit, and write tools."
)
SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
OPAQUE_TASK_ID = re.compile(r"^task_[a-z2-7]{26}$")


class RunnerError(RuntimeError):
    """The sealed execution contract could not be maintained."""


@dataclasses.dataclass(frozen=True)
class NetworkAttestation:
    provider_proxy: str
    docker_network: str
    allowed_provider_hosts: tuple[str, ...]
    agent_probe_image_id: str
    proof_sha256: str
    sha256: str
    topology: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int | None
    stdout: bytes
    stderr: bytes
    wall_seconds: float
    timed_out: bool


@dataclasses.dataclass(frozen=True)
class AgentRequest:
    arm: str
    workspace: pathlib.Path
    store: pathlib.Path
    raw_dir: pathlib.Path
    system_prompt: str
    user_prompt: str
    tools: str
    environment: Mapping[str, str]
    timeout_seconds: int


@dataclasses.dataclass(frozen=True)
class AgentOutcome:
    returncode: int | None
    stdout: bytes = b""
    stderr: bytes = b""
    wall_seconds: float = 0.0
    timed_out: bool = False
    metrics: Mapping[str, Any] = dataclasses.field(default_factory=dict)


class AgentExecutor(Protocol):
    def __call__(self, request: AgentRequest) -> AgentOutcome: ...


class TrustedCommandExecutor(Protocol):
    def __call__(
        self, argv: Sequence[str], *, cwd: pathlib.Path, timeout_seconds: int,
    ) -> CommandResult: ...


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def compact_canonical_json(value: Any) -> bytes:
    """Canonical bytes used by signed/audited evidence (without a trailing newline)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def commands_from_agents_md(manual: str) -> tuple[frozenset[str], frozenset[str]]:
    commands: set[str] = set()
    edits: set[str] = set()
    section = ""
    for line in manual.splitlines():
        header = re.match(r"^([A-Z][^:]*):", line)
        if header:
            section = header.group(1)
            continue
        match = re.match(r"^  ([a-z][a-z0-9-]+)(?:\s|$)", line)
        if match:
            commands.add(match.group(1))
            if section == "EDIT":
                edits.add(match.group(1))
    if not commands or not edits:
        raise RunnerError("runtime AGENTS.md did not yield command vocabulary")
    return frozenset(commands), frozenset(edits)


def parse_pi_metrics(
    raw: bytes, shipped_commands: frozenset[str] = frozenset(),
    shipped_edits: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    totals = {
        "input_tokens": 0, "uncached_input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "cache_write_tokens": 0, "turns": 0,
    }
    turn_inputs: list[int] = []
    greppy_calls = greppy_edit_calls = 0
    source_open_by_kind = {"builtin_read": 0, "shell_reader": 0, "greppy_source": 0}
    for line in raw.decode("utf-8", "replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "turn_end":
            continue
        usage = ((event.get("message") or {}).get("usage") or {})
        uncached = int(usage.get("input", 0) or 0)
        cache_read = int(usage.get("cacheRead", 0) or 0)
        cache_write = sum(int(usage.get(key, 0) or 0) for key in ("cacheWrite", "cacheWrite1h", "cacheWrite5m"))
        input_tokens = uncached + cache_read + cache_write
        totals["turns"] += 1
        totals["uncached_input_tokens"] += uncached
        totals["input_tokens"] += input_tokens
        totals["output_tokens"] += int(usage.get("output", 0) or 0)
        totals["cache_read_tokens"] += cache_read
        totals["cache_write_tokens"] += cache_write
        turn_inputs.append(input_tokens)
        for item in (event.get("message") or {}).get("content") or []:
            if not isinstance(item, dict) or item.get("type") != "toolCall":
                continue
            if item.get("name") == "read":
                source_open_by_kind["builtin_read"] += 1
                continue
            if item.get("name") != "bash":
                continue
            command = str((item.get("arguments") or {}).get("command", ""))
            source_open_by_kind["shell_reader"] += len(re.findall(
                r"(?:^|[;&|]\s*)(?:\S*/)?(?:cat|head|tail|sed)(?:\s|$)", command,
            ))
            for match in re.finditer(r"(?:^|[;&|]\s*)(?:\S*/)?greppy\s+([^;&|\n]+)", command):
                tokens = shlex.split(match.group(1))
                verb = next((token for token in tokens if token in shipped_commands), None)
                if verb:
                    greppy_calls += 1
                    greppy_edit_calls += verb in shipped_edits
                if any(token in {"read", "read-smart", "read-file"} for token in tokens) or "--code" in tokens:
                    source_open_by_kind["greppy_source"] += 1
    totals["turn_input_tokens"] = turn_inputs
    totals["provider_cost_usd"] = round(sum(
        totals[field] * rate / 1_000_000
        for field, rate in PROVIDER_PRICE_USD_PER_MILLION.items()
    ), 9)
    totals["greppy_calls"] = greppy_calls
    totals["greppy_edit_calls"] = greppy_edit_calls
    totals["source_open_by_kind"] = source_open_by_kind
    totals["source_open_events"] = sum(source_open_by_kind.values())
    totals["transactionality_observation"] = "unobservable_without_per_tool_interception"
    return totals


def run_command(
    argv: Sequence[str], *, cwd: pathlib.Path, timeout_seconds: int,
    env: Mapping[str, str] | None = None, input_bytes: bytes | None = None,
) -> CommandResult:
    started = time.monotonic()
    process = subprocess.Popen(
        list(argv), cwd=cwd, env=dict(env) if env is not None else None,
        stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(input=input_bytes, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, 9)
        stdout, stderr = process.communicate()
    return CommandResult(
        tuple(str(part) for part in argv), None if timed_out else process.returncode,
        stdout, stderr, time.monotonic() - started, timed_out,
    )


def checked(
    argv: Sequence[str], *, cwd: pathlib.Path, timeout_seconds: int,
    env: Mapping[str, str] | None = None, input_bytes: bytes | None = None,
) -> CommandResult:
    result = run_command(argv, cwd=cwd, timeout_seconds=timeout_seconds, env=env, input_bytes=input_bytes)
    if result.timed_out or result.returncode != 0:
        raise RunnerError(
            f"command failed: {shlex.join(result.argv)} "
            f"timeout={result.timed_out} rc={result.returncode}"
        )
    return result


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunnerError(f"{path} must contain an object")
    return value


def load_network_attestation(
    path: pathlib.Path, provider_proxy: str, docker_network: str,
) -> NetworkAttestation:
    raw = path.read_bytes()
    document = _load_json(path)
    if document.get("schema_version") != "greppy.provider-only-egress.v1":
        raise RunnerError("unsupported network isolation attestation")
    parsed = urllib.parse.urlsplit(provider_proxy)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RunnerError("provider proxy must be an explicit HTTP(S) endpoint")
    try:
        ipaddress.ip_address(parsed.hostname)
    except ValueError as exc:
        raise RunnerError("provider proxy host must be a preregistered internal IP; DNS is disabled") from exc
    if document.get("provider_proxy") != provider_proxy:
        raise RunnerError("provider proxy differs from network attestation")
    if document.get("docker_network") != docker_network:
        raise RunnerError("Docker network differs from network attestation")
    hosts = document.get("allowed_provider_hosts")
    if not isinstance(hosts, list) or not hosts or not all(isinstance(host, str) and host for host in hosts):
        raise RunnerError("attestation requires provider host allowlist")
    shell_probe = document.get("shell_public_egress_probe") or {}
    provider_probe = document.get("provider_connectivity_probe") or {}
    audit_evidence = document.get("audit_evidence") or {}
    probe_image_id = audit_evidence.get("agent_probe_image_id")
    proof_sha256 = audit_evidence.get("proof_sha256")
    proof_payload = dict(audit_evidence)
    proof_payload.pop("proof_sha256", None)
    computed_proof = sha256(compact_canonical_json(proof_payload))
    topology = audit_evidence.get("topology")
    if not (
        document.get("enforcement") == "docker-internal-network-plus-allowlist-proxy"
        and shell_probe.get("passed") is True
        and shell_probe.get("direct_public_egress_denied") is True
        and provider_probe.get("passed") is True
        and provider_probe.get("through_allowlist_proxy") is True
        and isinstance(probe_image_id, str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", probe_image_id)
        and isinstance(proof_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", proof_sha256)
        and proof_sha256 == computed_proof
        and isinstance(topology, dict)
        and isinstance(topology.get("internal_network"), dict)
        and isinstance(topology.get("egress_network"), dict)
        and isinstance(topology.get("proxy"), dict)
    ):
        raise RunnerError("network attestation does not prove denied shell egress and provider connectivity")
    return NetworkAttestation(
        provider_proxy, docker_network, tuple(hosts), probe_image_id, proof_sha256, sha256(raw), topology
    )


def verify_detached_signature(
    document: pathlib.Path, signature: pathlib.Path, public_key: pathlib.Path,
    openssl: pathlib.Path,
) -> None:
    for path in (document, signature, public_key, openssl):
        if not path.is_file():
            raise RunnerError(f"signed evidence input is missing: {path}")
    result = run_command(
        [
            str(openssl), "dgst", "-sha256", "-verify", str(public_key),
            "-signature", str(signature), str(document),
        ],
        cwd=pathlib.Path.cwd(), timeout_seconds=30,
    )
    if result.timed_out or result.returncode != 0:
        raise RunnerError("detached operations signature did not verify")


def load_signed_preflight(
    *, report_path: pathlib.Path, attestation_path: pathlib.Path,
    signature_path: pathlib.Path, public_key: pathlib.Path, openssl: pathlib.Path,
    runtime_bindings: Mapping[str, Any], network: NetworkAttestation,
) -> dict[str, str]:
    verify_detached_signature(attestation_path, signature_path, public_key, openssl)
    report = _load_json(report_path)
    attestation = _load_json(attestation_path)
    report_hash = sha256(report_path.read_bytes())
    network_check = ((report.get("checks") or {}).get("network") or {})
    network_proof = ((network_check.get("audit_evidence") or {}).get("proof_sha256"))
    if not (
        report.get("schema_version") == PREFLIGHT_REPORT_SCHEMA
        and report.get("ready") is True
        and not report.get("failures")
        and network_check.get("ready") is True
        and network_proof == network.proof_sha256
    ):
        raise RunnerError("gpu3 preflight is not ready or does not bind the live network audit")
    if not (
        attestation.get("schema_version") == PREFLIGHT_ATTESTATION_SCHEMA
        and attestation.get("ready") is True
        and attestation.get("preflight_report_sha256") == report_hash
        and attestation.get("runtime_bindings") == dict(runtime_bindings)
    ):
        raise RunnerError("signed gpu3 preflight does not match the current benchmark runtime")
    return {
        "report_sha256": report_hash,
        "attestation_sha256": sha256(attestation_path.read_bytes()),
        "signature_sha256": sha256(signature_path.read_bytes()),
    }


def load_signed_smoke_evidence(
    *, evidence_path: pathlib.Path, signature_path: pathlib.Path,
    public_key: pathlib.Path, openssl: pathlib.Path,
    runtime_bindings: Mapping[str, Any], preflight_attestation_sha256: str,
) -> dict[str, str]:
    verify_detached_signature(evidence_path, signature_path, public_key, openssl)
    evidence = _load_json(evidence_path)
    task_ids = evidence.get("task_ids")
    trace_hashes = evidence.get("arm_trace_sha256")
    smoke_archive_hash = evidence.get("smoke_run_archive_sha256")
    if not (
        evidence.get("schema_version") == SMOKE_EVIDENCE_SCHEMA
        and evidence.get("ready") is True
        and evidence.get("paired_trajectory_count") == 3
        and evidence.get("arm_trace_count") == 6
        and isinstance(task_ids, list)
        and all(isinstance(task_id, str) for task_id in task_ids)
        and len(task_ids) == len(set(task_ids)) == 3
        and isinstance(trace_hashes, list)
        and len(trace_hashes) == 6
        and all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in trace_hashes)
        and isinstance(smoke_archive_hash, str)
        and re.fullmatch(r"[0-9a-f]{64}", smoke_archive_hash)
        and (evidence.get("manual_review") or {}).get("passed") is True
        and (evidence.get("manual_review") or {}).get("read_all_six_arm_traces") is True
        and not (evidence.get("manual_review") or {}).get("open_findings")
        and evidence.get("runtime_bindings") == dict(runtime_bindings)
        and evidence.get("preflight_attestation_sha256") == preflight_attestation_sha256
    ):
        raise RunnerError("signed smoke evidence must contain exactly three reviewed paired trajectories for this runtime")
    return {
        "evidence_sha256": sha256(evidence_path.read_bytes()),
        "signature_sha256": sha256(signature_path.read_bytes()),
    }


def enforce_credential_boundary(full_release_population: bool) -> None:
    if full_release_population:
        raise RunnerError(
            "full run blocked: Pi and agent shell children can read the provider credential; "
            "a separately attestable broker is required for cost-valid release evidence"
        )


def _safe_relative(value: Any, field: str) -> pathlib.PurePosixPath:
    if not isinstance(value, str) or not value:
        raise RunnerError(f"{field} must be a non-empty relative path")
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
        raise RunnerError(f"{field} is unsafe")
    return path


def _validated_argv_commands(value: Any, field: str, *, allow_empty: bool = True) -> list[list[str]]:
    if not isinstance(value, list) or (not allow_empty and not value) or not all(
        isinstance(command, list) and command
        and all(isinstance(part, str) and part for part in command)
        for command in value
    ):
        qualifier = "" if allow_empty else " non-empty"
        raise RunnerError(f"{field} must be{qualifier} argv arrays")
    return [list(command) for command in value]


def load_release(public_dir: pathlib.Path, sealed_dir: pathlib.Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    public = _load_json(public_dir / "taskbank.json")
    sealed = _load_json(sealed_dir / "manifest.json")
    if public.get("schema_version") != SCHEMA_VERSION:
        raise RunnerError("unsupported public taskbank schema")
    if sealed.get("schema_version") != SEALED_SCHEMA:
        raise RunnerError("unsupported sealed manifest schema")
    public_freeze = (public.get("freeze") or {}).get("id")
    if not isinstance(public_freeze, str) or sealed.get("freeze_id") != public_freeze:
        raise RunnerError("public and sealed freeze identities differ")
    public_tasks = public.get("tasks")
    sealed_tasks = sealed.get("tasks")
    if not isinstance(public_tasks, list) or not isinstance(sealed_tasks, list):
        raise RunnerError("task manifests need task arrays")
    public_by_id = {task.get("id"): task for task in public_tasks if isinstance(task, dict)}
    sealed_by_id = {task.get("id"): task for task in sealed_tasks if isinstance(task, dict)}
    if len(public_by_id) != len(public_tasks) or len(sealed_by_id) != len(sealed_tasks):
        raise RunnerError("task IDs must be unique strings")
    if not all(isinstance(task_id, str) and OPAQUE_TASK_ID.fullmatch(task_id) for task_id in public_by_id):
        raise RunnerError("task IDs must be opaque and non-derived")
    if set(public_by_id) != set(sealed_by_id):
        raise RunnerError("public and sealed task IDs differ")
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for task_id in sorted(public_by_id):
        public_task = public_by_id[task_id]
        sealed_task = sealed_by_id[task_id]
        if not isinstance(public_task.get("user_task"), str) or not public_task["user_task"].strip():
            raise RunnerError("public user_task must be non-empty")
        workspace = public_task.get("workspace") or {}
        snapshot_rel = _safe_relative(workspace.get("snapshot"), "workspace.snapshot")
        snapshot = public_dir.joinpath(*snapshot_rel.parts)
        expected_snapshot_hash = workspace.get("snapshot_sha256")
        if not snapshot.is_file() or sha256(snapshot.read_bytes()) != expected_snapshot_hash:
            raise RunnerError("parent snapshot hash mismatch")
        artifacts = sealed_task.get("artifacts") or {}
        hashes = sealed_task.get("hashes") or {}
        test_rel = _safe_relative(artifacts.get("test_patch"), "artifacts.test_patch")
        test_patch = sealed_dir.joinpath(*test_rel.parts)
        if not test_patch.is_file() or sha256(test_patch.read_bytes()) != hashes.get("test_patch_sha256"):
            raise RunnerError("sealed test patch hash mismatch")
        evaluation = sealed_task.get("evaluation")
        if not isinstance(evaluation, dict):
            raise RunnerError("sealed task lacks evaluation spec")
        test_command = evaluation.get("test_command")
        if not isinstance(test_command, list) or not test_command or not all(
            isinstance(part, str) and part for part in test_command
        ):
            raise RunnerError("evaluation.test_command must be a non-empty argv array")
        _validated_argv_commands(evaluation.get("setup_commands"), "evaluation.setup_commands")
        _validated_argv_commands(
            evaluation.get("post_patch_commands"), "evaluation.post_patch_commands"
        )
        for group in ("fail_to_pass", "pass_to_pass"):
            if f"{group}_commands" in evaluation:
                _validated_argv_commands(
                    evaluation[f"{group}_commands"], f"evaluation.{group}_commands",
                    allow_empty=False,
                )
        expected_evaluation_hash = hashes.get("evaluation_sha256")
        if not isinstance(expected_evaluation_hash, str) or (
            sha256(compact_canonical_json(evaluation)) != expected_evaluation_hash
        ):
            raise RunnerError("sealed evaluation spec hash mismatch")
        pairs.append((public_task, sealed_task))
    return pairs


def _validate_tar_member(member: tarfile.TarInfo) -> None:
    name = pathlib.PurePosixPath(member.name)
    if name.is_absolute() or ".." in name.parts or ".git" in name.parts:
        raise RunnerError(f"unsafe snapshot member: {member.name}")
    if member.isdev() or member.isfifo() or member.islnk():
        raise RunnerError(f"unsupported snapshot member: {member.name}")
    if member.issym():
        target = pathlib.PurePosixPath(member.linkname)
        combined = pathlib.PurePosixPath(os.path.normpath(str(name.parent / target)))
        if target.is_absolute() or ".." in combined.parts:
            raise RunnerError(f"unsafe snapshot symlink: {member.name}")


def import_parent_snapshot(snapshot: pathlib.Path, workspace: pathlib.Path, timeout_seconds: int) -> str:
    if workspace.exists():
        raise RunnerError(f"workspace already exists: {workspace}")
    workspace.mkdir(parents=True)
    try:
        with tarfile.open(snapshot, "r:*") as archive:
            members = archive.getmembers()
            for member in members:
                _validate_tar_member(member)
            archive.extractall(workspace, members=members, filter="data")
    except (tarfile.TarError, OSError) as exc:
        raise RunnerError(f"cannot extract parent snapshot: {exc}") from exc
    checked(["git", "init", "--quiet"], cwd=workspace, timeout_seconds=timeout_seconds)
    checked(["git", "config", "user.name", "Greppy Benchmark"], cwd=workspace, timeout_seconds=timeout_seconds)
    checked(["git", "config", "user.email", "benchmark@example.invalid"], cwd=workspace, timeout_seconds=timeout_seconds)
    checked(["git", "add", "-A"], cwd=workspace, timeout_seconds=timeout_seconds)
    commit_env = dict(os.environ)
    commit_env.update({
        "GIT_AUTHOR_DATE": "2000-01-01T00:00:00Z",
        "GIT_COMMITTER_DATE": "2000-01-01T00:00:00Z",
    })
    checked(
        ["git", "commit", "--quiet", "-m", "benchmark parent snapshot"],
        cwd=workspace, timeout_seconds=timeout_seconds, env=commit_env,
    )
    assert_single_commit_workspace(workspace, timeout_seconds)
    return checked(["git", "rev-parse", "HEAD"], cwd=workspace, timeout_seconds=timeout_seconds).stdout.decode().strip()


def assert_single_commit_workspace(workspace: pathlib.Path, timeout_seconds: int) -> None:
    history = checked(
        ["git", "log", "--all", "--oneline"], cwd=workspace, timeout_seconds=timeout_seconds
    ).stdout.decode("utf-8", "replace").splitlines()
    remotes = checked(["git", "remote"], cwd=workspace, timeout_seconds=timeout_seconds).stdout.strip()
    if len(history) != 1 or remotes:
        raise RunnerError("agent workspace must expose exactly one commit and no remotes")


def apply_patch(workspace: pathlib.Path, patch: bytes, timeout_seconds: int, label: str) -> None:
    if not patch.strip():
        raise RunnerError(f"{label} is empty")
    checked(
        ["git", "apply", "--binary", "--check", "-"], cwd=workspace,
        timeout_seconds=timeout_seconds, input_bytes=patch,
    )
    checked(
        ["git", "apply", "--binary", "-"], cwd=workspace,
        timeout_seconds=timeout_seconds, input_bytes=patch,
    )


def capture_agent_diff(workspace: pathlib.Path, timeout_seconds: int) -> bytes:
    assert_single_commit_workspace(workspace, timeout_seconds)
    checked(["git", "add", "-A"], cwd=workspace, timeout_seconds=timeout_seconds)
    return checked(
        ["git", "diff", "--cached", "--binary", "--full-index", "HEAD"],
        cwd=workspace, timeout_seconds=timeout_seconds,
    ).stdout


def _clean_agent_environment(store: pathlib.Path, config: pathlib.Path) -> dict[str, str]:
    keep = {
        "PATH", "LANG", "LC_ALL", "LC_CTYPE", "SSL_CERT_FILE", "SSL_CERT_DIR",
    }
    env = {key: value for key, value in os.environ.items() if key in keep}
    env.update({
        "HOME": "/tmp/agent-home",
        "GREPPY_STORE_DIR": "/greppy-store",
        "PI_CODING_AGENT_DIR": "/pi-config",
    })
    return env


def system_prompt(arm: str, agents_md: str) -> str:
    if arm == "control":
        return SHARED_SYSTEM_PROMPT
    if arm == "treatment":
        return SHARED_SYSTEM_PROMPT + "\n\n" + agents_md
    raise RunnerError(f"unknown arm: {arm}")


def balanced_arm_orders(task_ids: Sequence[str]) -> dict[str, list[str]]:
    return {
        task_id: (["control", "treatment"] if index % 2 == 0 else ["treatment", "control"])
        for index, task_id in enumerate(sorted(task_ids))
    }


def _run_setup(
    commands: Any, workspace: pathlib.Path, timeout_seconds: int,
    command_executor: TrustedCommandExecutor = run_command,
) -> None:
    for command in _validated_argv_commands(commands, "setup_commands"):
        result = command_executor(command, cwd=workspace, timeout_seconds=timeout_seconds)
        if result.timed_out or result.returncode != 0:
            raise RunnerError("containerized setup command failed")
    status = checked(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=workspace, timeout_seconds=timeout_seconds,
    ).stdout
    if status.strip():
        raise RunnerError("setup modified tracked parent sources")


def _run_post_patch_commands(
    commands: Any, workspace: pathlib.Path, timeout_seconds: int,
    command_executor: TrustedCommandExecutor = run_command,
) -> None:
    for command in _validated_argv_commands(commands, "post_patch_commands"):
        result = command_executor(command, cwd=workspace, timeout_seconds=timeout_seconds)
        if result.timed_out or result.returncode != 0:
            raise RunnerError("containerized post-patch command failed")


def _group_commands(sealed_task: dict[str, Any], group: str) -> list[list[str]]:
    evaluation = sealed_task.get("evaluation") or {}
    explicit = evaluation.get(f"{group}_commands")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise RunnerError(f"evaluation.{group}_commands must be non-empty")
        commands = explicit
    else:
        base = evaluation.get("test_command")
        selectors = (sealed_task.get("validation_evidence") or {}).get(group)
        if not isinstance(base, list) or not base or not all(isinstance(part, str) for part in base):
            raise RunnerError("evaluation.test_command must be a non-empty argv array")
        if not isinstance(selectors, list) or not selectors or not all(isinstance(item, str) for item in selectors):
            raise RunnerError(f"validation_evidence.{group} must be non-empty")
        commands = [[*base, selector] for selector in selectors]
    if not all(
        isinstance(command, list) and command and all(isinstance(part, str) for part in command)
        for command in commands
    ):
        raise RunnerError(f"invalid {group} grader commands")
    return [list(command) for command in commands]


def _run_grade_group(
    commands: Sequence[Sequence[str]], workspace: pathlib.Path, timeout_seconds: int,
    command_executor: TrustedCommandExecutor = run_command,
) -> dict[str, Any]:
    results = [command_executor(command, cwd=workspace, timeout_seconds=timeout_seconds) for command in commands]
    return {
        "passed": all(not result.timed_out and result.returncode == 0 for result in results),
        "commands": [
            {
                "argv": list(result.argv), "returncode": result.returncode,
                "timed_out": result.timed_out, "wall_seconds": round(result.wall_seconds, 3),
                "output_sha256": sha256(result.stdout + result.stderr),
            }
            for result in results
        ],
    }


def grade_agent_diff(
    *, snapshot: pathlib.Path, sealed_dir: pathlib.Path, sealed_task: dict[str, Any],
    agent_diff: bytes, workspace: pathlib.Path,
    command_executor: TrustedCommandExecutor = run_command,
) -> dict[str, Any]:
    evaluation = sealed_task.get("evaluation") or {}
    timeout_seconds = int(evaluation.get("timeout_seconds", 1800))
    import_parent_snapshot(snapshot, workspace, timeout_seconds)
    _run_setup(evaluation.get("setup_commands", []), workspace, timeout_seconds, command_executor)
    try:
        if agent_diff.strip():
            apply_patch(workspace, agent_diff, timeout_seconds, "agent diff")
    except RunnerError as exc:
        return {
            "correctness": False, "agent_diff_applied_before_test_patch": False,
            "test_patch_applied_after_agent": False, "patch_application_error": str(exc),
            "post_patch_commands_executed": False,
            "fail_to_pass": None, "pass_to_pass": None,
        }
    test_rel = _safe_relative((sealed_task.get("artifacts") or {}).get("test_patch"), "artifacts.test_patch")
    test_patch = sealed_dir.joinpath(*test_rel.parts).read_bytes()
    try:
        apply_patch(workspace, test_patch, timeout_seconds, "sealed test patch")
    except RunnerError as exc:
        return {
            "correctness": False, "agent_diff_applied_before_test_patch": True,
            "test_patch_applied_after_agent": False, "patch_application_error": str(exc),
            "post_patch_commands_executed": False,
            "fail_to_pass": None, "pass_to_pass": None,
        }
    _run_post_patch_commands(
        evaluation.get("post_patch_commands"), workspace, timeout_seconds, command_executor
    )
    fail_to_pass = _run_grade_group(
        _group_commands(sealed_task, "fail_to_pass"), workspace, timeout_seconds, command_executor
    )
    pass_to_pass = _run_grade_group(
        _group_commands(sealed_task, "pass_to_pass"), workspace, timeout_seconds, command_executor
    )
    return {
        "correctness": fail_to_pass["passed"] and pass_to_pass["passed"],
        "agent_diff_applied_before_test_patch": True,
        "test_patch_applied_after_agent": True,
        "post_patch_commands_executed": True,
        "fail_to_pass": fail_to_pass,
        "pass_to_pass": pass_to_pass,
    }


def execute_pair(
    *, public_task: dict[str, Any], sealed_task: dict[str, Any], public_dir: pathlib.Path,
    sealed_dir: pathlib.Path, slot_dir: pathlib.Path, agents_md: str,
    executor: AgentExecutor, arm_order: Sequence[str] = ARMS,
    command_executor: TrustedCommandExecutor = run_command,
    forbidden_secrets: Sequence[bytes] = (),
) -> list[dict[str, Any]]:
    snapshot_rel = _safe_relative((public_task.get("workspace") or {}).get("snapshot"), "workspace.snapshot")
    snapshot = public_dir.joinpath(*snapshot_rel.parts)
    timeout_seconds = int((sealed_task.get("evaluation") or {}).get("timeout_seconds", 1800))
    rows: list[dict[str, Any]] = []
    if set(arm_order) != set(ARMS) or len(arm_order) != len(ARMS):
        raise RunnerError("arm order must contain control and treatment exactly once")
    for arm in arm_order:
        try:
            rows.append(_execute_arm(
                arm=arm, public_task=public_task, sealed_task=sealed_task,
                sealed_dir=sealed_dir, snapshot=snapshot, slot_dir=slot_dir,
                agents_md=agents_md, executor=executor,
                command_executor=command_executor, forbidden_secrets=forbidden_secrets,
                timeout_seconds=timeout_seconds,
            ))
        except Exception as exc:
            rows.append({
                "arm": arm, "valid": False,
                "agent": {
                    "metrics": {"provider_cost_usd": None}, "timed_out": False,
                    "wall_seconds": 0.0,
                },
                "agent_diff_sha256": None, "agent_diff_bytes": 0,
                "grading": {"correctness": False, "harness_error": exc.__class__.__name__},
            })
    return rows


def _execute_arm(
    *, arm: str, public_task: dict[str, Any], sealed_task: dict[str, Any],
    sealed_dir: pathlib.Path, snapshot: pathlib.Path, slot_dir: pathlib.Path,
    agents_md: str, executor: AgentExecutor, command_executor: TrustedCommandExecutor,
    forbidden_secrets: Sequence[bytes], timeout_seconds: int,
) -> dict[str, Any]:
    arm_dir = slot_dir / arm
    agent_workspace = arm_dir / "agent-workspace"
    grading_workspace = arm_dir / "grading-workspace"
    store, raw_dir, config = arm_dir / "store", arm_dir / "raw", arm_dir / "pi-config"
    for directory in (store, raw_dir, config):
        directory.mkdir(parents=True, exist_ok=True)
    import_parent_snapshot(snapshot, agent_workspace, timeout_seconds)
    _run_setup(
        (sealed_task.get("evaluation") or {}).get("setup_commands", []),
        agent_workspace, timeout_seconds, command_executor,
    )
    request = AgentRequest(
        arm=arm, workspace=agent_workspace, store=store, raw_dir=raw_dir,
        system_prompt=system_prompt(arm, agents_md), user_prompt=public_task["user_task"].strip(),
        tools=ARM_TOOLS[arm], environment=_clean_agent_environment(store, config),
        timeout_seconds=timeout_seconds,
    )
    outcome = executor(request)
    stdout, stderr = outcome.stdout, outcome.stderr
    for secret in forbidden_secrets:
        if secret:
            stdout = stdout.replace(secret, b"<redacted>")
            stderr = stderr.replace(secret, b"<redacted>")
    (raw_dir / "agent.stdout").write_bytes(stdout)
    (raw_dir / "agent.stderr").write_bytes(stderr)
    agent_diff = capture_agent_diff(agent_workspace, timeout_seconds)
    if any(secret and secret in agent_diff for secret in forbidden_secrets):
        raise RunnerError("provider credential appeared in agent diff")
    (raw_dir / "agent.patch").write_bytes(agent_diff)
    grade = grade_agent_diff(
        snapshot=snapshot, sealed_dir=sealed_dir, sealed_task=sealed_task,
        agent_diff=agent_diff, workspace=grading_workspace,
        command_executor=command_executor,
    )
    return {
        "arm": arm, "valid": outcome.returncode == 0 and not outcome.timed_out,
        "agent": {
            "returncode": outcome.returncode, "timed_out": outcome.timed_out,
            "wall_seconds": round(outcome.wall_seconds, 3),
            "stdout_sha256": sha256(stdout), "stderr_sha256": sha256(stderr),
            "metrics": dict(outcome.metrics),
        },
        "agent_diff_sha256": sha256(agent_diff), "agent_diff_bytes": len(agent_diff),
        "grading": grade,
    }


class DockerPiExecutor:
    """Run Pi on an internal network whose only egress is an allowlist proxy."""

    def __init__(
        self, *, docker: pathlib.Path, image: str, network: NetworkAttestation,
        pi_command: str, greppy_bin: pathlib.Path, provider_extension: pathlib.Path,
        provider_key_file: pathlib.Path, provider: str, model: str,
        shipped_commands: frozenset[str], shipped_edits: frozenset[str],
        readonly_roots: Sequence[pathlib.Path] = (),
    ) -> None:
        self.docker = docker
        self.image = image
        self.network = network
        self.pi_command = pi_command
        self.greppy_bin = greppy_bin.resolve()
        self.provider_extension = provider_extension.resolve()
        self.provider_key_file = provider_key_file.resolve()
        self.provider_key = self.provider_key_file.read_bytes().strip()
        if len(self.provider_key) < 16:
            raise RunnerError("ephemeral provider key is missing or too short")
        self.provider = provider
        self.model = model
        self.shipped_commands = shipped_commands
        self.shipped_edits = shipped_edits
        self.readonly_roots = tuple(path.resolve() for path in readonly_roots)

    def __call__(self, request: AgentRequest) -> AgentOutcome:
        argv = [
            str(self.docker), "run", "--rm", "--init", "--read-only",
            "--network", self.network.docker_network,
            "--dns", "127.0.0.1",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "512", "--user", f"{os.getuid()}:{os.getgid()}",
            "--workdir", "/workspace", "--tmpfs", "/tmp:rw,nosuid,nodev",
            "--mount", f"type=bind,src={request.workspace},dst=/workspace",
            "--mount", f"type=bind,src={request.store},dst=/greppy-store",
            "--mount", f"type=bind,src={self.provider_extension},dst=/provider/provider.js,readonly",
            "--mount", f"type=bind,src={self.provider_key_file},dst=/run/secrets/minimax_api_key,readonly",
        ]
        if request.arm == "treatment":
            argv.extend(["--mount", f"type=bind,src={self.greppy_bin},dst=/tools/greppy,readonly"])
        for index, path in enumerate(self.readonly_roots):
            argv.extend(["--mount", f"type=bind,src={path},dst=/dependencies/{index},readonly"])
        environment = dict(request.environment)
        environment.update({
            "HOME": "/tmp/agent-home",
            "PATH": "/tools:/usr/local/bin:/usr/bin:/bin",
            "GREPPY_STORE_DIR": "/greppy-store",
            "PI_CODING_AGENT_DIR": "/tmp/pi-config",
            "HTTP_PROXY": self.network.provider_proxy,
            "HTTPS_PROXY": self.network.provider_proxy,
            "NO_PROXY": "",
        })
        for key, value in environment.items():
            argv.extend(["--env", f"{key}={value}"])
        pi_argv = [
            self.pi_command, "-p", "--extension", "/provider/provider.js",
            "--provider", self.provider, "--model", self.model, "--mode", "json",
            "--no-session", "--tools", request.tools, "--no-context-files", "--no-skills",
            "--no-prompt-templates", "--no-extensions", "--approve",
            "--append-system-prompt", request.system_prompt, request.user_prompt,
        ]
        result = run_command(
            [
                *argv, self.image, "/bin/sh", "-c",
                'MINIMAX_API_KEY="$(cat /run/secrets/minimax_api_key)"; export MINIMAX_API_KEY; exec "$@"',
                "benchmark-provider", *pi_argv,
            ],
            cwd=request.workspace, timeout_seconds=request.timeout_seconds,
        )
        stdout = result.stdout.replace(self.provider_key, b"<redacted>")
        stderr = result.stderr.replace(self.provider_key, b"<redacted>")
        return AgentOutcome(
            result.returncode, stdout, stderr, result.wall_seconds,
            result.timed_out, parse_pi_metrics(stdout, self.shipped_commands, self.shipped_edits),
        )


class DockerTrustedCommandExecutor:
    """Run setup and graders offline in the exact sealed validation image."""

    def __init__(
        self, *, docker: pathlib.Path, image: str,
        readonly_roots: Sequence[pathlib.Path] = (),
    ) -> None:
        self.docker = docker
        self.image = image
        self.readonly_roots = tuple(path.resolve() for path in readonly_roots)

    def __call__(
        self, argv: Sequence[str], *, cwd: pathlib.Path, timeout_seconds: int,
    ) -> CommandResult:
        command = [
            str(self.docker), "run", "--rm", "--init", "--network", "none",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", f"{os.getuid()}:{os.getgid()}", "--workdir", "/workspace",
            "--mount", f"type=bind,src={cwd},dst=/workspace",
        ]
        for index, path in enumerate(self.readonly_roots):
            command.extend(["--mount", f"type=bind,src={path},dst=/dependencies/{index},readonly"])
        command.extend([self.image, *argv])
        return run_command(command, cwd=cwd, timeout_seconds=timeout_seconds)


def _docker_inspect_one(docker: pathlib.Path, kind: str, name: str) -> dict[str, Any]:
    result = checked(
        [str(docker), kind, "inspect", name], cwd=pathlib.Path.cwd(), timeout_seconds=30,
    )
    try:
        values = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RunnerError(f"Docker returned invalid {kind} inspection JSON") from exc
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
        raise RunnerError(f"Docker {kind} inspection did not return exactly one object")
    return values[0]


def verify_live_network_attestation(docker: pathlib.Path, network: NetworkAttestation) -> None:
    topology = network.topology
    internal_expected = topology.get("internal_network") or {}
    egress_expected = topology.get("egress_network") or {}
    proxy_expected = topology.get("proxy") or {}
    if not all(isinstance(row.get("name"), str) and isinstance(row.get("id"), str)
               for row in (internal_expected, egress_expected, proxy_expected)):
        raise RunnerError("network attestation lacks inspectable topology identities")
    internal = _docker_inspect_one(docker, "network", internal_expected["name"])
    egress = _docker_inspect_one(docker, "network", egress_expected["name"])
    proxy = _docker_inspect_one(docker, "container", proxy_expected["name"])
    if (
        internal_expected["name"] != network.docker_network
        or internal.get("Id") != internal_expected["id"]
        or internal.get("Internal") is not True
        or egress.get("Id") != egress_expected["id"]
        or egress.get("Internal") is True
        or proxy.get("Id") != proxy_expected["id"]
        or proxy.get("Image") != proxy_expected.get("image_id")
        or proxy.get("State", {}).get("Running") is not True
    ):
        raise RunnerError("live Docker network/proxy identities differ from the signed audit")
    live_networks = (proxy.get("NetworkSettings") or {}).get("Networks") or {}
    expected_networks = proxy_expected.get("networks")
    proxy_ip = live_networks.get(network.docker_network, {}).get("IPAddress")
    if (
        sorted(live_networks) != expected_networks
        or proxy_expected.get("mount_count") != len(proxy.get("Mounts") or [])
        or proxy_ip != urllib.parse.urlsplit(network.provider_proxy).hostname
    ):
        raise RunnerError("live proxy topology differs from the signed audit")


def verify_internal_docker_network(docker: pathlib.Path, network: str) -> None:
    """Compatibility helper for callers that only need the internal-network invariant."""
    inspected = _docker_inspect_one(docker, "network", network)
    if inspected.get("Internal") is not True:
        raise RunnerError("agent Docker network must be internal")


def verify_network_attestation_image(network: NetworkAttestation, image_id: str) -> None:
    if network.agent_probe_image_id != image_id:
        raise RunnerError("network audit evidence was produced with a different agent image")


def docker_image_id(docker: pathlib.Path, image: str) -> str:
    return checked(
        [str(docker), "image", "inspect", "--format", "{{.Id}}", image],
        cwd=pathlib.Path.cwd(), timeout_seconds=60,
    ).stdout.decode("ascii", "replace").strip()


def hash_readonly_root(root: pathlib.Path) -> dict[str, Any]:
    """Bind the bytes and symlink targets mounted into both arms without publishing host paths."""
    resolved = root.resolve()
    if not resolved.is_dir():
        raise RunnerError(f"readonly dependency root is not a directory: {root}")
    digest = hashlib.sha256()
    files = symlinks = 0
    for entry in sorted(resolved.rglob("*"), key=lambda item: item.relative_to(resolved).as_posix()):
        relative = entry.relative_to(resolved).as_posix().encode()
        stat = entry.lstat()
        if entry.is_symlink():
            kind = b"L"
            payload = os.readlink(entry).encode()
            symlinks += 1
        elif entry.is_file():
            kind = b"F"
            file_digest = hashlib.sha256()
            with entry.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    file_digest.update(chunk)
            payload = file_digest.digest()
            files += 1
        elif entry.is_dir():
            kind = b"D"
            payload = b""
        else:
            raise RunnerError(f"readonly dependency contains unsupported file type: {entry}")
        digest.update(kind + b"\0" + relative + b"\0" + str(stat.st_mode & 0o7777).encode() + b"\0" + payload + b"\0")
    return {
        "tree_sha256": digest.hexdigest(),
        "file_count": files,
        "symlink_count": symlinks,
    }


def runtime_bindings(
    *, agents_md_path: pathlib.Path, greppy_bin: pathlib.Path,
    provider_extension: pathlib.Path, provider: str, model: str,
    image_id: str, network: NetworkAttestation,
    readonly_roots: Sequence[pathlib.Path],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    readonly = [
        {"mount_index": index, **hash_readonly_root(root)}
        for index, root in enumerate(readonly_roots)
    ]
    bindings = {
        "runner_source_sha256": sha256(pathlib.Path(__file__).read_bytes()),
        "gate_contract_sha256": sha256(compact_canonical_json(GATE_CONTRACT)),
        "pricing_contract_sha256": sha256(compact_canonical_json(PROVIDER_PRICE_USD_PER_MILLION)),
        "greppy_binary_sha256": sha256(greppy_bin.read_bytes()),
        "agents_md_sha256": sha256(agents_md_path.read_bytes()),
        "provider_extension_sha256": sha256(provider_extension.read_bytes()),
        "model": {"provider": provider, "model": model},
        "agent_image_id": image_id,
        "network_attestation_sha256": network.sha256,
        "network_audit_proof_sha256": network.proof_sha256,
        "readonly_dependencies_sha256": sha256(compact_canonical_json(readonly)),
    }
    return bindings, readonly


def probe_agent_image(
    docker: pathlib.Path, image: str, greppy_bin: pathlib.Path, agents_md_path: pathlib.Path,
) -> None:
    result = checked(
        [
            str(docker), "run", "--rm", "--network", "none", "--read-only",
            "--mount", f"type=bind,src={greppy_bin.resolve()},dst=/tools/greppy,readonly",
            "--mount", f"type=bind,src={agents_md_path.resolve()},dst=/manual/AGENTS.md,readonly",
            image, "/bin/sh", "-c",
            "set -eu; rg --version; pi --version; /tools/greppy --version; sha256sum /manual/AGENTS.md",
        ],
        cwd=pathlib.Path.cwd(), timeout_seconds=120,
    )
    expected = sha256(agents_md_path.read_bytes())
    if expected not in result.stdout.decode("utf-8", "replace"):
        raise RunnerError("runtime AGENTS.md hash probe failed inside agent image")


def _describe(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "denominator_status": "zero_fail"}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 9),
        "median": round(statistics.median(values), 9),
        "denominator_status": "measured",
    }


def _paired_bootstrap_lower(differences: Sequence[int], samples: int = 10_000) -> float | None:
    if not differences:
        return None
    generator = random.Random(0x0300)
    estimates = sorted(
        statistics.fmean(generator.choice(differences) for _ in differences)
        for _ in range(samples)
    )
    return round(estimates[int(0.025 * samples)], 6)


def _paired_bootstrap_interval(
    differences: Sequence[float], *, seed: int, samples: int = 10_000,
) -> dict[str, Any]:
    if not differences:
        return {"n": 0, "mean_treatment_minus_control": None, "bootstrap_95pct": None}
    generator = random.Random(seed)
    estimates = sorted(
        statistics.fmean(generator.choice(differences) for _ in differences)
        for _ in range(samples)
    )
    return {
        "n": len(differences),
        "mean_treatment_minus_control": round(statistics.fmean(differences), 6),
        "bootstrap_95pct": [
            round(estimates[int(0.025 * samples)], 6),
            round(estimates[int(0.975 * samples)], 6),
        ],
        "bootstrap_samples": samples,
    }


def _repo_clustered_cost_upper(rows: Sequence[dict[str, Any]], samples: int = 10_000) -> float | None:
    clusters: dict[str, list[dict[str, Any]]] = {}
    for task in rows:
        clusters.setdefault(str(task["strata"].get("repository", "unknown")), []).append(task)
    if not clusters:
        return None
    names = sorted(clusters)
    generator = random.Random(0x0301)
    ratios: list[float] = []
    for _ in range(samples):
        control = treatment = 0.0
        for name in (generator.choice(names) for _ in names):
            for task in clusters[name]:
                entries = {entry["arm"]: entry for entry in task["arms"]}
                control_value = entries["control"]["agent"]["metrics"].get("provider_cost_usd")
                treatment_value = entries["treatment"]["agent"]["metrics"].get("provider_cost_usd")
                if control_value is None or treatment_value is None:
                    return None
                control += float(control_value)
                treatment += float(treatment_value)
        if control <= 0:
            return None
        ratios.append(treatment / control)
    ratios.sort()
    return round(ratios[int(0.975 * samples)], 6)


def _stratum_arm_summary(tasks: Sequence[dict[str, Any]], arm: str) -> dict[str, Any]:
    entries = [next(entry for entry in task["arms"] if entry["arm"] == arm) for task in tasks]
    costs = [entry["agent"]["metrics"].get("provider_cost_usd") for entry in entries]
    return {
        "n": len(entries),
        "solved": sum(bool(entry["grading"]["correctness"]) for entry in entries),
        "total_provider_cost_usd": (
            round(sum(float(value) for value in costs), 9)
            if all(value is not None for value in costs) else None
        ),
    }


def summarize_results(rows: Sequence[dict[str, Any]], agents_md: str) -> dict[str, Any]:
    arm_rows = {
        arm: [next(entry for entry in task["arms"] if entry["arm"] == arm) for task in rows]
        for arm in ARMS
    }
    arm_summary: dict[str, Any] = {}
    for arm, entries in arm_rows.items():
        raw_costs = [entry["agent"]["metrics"].get("provider_cost_usd") for entry in entries]
        costs_complete = all(value is not None for value in raw_costs)
        costs = [float(value) for value in raw_costs if value is not None]
        solved = sum(bool(entry["grading"]["correctness"]) for entry in entries)
        gross_input = [float(entry["agent"]["metrics"].get("input_tokens", 0) or 0) for entry in entries]
        wall = [float(entry["agent"].get("wall_seconds", 0) or 0) for entry in entries]
        source_opens = [float(entry["agent"]["metrics"].get("source_open_events", 0) or 0) for entry in entries]
        arm_summary[arm] = {
            "intention_to_treat": True,
            "attempted_tasks": len(entries),
            "solved_tasks": solved,
            "correctness_rate": round(solved / len(entries), 6) if entries else None,
            "correctness_denominator_status": "measured" if entries else "zero_fail",
            "total_provider_cost_usd_all_tasks_all_attempts": round(sum(costs), 9) if costs_complete else None,
            "cost_per_solve_usd": round(sum(costs) / solved, 9) if solved and costs_complete else None,
            "cost_per_solve_denominator_status": "measured" if solved and costs_complete else "zero_or_missing_fail",
            "provider_cost_usd": _describe(costs),
            "gross_input_tokens": _describe(gross_input),
            "agent_wall_seconds": _describe(wall),
            "source_open_events_all_tools": _describe(source_opens),
        }
    paired_differences = [
        int(next(entry for entry in task["arms"] if entry["arm"] == "treatment")["grading"]["correctness"])
        - int(next(entry for entry in task["arms"] if entry["arm"] == "control")["grading"]["correctness"])
        for task in rows
    ]
    lower = _paired_bootstrap_lower(paired_differences)
    complete_valid = all(entry.get("valid") is True for task in rows for entry in task["arms"])
    all_costs_complete = all(
        entry["agent"]["metrics"].get("provider_cost_usd") is not None
        for task in rows for entry in task["arms"]
    )
    control_total = arm_summary["control"]["total_provider_cost_usd_all_tasks_all_attempts"]
    treatment_total = arm_summary["treatment"]["total_provider_cost_usd_all_tasks_all_attempts"]
    observed_cost_ratio = (
        treatment_total / control_total
        if all_costs_complete and control_total is not None and control_total > 0 and treatment_total is not None
        else None
    )
    cost_upper = _repo_clustered_cost_upper(rows)
    treatment_entries = arm_rows["treatment"]
    adopted = sum(entry["agent"]["metrics"].get("greppy_calls", 0) > 0 for entry in treatment_entries)
    adoption_rate = adopted / len(treatment_entries) if treatment_entries else None
    token_differences = []
    wall_differences = []
    for task in rows:
        entries = {entry["arm"]: entry for entry in task["arms"]}
        token_differences.append(
            float(entries["treatment"]["agent"]["metrics"].get("input_tokens", 0) or 0)
            - float(entries["control"]["agent"]["metrics"].get("input_tokens", 0) or 0)
        )
        wall_differences.append(
            float(entries["treatment"]["agent"].get("wall_seconds", 0) or 0)
            - float(entries["control"]["agent"].get("wall_seconds", 0) or 0)
        )
    solved_pairs = [
        task for task in rows
        if all(entry["grading"]["correctness"] for entry in task["arms"])
    ]
    solved_pair_cost_ratios: list[float] = []
    for task in solved_pairs:
        entries = {entry["arm"]: entry for entry in task["arms"]}
        control_cost = float(entries["control"]["agent"]["metrics"].get("provider_cost_usd", 0) or 0)
        treatment_cost = float(entries["treatment"]["agent"]["metrics"].get("provider_cost_usd", 0) or 0)
        if control_cost > 0:
            solved_pair_cost_ratios.append(treatment_cost / control_cost)
    strata: dict[str, Any] = {}
    for dimension in ("repository", "language", "task_class"):
        values: dict[str, list[dict[str, Any]]] = {}
        for task in rows:
            values.setdefault(str(task["strata"].get(dimension, "unknown")), []).append(task)
        strata[dimension] = {
            value: {
                arm: _stratum_arm_summary(tasks, arm)
                for arm in ARMS
            }
            for value, tasks in sorted(values.items())
        }
    treatment_bytes = len(system_prompt("treatment", agents_md).encode())
    control_bytes = len(system_prompt("control", agents_md).encode())
    return {
        "primary_cost_population": "all_tasks_all_provider_attempts_intention_to_treat",
        "arms": arm_summary,
        "correctness_noninferiority": {
            "margin_percentage_points": V3_CORRECTNESS_MARGIN * 100,
            "observed_treatment_minus_control": round(statistics.fmean(paired_differences), 6) if paired_differences else None,
            "paired_bootstrap_95pct_lower": lower,
            "bootstrap_samples": 10_000,
            "passes": complete_valid and lower is not None and lower >= V3_CORRECTNESS_MARGIN,
            "n": len(paired_differences),
            "all_trajectories_valid": complete_valid,
        },
        "solved_pair_cost_ratio_descriptive_only": _describe(solved_pair_cost_ratios),
        "primary_cost_gate": {
            "population": "all_tasks_all_provider_attempts_intention_to_treat",
            "observed_treatment_to_control": round(observed_cost_ratio, 6) if observed_cost_ratio is not None else None,
            "repo_clustered_bootstrap_95pct_upper": cost_upper,
            "threshold": V3_COST_RATIO_MAX,
            "passes": complete_valid and all_costs_complete and cost_upper is not None and cost_upper <= V3_COST_RATIO_MAX,
        },
        "greppy_task_adoption": {
            "adopted_treatment_tasks": adopted,
            "treatment_tasks": len(treatment_entries),
            "rate": round(adoption_rate, 6) if adoption_rate is not None else None,
            "minimum": V3_MIN_GREPPY_ADOPTION,
            "passes": adoption_rate is not None and adoption_rate >= V3_MIN_GREPPY_ADOPTION,
        },
        "greppy_edit_adoption_diagnostic_only": {
            "tasks": sum(entry["agent"]["metrics"].get("greppy_edit_calls", 0) > 0 for entry in treatment_entries),
            "is_gate_metric": False,
        },
        "paired_efficiency_intervals": {
            "gross_input_tokens": _paired_bootstrap_interval(token_differences, seed=0x0302),
            "agent_wall_seconds": _paired_bootstrap_interval(wall_differences, seed=0x0303),
        },
        "transactionality_observation": {
            "observable": False,
            "is_release_gate": False,
            "partial_workspace_incidents": None,
            "reason": "Pi JSON is post-hoc and provides no before/after hook around each failed Greppy edit; log text is not state evidence.",
            "required_future_instrumentation": "Capture git diff and git status hashes immediately before and after every failed edit tool result.",
        },
        "prompt_overhead": {
            "static_utf8_bytes": {"control": control_bytes, "treatment": treatment_bytes},
            "treatment_delta_utf8_bytes": treatment_bytes - control_bytes,
            "treatment_delta_token_estimate_per_turn": (treatment_bytes - control_bytes + 3) // 4,
            "provider_reported_treatment_prompt_tokens": None,
            "provider_split_unavailable_reason": "provider usage reports gross turn input, not system-prompt components",
            "headline_cost_includes_prompt_overhead": True,
        },
        "strata": strata,
    }


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True).encode() + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)


def release_process_exit_code(full_release_population: bool, gate_passed: bool | None) -> int:
    return 3 if full_release_population and gate_passed is not True else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, help="sealed release ID below the configured NAS releases root")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--storage-config", type=pathlib.Path)
    parser.add_argument("--agents-md", type=pathlib.Path, required=True)
    parser.add_argument("--greppy-bin", type=pathlib.Path, required=True)
    parser.add_argument("--provider-extension", type=pathlib.Path, required=True)
    parser.add_argument("--provider-key-file", type=pathlib.Path, required=True, help="0600 scoped ephemeral benchmark key")
    parser.add_argument("--docker", type=pathlib.Path, default=pathlib.Path("/usr/bin/docker"))
    parser.add_argument("--agent-image", required=True, help="immutable agent image reference with @sha256 digest")
    parser.add_argument("--agent-network", required=True, help="precreated internal Docker network")
    parser.add_argument("--provider-proxy", required=True, help="allowlist proxy reachable only on agent-network")
    parser.add_argument("--network-isolation-attestation", type=pathlib.Path, required=True)
    parser.add_argument("--gpu3-preflight-report", type=pathlib.Path)
    parser.add_argument("--gpu3-preflight-attestation", type=pathlib.Path)
    parser.add_argument("--gpu3-preflight-signature", type=pathlib.Path)
    parser.add_argument("--operations-public-key", type=pathlib.Path)
    parser.add_argument("--openssl", type=pathlib.Path, default=pathlib.Path("/usr/bin/openssl"))
    parser.add_argument("--smoke-evidence", type=pathlib.Path)
    parser.add_argument("--smoke-signature", type=pathlib.Path)
    parser.add_argument("--emit-runtime-bindings", type=pathlib.Path)
    parser.add_argument("--pi-command", default="pi")
    parser.add_argument("--provider", default="minimax")
    parser.add_argument("--model", default="MiniMax-M3")
    parser.add_argument("--readonly-root", action="append", type=pathlib.Path, default=[])
    parser.add_argument("--task", action="append", dest="task_ids")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if not SAFE_RUN_ID.fullmatch(args.run_id) or not SAFE_RUN_ID.fullmatch(args.release):
            raise RunnerError("run-id and release must be opaque safe path components")
        for executable in (
            args.agents_md, args.greppy_bin, args.provider_extension, args.provider_key_file,
            args.docker, args.openssl,
        ):
            if not executable.is_file():
                raise RunnerError(f"required file is missing: {executable}")
        if args.provider_key_file.stat().st_mode & 0o077:
            raise RunnerError("provider key file must not be group/world accessible")
        if "@sha256:" not in args.agent_image:
            raise RunnerError("agent image must be pinned by sha256 digest")
        layout: StorageLayout = load_storage(args.storage_config)
        release_dir = layout.releases / args.release
        public_dir, sealed_dir = release_dir / "public", release_dir / "sealed"
        pairs = load_release(public_dir, sealed_dir)
        selected = set(args.task_ids or [])
        if selected:
            known = {public["id"] for public, _ in pairs}
            if not selected <= known:
                raise RunnerError("unknown selected task ID")
            pairs = [pair for pair in pairs if pair[0]["id"] in selected]
        full_release_population = not selected and len(pairs) == 144
        agents_md = args.agents_md.read_text(encoding="utf-8")
        shipped_commands, shipped_edits = commands_from_agents_md(agents_md)
        network = load_network_attestation(
            args.network_isolation_attestation, args.provider_proxy, args.agent_network
        )
        verify_live_network_attestation(args.docker, network)
        image_id = docker_image_id(args.docker, args.agent_image)
        verify_network_attestation_image(network, image_id)
        probe_agent_image(args.docker, args.agent_image, args.greppy_bin, args.agents_md)
        sealed_image_digests = {
            str((sealed.get("validation_evidence") or {}).get("runner_image_digest", ""))
            for _, sealed in pairs
        }
        if sealed_image_digests != {image_id}:
            raise RunnerError(
                "agent/setup/grader image must exactly equal every sealed runner_image_digest"
            )
        for root in args.readonly_root:
            resolved = root.resolve()
            dependency_root = (layout.nvme_root / "dependency-caches").resolve()
            if resolved != dependency_root and dependency_root not in resolved.parents:
                raise RunnerError(
                    "agent readonly roots must be narrow children of the configured NVMe dependency-caches root"
                )
        bindings, readonly_dependency_identities = runtime_bindings(
            agents_md_path=args.agents_md, greppy_bin=args.greppy_bin,
            provider_extension=args.provider_extension, provider=args.provider,
            model=args.model, image_id=image_id, network=network,
            readonly_roots=args.readonly_root,
        )
        if args.emit_runtime_bindings is not None:
            atomic_json(args.emit_runtime_bindings, {
                "runtime_bindings": bindings,
                "readonly_dependency_identities": readonly_dependency_identities,
            })
            print(args.emit_runtime_bindings)
            return 0
        operational_paths = (
            args.gpu3_preflight_report, args.gpu3_preflight_attestation,
            args.gpu3_preflight_signature, args.operations_public_key,
        )
        if any(path is None or not path.is_file() for path in operational_paths):
            raise RunnerError("run requires signed gpu3 preflight report and attestation evidence")
        if full_release_population and (
            args.smoke_evidence is None or not args.smoke_evidence.is_file()
            or args.smoke_signature is None or not args.smoke_signature.is_file()
        ):
            raise RunnerError("full run requires signed exactly-three-pair smoke evidence")
        preflight_evidence = load_signed_preflight(
            report_path=args.gpu3_preflight_report,
            attestation_path=args.gpu3_preflight_attestation,
            signature_path=args.gpu3_preflight_signature,
            public_key=args.operations_public_key, openssl=args.openssl,
            runtime_bindings=bindings, network=network,
        )
        smoke_evidence = None
        if full_release_population:
            smoke_evidence = load_signed_smoke_evidence(
                evidence_path=args.smoke_evidence, signature_path=args.smoke_signature,
                public_key=args.operations_public_key, openssl=args.openssl,
                runtime_bindings=bindings,
                preflight_attestation_sha256=preflight_evidence["attestation_sha256"],
            )
        enforce_credential_boundary(full_release_population)
        executor = DockerPiExecutor(
            docker=args.docker, image=args.agent_image, network=network,
            pi_command=args.pi_command, greppy_bin=args.greppy_bin,
            provider_extension=args.provider_extension, provider_key_file=args.provider_key_file,
            provider=args.provider, model=args.model,
            shipped_commands=shipped_commands, shipped_edits=shipped_edits,
            readonly_roots=args.readonly_root,
        )
        run_root = layout.worktrees / args.run_id
        checkpoint = layout.scratch / "runs" / args.run_id / "checkpoint.json"
        archive = layout.nas_root / "agent-coding-v3" / "run-archives" / args.release / f"{args.run_id}.json"
        arm_orders = balanced_arm_orders([public["id"] for public, _ in pairs])
        checkpoint_identity = {
            "run_id": args.run_id,
            "release": args.release,
            "public_taskbank_sha256": sha256((public_dir / "taskbank.json").read_bytes()),
            "sealed_manifest_sha256": sha256((sealed_dir / "manifest.json").read_bytes()),
            "agents_md_sha256": sha256(agents_md.encode()),
            "agent_image_id": image_id,
            "model": {"provider": args.provider, "model": args.model},
            "greppy_binary_sha256": sha256(args.greppy_bin.read_bytes()),
            "provider_extension_sha256": sha256(args.provider_extension.read_bytes()),
            "network_isolation_attestation_sha256": network.sha256,
            "network_audit_proof_sha256": network.proof_sha256,
            "network_audit_agent_probe_image_id": network.agent_probe_image_id,
            "runtime_bindings": bindings,
            "readonly_dependency_identities": readonly_dependency_identities,
            "gpu3_preflight_evidence": preflight_evidence,
            "smoke_evidence": smoke_evidence,
            "arm_orders": arm_orders,
            "selected_task_ids": [public["id"] for public, _ in pairs],
        }
        if archive.exists():
            raise RunnerError("immutable run archive already exists")
        if (run_root.exists() or checkpoint.exists()) and not args.resume:
            raise RunnerError("run workspace/checkpoint exists; pass --resume")
        rows: list[dict[str, Any]] = []
        if args.resume and checkpoint.is_file():
            checkpoint_doc = _load_json(checkpoint)
            if checkpoint_doc.get("identity") != checkpoint_identity:
                raise RunnerError("checkpoint identity mismatch")
            loaded_rows = checkpoint_doc.get("rows")
            if not isinstance(loaded_rows, list):
                raise RunnerError("checkpoint rows are invalid")
            rows = loaded_rows
        completed = {row.get("task_id") for row in rows}
        command_executor = DockerTrustedCommandExecutor(
            docker=args.docker, image=args.agent_image, readonly_roots=args.readonly_root,
        )
        for index, (public_task, sealed_task) in enumerate(pairs, 1):
            if public_task["id"] in completed:
                continue
            slot = f"slot-{index:04d}"
            slot_path = run_root / slot
            if slot_path.exists():
                shutil.rmtree(slot_path)
            try:
                arm_rows = execute_pair(
                    public_task=public_task, sealed_task=sealed_task,
                    public_dir=public_dir, sealed_dir=sealed_dir,
                    slot_dir=slot_path, agents_md=agents_md, executor=executor,
                    arm_order=arm_orders[public_task["id"]],
                    command_executor=command_executor,
                    forbidden_secrets=(executor.provider_key,),
                )
            except Exception as exc:
                arm_rows = [
                    {
                        "arm": arm, "valid": False,
                        "agent": {"metrics": {}, "timed_out": False, "wall_seconds": 0.0},
                        "agent_diff_sha256": None, "agent_diff_bytes": 0,
                        "grading": {"correctness": False, "harness_error": exc.__class__.__name__},
                    }
                    for arm in arm_orders[public_task["id"]]
                ]
            rows.append({
                "task_id": public_task["id"], "slot": slot,
                "strata": {
                    "repository": ((sealed_task.get("repository") or {}).get("key", "unknown")),
                    "language": public_task.get("language", "unknown"),
                    "task_class": public_task.get("task_class", "unknown"),
                },
                "arms": arm_rows,
            })
            atomic_json(checkpoint, {"identity": checkpoint_identity, "rows": rows})
        summary = summarize_results(rows, agents_md)
        full_release_population = full_release_population and len(rows) == 144
        release_gate = {
            "applicable": full_release_population,
            "full_144_task_population": full_release_population,
            "subset_runs_are_smoke_only": True,
            "correctness_noninferiority_passes": summary["correctness_noninferiority"]["passes"],
            "itt_cost_passes": summary["primary_cost_gate"]["passes"],
            "greppy_adoption_passes": summary["greppy_task_adoption"]["passes"],
        }
        release_gate["passed"] = (all(
            value is True for key, value in release_gate.items()
            if key not in {"applicable", "full_144_task_population", "subset_runs_are_smoke_only"}
        ) if full_release_population else None)
        evidence = {
            "schema_version": "greppy.agent-coding-results.v3",
            "run_id": args.run_id,
            "release": args.release,
            "run_mode": "full_release" if full_release_population else "smoke_only_subset",
            "cost_gate_valid": False,
            "public_taskbank_sha256": sha256((public_dir / "taskbank.json").read_bytes()),
            "sealed_manifest_sha256": sha256((sealed_dir / "manifest.json").read_bytes()),
            "agents_md_sha256": sha256(agents_md.encode()),
            "agent_image": args.agent_image,
            "agent_image_id": image_id,
            "agent_network": args.agent_network,
            "network_isolation_attestation_sha256": network.sha256,
            "runtime_bindings": bindings,
            "readonly_dependency_identities": readonly_dependency_identities,
            "gpu3_preflight_evidence": preflight_evidence,
            "smoke_evidence": smoke_evidence,
            "allowed_provider_hosts": list(network.allowed_provider_hosts),
            "credential_contract": {
                "status": CREDENTIAL_BOUNDARY_STATUS,
                "scoped_ephemeral_provider_key": True,
                "mounted_as_read_only_docker_secret": True,
                "redacted_from_stdout_stderr_and_rejected_in_diff": True,
                "agent_child_process_can_read_provider_environment": True,
                "cost_gate_valid": False,
                "full_release_run_blocked": True,
            },
            "tools_per_arm": ARM_TOOLS,
            "identical_arm_tools": len(set(ARM_TOOLS.values())) == 1,
            "prompt_overhead_utf8_bytes": {
                arm: len(system_prompt(arm, agents_md).encode()) for arm in ARMS
            },
            "summary": summary,
            "release_gate": release_gate,
            "rows": rows,
        }
        atomic_json(archive, evidence)
        output = archive
    except (RunnerError, StorageError) as exc:
        print(f"v3 run failed: {exc}", file=sys.stderr)
        return 2
    print(output)
    return release_process_exit_code(full_release_population, release_gate["passed"])


if __name__ == "__main__":
    sys.exit(main())
