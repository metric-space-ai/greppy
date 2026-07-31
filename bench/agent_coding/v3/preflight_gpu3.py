#!/usr/bin/env python3
"""Fail-closed gpu3 readiness gate for the sealed V3 coding benchmark."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .audit_network_isolation import run_audit as run_network_audit
    from .storage import NAS_ENV, NVME_ENV, StorageError, load_storage
except ImportError:
    from audit_network_isolation import run_audit as run_network_audit
    from storage import NAS_ENV, NVME_ENV, StorageError, load_storage


REPORT_SCHEMA = "greppy.agent-coding-v3.gpu3-preflight.1"
CONFIG_SCHEMA = "greppy.agent-coding-v3.gpu3-preflight-config.1"
ADAPTER_SCHEMA = "greppy.agent-coding-v3.adapter-manifest.1"
UTC = dt.timezone.utc
HEX64 = re.compile(r"^[0-9a-f]{64}$")

DEFAULT_MINIMUM_VERSIONS: dict[str, str | None] = {
    "git": "2.30",
    "rg": "13.0",
    "python3": "3.11",
    "uv": "0.5",
    "cargo": "1.75",
    "rustc": "1.75",
    "go": "1.22",
    "java": "17",
    "mvn": "3.8.6",
    "node": "20",
    "pnpm": "9",
    "cmake": "3.25",
    "ninja": "1.10",
    "c++": None,
    "ruby": "3.2",
    "bundle": "2.5",
    "pi": None,
    "greppy": "0.3.0",
    "docker": "27.2.0",
}

TOOL_PROBES: dict[str, tuple[str, ...]] = {
    "git": ("--version",),
    "rg": ("--version",),
    "python3": ("--version",),
    "uv": ("--version",),
    "cargo": ("--version",),
    "rustc": ("--version",),
    "go": ("version",),
    "java": ("-version",),
    "mvn": ("-version",),
    "node": ("--version",),
    "pnpm": ("--version",),
    "cmake": ("--version",),
    "ninja": ("--version",),
    "c++": ("--version",),
    "ruby": ("--version",),
    "bundle": ("--version",),
    "pi": ("--version",),
    "greppy": ("--version",),
    "docker": ("version", "--format", "{{.Server.Version}}"),
}

CONTAINER_HOST_TOOLS = ("git", "rg", "docker", "pi", "greppy")

PROFILE_TOOLS: dict[str, tuple[str, ...]] = {
    "python-pip": ("python3", "uv"),
    "rust-cargo": ("cargo", "rustc"),
    "go-test": ("go",),
    "java-maven": ("java", "mvn"),
    "java-gradle": ("java",),
    "ts-pnpm": ("node", "pnpm"),
    "javascript-node": ("node",),
    "cpp-cmake": ("cmake", "ninja", "c++"),
    "ruby-bundler": ("ruby", "bundle"),
}

LANGUAGE_TOOLS: dict[str, tuple[str, ...]] = {
    "rust": ("cargo", "rustc"),
    "go": ("go",),
    "python": ("python3", "uv"),
    "typescript": ("node", "pnpm"),
    "java": ("java", "mvn"),
    "cpp": ("cmake", "ninja", "c++"),
    "javascript": ("node", "pnpm"),
    "ruby": ("ruby", "bundle"),
}


class PreflightConfigError(ValueError):
    """The gate cannot interpret its explicit configuration."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightConfigError(f"cannot read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightConfigError(f"{path} must contain a JSON object")
    return value


def parse_version(text: str) -> tuple[int, ...] | None:
    match = re.search(r"(?<!\d)(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:\.(\d+))?", text)
    if match is None:
        return None
    return tuple(int(value) for value in match.groups(default="0"))


def version_at_least(observed: tuple[int, ...], minimum: tuple[int, ...]) -> bool:
    width = max(len(observed), len(minimum))
    return observed + (0,) * (width - len(observed)) >= minimum + (0,) * (width - len(minimum))


def java_major(text: str) -> int | None:
    match = re.search(r'(?:version\s+")?(1\.)?(\d+)(?:\.\d+)', text, re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(2))


def mount_identity(path: Path) -> dict[str, Any]:
    """Return the most specific Linux mount record without invoking a shell."""
    best: tuple[int, dict[str, Any]] | None = None
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    resolved = str(path.resolve())
    for line in lines:
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        fields = left.split()
        suffix = right.split()
        if len(fields) < 5 or len(suffix) < 2:
            continue
        mountpoint = fields[4].replace("\\040", " ")
        try:
            inside = os.path.commonpath((resolved, mountpoint)) == mountpoint
        except ValueError:
            inside = False
        if not inside:
            continue
        record = {
            "mountpoint": mountpoint,
            "filesystem_type": suffix[0],
            "source": suffix[1],
        }
        if best is None or len(mountpoint) > best[0]:
            best = (len(mountpoint), record)
    return best[1] if best else {"mountpoint": None, "filesystem_type": None, "source": None}


def writable_probe(path: Path) -> tuple[bool, str | None]:
    probe: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".greppy-v3-preflight-", dir=path)
        probe = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(b"greppy-v3-preflight\n")
            handle.flush()
            os.fsync(handle.fileno())
        probe.unlink()
        probe = None
        return True, None
    except OSError as exc:
        return False, str(exc)
    finally:
        if probe is not None:
            probe.unlink(missing_ok=True)


def check_storage(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"ready": False, "roots": {}, "failures": []}
    try:
        layout = load_storage(config_path, create=False)
    except StorageError as exc:
        result["failures"].append({"code": "storage_config", "message": str(exc)})
        return result

    roots = {"nvme": layout.nvme_root, "nas": layout.nas_root}
    identities: dict[str, dict[str, Any]] = {}
    for tier, root in roots.items():
        tier_config = config.get(tier, {})
        minimum_gib = tier_config.get("minimum_free_gib", 0) if isinstance(tier_config, dict) else 0
        if not isinstance(minimum_gib, (int, float)) or isinstance(minimum_gib, bool) or minimum_gib <= 0:
            result["failures"].append({"code": "storage_minimum", "tier": tier, "message": "minimum_free_gib must be positive and explicit"})
            continue
        entry: dict[str, Any] = {"path": str(root), "minimum_free_gib": minimum_gib}
        if not root.is_dir():
            entry.update({"exists": False, "writable": False})
            result["failures"].append({"code": "storage_missing", "tier": tier, "message": f"configured {tier} root is not a directory"})
            result["roots"][tier] = entry
            continue
        stat = root.stat()
        usage = shutil.disk_usage(root)
        mount = mount_identity(root)
        writable, write_error = writable_probe(root)
        entry.update({
            "exists": True,
            "writable": writable,
            "write_error": write_error,
            "device_id": stat.st_dev,
            "free_bytes": usage.free,
            "free_gib": round(usage.free / (1024 ** 3), 3),
            **mount,
        })
        identities[tier] = entry
        if not writable:
            result["failures"].append({"code": "storage_not_writable", "tier": tier, "message": write_error or "write probe failed"})
        required = int(float(minimum_gib) * (1024 ** 3))
        if usage.free < required:
            result["failures"].append({"code": "storage_space", "tier": tier, "required_bytes": required, "observed_bytes": usage.free, "message": "insufficient free space"})
        result["roots"][tier] = entry

    if set(identities) == {"nvme", "nas"}:
        distinct = identities["nvme"]["device_id"] != identities["nas"]["device_id"]
        result["distinct_devices"] = distinct
        if not distinct:
            result["failures"].append({"code": "storage_same_device", "message": "NVMe and NAS roots resolve to the same device/filesystem"})
    result["ready"] = not result["failures"]
    return result


def command_environment() -> dict[str, str]:
    env = dict(os.environ)
    java_home = env.get("JAVA_HOME")
    if java_home:
        java_bin = str(Path(java_home).expanduser() / "bin")
        env["PATH"] = java_bin + os.pathsep + env.get("PATH", "")
    return env


def resolve_command(name: str, overrides: Mapping[str, Any], env: Mapping[str, str]) -> str | None:
    override = overrides.get(name)
    if override is not None:
        if not isinstance(override, str) or not override:
            return None
        candidate = Path(override).expanduser()
        if candidate.is_absolute() or "/" in override:
            return str(candidate.resolve()) if candidate.is_file() and os.access(candidate, os.X_OK) else None
        return shutil.which(override, path=env.get("PATH"))
    if name == "java" and env.get("JAVA_HOME"):
        candidate = Path(env["JAVA_HOME"]).expanduser() / "bin" / "java"
        return str(candidate.resolve()) if candidate.is_file() and os.access(candidate, os.X_OK) else None
    return shutil.which(name, path=env.get("PATH"))


def probe_tool(name: str, executable: str, minimum: str | None, timeout: int, env: Mapping[str, str]) -> dict[str, Any]:
    entry: dict[str, Any] = {"name": name, "executable": executable, "minimum_version": minimum, "ready": False}
    try:
        proc = subprocess.run(
            [executable, *TOOL_PROBES[name]], capture_output=True, text=True,
            errors="replace", timeout=timeout, env=dict(env),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        entry["error"] = str(exc)
        return entry
    output = (proc.stdout + "\n" + proc.stderr).strip()
    entry.update({"returncode": proc.returncode, "version_output": output[:1000]})
    if proc.returncode != 0:
        entry["error"] = "version probe returned nonzero"
        return entry
    observed = parse_version(output)
    entry["observed_version"] = ".".join(map(str, observed)) if observed else None
    if minimum is not None:
        required = parse_version(minimum)
        if observed is None or required is None or not version_at_least(observed, required):
            entry["error"] = "version is below configured minimum or could not be parsed"
            return entry
    entry["ready"] = True
    return entry


def check_tools(config: Mapping[str, Any], registry: Mapping[str, Any], execution_mode: str = "host") -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    tools_config = config.get("tools", {})
    if not isinstance(tools_config, dict):
        return {"ready": False, "tools": {}, "language_coverage": {}, "failures": [{"code": "tools_config", "message": "tools must be an object"}]}
    overrides = tools_config.get("command_overrides", {})
    minimums = dict(DEFAULT_MINIMUM_VERSIONS)
    configured_minimums = tools_config.get("minimum_versions", {})
    configured_exact_versions = tools_config.get("exact_versions", {})
    if not isinstance(overrides, dict) or not isinstance(configured_minimums, dict) or not isinstance(configured_exact_versions, dict):
        return {"ready": False, "tools": {}, "language_coverage": {}, "failures": [{"code": "tools_config", "message": "command_overrides, minimum_versions and exact_versions must be objects"}]}
    exact_versions = {"greppy": "0.3.0", **configured_exact_versions}
    if exact_versions.get("greppy") != "0.3.0":
        failures.append({"code": "tool_exact_version", "tool": "greppy", "message": "this release preflight requires exact Greppy 0.3.0"})
    for name, value in configured_minimums.items():
        if name not in minimums or (value is not None and not isinstance(value, str)):
            failures.append({"code": "tool_minimum", "tool": name, "message": "unknown tool or invalid minimum version"})
        else:
            minimums[name] = value
    timeout = tools_config.get("probe_timeout_seconds", 20)
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 1:
        timeout = 20
        failures.append({"code": "tool_timeout", "message": "probe_timeout_seconds must be a positive integer"})
    env = command_environment()
    entries: dict[str, Any] = {}
    required_names = tuple(TOOL_PROBES) if execution_mode == "host" else CONTAINER_HOST_TOOLS
    for name in required_names:
        executable = resolve_command(name, overrides, env)
        if executable is None:
            entries[name] = {"name": name, "ready": False, "error": "executable not found", "minimum_version": minimums[name]}
            failures.append({"code": "tool_missing", "tool": name, "message": "required executable not found"})
            continue
        entry = probe_tool(name, executable, minimums[name], timeout, env)
        entries[name] = entry
        if not entry["ready"]:
            failures.append({"code": "tool_probe", "tool": name, "message": entry.get("error", "probe failed")})
    for name, required_text in exact_versions.items():
        if name not in entries or not isinstance(required_text, str) or not required_text:
            failures.append({"code": "tool_exact_version", "tool": name, "message": "invalid exact-version requirement"})
            continue
        observed = parse_version(entries[name].get("version_output", ""))
        required = parse_version(required_text)
        if observed is None or required is None or observed != required:
            entries[name]["ready"] = False
            entries[name]["exact_version"] = required_text
            failures.append({"code": "tool_exact_version", "tool": name, "message": f"required exact version {required_text}"})

    required_languages = registry.get("primary_languages", [])
    coverage: dict[str, Any] = {}
    for language in required_languages:
        if execution_mode == "container":
            coverage[language] = {"ready": None, "delegated_to": "pinned per-adapter container probe"}
            continue
        required = LANGUAGE_TOOLS.get(language)
        if required is None:
            coverage[language] = {"ready": False, "tools": [], "error": "no preflight tool mapping"}
            failures.append({"code": "language_tools", "language": language, "message": "no toolchain mapping"})
            continue
        ready = all(entries.get(name, {}).get("ready") for name in required)
        coverage[language] = {"ready": ready, "tools": list(required)}
        if not ready:
            failures.append({"code": "language_tools", "language": language, "message": "one or more language tools failed"})

    if execution_mode == "container":
        return {
            "ready": not failures,
            "execution_mode": execution_mode,
            "tools": entries,
            "language_coverage": coverage,
            "jdk_maven": {"ready": None, "delegated_to": "java adapter image probes"},
            "failures": failures,
        }

    java_output = entries.get("java", {}).get("version_output", "")
    mvn_output = entries.get("mvn", {}).get("version_output", "")
    required_java_major = tools_config.get("required_java_major", 17)
    java_observed = java_major(java_output)
    maven_java_observed = java_major(re.search(r"Java version:.*", mvn_output, re.IGNORECASE).group(0)) if re.search(r"Java version:.*", mvn_output, re.IGNORECASE) else None
    jvm = {
        "java_home": env.get("JAVA_HOME"),
        "required_major": required_java_major,
        "java_observed_major": java_observed,
        "maven_java_observed_major": maven_java_observed,
        "ready": java_observed == required_java_major and maven_java_observed == required_java_major,
    }
    if not jvm["ready"]:
        failures.append({"code": "jdk_maven_version", "message": "java and Maven must both run on the configured JDK major", **jvm})

    return {"ready": not failures, "execution_mode": execution_mode, "tools": entries, "language_coverage": coverage, "jdk_maven": jvm, "failures": failures}


def check_adapters(
    config_path: Path,
    config: Mapping[str, Any],
    registry: Mapping[str, Any],
    execution_mode: str = "host",
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    configured = config.get("adapter_manifest")
    expected_rows = registry.get("repositories", [])
    expected = {row.get("id"): row for row in expected_rows if isinstance(row, dict) and isinstance(row.get("id"), str)}
    result: dict[str, Any] = {"ready": False, "expected": len(expected), "ready_count": 0, "repositories": {}, "failures": failures}
    if not isinstance(configured, str) or not configured:
        failures.append({"code": "adapter_manifest", "message": "adapter_manifest path is required"})
        return result
    manifest_path = Path(configured).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = config_path.parent / manifest_path
    try:
        manifest = load_json(manifest_path.resolve())
    except PreflightConfigError as exc:
        failures.append({"code": "adapter_manifest", "message": str(exc)})
        return result
    if manifest.get("schema_version") != ADAPTER_SCHEMA or not isinstance(manifest.get("adapters"), list):
        failures.append({"code": "adapter_manifest", "message": "unsupported adapter manifest schema"})
        return result
    provided: dict[str, dict[str, Any]] = {}
    for row in manifest["adapters"]:
        key = row.get("repository_id") if isinstance(row, dict) else None
        if not isinstance(key, str) or key in provided:
            failures.append({"code": "adapter_duplicate", "repository_id": key, "message": "invalid or duplicate adapter row"})
            continue
        provided[key] = row
    for extra in sorted(set(provided) - set(expected)):
        failures.append({"code": "adapter_extra", "repository_id": extra, "message": "adapter is not in the frozen registry"})

    timeout = config.get("adapter_probe_timeout_seconds", 30)
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 1:
        timeout = 30
        failures.append({"code": "adapter_timeout", "message": "adapter_probe_timeout_seconds must be positive"})
    env = command_environment()
    tool_config = config.get("tools", {}) if isinstance(config.get("tools", {}), dict) else {}
    overrides = tool_config.get("command_overrides", {}) if isinstance(tool_config.get("command_overrides", {}), dict) else {}
    docker_binary = resolve_command("docker", overrides, env)
    for key, registry_row in expected.items():
        row = provided.get(key)
        entry: dict[str, Any] = {"ready": False}
        if row is None:
            entry["error"] = "missing adapter row"
            failures.append({"code": "adapter_missing", "repository_id": key, "message": entry["error"]})
        elif row.get("status") != "ready":
            entry["error"] = f"adapter status is {row.get('status')!r}, expected 'ready'"
            failures.append({"code": "adapter_not_ready", "repository_id": key, "message": entry["error"]})
        elif row.get("toolchain_profile") != registry_row.get("toolchain_profile"):
            entry["error"] = "toolchain profile does not match registry"
            failures.append({"code": "adapter_profile", "repository_id": key, "message": entry["error"]})
        elif not isinstance(row.get("image"), str) or "@sha256:" not in row["image"]:
            entry["error"] = "ready adapter lacks a digest-pinned image reference"
            failures.append({"code": "adapter_image", "repository_id": key, "message": entry["error"]})
        elif not isinstance(row.get("image_id"), str) or not row["image_id"].startswith("sha256:"):
            entry["error"] = "ready adapter lacks a local sha256 image_id"
            failures.append({"code": "adapter_image", "repository_id": key, "message": entry["error"]})
        elif not isinstance(row.get("proof_sha256"), str) or HEX64.fullmatch(row["proof_sha256"]) is None:
            entry["error"] = "ready adapter lacks a 64-hex proof_sha256"
            failures.append({"code": "adapter_proof", "repository_id": key, "message": entry["error"]})
        elif (
            not isinstance(row.get("commands"), dict)
            or set(row["commands"]) != {"probe", "metadata", "validation"}
            or any(
                not isinstance(command, list)
                or not command
                or not all(isinstance(arg, str) and arg for arg in command)
                for command in row["commands"].values()
            )
        ):
            entry["error"] = "ready adapter needs complete argv arrays for probe, metadata and validation"
            failures.append({"code": "adapter_command", "repository_id": key, "message": entry["error"]})
        else:
            try:
                greppy_binary = resolve_command("greppy", overrides, env)
                if execution_mode == "container":
                    if docker_binary is None or greppy_binary is None:
                        raise OSError("Docker and exact Greppy binary are required for adapter image probes")
                    inspect = subprocess.run(
                        [docker_binary, "image", "inspect", row["image"]], capture_output=True,
                        text=True, errors="replace", timeout=timeout, env=env,
                    )
                    inspected = json.loads(inspect.stdout) if inspect.returncode == 0 else None
                    if not isinstance(inspected, list) or len(inspected) != 1 or inspected[0].get("Id") != row["image_id"]:
                        raise OSError("adapter image is missing or local image ID differs from manifest")
                command_reports: dict[str, Any] = {}
                valid_payload = True
                last_proc: subprocess.CompletedProcess[str] | None = None
                for role in ("probe", "metadata", "validation"):
                    if execution_mode == "container":
                        command = [
                            docker_binary, "run", "--rm", "--network", "none", "--read-only",
                            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                            "--pids-limit", "128",
                            "--mount", f"type=bind,src={greppy_binary},dst=/tools/greppy,readonly",
                            row["image"], *row["commands"][role], "--preflight",
                        ]
                    else:
                        command = [*row["commands"][role], "--preflight"]
                    proc = subprocess.run(command, cwd=config_path.parent, capture_output=True, text=True, errors="replace", timeout=timeout, env=env)
                    last_proc = proc
                    payload = json.loads(proc.stdout) if proc.stdout.strip() else None
                    role_valid = (
                        proc.returncode == 0
                        and isinstance(payload, dict)
                        and payload.get("ready") is True
                        and payload.get("repository_id") == key
                        and payload.get("command_role") == role
                        and payload.get("proof_sha256") == row["proof_sha256"]
                    )
                    command_reports[role] = payload
                    valid_payload = valid_payload and role_valid
                probe_payload = command_reports.get("probe") or {}
                if valid_payload and execution_mode == "container":
                    observed_tools = probe_payload.get("tools")
                    required_tools = PROFILE_TOOLS.get(str(registry_row.get("toolchain_profile")))
                    if not isinstance(observed_tools, dict) or required_tools is None:
                        valid_payload = False
                    else:
                        for tool_name in required_tools:
                            version_text = observed_tools.get(tool_name)
                            minimum_text = DEFAULT_MINIMUM_VERSIONS.get(tool_name)
                            observed_version = parse_version(version_text) if isinstance(version_text, str) else None
                            minimum_version = parse_version(minimum_text) if minimum_text else None
                            if observed_version is None or (minimum_version is not None and not version_at_least(observed_version, minimum_version)):
                                valid_payload = False
                        if registry_row.get("toolchain_profile") == "java-maven":
                            valid_payload = valid_payload and java_major(str(observed_tools.get("java", ""))) == 17 and java_major(str(observed_tools.get("mvn", ""))) == 17
                    agent_tools = probe_payload.get("agent_tools")
                    if not isinstance(agent_tools, dict):
                        valid_payload = False
                    else:
                        for tool_name in ("rg", "pi", "greppy"):
                            observed_version = parse_version(str(agent_tools.get(tool_name, "")))
                            minimum_text = DEFAULT_MINIMUM_VERSIONS.get(tool_name)
                            minimum_version = parse_version(minimum_text) if minimum_text else None
                            if observed_version is None or (minimum_version is not None and not version_at_least(observed_version, minimum_version)):
                                valid_payload = False
                        valid_payload = valid_payload and parse_version(str(agent_tools.get("greppy", ""))) == parse_version("0.3.0")
                entry.update({
                    "returncode": last_proc.returncode if last_proc else None,
                    "command_reports": command_reports,
                    "stderr": last_proc.stderr[-1000:] if last_proc else "",
                    "image": row["image"],
                    "image_id": row["image_id"],
                })
                if valid_payload:
                    entry["ready"] = True
                else:
                    entry["error"] = "adapter commands/toolchain probes did not return exact proof-bound ready JSON"
                    failures.append({"code": "adapter_probe", "repository_id": key, "message": entry["error"]})
            except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                entry["error"] = str(exc)
                failures.append({"code": "adapter_probe", "repository_id": key, "message": str(exc)})
        result["repositories"][key] = entry
    result["ready_count"] = sum(1 for entry in result["repositories"].values() if entry["ready"])
    result["execution_mode"] = execution_mode
    result["ready"] = not failures and result["ready_count"] == len(expected)
    return result


def run_preflight(config_path: Path) -> dict[str, Any]:
    checked_at = dt.datetime.now(UTC).isoformat().replace("+00:00", "Z")
    try:
        config = load_json(config_path)
        if config.get("schema_version") != CONFIG_SCHEMA:
            raise PreflightConfigError("unsupported preflight config schema_version")
        execution_mode = config.get("execution_mode")
        if execution_mode not in ("host", "container"):
            raise PreflightConfigError("execution_mode must be explicitly 'host' or 'container'")
        registry_value = config.get("registry")
        if not isinstance(registry_value, str) or not registry_value:
            raise PreflightConfigError("registry path is required")
        registry_path = Path(registry_value).expanduser()
        if not registry_path.is_absolute():
            registry_path = config_path.parent / registry_path
        registry = load_json(registry_path.resolve())
        if registry.get("schema_version") != "greppy.agent-coding-v3.repository-registry.1":
            raise PreflightConfigError("unsupported repository registry schema")
        rows = registry.get("repositories")
        languages = registry.get("primary_languages")
        if (
            registry.get("target_task_count") != 144
            or registry.get("repository_count") != 24
            or registry.get("tasks_per_repository") != 6
            or not isinstance(rows, list)
            or len(rows) != 24
            or not isinstance(languages, list)
            or set(languages) != set(LANGUAGE_TOOLS)
        ):
            raise PreflightConfigError("registry must be the fixed 144-task, 24-repository, eight-language V3 design")
        repo_ids = [row.get("id") for row in rows if isinstance(row, dict)]
        repo_languages = [row.get("primary_language") for row in rows if isinstance(row, dict)]
        if len(repo_ids) != 24 or len(set(repo_ids)) != 24 or any(repo_languages.count(language) != 3 for language in languages):
            raise PreflightConfigError("registry needs 24 unique repository IDs and exactly three repositories per language")
        network_value = config.get("network_policy")
        if not isinstance(network_value, str) or not network_value:
            raise PreflightConfigError("network_policy path is required")
        network_path = Path(network_value).expanduser()
        if not network_path.is_absolute():
            network_path = config_path.parent / network_path
        network_path = network_path.resolve()
    except PreflightConfigError as exc:
        return {
            "schema_version": REPORT_SCHEMA, "ready": False, "checked_at": checked_at,
            "host": {"hostname": socket.gethostname(), "platform": platform.platform()},
            "checks": {}, "failures": [{"code": "configuration", "message": str(exc)}],
        }

    checks: dict[str, dict[str, Any]] = {}
    probes = {
        "storage": lambda: check_storage(config_path, config),
        "tools": lambda: check_tools(config, registry, execution_mode),
        "adapters": lambda: check_adapters(config_path, config, registry, execution_mode),
        "network": lambda: run_network_audit(network_path),
    }
    for section, probe in probes.items():
        try:
            checks[section] = probe()
        except Exception as exc:  # a preflight bug must fail closed with JSON
            checks[section] = {
                "ready": False,
                "failures": [{"code": "internal_probe_error", "message": f"{type(exc).__name__}: {exc}"}],
            }
    failures = [
        {"section": section, **failure}
        for section, check in checks.items()
        for failure in check.get("failures", [])
    ]
    return {
        "schema_version": REPORT_SCHEMA,
        "ready": not failures and all(check.get("ready") for check in checks.values()),
        "checked_at": checked_at,
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(), "machine": platform.machine()},
        "inputs": {
            "config_sha256": hashlib.sha256(canonical_json(config)).hexdigest(),
            "registry_sha256": hashlib.sha256(canonical_json(registry)).hexdigest(),
            "execution_mode": execution_mode,
            "nvme_env_set": bool(os.environ.get(NVME_ENV)),
            "nas_env_set": bool(os.environ.get(NAS_ENV)),
        },
        "checks": checks,
        "failures": failures,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="also atomically write the JSON report")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_preflight(args.config.resolve())
    if args.report:
        try:
            target = args.report.resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        except OSError as exc:
            report["ready"] = False
            report["failures"].append({"section": "report", "code": "report_write", "message": str(exc)})
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    sys.stdout.write(payload)
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    sys.exit(main())
