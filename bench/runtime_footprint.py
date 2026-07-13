#!/usr/bin/env python3
"""Measure Greppy's runtime footprint without publishing repository content."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = "greppy.runtime-footprint.v1"
MAX_CAPTURE_BYTES = 64 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 900.0


class MeasurementError(RuntimeError):
    """A redaction-safe measurement failure."""


@dataclass(frozen=True)
class Config:
    greppy: Path
    repo: Path
    semantic_query: str
    brief_symbol: str
    output: Path
    warm_repeats: int
    device: str = "auto"
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class CommandResult:
    stdout: bytes
    elapsed_ms: float
    record: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _argv_sha256(argv: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in argv:
        encoded = os.fsencode(value)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _argv_template_sha256(
    argv: Sequence[str], replacements: dict[str, str]
) -> str:
    return _argv_sha256([replacements.get(value, value) for value in argv])


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _private_store(repo: Path) -> Path:
    candidates = [Path(tempfile.gettempdir()).resolve(), repo.parent.resolve()]
    for parent in candidates:
        if _is_within(parent, repo):
            continue
        store = Path(tempfile.mkdtemp(prefix="greppy-footprint-", dir=parent)).resolve()
        if _is_within(store, repo):
            shutil.rmtree(store)
            continue
        os.chmod(store, 0o700)
        return store
    raise MeasurementError("could not create a private store outside the repository")


def _child_environment(store: Path, device: str) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GREPPY_")
    }
    env["GREPPY_STORE_DIR"] = str(store)
    env["GREPPY_DEVICE"] = device
    return env


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if process.poll() is None:
            process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=2.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_command(
    argv: Sequence[str],
    *,
    phase: str,
    env: dict[str, str],
    timeout_seconds: float,
    accepted_exit_codes: Iterable[int] = (0,),
    iteration: int | None = None,
    publication_replacements: dict[str, str] | None = None,
) -> CommandResult:
    if not argv or any(not isinstance(value, str) or "\x00" in value for value in argv):
        raise MeasurementError(f"{phase}: invalid command arguments")
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    started = time.perf_counter_ns()
    try:
        process = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=os.name != "nt",
            creationflags=creationflags,
        )
    except OSError as error:
        raise MeasurementError(f"{phase}: command could not be started") from error
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as error:
        _kill_process_group(process)
        process.communicate()
        raise MeasurementError(f"{phase}: command timed out") from error
    elapsed_ms = round((time.perf_counter_ns() - started) / 1_000_000, 3)
    if len(stdout) > MAX_CAPTURE_BYTES or len(stderr) > MAX_CAPTURE_BYTES:
        raise MeasurementError(f"{phase}: command output exceeded the safety limit")
    if process.returncode not in set(accepted_exit_codes):
        raise MeasurementError(f"{phase}: command failed with exit status {process.returncode}")
    record: dict[str, Any] = {
        "phase": phase,
        "argv_template_sha256": _argv_template_sha256(
            argv, publication_replacements or {}
        ),
        "argument_count": len(argv),
        "exit_code": process.returncode,
        "wall_time_ms": elapsed_ms,
    }
    if iteration is not None:
        record["iteration"] = iteration
    return CommandResult(stdout=stdout, elapsed_ms=elapsed_ms, record=record)


def _json_object(raw: bytes, phase: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MeasurementError(f"{phase}: command did not return valid JSON") from error
    if not isinstance(value, dict):
        raise MeasurementError(f"{phase}: JSON root must be an object")
    return value


def _dict(value: Any, phase: str, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MeasurementError(f"{phase}: invalid JSON field {field}")
    return value


def _list(value: Any, phase: str, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise MeasurementError(f"{phase}: invalid JSON field {field}")
    return value


def _string(value: Any, phase: str, field: str) -> str:
    if not isinstance(value, str):
        raise MeasurementError(f"{phase}: invalid JSON field {field}")
    return value


def _boolean(value: Any, phase: str, field: str) -> bool:
    if not isinstance(value, bool):
        raise MeasurementError(f"{phase}: invalid JSON field {field}")
    return value


def _integer(value: Any, phase: str, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MeasurementError(f"{phase}: invalid JSON field {field}")
    return value


def _optional_scalar(source: dict[str, Any], target: dict[str, Any], field: str) -> None:
    value = source.get(field)
    if value is None or isinstance(value, (str, bool, int)):
        target[field] = value
    else:
        raise MeasurementError(f"invalid diagnostic scalar field {field}")


def _hashed_optional_text(source: dict[str, Any], target: dict[str, Any], field: str) -> None:
    value = source.get(field)
    if value is None:
        target[f"{field}_present"] = False
        target[f"{field}_sha256"] = None
    elif isinstance(value, str):
        target[f"{field}_present"] = True
        target[f"{field}_sha256"] = hashlib.sha256(value.encode("utf-8")).hexdigest()
    else:
        raise MeasurementError(f"invalid diagnostic text field {field}")


def _sanitize_registry(value: Any, phase: str) -> dict[str, Any]:
    registry = _dict(value, phase, "inference.registry")
    result: dict[str, Any] = {}
    for field in (
        "version",
        "preference",
        "explicit",
        "required_gpu_memory",
        "selected_backend",
        "selected_device_id",
    ):
        _optional_scalar(registry, result, field)
    probes = _list(registry.get("probes"), phase, "inference.registry.probes")
    result["probes"] = []
    for probe_value in probes:
        probe = _dict(probe_value, phase, "inference.registry.probes[]")
        clean_probe: dict[str, Any] = {}
        for field in (
            "backend",
            "backend_id",
            "build_info",
            "abi_version",
            "compiled",
            "available",
            "score",
        ):
            _optional_scalar(probe, clean_probe, field)
        _hashed_optional_text(probe, clean_probe, "reason")
        devices = _list(probe.get("devices"), phase, "inference.registry.probes[].devices")
        clean_probe["devices"] = []
        for device_value in devices:
            device = _dict(device_value, phase, "inference.registry.probes[].devices[]")
            clean_device: dict[str, Any] = {}
            for field in (
                "backend",
                "id",
                "name",
                "description",
                "device_type",
                "memory_free",
                "memory_total",
                "compute_capability",
                "metal_family",
            ):
                _optional_scalar(device, clean_device, field)
            _hashed_optional_text(device, clean_device, "rejection_reason")
            capabilities = _list(
                device.get("capabilities"), phase, "inference.registry.probes[].devices[].capabilities"
            )
            if not all(isinstance(item, str) for item in capabilities):
                raise MeasurementError(f"{phase}: invalid device capabilities")
            clean_device["capabilities"] = capabilities
            clean_probe["devices"].append(clean_device)
        result["probes"].append(clean_probe)
    return result


def _sanitize_models(value: Any, phase: str) -> dict[str, Any]:
    models = _dict(value, phase, "inference.models")
    result: dict[str, Any] = {}
    for name in ("embedding", "summary"):
        model = _dict(models.get(name), phase, f"inference.models.{name}")
        clean: dict[str, Any] = {}
        for field in (
            "model_id",
            "format",
            "embedded",
            "model_sha256",
            "tokenizer_sha256",
            "model_bytes",
            "runtime_state",
            "prompt_version",
            "task_profile",
            "state",
        ):
            _optional_scalar(model, clean, field)
        result[name] = clean
    return result


def _sanitize_daemons(value: Any, phase: str) -> tuple[dict[str, Any], set[int]]:
    daemons = _dict(value, phase, "inference.daemons")
    result: dict[str, Any] = {}
    pids: set[int] = set()
    for name in ("embedding", "summary"):
        daemon = _dict(daemons.get(name), phase, f"inference.daemons.{name}")
        clean: dict[str, Any] = {}
        for field in (
            "protocol",
            "state",
            "state_elapsed_ms",
            "active_request_elapsed_ms",
            "completed_requests",
            "rejected_requests",
            "queue_capacity",
            "pending_requests",
        ):
            _optional_scalar(daemon, clean, field)
        pid = daemon.get("daemon_pid")
        clean["pid_reported"] = isinstance(pid, int) and not isinstance(pid, bool) and pid > 1
        if clean["pid_reported"]:
            pids.add(pid)
        result[name] = clean
    return result, pids


def _doctor_daemon_pids(value: dict[str, Any]) -> set[int]:
    try:
        inference = value["inference"]
        daemons = inference["daemons"]
    except (KeyError, TypeError):
        return set()
    if not isinstance(daemons, dict):
        return set()
    pids: set[int] = set()
    for name in ("embedding", "summary"):
        daemon = daemons.get(name)
        if not isinstance(daemon, dict):
            continue
        pid = daemon.get("daemon_pid")
        if isinstance(pid, int) and not isinstance(pid, bool) and pid > 1:
            pids.add(pid)
    return pids


def _doctor_background_pids(value: dict[str, Any]) -> set[int]:
    job = value.get("background_job")
    if not isinstance(job, dict):
        return set()
    pid = job.get("pid")
    if isinstance(pid, int) and not isinstance(pid, bool) and pid > 1:
        return {pid}
    return set()


def _sanitize_doctor(
    value: dict[str, Any], phase: str
) -> tuple[dict[str, Any], set[int]]:
    if _string(value.get("command"), phase, "command") != "doctor":
        raise MeasurementError(f"{phase}: unexpected command JSON")
    result: dict[str, Any] = {
        "status": _string(value.get("status"), phase, "status"),
        "healthy": _boolean(value.get("healthy"), phase, "healthy"),
        "store_exists": _boolean(value.get("store_exists"), phase, "store_exists"),
        "store_bytes": _integer(value.get("store_bytes"), phase, "store_bytes"),
    }
    for field in (
        "embedding_complete",
        "fresh",
        "schema_version",
        "expected_schema_version",
        "schema_current",
        "integrity_ok",
        "project_present",
        "graph_generation",
        "current_embedding_rows",
        "incomplete_provider_count",
        "provider_failure_count",
        "git_tracked_files",
        "vectors_missing_with_model",
    ):
        _optional_scalar(value, result, field)
    stats = value.get("stats")
    if stats is None:
        result["stats"] = None
    else:
        stats = _dict(stats, phase, "stats")
        result["stats"] = {
            field: _integer(stats.get(field), phase, f"stats.{field}")
            for field in ("files", "nodes", "edges")
        }
    inference = _dict(value.get("inference"), phase, "inference")
    daemons, pids = _sanitize_daemons(inference.get("daemons"), phase)
    result["inference"] = {
        "registry": _sanitize_registry(inference.get("registry"), phase),
        "models": _sanitize_models(inference.get("models"), phase),
        "daemons": daemons,
    }
    return result, pids


def _sanitize_cache(value: dict[str, Any], phase: str) -> dict[str, Any]:
    result = {
        field: _integer(value.get(field), phase, field)
        for field in (
            "managed_bytes",
            "unmanaged_bytes",
            "locked_bytes",
            "quota_bytes",
            "low_water_bytes",
            "ttl_secs",
        )
    }
    entries = _list(value.get("entries"), phase, "entries")
    by_kind: dict[str, dict[str, int]] = {}
    for entry_value in entries:
        entry = _dict(entry_value, phase, "entries[]")
        kind = _string(entry.get("kind"), phase, "entries[].kind")
        bucket = by_kind.setdefault(
            kind, {"entries": 0, "bytes": 0, "locked": 0, "orphaned": 0}
        )
        bucket["entries"] += 1
        bucket["bytes"] += _integer(entry.get("bytes"), phase, "entries[].bytes")
        bucket["locked"] += int(_boolean(entry.get("locked"), phase, "entries[].locked"))
        bucket["orphaned"] += int(
            _boolean(entry.get("orphaned"), phase, "entries[].orphaned")
        )
    result["entries"] = {
        kind: by_kind[kind]
        for kind in sorted(by_kind)
    }
    unmanaged = _list(value.get("unmanaged"), phase, "unmanaged")
    result["unmanaged_entry_count"] = len(unmanaged)
    return result


def _sanitize_semantic(value: dict[str, Any], phase: str) -> dict[str, Any]:
    if _string(value.get("command"), phase, "command") != "semantic-search":
        raise MeasurementError(f"{phase}: unexpected command JSON")
    if _string(value.get("schema_version"), phase, "schema_version") != "greppy.semantic-search.v1":
        raise MeasurementError(f"{phase}: unsupported semantic schema")
    status = _string(value.get("status"), phase, "status")
    mode = _string(value.get("mode"), phase, "mode")
    hits = _list(value.get("hits"), phase, "hits")
    shown = _integer(value.get("shown"), phase, "shown")
    if status != "ok" or mode != "vector" or shown != len(hits) or not hits:
        raise MeasurementError(f"{phase}: semantic result is not a successful vector result")
    summary_count = 0
    summary_line_count = 0
    for hit_value in hits:
        hit = _dict(hit_value, phase, "hits[]")
        summaries = _list(hit.get("summary"), phase, "hits[].summary")
        if not all(isinstance(summary, str) for summary in summaries):
            raise MeasurementError(f"{phase}: invalid semantic summary")
        summary_count += int(bool(summaries))
        summary_line_count += len(summaries)
    if summary_count == 0:
        raise MeasurementError(f"{phase}: semantic result did not exercise purpose summaries")
    result: dict[str, Any] = {
        "status": status,
        "mode": mode,
        "hit_count": len(hits),
        "hits_with_summary": summary_count,
        "summary_line_count": summary_line_count,
        "expand_available": isinstance(value.get("expand_id"), str)
        and bool(value.get("expand_id")),
    }
    for field in (
        "candidate_total",
        "total_exact",
        "retrieved",
        "shown",
        "omitted",
        "unranked_candidates",
    ):
        result[field] = _integer(value.get(field), phase, field)
    result["truncated"] = _boolean(value.get("truncated"), phase, "truncated")
    result["fresh"] = _boolean(value.get("fresh"), phase, "fresh")
    return result


def _sanitize_brief(value: dict[str, Any], phase: str) -> dict[str, Any]:
    if _string(value.get("command"), phase, "command") != "brief":
        raise MeasurementError(f"{phase}: unexpected command JSON")
    if _string(value.get("schema_version"), phase, "schema_version") != "greppy.brief.v1":
        raise MeasurementError(f"{phase}: unsupported brief schema")
    status = _string(value.get("status"), phase, "status")
    definitions = _list(value.get("definitions"), phase, "definitions")
    if status != "ok" or not definitions:
        raise MeasurementError(f"{phase}: brief did not resolve a definition")
    definitions_with_summary = 0
    summary_line_count = 0
    for definition_value in definitions:
        definition = _dict(definition_value, phase, "definitions[]")
        summaries = _list(definition.get("summary"), phase, "definitions[].summary")
        if not all(isinstance(summary, str) for summary in summaries):
            raise MeasurementError(f"{phase}: invalid brief summary")
        definitions_with_summary += int(bool(summaries))
        summary_line_count += len(summaries)
    if definitions_with_summary == 0:
        raise MeasurementError(f"{phase}: brief did not exercise purpose summaries")
    result = {
        "status": status,
        "definition_count": len(definitions),
        "definitions_with_summary": definitions_with_summary,
        "summary_line_count": summary_line_count,
        "expand_available": isinstance(value.get("expand_id"), str)
        and bool(value.get("expand_id")),
    }
    for field in ("callers", "references", "calls"):
        result[f"{field}_count"] = len(_list(value.get(field), phase, field))
    return result


def _series(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        raise MeasurementError("timing series is empty")
    return {
        "samples_ms": list(values),
        "minimum_ms": min(values),
        "median_ms": round(statistics.median(values), 3),
        "maximum_ms": max(values),
    }


def _cpu_description() -> str:
    description = platform.processor().strip()
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("model name") and ":" in line:
                    description = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            output = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stdin=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
                text=True,
            ).strip()
            if output:
                description = output
        except (OSError, subprocess.SubprocessError):
            pass
    return description or platform.machine()


def _rss_bytes(pid: int) -> int | None:
    if pid <= 1:
        return None
    if sys.platform.startswith("linux"):
        try:
            for line in Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines():
                if line.startswith("VmRSS:"):
                    fields = line.split()
                    if len(fields) == 3 and fields[2] == "kB":
                        return int(fields[1]) * 1024
        except (OSError, ValueError):
            return None
    elif sys.platform == "darwin":
        try:
            value = subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(pid)],
                stdin=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
                text=True,
            ).strip()
            return int(value) * 1024 if value else None
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
    return None


def _pid_executable(pid: int) -> Path | None:
    if pid <= 1:
        return None
    if sys.platform.startswith("linux"):
        try:
            return Path(f"/proc/{pid}/exe").resolve(strict=True)
        except OSError:
            return None
    if sys.platform == "darwin":
        try:
            value = subprocess.check_output(
                ["ps", "-o", "comm=", "-p", str(pid)],
                stdin=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
                text=True,
            ).strip()
            return Path(value).resolve(strict=True) if value else None
        except (OSError, subprocess.SubprocessError):
            return None
    if os.name == "nt":
        try:
            import ctypes

            query_limited_information = 0x1000
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(query_limited_information, False, pid)
            if not handle:
                return None
            try:
                capacity = 32_768
                buffer = ctypes.create_unicode_buffer(capacity)
                size = ctypes.c_ulong(capacity)
                if not kernel32.QueryFullProcessImageNameW(
                    handle, 0, buffer, ctypes.byref(size)
                ):
                    return None
                return Path(buffer.value).resolve(strict=True)
            finally:
                kernel32.CloseHandle(handle)
        except (AttributeError, OSError, ValueError):
            return None
    return None


def _terminate_pid(pid: int, expected_executable: Path) -> bool:
    if pid <= 1 or pid == os.getpid():
        return False
    actual = _pid_executable(pid)
    try:
        expected = expected_executable.resolve(strict=True)
    except OSError:
        return False
    if actual is None:
        # Unlike POSIX, os.kill(pid, 0) terminates the process on Windows.
        # An unverified Windows PID must never be touched.
        if os.name == "nt":
            return False
        try:
            os.kill(pid, 0)
        except (OSError, ProcessLookupError):
            return True
        return False
    if actual != expected:
        return False
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if _pid_executable(pid) is None:
                return True
            time.sleep(0.05)
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if _pid_executable(pid) is None:
            return True
        time.sleep(0.05)
    return False


def _terminate_daemons(
    pids: Iterable[int], expected_executable: Path, *, strict: bool
) -> None:
    for pid in sorted(set(pids)):
        if not _terminate_pid(pid, expected_executable) and strict:
            raise MeasurementError("daemon identity or termination check failed")


def _daemon_processes(
    doctor_json: dict[str, Any], doctor: dict[str, Any], phase: str
) -> list[dict[str, Any]]:
    raw_inference = _dict(doctor_json.get("inference"), phase, "inference")
    raw_daemons = _dict(raw_inference.get("daemons"), phase, "inference.daemons")
    processes = []
    for name, daemon in doctor["inference"]["daemons"].items():
        raw_daemon = _dict(raw_daemons.get(name), phase, f"inference.daemons.{name}")
        pid = raw_daemon.get("daemon_pid")
        rss = _rss_bytes(pid) if isinstance(pid, int) and not isinstance(pid, bool) else None
        processes.append(
            {
                "daemon": name,
                "state": daemon.get("state"),
                "pid_reported": daemon.get("pid_reported", False),
                "rss_bytes": rss,
            }
        )
    return processes


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_config(config: Config) -> Config:
    greppy = config.greppy.expanduser().resolve(strict=True)
    repo = config.repo.expanduser().resolve(strict=True)
    output = config.output.expanduser().resolve(strict=False)
    if not greppy.is_file() or not os.access(greppy, os.X_OK):
        raise MeasurementError("greppy binary must be an executable regular file")
    if not repo.is_dir() or not (repo / ".git").exists():
        raise MeasurementError("repository must be a real git repository root")
    if not config.semantic_query or not config.brief_symbol:
        raise MeasurementError("semantic query and brief symbol must be non-empty")
    if config.warm_repeats < 1:
        raise MeasurementError("warm repeat count must be at least one")
    if not re.fullmatch(r"(?:auto|cpu|metal|cuda(?::[0-9]+)?)", config.device):
        raise MeasurementError("device must be auto, cpu, metal, cuda, or cuda:INDEX")
    if not (0 < config.timeout_seconds <= 86_400):
        raise MeasurementError("timeout must be between 0 and 86400 seconds")
    return Config(
        greppy=greppy,
        repo=repo,
        semantic_query=config.semantic_query,
        brief_symbol=config.brief_symbol,
        output=output,
        warm_repeats=config.warm_repeats,
        device=config.device,
        timeout_seconds=config.timeout_seconds,
    )


def measure(config: Config) -> dict[str, Any]:
    config = _validate_config(config)
    store = _private_store(config.repo)
    env = _child_environment(store, config.device)
    binary = str(config.greppy)
    repo = str(config.repo)
    publication_replacements = {
        binary: "<greppy>",
        repo: "<repository>",
        config.semantic_query: "<semantic-query>",
        config.brief_symbol: "<brief-symbol>",
    }
    command_records: list[dict[str, Any]] = []
    daemon_pids: set[int] = set()
    result: dict[str, Any] | None = None
    try:
        before_run = _run_command(
            [binary, "doctor", "--json", "--root", repo],
            phase="doctor_before",
            env=env,
            timeout_seconds=config.timeout_seconds,
            accepted_exit_codes=(1,),
            publication_replacements=publication_replacements,
        )
        command_records.append(before_run.record)
        doctor_before, before_pids = _sanitize_doctor(
            _json_object(before_run.stdout, "doctor_before"), "doctor_before"
        )
        daemon_pids.update(before_pids)
        if doctor_before["store_exists"] or doctor_before["status"] != "no_index":
            raise MeasurementError("doctor_before: private store was not empty")

        index_run = _run_command(
            [binary, "index", repo],
            phase="index",
            env=env,
            timeout_seconds=config.timeout_seconds,
            publication_replacements=publication_replacements,
        )
        command_records.append(index_run.record)

        after_index_run = _run_command(
            [binary, "doctor", "--json", "--root", repo],
            phase="doctor_after_index",
            env=env,
            timeout_seconds=config.timeout_seconds,
            accepted_exit_codes=(0, 1, 73),
            publication_replacements=publication_replacements,
        )
        command_records.append(after_index_run.record)
        after_index_json = _json_object(after_index_run.stdout, "doctor_after_index")
        doctor_after_index, index_pids = _sanitize_doctor(
            after_index_json, "doctor_after_index"
        )
        daemon_pids.update(index_pids)
        daemon_pids.update(_doctor_background_pids(after_index_json))

        embedding_wait_started = time.perf_counter_ns()
        embedding_wait_polls = 0
        after_embedding_json = after_index_json
        doctor_after_embedding = doctor_after_index
        embedding_deadline = time.monotonic() + config.timeout_seconds
        while doctor_after_embedding.get("embedding_complete") is not True:
            background_job = after_embedding_json.get("background_job")
            if isinstance(background_job, dict) and background_job.get("state") == "failed":
                raise MeasurementError("embedding build reported a background failure")
            remaining = embedding_deadline - time.monotonic()
            if remaining <= 0:
                raise MeasurementError("embedding build timed out")
            time.sleep(min(1.0, remaining))
            embedding_wait_polls += 1
            poll_run = _run_command(
                [binary, "doctor", "--json", "--root", repo],
                phase="embedding_wait",
                env=env,
                timeout_seconds=min(config.timeout_seconds, max(remaining, 0.001)),
                accepted_exit_codes=(0, 1, 73),
                iteration=embedding_wait_polls,
                publication_replacements=publication_replacements,
            )
            after_embedding_json = _json_object(
                poll_run.stdout, "embedding_wait"
            )
            doctor_after_embedding, poll_pids = _sanitize_doctor(
                after_embedding_json, "embedding_wait"
            )
            daemon_pids.update(poll_pids)
            daemon_pids.update(_doctor_background_pids(after_embedding_json))
        embedding_wait_ms = round(
            (time.perf_counter_ns() - embedding_wait_started) / 1_000_000,
            3,
        )
        _terminate_daemons(index_pids, config.greppy, strict=True)
        daemon_pids.difference_update(index_pids)

        semantic_argv = [
            binary,
            "semantic-search",
            config.semantic_query,
            "--json",
            "--root",
            repo,
        ]
        semantic_first_run = _run_command(
            semantic_argv,
            phase="semantic_first",
            env=env,
            timeout_seconds=config.timeout_seconds,
            publication_replacements=publication_replacements,
        )
        command_records.append(semantic_first_run.record)
        semantic_result = _sanitize_semantic(
            _json_object(semantic_first_run.stdout, "semantic_first"), "semantic_first"
        )
        semantic_warm: list[float] = []
        for iteration in range(1, config.warm_repeats + 1):
            run = _run_command(
                semantic_argv,
                phase="semantic_warm",
                env=env,
                timeout_seconds=config.timeout_seconds,
                iteration=iteration,
                publication_replacements=publication_replacements,
            )
            command_records.append(run.record)
            _sanitize_semantic(
                _json_object(run.stdout, "semantic_warm"), "semantic_warm"
            )
            semantic_warm.append(run.elapsed_ms)

        after_semantic_run = _run_command(
            [binary, "doctor", "--json", "--root", repo],
            phase="doctor_after_semantic",
            env=env,
            timeout_seconds=config.timeout_seconds,
            publication_replacements=publication_replacements,
        )
        command_records.append(after_semantic_run.record)
        after_semantic_json = _json_object(
            after_semantic_run.stdout, "doctor_after_semantic"
        )
        doctor_after_semantic, semantic_pids = _sanitize_doctor(
            after_semantic_json, "doctor_after_semantic"
        )
        daemon_pids.update(semantic_pids)
        semantic_daemon_processes = _daemon_processes(
            after_semantic_json, doctor_after_semantic, "doctor_after_semantic"
        )
        _terminate_daemons(semantic_pids, config.greppy, strict=True)
        daemon_pids.difference_update(semantic_pids)

        brief_argv = [
            binary,
            "brief",
            config.brief_symbol,
            "--json",
            "--root",
            repo,
        ]
        brief_first_run = _run_command(
            brief_argv,
            phase="brief_first",
            env=env,
            timeout_seconds=config.timeout_seconds,
            publication_replacements=publication_replacements,
        )
        command_records.append(brief_first_run.record)
        brief_result = _sanitize_brief(
            _json_object(brief_first_run.stdout, "brief_first"), "brief_first"
        )
        brief_warm: list[float] = []
        for iteration in range(1, config.warm_repeats + 1):
            run = _run_command(
                brief_argv,
                phase="brief_warm",
                env=env,
                timeout_seconds=config.timeout_seconds,
                iteration=iteration,
                publication_replacements=publication_replacements,
            )
            command_records.append(run.record)
            _sanitize_brief(_json_object(run.stdout, "brief_warm"), "brief_warm")
            brief_warm.append(run.elapsed_ms)

        cache_run = _run_command(
            [binary, "cache", "status", "--json", "--root", repo],
            phase="cache_status",
            env=env,
            timeout_seconds=config.timeout_seconds,
            publication_replacements=publication_replacements,
        )
        command_records.append(cache_run.record)
        cache = _sanitize_cache(
            _json_object(cache_run.stdout, "cache_status"), "cache_status"
        )

        after_run = _run_command(
            [binary, "doctor", "--json", "--root", repo],
            phase="doctor_after",
            env=env,
            timeout_seconds=config.timeout_seconds,
            publication_replacements=publication_replacements,
        )
        command_records.append(after_run.record)
        after_json = _json_object(after_run.stdout, "doctor_after")
        doctor_after, after_pids = _sanitize_doctor(after_json, "doctor_after")
        daemon_pids.update(after_pids)
        if not doctor_after["healthy"] or not doctor_after["store_exists"]:
            raise MeasurementError("doctor_after: indexed runtime is not healthy")

        daemon_rss = _daemon_processes(after_json, doctor_after, "doctor_after")

        result = {
            "schema_version": SCHEMA_VERSION,
            "harness_sha256": _sha256_file(Path(__file__).resolve()),
            "binary": {
                "sha256": _sha256_file(config.greppy),
                "bytes": config.greppy.stat().st_size,
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "cpu": _cpu_description(),
                "logical_cpu_count": os.cpu_count(),
            },
            "configuration": {
                "warm_repeat_count": config.warm_repeats,
                "command_timeout_seconds": config.timeout_seconds,
                "device": config.device,
                "private_store": True,
            },
            "commands": command_records,
            "measurements": {
                "index": {"wall_time_ms": index_run.elapsed_ms},
                "embedding_build": {
                    "wait_wall_time_ms": embedding_wait_ms,
                    "poll_count": embedding_wait_polls,
                    "deferred": embedding_wait_polls > 0,
                },
                "semantic_search": {
                    "first_wall_time_ms": semantic_first_run.elapsed_ms,
                    "warm": _series(semantic_warm),
                    "result": semantic_result,
                    "daemon_processes": semantic_daemon_processes,
                },
                "brief": {
                    "first_wall_time_ms": brief_first_run.elapsed_ms,
                    "warm": _series(brief_warm),
                    "result": brief_result,
                },
                "cache": cache,
                "daemon_processes": daemon_rss,
            },
            "doctor": {
                "before": doctor_before,
                "after_index": doctor_after_index,
                "after_embedding": doctor_after_embedding,
                "after_semantic": doctor_after_semantic,
                "after": doctor_after,
            },
        }
    finally:
        if result is None:
            try:
                cleanup_run = _run_command(
                    [binary, "doctor", "--json", "--root", repo],
                    phase="cleanup_doctor",
                    env=env,
                    timeout_seconds=min(config.timeout_seconds, 5.0),
                    accepted_exit_codes=(0, 1, 73),
                    publication_replacements=publication_replacements,
                )
                cleanup_json = _json_object(cleanup_run.stdout, "cleanup_doctor")
                if cleanup_json.get("command") == "doctor":
                    daemon_pids.update(_doctor_daemon_pids(cleanup_json))
                    daemon_pids.update(_doctor_background_pids(cleanup_json))
            except (MeasurementError, OSError):
                pass
        for pid in daemon_pids:
            _terminate_pid(pid, config.greppy)
        shutil.rmtree(store, ignore_errors=False)
    if result is None:
        raise MeasurementError("measurement did not produce a result")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure Greppy runtime footprint without retaining repository content."
    )
    parser.add_argument("--greppy", required=True, type=Path, help="exact Greppy binary")
    parser.add_argument("--repo", required=True, type=Path, help="real git repository root")
    parser.add_argument("--semantic-query", required=True, help="semantic query (never recorded)")
    parser.add_argument("--brief-symbol", required=True, help="brief symbol (never recorded)")
    parser.add_argument("--output", required=True, type=Path, help="output JSON path")
    parser.add_argument("--warm-repeats", required=True, type=int, help="warm runs per command")
    parser.add_argument(
        "--device",
        default="auto",
        help="shared inference device: auto, cpu, metal, cuda, or cuda:INDEX",
    )
    parser.add_argument(
        "--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="per-command timeout"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = Config(
            greppy=args.greppy,
            repo=args.repo,
            semantic_query=args.semantic_query,
            brief_symbol=args.brief_symbol,
            output=args.output,
            warm_repeats=args.warm_repeats,
            device=args.device,
            timeout_seconds=args.timeout_seconds,
        )
        validated = _validate_config(config)
        result = measure(validated)
        _atomic_write_json(validated.output, result)
    except MeasurementError as error:
        print(f"runtime footprint measurement failed: {error}", file=sys.stderr)
        return 1
    except OSError:
        print("runtime footprint measurement failed: local filesystem operation failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
