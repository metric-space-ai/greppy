#!/usr/bin/env python3
"""Reproducible paired edit-and-test benchmark for Greppy and an explorer arm."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit


HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
# Exploratory model override (forensics runs, e.g. Kimi K3 whose responses
# carry visible reasoning). Gate evidence stays on the registered MiniMax
# defaults; the manifest records whatever was actually used.
_EXT_OVERRIDE = os.environ.get("GREPPY_BENCH_EXTENSION")
PROVIDER_EXTENSION = (
    pathlib.Path(_EXT_OVERRIDE)
    if _EXT_OVERRIDE
    else REPO_ROOT / "bench" / "agent_efficiency" / "minimax-provider.js"
)
TASK_SCHEMA_VERSION = "greppy.agent-coding-tasks.v1"
TASK_SCHEMA_VERSION_V2 = "greppy.agent-coding-tasks.v2"
FLASK_STALE_VENV_MARKER = "/validation-v2/venvs/flask/bin/"
FLASK_LOCAL_PYTHON = ".tox/greppy-bench-venv/bin/python3"
RESULT_SCHEMA_VERSION = "greppy.agent-coding-results.v1"
MANIFEST_SCHEMA_VERSION = "greppy.agent-coding-manifest.v1"
GATE_SCHEMA_VERSION = "greppy.agent-coding-gate.v4"
# v4 (registered in docs/contracts/EDIT_CONTRACT.md): the greppy-edit arm
# must beat the explorer on billed provider dollars (<= 0.80), keep post-
# edit re-reads at <= 0.1 per edit, and hold exact-McNemar correctness
# parity. The v3 contrast (greppy vs explorer, cost <= 1.0) stays as a
# reported check so the navigation arm's non-inferiority remains visible.
EDIT_COST_RATIO_MAX = 0.80
POST_EDIT_REREADS_MAX = 0.1
# Frozen MiniMax-M3 standard-tier billed rates (USD per million tokens,
# snapshot 2026-07-14, <=512k context; output counts are provider-billed and
# include reasoning tokens). The gate compares both arms under the same model
# and prices, so the verdict is invariant to the absolute price level.
PROVIDER_PRICE_USD_PER_MILLION = {
    "uncached_input_tokens": 0.30,
    "output_tokens": 1.20,
    "cache_read_tokens": 0.06,
    "cache_write_tokens": 0.00,
}
PRICING_AS_OF = "2026-07-14"


def provider_cost_usd(agent: dict) -> float:
    return sum(
        float(agent.get(field, 0) or 0) * rate / 1_000_000
        for field, rate in PROVIDER_PRICE_USD_PER_MILLION.items()
    )
HARNESS_VERSION = "2"
DEFAULT_MODEL = os.environ.get("GREPPY_BENCH_MODEL", "MiniMax-M3")
DEFAULT_PROVIDER = os.environ.get("GREPPY_BENCH_PROVIDER", "minimax")
# "off" for the registered MiniMax gate; forensics runs set e.g. "medium" so
# models that RETURN reasoning (Kimi K3) leave it in the recorded trace.
DEFAULT_THINKING = os.environ.get("GREPPY_BENCH_THINKING", "off")
RAW_ROOT = HERE / "raw_traces"
ARMS = ("explorer", "greppy", "greppy-edit")
MIN_COMPLETE_PAIRS = 30
MIN_SOLVED_PAIRS = 20
PROMPT_USAGE_KEYS = ("input", "cacheRead", "cacheWrite", "cacheWrite1h", "cacheWrite5m")

SHARED_SYSTEM_PROMPT = (
    "You are a coding agent working in the current Git worktree. Implement the "
    "user's task directly in this worktree. You may inspect and edit source files "
    "and run shell commands. Do not commit, switch revisions, inspect environment "
    "variables, or print secrets. Keep the change focused. Do not inspect or modify "
    "repository content outside this worktree. "
    "Run the supplied verification command when useful, but the benchmark "
    "harness will run it independently after you finish."
)

EXPLORER_POLICY = (
    "Explore the repository as needed with the normal shell and file-reading tools "
    "available to you, then implement and verify the fix."
)

GREPPY_EDIT_POLICY_TEMPLATE = (
    "This system does not ship grep, ripgrep, or a patch utility for source files. "
    "Greppy replaces all of them; the executable is {greppy}, always with `--root .`. "
    "If you need original grep behavior, call greppy exactly like grep (byte-exact "
    "passthrough). "
    "NAVIGATE: `semantic-search QUERY` for behavior questions; `brief SYMBOL`, "
    "`who-calls`, `callees`, `find-usages` for known symbols. "
    "READ: code here is read by symbol - `read SYMBOL --handle --json` returns the "
    "definition's exact source span plus a HANDLE; do not open whole source files to "
    "find a definition. "
    "EDIT: source files here are changed transactionally through greppy - there is no "
    "apply_patch and no manual patching of source files: "
    "`edit replace-body --symbol SYM --source-file F`, "
    "`edit replace-span --target HANDLE --source-file F`, "
    "`edit rename-call --in SYM --from A --to B`, `edit ensure-import --file P --module M --name N`, "
    "`edit text-cas --file P --old 'OLD' --new 'NEW'` for small exact swaps "
    "(file forms --old-file/--new-file exist too; --source-file - reads stdin for heredocs). "
    "Several edits to the SAME file go into ONE `edit apply --plan PLAN.json` call: a plan "
    "applies all operations against one snapshot, all-or-nothing, and each plan operation "
    "declares the file_sha256 from its handle (a plan without hashes is refused as stale). "
    "Every edit prints a certificate: matched exactly once, hashes before/after, changed "
    "ranges, and syntax verification. The certificate includes the resulting span "
    "(`result_span`) — you have already seen the result; do not re-read the file after a "
    "successful edit. Exit codes: 0 ok/already-satisfied, 10 not "
    "found, 11 ambiguous (candidates listed), 12 stale (re-read the span, then retry), "
    "13 result rejected (postcondition or overlapping operations - nothing was written), "
    "14 validator failed, 16 publish blocked (workspace lock - nothing was written). "
    "Verify the final result with the supplied verification command."
)

GREPPY_POLICY_TEMPLATE = (
    "Use Greppy as the primary code-navigation surface before opening source files. "
    "The executable is {greppy}. Always pass `--root .`. For behavior questions use "
    "`semantic-search QUERY`; for known symbols use `brief SYMBOL`, `who-calls SYMBOL`, "
    "`callees SYMBOL`, `find-usages SYMBOL`, or `impact SYMBOL`; for partial names use "
    "`search-symbols NAME`; for literals use `search-code TEXT`. Inspect the returned "
    "source evidence, then implement and verify the fix. If one Greppy call fails, use "
    "at most one documented fallback and continue with the evidence available."
)


class HarnessError(RuntimeError):
    """Expected benchmark setup or execution failure."""


class SetupCommandError(HarnessError):
    """A setup command failed after its redacted evidence was captured."""

    def __init__(self, summary: dict[str, Any]) -> None:
        super().__init__("setup command failed")
        self.summary = summary


@dataclass(frozen=True)
class ProcessResult:
    returncode: int | None
    stdout: bytes
    stderr: bytes
    wall_seconds: float
    timed_out: bool


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _termination_handler(signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt(f"received signal {signum}")


def install_cleanup_signal_handlers() -> None:
    for name in ("SIGTERM", "SIGHUP"):
        signum = getattr(signal, name, None)
        if signum is not None:
            signal.signal(signum, _termination_handler)


def atomic_write_bytes(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)


def atomic_write_json(path: pathlib.Path, value: Any) -> None:
    atomic_write_bytes(path, json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n")


def redact(data: bytes, secrets: Sequence[str]) -> bytes:
    redacted = data
    for secret in secrets:
        if secret and len(secret) >= 4:
            redacted = redacted.replace(secret.encode("utf-8"), b"<redacted>")
    return redacted


def environment_without_provider_key() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("MINIMAX_API_KEY", None)
    return env


def run_process(
    argv: Sequence[str],
    *,
    cwd: pathlib.Path,
    timeout_seconds: int,
    env: dict[str, str] | None = None,
    input_bytes: bytes | None = None,
) -> ProcessResult:
    started = time.monotonic()
    process = subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(input=input_bytes, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    return ProcessResult(
        returncode=None if timed_out else process.returncode,
        stdout=stdout,
        stderr=stderr,
        wall_seconds=time.monotonic() - started,
        timed_out=timed_out,
    )


def run_checked(
    argv: Sequence[str],
    *,
    cwd: pathlib.Path,
    timeout_seconds: int,
    input_bytes: bytes | None = None,
    operation: str,
) -> ProcessResult:
    result = run_process(argv, cwd=cwd, timeout_seconds=timeout_seconds, input_bytes=input_bytes)
    if result.timed_out:
        raise HarnessError(f"{operation} timed out")
    if result.returncode != 0:
        raise HarnessError(f"{operation} failed")
    return result


def validate_task_document(document: Any) -> list[dict[str, Any]]:
    if not isinstance(document, dict):
        raise HarnessError("task document must be an object")
    if set(document) != {"schema_version", "tasks"}:
        raise HarnessError("task document must contain only schema_version and tasks")
    schema_version = document.get("schema_version")
    if schema_version not in (TASK_SCHEMA_VERSION, TASK_SCHEMA_VERSION_V2):
        raise HarnessError(
            f"schema_version must be {TASK_SCHEMA_VERSION} or {TASK_SCHEMA_VERSION_V2}"
        )
    is_v2 = schema_version == TASK_SCHEMA_VERSION_V2
    tasks = document.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise HarnessError("tasks must be a non-empty array")

    seen: set[str] = set()
    for index, task in enumerate(tasks):
        prefix = f"tasks[{index}]"
        if not isinstance(task, dict):
            raise HarnessError(f"{prefix} must be an object")
        if is_v2:
            required = {
                "id",
                "class",
                "type",
                "repository",
                "setup_commands",
                "test_patch",
                "user_task",
                "test_command",
                "timeout_seconds",
            }
        else:
            required = {
                "id",
                "repository",
                "setup_commands",
                "mutation_patch",
                "user_task",
                "test_command",
                "timeout_seconds",
            }
        if set(task) != required:
            raise HarnessError(f"{prefix} must contain exactly {sorted(required)}")
        if is_v2:
            if task.get("class") not in ("S", "M"):
                raise HarnessError(f"{prefix}.class must be S or M")
            if not isinstance(task.get("type"), str) or not task["type"].strip():
                raise HarnessError(f"{prefix}.type must be a non-empty string")
        task_id = task["id"]
        if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", task_id):
            raise HarnessError(f"{prefix}.id is invalid")
        if task_id in seen:
            raise HarnessError(f"duplicate task id: {task_id}")
        seen.add(task_id)

        repository = task["repository"]
        if not isinstance(repository, dict) or set(repository) != {"url", "commit"}:
            raise HarnessError(f"{prefix}.repository must contain exactly url and commit")
        url = repository["url"]
        commit = repository["commit"]
        if not isinstance(url, str) or not url.strip() or "\n" in url or "\r" in url:
            raise HarnessError(f"{prefix}.repository.url is invalid")
        parsed = urlsplit(url) if "://" in url else None
        if parsed and (parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise HarnessError(f"{prefix}.repository.url must not contain credentials, query, or fragment")
        if not isinstance(commit, str) or not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", commit):
            raise HarnessError(f"{prefix}.repository.commit must be a full 40- or 64-hex object id")
        patch_field = "test_patch" if is_v2 else "mutation_patch"
        for field in (patch_field, "user_task"):
            if not isinstance(task[field], str) or not task[field].strip():
                raise HarnessError(f"{prefix}.{field} must be a non-empty string")
        setup_commands = task["setup_commands"]
        if not isinstance(setup_commands, list):
            raise HarnessError(f"{prefix}.setup_commands must be an array of argv arrays")
        for command_index, setup_command in enumerate(setup_commands):
            if (
                not isinstance(setup_command, list)
                or not setup_command
                or any(not isinstance(part, str) or not part for part in setup_command)
            ):
                raise HarnessError(
                    f"{prefix}.setup_commands[{command_index}] must be a non-empty argv array"
                )
        command = task["test_command"]
        if not isinstance(command, list) or not command or any(not isinstance(part, str) or not part for part in command):
            raise HarnessError(f"{prefix}.test_command must be a non-empty argv array")
        timeout = task["timeout_seconds"]
        if isinstance(timeout, bool) or not isinstance(timeout, int) or not 1 <= timeout <= 7200:
            raise HarnessError(f"{prefix}.timeout_seconds must be an integer from 1 to 7200")
        if is_v2:
            # Normalize: downstream patch application, cross-arm hash checks,
            # and diff capture run on one field regardless of the bank.
            task["mutation_patch"] = task["test_patch"]
            task["task_bank"] = "v2"
            # The validation stage recorded an absolute scratch-venv python for
            # the flask tasks; rewrite it to the task-local venv the
            # setup_commands create (load-time only — the bank file is pinned).
            if command and FLASK_STALE_VENV_MARKER in command[0]:
                task["test_command"] = [FLASK_LOCAL_PYTHON, *command[1:]]
        else:
            task["task_bank"] = "v1"
    return tasks


def load_tasks(path: pathlib.Path, selected_ids: set[str] | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HarnessError(f"cannot read task file: {error.__class__.__name__}") from error
    tasks = validate_task_document(document)
    if selected_ids:
        known = {task["id"] for task in tasks}
        missing = selected_ids - known
        if missing:
            raise HarnessError(f"unknown task ids: {', '.join(sorted(missing))}")
        tasks = [task for task in tasks if task["id"] in selected_ids]
    return document, tasks


def resolve_executable(value: str) -> pathlib.Path:
    candidate = pathlib.Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() or candidate.parent != pathlib.Path(".") else None
    if resolved is None or not resolved.is_file():
        found = shutil.which(value)
        resolved = pathlib.Path(found).resolve() if found else None
    if resolved is None or not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise HarnessError(f"executable not found: {value}")
    return resolved


def executable_version(executable: pathlib.Path) -> str | None:
    try:
        result = run_process(
            [str(executable), "--version"],
            cwd=REPO_ROOT,
            timeout_seconds=10,
            env=environment_without_provider_key(),
        )
    except OSError:
        return None
    if result.timed_out or result.returncode != 0:
        return None
    for line in (result.stdout + result.stderr).decode("utf-8", "replace").splitlines():
        cleaned = re.sub(r"[\x00-\x1f\x7f]+", " ", line).strip()
        if cleaned:
            return cleaned[:200]
    return None


def greppy_source_identity() -> dict[str, Any]:
    commit = run_checked(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        timeout_seconds=30,
        operation="read Greppy source commit",
    ).stdout.decode("ascii", errors="strict").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise HarnessError("Greppy source commit is not a full Git object ID")
    status = run_checked(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        timeout_seconds=30,
        operation="read Greppy tracked worktree status",
    ).stdout
    return {"git_commit": commit, "tracked_worktree_dirty": bool(status.strip())}


def verify_provider_registration(pi_bin: pathlib.Path) -> None:
    # The probe is purely local (no API call), but a shared ~/.pi config
    # dir is lock-contended and node startup is IO-starved when other pi
    # fleets run on the same host (2026-07-17: 73s wall at 3s CPU next to
    # five panel runners; one probe exceeded even 120s at peak load).
    # Probe in a private config dir, tolerate load with retries.
    last = "no attempt"
    for attempt in range(3):
        with tempfile.TemporaryDirectory(prefix="greppy-probe-pi-") as probe_dir:
            env = os.environ.copy()
            env["PI_CODING_AGENT_DIR"] = probe_dir
            result = run_process(
                [
                    str(pi_bin),
                    "--extension",
                    str(PROVIDER_EXTENSION),
                    "--no-extensions",
                    "--list-models",
                    DEFAULT_MODEL,
                ],
                cwd=REPO_ROOT,
                timeout_seconds=120,
                env=env,
            )
        output = (result.stdout + result.stderr).decode("utf-8", "replace")
        pattern = rf"(?m)^{re.escape(DEFAULT_PROVIDER)}\s+{re.escape(DEFAULT_MODEL)}\s"
        if not result.timed_out and result.returncode == 0 and re.search(pattern, output):
            return
        last = f"timed_out={result.timed_out} rc={result.returncode} tail={output[-200:]!r}"
        time.sleep(10 * (attempt + 1))
    raise HarnessError(f"explicit MiniMax provider registration probe failed: {last}")


def clone_pinned_repository(task: dict[str, Any], parent: pathlib.Path) -> pathlib.Path:
    backing = parent / "repo.git"
    timeout = task["timeout_seconds"]
    run_checked(
        ["git", "clone", "--mirror", "--no-local", "--", task["repository"]["url"], str(backing)],
        cwd=parent,
        timeout_seconds=timeout,
        operation="repository clone",
    )
    resolved = run_checked(
        ["git", "--git-dir", str(backing), "rev-parse", "--verify", f"{task['repository']['commit']}^{{commit}}"],
        cwd=parent,
        timeout_seconds=timeout,
        operation="pinned commit verification",
    ).stdout.decode("ascii", "replace").strip()
    if resolved.lower() != task["repository"]["commit"].lower():
        raise HarnessError("repository did not resolve to the pinned commit")
    return backing


@contextlib.contextmanager
def temporary_worktree(
    backing: pathlib.Path,
    commit: str,
    path: pathlib.Path,
    timeout_seconds: int,
) -> Iterator[pathlib.Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    run_checked(
        ["git", "--git-dir", str(backing), "worktree", "add", "--detach", str(path), commit],
        cwd=path.parent,
        timeout_seconds=timeout_seconds,
        operation="worktree creation",
    )
    try:
        yield path
    finally:
        run_process(
            ["git", "--git-dir", str(backing), "worktree", "remove", "--force", str(path)],
            cwd=path.parent,
            timeout_seconds=min(timeout_seconds, 120),
        )
        shutil.rmtree(path, ignore_errors=True)
        run_process(
            ["git", "--git-dir", str(backing), "worktree", "prune", "--expire", "now"],
            cwd=path.parent,
            timeout_seconds=min(timeout_seconds, 120),
        )
        if path.exists():
            raise HarnessError("worktree cleanup failed")


def apply_mutation(worktree: pathlib.Path, patch: str, timeout_seconds: int) -> None:
    patch_bytes = patch.encode("utf-8")
    base = ["git", "apply", "--binary", "--whitespace=nowarn"]
    run_checked(
        [*base, "--check", "-"],
        cwd=worktree,
        timeout_seconds=timeout_seconds,
        input_bytes=patch_bytes,
        operation="mutation patch check",
    )
    run_checked(
        [*base, "-"],
        cwd=worktree,
        timeout_seconds=timeout_seconds,
        input_bytes=patch_bytes,
        operation="mutation patch application",
    )


def capture_binary_diff(worktree: pathlib.Path, base_commit: str, timeout_seconds: int) -> bytes:
    # Intent-to-add makes untracked files visible without staging their contents.
    run_checked(
        ["git", "add", "--intent-to-add", "--all", "--", "."],
        cwd=worktree,
        timeout_seconds=timeout_seconds,
        operation="untracked-file registration",
    )
    return run_checked(
        ["git", "diff", "--binary", "--full-index", "--no-ext-diff", base_commit, "--"],
        cwd=worktree,
        timeout_seconds=timeout_seconds,
        operation="binary diff capture",
    ).stdout


@dataclass(frozen=True)
class TestFileState:
    content: bytes | None
    mode: int | None


def snapshot_test_files(worktree: pathlib.Path, paths: Sequence[str]) -> dict[str, TestFileState]:
    snapshots: dict[str, TestFileState] = {}
    for relative in paths:
        path = worktree / relative
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise HarnessError(f"test patch path is not a regular file: {relative}")
        snapshots[relative] = (
            TestFileState(path.read_bytes(), path.stat().st_mode & 0o777)
            if path.is_file()
            else TestFileState(None, None)
        )
    return snapshots


def restore_test_files(worktree: pathlib.Path, snapshots: dict[str, TestFileState]) -> list[str]:
    modified: list[str] = []
    for relative, expected in snapshots.items():
        path = worktree / relative
        regular = path.is_file() and not path.is_symlink()
        current_content = path.read_bytes() if regular else None
        current_mode = path.stat().st_mode & 0o777 if regular else None
        if current_content == expected.content and current_mode == expected.mode:
            continue
        modified.append(relative)
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()
        if expected.content is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_bytes(path, expected.content)
            if expected.mode is not None:
                path.chmod(expected.mode)
    return modified


def patch_touched_paths(patch: str) -> list[str]:
    """Return normalized repository paths named by a git-style patch."""
    paths: set[str] = set()
    for line in patch.splitlines():
        if not line.startswith("diff --git "):
            continue
        try:
            fields = shlex.split(line)
        except ValueError as error:
            raise HarnessError("test patch has an invalid diff header") from error
        if len(fields) != 4:
            raise HarnessError("test patch has an invalid diff header")
        for field, prefix in ((fields[2], "a/"), (fields[3], "b/")):
            if field == "/dev/null":
                continue
            path = field[len(prefix):] if field.startswith(prefix) else field
            candidate = pathlib.PurePosixPath(path)
            if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
                raise HarnessError("test patch contains an unsafe path")
            paths.add(candidate.as_posix())
    if not paths:
        raise HarnessError("test patch contains no diff paths")
    return sorted(paths)


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _clean_output_lines(output: bytes) -> list[str]:
    text = _ANSI_ESCAPE_RE.sub("", output.decode("utf-8", "replace"))
    return [line.strip() for line in text.splitlines() if line.strip()]


def _first_matching_line(lines: Sequence[str], pattern: str, flags: int = 0) -> str | None:
    expression = re.compile(pattern, flags)
    return next((line for line in lines if expression.search(line)), None)


def classify_v2_patched_failure(
    test_command: Sequence[str],
    output: bytes,
    patched_paths: Sequence[str],
    *,
    spawn_error: bool = False,
) -> dict[str, Any]:
    """Accept only framework-proven v2 test failures, never bare nonzero exits."""
    lines = _clean_output_lines(output)
    command = " ".join(test_command).lower()
    collected_line = _first_matching_line(lines, r"\bcollected\s+[1-9]\d*\s+items?\b", re.IGNORECASE)

    if spawn_error:
        return {
            "verdict": "preflight_infra_failure",
            "framework": None,
            "signature": "test_process_spawn_error",
            "proof_line": lines[0] if lines else "test process could not start",
        }

    hard_infra_signatures = (
        ("command_not_found", r"\bcommand not found\b"),
        ("missing_file_or_interpreter", r"\bno such file or directory\b"),
        ("missing_python_module", r"\bno module named\s+(?:['\"])?(?:pytest|unittest|nose)\b"),
        ("bad_interpreter", r"\bbad interpreter\b"),
        ("cannot_execute", r"\bcannot execute(?: binary file|: required file not found)?\b"),
        ("windows_command_not_found", r"\bis not recognized as an internal or external command\b"),
    )
    for signature, pattern in hard_infra_signatures:
        proof = _first_matching_line(lines, pattern, re.IGNORECASE)
        if proof is not None:
            return {
                "verdict": "preflight_infra_failure",
                "framework": None,
                "signature": signature,
                "proof_line": proof,
            }

    framework: str | None = None
    proof_line: str | None = None
    supporting_line: str | None = None
    signature: str | None = None
    if "pytest" in command:
        framework = "pytest"
        failed_line = _first_matching_line(lines, r"\b(?:FAILED|failed)\b")
        if collected_line and failed_line:
            proof_line = failed_line
            supporting_line = collected_line
            signature = "pytest_failed_with_collected_tests"
    elif re.search(r"(?:^|\s)go\s+test(?:\s|$)", command):
        framework = "go-test"
        proof_line = _first_matching_line(lines, r"^(?:--- FAIL:|FAIL\t)")
        if proof_line:
            signature = "go_test_fail_marker"
    elif re.search(r"(?:^|\s)cargo\s+test(?:\s|$)", command):
        framework = "cargo-test"
        proof_line = _first_matching_line(lines, r"\btest result: FAILED\b")
        if proof_line:
            signature = "cargo_test_failed_result"
        else:
            compile_line = _first_matching_line(lines, r"^error: could not compile\b", re.IGNORECASE)
            path_line = next(
                (line for line in lines if any(path in line.replace("\\", "/") for path in patched_paths)),
                None,
            )
            if compile_line and path_line:
                proof_line = path_line
                supporting_line = compile_line
                signature = "cargo_patched_test_compile_failure"
    elif "vitest" in command or re.search(r"(?:^|\s)jest(?:\s|$)", command):
        framework = "vitest-jest"
        proof_line = _first_matching_line(lines, r"(?:✕|×|\bfailed\b)", re.IGNORECASE)
        if proof_line:
            signature = "vitest_jest_failure_marker"
    elif re.search(r"(?:^|\s)(?:mvn|mvnw)(?:\s|$)", command) or "surefire" in command:
        framework = "maven-surefire"
        proof_line = _first_matching_line(lines, r"Tests run:.*Failures:\s*[1-9]\d*", re.IGNORECASE)
        if proof_line:
            signature = "surefire_nonzero_failures"

    if proof_line is not None:
        classification = {
            "verdict": "test_failure",
            "framework": framework,
            "signature": signature,
            "proof_line": proof_line,
        }
        if supporting_line is not None:
            classification["supporting_line"] = supporting_line
        return classification

    import_line = _first_matching_line(lines, r"\b(?:ImportError|ModuleNotFoundError)\b")
    if import_line is not None and collected_line is None:
        return {
            "verdict": "preflight_infra_failure",
            "framework": framework,
            "signature": "import_error_without_collected_tests",
            "proof_line": import_line,
        }
    compile_line = _first_matching_line(lines, r"^error: could not compile\b", re.IGNORECASE)
    if compile_line is not None:
        return {
            "verdict": "preflight_infra_failure",
            "framework": framework,
            "signature": "compile_failure_outside_patched_tests",
            "proof_line": compile_line,
        }
    return {
        "verdict": "preflight_infra_failure",
        "framework": framework,
        "signature": "no_framework_failure_evidence",
        "proof_line": lines[0] if lines else "nonzero exit without test-framework output",
    }


def parse_pi_jsonl(raw: bytes) -> dict[str, Any]:
    input_tokens = uncached_input_tokens = output_tokens = tool_calls = source_opens = turns = 0
    cache_read = cache_write = 0
    edit_calls = 0
    post_edit_source_opens = 0
    edited_files: set[str] = set()
    error: str | None = None
    last_error_text: str | None = None
    for line in raw.decode("utf-8", "replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "turn_end":
            continue
        turns += 1
        results = event.get("toolResults") or []
        tool_calls += len(results)
        message = event.get("message") or {}
        usage = message.get("usage") or {}
        uncached_input_tokens += int(usage.get("input", 0) or 0)
        input_tokens += sum(int(usage.get(key, 0) or 0) for key in PROMPT_USAGE_KEYS)
        output_tokens += int(usage.get("output", 0) or 0)
        cache_read += int(usage.get("cacheRead", 0) or 0)
        cache_write += sum(int(usage.get(key, 0) or 0) for key in ("cacheWrite", "cacheWrite1h", "cacheWrite5m"))
        for item in message.get("content") or []:
            if not isinstance(item, dict) or item.get("type") != "toolCall":
                continue
            name = item.get("name")
            opened_file = None
            if name == "read":
                source_opens += 1
                opened_file = str((item.get("arguments") or {}).get("path", ""))
            elif name == "bash":
                command = str((item.get("arguments") or {}).get("command", ""))
                if re.search(r"(^|[;&|]\s*)(cat|head|tail|sed\s+-n)\s", command):
                    source_opens += 1
                    m = re.search(r"(?:cat|head|tail|sed\s+-n\s+\S+)\s+([\w./-]+)", command)
                    opened_file = m.group(1) if m else None
                if re.search(r"greppy[^\n]*\bedit\b", command):
                    edit_calls += 1
                    m = re.search(r"--file\s+([\w./-]+)", command)
                    if m:
                        edited_files.add(m.group(1))
                    # symbol-adressierte edits: datei unbekannt bis zertifikat;
                    # konservativ nicht zaehlen
            if opened_file and opened_file in edited_files:
                post_edit_source_opens += 1
        if message.get("errorMessage"):
            error = str(message["errorMessage"])
            last_error_text = error[:300]
    return {
        "input_tokens": input_tokens,
        "uncached_input_tokens": uncached_input_tokens,
        "output_tokens": output_tokens,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "tool_calls": tool_calls,
        "source_opens": source_opens,
        "turns": turns,
        "reported_error": bool(error),
        "edit_calls": edit_calls,
        "post_edit_source_opens": post_edit_source_opens,
        # provider error text, redacted upstream with the rest of stdout;
        # 10 consecutive identical failures on one task are invisible
        # without it (2026-07-17: serde-range-start-field, both arms)
        "last_error_text": last_error_text,
    }


def shared_user_prompt(task: dict[str, Any]) -> str:
    command = shlex.join(task["test_command"])
    return f"Task:\n{task['user_task'].strip()}\n\nVerification command:\n{command}\n"


# Per-arm pi tool surface. The greppy-edit arm's displacement prompt states
# "there is no apply_patch and no manual patching"; with pi's builtin
# edit/write/read in the tool palette that statement is visibly false and
# gets ignored (trace forensics 2026-07-17: the arm's agent used builtin
# edit and never called greppy). Displacement only works when the
# environment matches the claim - the MSCC panel proves prompt-driven
# adoption (78-87% greppy usage) exactly where the prompt's claim is true.
# The edit arm therefore ships bash only: greppy read/edit is the paved
# road, bash remains the honest escape hatch (and bash-side manual edits
# stay measurable as fallback_edits in trace forensics).
ARM_TOOLS = {
    "explorer": "bash,read,edit,write",
    "greppy": "bash,read,edit,write",
    "greppy-edit": "bash",
}


def system_prompt(arm: str, greppy_bin: pathlib.Path) -> str:
    if arm == "explorer":
        policy = EXPLORER_POLICY
    elif arm == "greppy":
        policy = GREPPY_POLICY_TEMPLATE.format(greppy=shlex.quote(str(greppy_bin)))
    elif arm == "greppy-edit":
        policy = GREPPY_EDIT_POLICY_TEMPLATE.format(greppy=shlex.quote(str(greppy_bin)))
    else:
        raise HarnessError(f"unknown arm: {arm}")
    return f"{SHARED_SYSTEM_PROMPT}\n\nNavigation treatment:\n{policy}"


def run_pi_agent(
    *,
    arm: str,
    task: dict[str, Any],
    worktree: pathlib.Path,
    store_dir: pathlib.Path,
    pi_config_dir: pathlib.Path,
    pi_bin: pathlib.Path,
    greppy_bin: pathlib.Path,
    raw_dir: pathlib.Path,
    secrets: Sequence[str],
) -> tuple[dict[str, Any], ProcessResult]:
    env = os.environ.copy()
    env["GREPPY_STORE_DIR"] = str(store_dir)
    env["PI_CODING_AGENT_DIR"] = str(pi_config_dir)
    # Pin bare `greppy` on PATH to the binary under test. The system prompt
    # gives the agent an absolute path, but agents drift to bare `greppy`;
    # without this they silently hit a stale system/ctox shim whose passthrough
    # routes unknown subcommands to grep -> contaminated measurement.
    binshim = raw_dir / ".binshim"
    binshim.mkdir(parents=True, exist_ok=True)
    shim = binshim / "greppy"
    if shim.is_symlink() or shim.exists():
        shim.unlink()
    shim.symlink_to(greppy_bin.resolve())
    env["PATH"] = f"{binshim}{os.pathsep}{env.get('PATH', '')}"
    argv = [
        str(pi_bin),
        "-p",
        "--extension",
        str(PROVIDER_EXTENSION),
        "--provider",
        DEFAULT_PROVIDER,
        "--model",
        DEFAULT_MODEL,
        "--mode",
        "json",
        "--no-session",
        "--thinking",
        DEFAULT_THINKING,
        "--tools",
        ARM_TOOLS[arm],
        "--no-context-files",
        "--no-skills",
        "--no-prompt-templates",
        # Pi 0.80.2 was verified to keep explicit --extension entries active;
        # --no-extensions disables discovery only.
        "--no-extensions",
        "--approve",
        "--append-system-prompt",
        system_prompt(arm, greppy_bin),
        shared_user_prompt(task),
    ]
    result = run_process(
        argv,
        cwd=worktree,
        timeout_seconds=task["timeout_seconds"],
        env=env,
    )
    stdout = redact(result.stdout, secrets)
    stderr = redact(result.stderr, secrets)
    atomic_write_bytes(raw_dir / "agent.jsonl", stdout)
    atomic_write_bytes(raw_dir / "agent.stderr", stderr)
    metrics = parse_pi_jsonl(stdout)
    metrics.update(
        {
            "wall_seconds": round(result.wall_seconds, 3),
            "timed_out": result.timed_out,
            "return_code": result.returncode,
            "success": not result.timed_out and result.returncode == 0 and not metrics["reported_error"] and metrics["turns"] > 0,
            "stdout_sha256": sha256_bytes(stdout),
            "stderr_sha256": sha256_bytes(stderr),
        }
    )
    return metrics, result


def process_summary(result: ProcessResult, output: bytes) -> dict[str, Any]:
    return {
        "return_code": result.returncode,
        "timed_out": result.timed_out,
        "wall_seconds": round(result.wall_seconds, 3),
        "output_sha256": sha256_bytes(output),
    }


def setup_summary(commands: list[dict[str, Any]], success: bool) -> dict[str, Any]:
    core = {
        "success": success,
        "command_count": len(commands),
        "wall_seconds": round(sum(float(command["wall_seconds"]) for command in commands), 3),
        "excluded_from_agent_wall": True,
        "commands": commands,
    }
    return {**core, "summary_sha256": sha256_bytes(canonical_json_bytes(core))}


def run_setup_commands(
    *,
    task: dict[str, Any],
    worktree: pathlib.Path,
    store_dir: pathlib.Path,
    raw_dir: pathlib.Path,
    secrets: Sequence[str],
) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    store_dir.mkdir(parents=True, exist_ok=True)
    env = environment_without_provider_key()
    env["GREPPY_STORE_DIR"] = str(store_dir)
    commands: list[dict[str, Any]] = []
    for index, argv in enumerate(task["setup_commands"]):
        started = time.monotonic()
        spawn_error = False
        try:
            result = run_process(
                argv,
                cwd=worktree,
                timeout_seconds=task["timeout_seconds"],
                env=env,
            )
        except OSError:
            spawn_error = True
            result = ProcessResult(
                returncode=None,
                stdout=b"",
                stderr=b"setup process could not start\n",
                wall_seconds=time.monotonic() - started,
                timed_out=False,
            )
        output = redact(result.stdout + result.stderr, secrets)
        atomic_write_bytes(raw_dir / f"setup-{index:02d}.log", output)
        status = {
            "return_code": result.returncode,
            "timed_out": result.timed_out,
            "spawn_error": spawn_error,
        }
        command = {
            "index": index,
            "argv_sha256": sha256_bytes(canonical_json_bytes(argv)),
            **process_summary(result, output),
            "spawn_error": spawn_error,
            "status_sha256": sha256_bytes(canonical_json_bytes(status)),
        }
        commands.append(command)
        if spawn_error or result.timed_out or result.returncode != 0:
            raise SetupCommandError(setup_summary(commands, False))
    return setup_summary(commands, True)


def run_mutation_preflight(
    task: dict[str, Any],
    backing: pathlib.Path,
    task_tmp: pathlib.Path,
    raw_dir: pathlib.Path,
    secrets: Sequence[str],
) -> dict[str, Any]:
    worktree_path = task_tmp / "preflight-worktree"
    preflight_store = task_tmp / "preflight-greppy-store"
    with temporary_worktree(backing, task["repository"]["commit"], worktree_path, task["timeout_seconds"]) as worktree:
        try:
            setup = run_setup_commands(
                task=task,
                worktree=worktree,
                store_dir=preflight_store,
                raw_dir=raw_dir / "setup",
                secrets=secrets,
            )
        except SetupCommandError as error:
            return {
                "valid": False,
                "failure_kind": "setup_failed",
                "setup": error.summary,
                "clean_test": None,
                "mutated_test": None,
            }

        test_env = environment_without_provider_key()
        test_env["GREPPY_STORE_DIR"] = str(preflight_store)
        if task.get("task_bank") == "v2":
            # v2 real-commit tasks: the deciding tests arrive WITH the test
            # patch, so there is no clean-source pass to prove. The proof is
            # solely that the patched tests fail before the agent works.
            clean_summary = None
        else:
            clean_result = run_process(
                task["test_command"],
                cwd=worktree,
                timeout_seconds=task["timeout_seconds"],
                env=test_env,
            )
            clean_output = redact(clean_result.stdout + clean_result.stderr, secrets)
            atomic_write_bytes(raw_dir / "preflight-clean-test.log", clean_output)
            clean_summary = process_summary(clean_result, clean_output)
            if clean_result.timed_out or clean_result.returncode != 0:
                return {
                    "valid": False,
                    "failure_kind": "clean_source_test_failed",
                    "setup": setup,
                    "clean_test": clean_summary,
                    "mutated_test": None,
                }

        apply_mutation(worktree, task["mutation_patch"], task["timeout_seconds"])
        mutation_diff = capture_binary_diff(worktree, task["repository"]["commit"], task["timeout_seconds"])
        mutated_spawn_error = False
        try:
            mutated_result = run_process(
                task["test_command"],
                cwd=worktree,
                timeout_seconds=task["timeout_seconds"],
                env=test_env,
            )
        except OSError as error:
            mutated_spawn_error = True
            mutated_result = ProcessResult(
                returncode=None,
                stdout=b"",
                stderr=f"test process could not start: {error.__class__.__name__}\n".encode("utf-8"),
                wall_seconds=0.0,
                timed_out=False,
            )
        mutated_output = redact(mutated_result.stdout + mutated_result.stderr, secrets)
        atomic_write_bytes(raw_dir / "preflight-mutated-test.log", mutated_output)
    mutated_summary = process_summary(mutated_result, mutated_output)
    classification = None
    if task.get("task_bank") == "v2" and not mutated_result.timed_out and mutated_result.returncode != 0:
        classification = classify_v2_patched_failure(
            task["test_command"],
            mutated_output,
            patch_touched_paths(task["mutation_patch"]),
            spawn_error=mutated_spawn_error,
        )
        valid = classification["verdict"] == "test_failure"
        failure_kind = None if valid else "preflight_infra_failure"
    else:
        valid = not mutated_result.timed_out and mutated_result.returncode not in (None, 0)
        failure_kind = None if valid else (
            "patched_test_did_not_fail" if task.get("task_bank") == "v2" else "mutation_test_did_not_fail"
        )
    return {
        "valid": valid,
        "failure_kind": failure_kind,
        "setup": setup,
        "clean_test": clean_summary,
        "mutated_test": mutated_summary,
        "patched_failure_classification": classification,
        "mutation_diff_sha256": sha256_bytes(mutation_diff),
    }


def run_arm(
    *,
    arm: str,
    task: dict[str, Any],
    backing: pathlib.Path,
    task_tmp: pathlib.Path,
    raw_dir: pathlib.Path,
    pi_bin: pathlib.Path,
    greppy_bin: pathlib.Path,
    warm_greppy: bool,
    expected_mutation_hash: str,
    secrets: Sequence[str],
) -> dict[str, Any]:
    arm_tmp = task_tmp / arm
    worktree_path = arm_tmp / "worktree"
    store_dir = arm_tmp / "greppy-store"
    pi_config_dir = arm_tmp / "pi-config"
    store_dir.mkdir(parents=True, exist_ok=True)
    pi_config_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    with temporary_worktree(backing, task["repository"]["commit"], worktree_path, task["timeout_seconds"]) as worktree:
        setup = run_setup_commands(
            task=task,
            worktree=worktree,
            store_dir=store_dir,
            raw_dir=raw_dir / "setup",
            secrets=secrets,
        )
        apply_mutation(worktree, task["mutation_patch"], task["timeout_seconds"])
        mutation_diff = capture_binary_diff(worktree, task["repository"]["commit"], task["timeout_seconds"])
        mutation_hash = sha256_bytes(mutation_diff)
        if mutation_hash != expected_mutation_hash:
            raise HarnessError("arm mutation differs from preflight mutation")
        test_file_snapshots = (
            snapshot_test_files(worktree, patch_touched_paths(task["test_patch"]))
            if task.get("task_bank") == "v2"
            else {}
        )

        warmup: dict[str, Any] = {"enabled": bool(warm_greppy and arm == "greppy")}
        if warmup["enabled"]:
            env = environment_without_provider_key()
            env["GREPPY_STORE_DIR"] = str(store_dir)
            warm_result = run_process(
                [str(greppy_bin), "--root", str(worktree), "index", str(worktree)],
                cwd=worktree,
                timeout_seconds=task["timeout_seconds"],
                env=env,
            )
            warm_output = redact(warm_result.stdout + warm_result.stderr, secrets)
            atomic_write_bytes(raw_dir / "greppy-warmup.log", warm_output)
            warmup.update(process_summary(warm_result, warm_output))
            if warm_result.timed_out or warm_result.returncode != 0:
                raise HarnessError("Greppy warmup failed")

        agent, _ = run_pi_agent(
            arm=arm,
            task=task,
            worktree=worktree,
            store_dir=store_dir,
            pi_config_dir=pi_config_dir,
            pi_bin=pi_bin,
            greppy_bin=greppy_bin,
            raw_dir=raw_dir,
            secrets=secrets,
        )
        pretest_diff = capture_binary_diff(worktree, task["repository"]["commit"], task["timeout_seconds"])
        safe_pretest_diff = redact(pretest_diff, secrets)
        atomic_write_bytes(raw_dir / "pretest.patch", safe_pretest_diff)
        if safe_pretest_diff != pretest_diff:
            raise HarnessError("provider key appeared in agent diff")
        modified_test_files = restore_test_files(worktree, test_file_snapshots)

        test_env = environment_without_provider_key()
        test_env["GREPPY_STORE_DIR"] = str(store_dir)
        test_result = run_process(
            task["test_command"],
            cwd=worktree,
            timeout_seconds=task["timeout_seconds"],
            env=test_env,
        )
        test_output = redact(test_result.stdout + test_result.stderr, secrets)
        atomic_write_bytes(raw_dir / "test.log", test_output)
        final_diff = capture_binary_diff(worktree, task["repository"]["commit"], task["timeout_seconds"])
        safe_final_diff = redact(final_diff, secrets)
        atomic_write_bytes(raw_dir / "final.patch", safe_final_diff)
        if safe_final_diff != final_diff:
            raise HarnessError("provider key appeared in final diff")
        final_head = run_checked(
            ["git", "rev-parse", "HEAD"],
            cwd=worktree,
            timeout_seconds=task["timeout_seconds"],
            operation="final HEAD capture",
        ).stdout.decode("ascii", "replace").strip()

        row = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "task_id": task["id"],
            "arm": arm,
            "valid": agent_result_is_valid(agent),
            "correctness": not test_result.timed_out and test_result.returncode == 0,
            "agent": agent,
            "setup": setup,
            "test": process_summary(test_result, test_output),
            "warmup": warmup,
            "mutation_diff_sha256": mutation_hash,
            "test_files_modified_by_agent": bool(modified_test_files),
            "test_files_modified_by_agent_paths": modified_test_files,
            "pretest_diff_sha256": sha256_bytes(pretest_diff),
            "pretest_diff_bytes": len(pretest_diff),
            "final_diff_sha256": sha256_bytes(final_diff),
            "final_diff_bytes": len(final_diff),
            "final_head": final_head,
            "worktree_cleaned": True,
            "completed_at": utc_now(),
        }
    row["worktree_cleaned"] = not worktree_path.exists()
    return row


def agent_result_is_valid(agent: dict[str, Any]) -> bool:
    return agent.get("success") is True


def exact_regression_p_value(baseline_only: int, candidate_only: int) -> float:
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    return sum(math.comb(discordant, k) for k in range(baseline_only, discordant + 1)) / (2**discordant)


def ratio(numerator: float, denominator: float) -> float | None:
    return round(numerator / denominator, 6) if denominator > 0 else None


def grade_results(rows: Sequence[dict[str, Any]], expected_task_ids: Sequence[str]) -> dict[str, Any]:
    by_key = {(row.get("task_id"), row.get("arm")): row for row in rows}
    complete_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    invalid_or_missing: list[str] = []
    nav_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for task_id in expected_task_ids:
        baseline = by_key.get((task_id, "explorer"))
        candidate = by_key.get((task_id, "greppy-edit"))
        nav = by_key.get((task_id, "greppy"))
        if not baseline or not candidate or not baseline.get("valid") or not candidate.get("valid"):
            invalid_or_missing.append(task_id)
            continue
        complete_pairs.append((baseline, candidate))
        if nav and nav.get("valid") and baseline.get("valid"):
            nav_pairs.append((baseline, nav))

    baseline_only = sum(bool(base["correctness"]) and not bool(cand["correctness"]) for base, cand in complete_pairs)
    candidate_only = sum(bool(cand["correctness"]) and not bool(base["correctness"]) for base, cand in complete_pairs)
    p_value = exact_regression_p_value(baseline_only, candidate_only)
    solved = [(base, cand) for base, cand in complete_pairs if base["correctness"] and cand["correctness"]]

    base_tools = sum(row[0]["agent"]["tool_calls"] for row in solved)
    cand_tools = sum(row[1]["agent"]["tool_calls"] for row in solved)
    base_source_opens = sum(row[0]["agent"]["source_opens"] for row in solved)
    cand_source_opens = sum(row[1]["agent"]["source_opens"] for row in solved)
    base_input = sum(row[0]["agent"]["input_tokens"] for row in solved)
    cand_input = sum(row[1]["agent"]["input_tokens"] for row in solved)
    tool_ratio = ratio(cand_tools, base_tools)
    source_open_ratio = ratio(cand_source_opens, base_source_opens)
    input_ratio = ratio(cand_input, base_input)
    # Re-registered 2026-07-16 (owner decision): the release claim this gate
    # guards is cost non-inferiority at correctness parity on edit tasks,
    # measured in billed provider dollars (reasoning tokens included). The
    # former all-token-ratios <= 0.8 bar tied a navigation-scale efficiency
    # claim to the edit regime; that claim is carried by the agent-efficiency
    # gate and the MSCC panel, where it is measured at full strength.
    base_cost = sum(provider_cost_usd(row[0]["agent"]) for row in solved)
    cand_cost = sum(provider_cost_usd(row[1]["agent"]) for row in solved)
    cost_ratio = ratio(cand_cost, base_cost)
    # v4: the edit arm must be decisively cheaper, not merely non-inferior
    efficiency_pass = cost_ratio is not None and cost_ratio <= EDIT_COST_RATIO_MAX
    # edit-loop metrics over the edit arm's solved rows
    edit_calls_total = sum(row[1]["agent"].get("edit_calls", 0) for row in solved)
    post_edit_rereads = sum(row[1]["agent"].get("post_edit_source_opens", 0) for row in solved)
    reread_rate = post_edit_rereads / edit_calls_total if edit_calls_total else 0.0
    reread_pass = reread_rate <= POST_EDIT_REREADS_MAX
    # navigation arm (v3 semantics) stays visible as a reported check
    nav_solved = [(b, n) for b, n in nav_pairs if b["correctness"] and n["correctness"]]
    nav_cost_ratio = ratio(
        sum(provider_cost_usd(row[1]["agent"]) for row in nav_solved),
        sum(provider_cost_usd(row[0]["agent"]) for row in nav_solved),
    ) if nav_solved else None
    no_significant_regression = p_value >= 0.05
    observed_not_lower = candidate_only >= baseline_only
    credited_wall_wins = sum(cand["agent"]["wall_seconds"] < base["agent"]["wall_seconds"] for base, cand in solved)
    wall_ratio = ratio(
        sum(cand["agent"]["wall_seconds"] for base, cand in solved),
        sum(base["agent"]["wall_seconds"] for base, cand in solved),
    )
    complete = not invalid_or_missing and len(complete_pairs) == len(expected_task_ids)
    sample_size_pass = len(complete_pairs) >= MIN_COMPLETE_PAIRS and len(solved) >= MIN_SOLVED_PAIRS
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "passed": complete
        and sample_size_pass
        and observed_not_lower
        and no_significant_regression
        and efficiency_pass
        and reread_pass,
        "complete": complete,
        "sample_size": {
            "minimum_complete_pairs": MIN_COMPLETE_PAIRS,
            "minimum_solved_pairs": MIN_SOLVED_PAIRS,
            "passes": sample_size_pass,
        },
        "invalid_or_missing_task_ids": invalid_or_missing,
        "complete_pair_count": len(complete_pairs),
        "solved_pair_count": len(solved),
        "correctness": {
            "baseline_only_passes": baseline_only,
            "greppy_only_passes": candidate_only,
            "one_sided_exact_mcnemar_p": round(p_value, 8),
            "alpha": 0.05,
            "greppy_observed_correctness_not_lower": observed_not_lower,
            "no_significant_regression": no_significant_regression,
        },
        "edit_loop": {
            "edit_calls_total": edit_calls_total,
            "post_edit_source_opens": post_edit_rereads,
            "post_edit_reread_rate": round(reread_rate, 4),
            "threshold": POST_EDIT_REREADS_MAX,
            "passes": reread_pass,
        },
        "navigation_arm_v3_check": {
            "greppy_to_explorer_provider_cost": nav_cost_ratio,
            "threshold_ratio": 1.0,
            "is_gate_metric": False,
        },
        "cost_on_solved_pairs": {
            "metric": "provider_cost_usd",
            "pricing_as_of": PRICING_AS_OF,
            "threshold_ratio": EDIT_COST_RATIO_MAX,
            "greppy_edit_to_explorer_provider_cost": cost_ratio,
            "greppy_total_usd": round(cand_cost, 6),
            "explorer_total_usd": round(base_cost, 6),
            "passes": efficiency_pass,
        },
        "token_ratios_on_solved_pairs": {
            "greppy_to_explorer_tool_calls": tool_ratio,
            "greppy_to_explorer_source_opens": source_open_ratio,
            "greppy_to_explorer_input_tokens": input_ratio,
            "is_gate_metric": False,
        },
        "wall_time_on_solved_pairs_only": {
            "greppy_to_explorer": wall_ratio,
            "credited_greppy_wins": credited_wall_wins,
            "is_gate_metric": False,
        },
        "failed_tests_receive_speed_credit": False,
    }


def deterministic_arm_order(task_id: str) -> list[str]:
    return list(ARMS if hashlib.sha256(task_id.encode("utf-8")).digest()[0] % 2 == 0 else reversed(ARMS))


def task_manifest_entry(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": task["id"],
        "repository": task["repository"],
        "task_bank": task.get("task_bank", "v1"),
        "mutation_patch_sha256": sha256_text(task["mutation_patch"]),
        "user_task_sha256": sha256_text(task["user_task"]),
        "setup_commands_sha256": sha256_bytes(canonical_json_bytes(task["setup_commands"])),
        "test_command_sha256": sha256_bytes(canonical_json_bytes(task["test_command"])),
        "timeout_seconds": task["timeout_seconds"],
        "shared_user_prompt_sha256": sha256_text(shared_user_prompt(task)),
        "arm_order": deterministic_arm_order(task["id"]),
    }


def public_result(row: dict[str, Any]) -> dict[str, Any]:
    return row


def save_checkpoint(
    *,
    run_dir: pathlib.Path,
    run_id: str,
    rows: list[dict[str, Any]],
    base_manifest: dict[str, Any],
    expected_task_ids: list[str],
) -> None:
    ordered = sorted(rows, key=lambda row: (expected_task_ids.index(row["task_id"]), ARMS.index(row["arm"])))
    result_document = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "updated_at": utc_now(),
        "results": ordered,
    }
    gate = grade_results(ordered, expected_task_ids)
    manifest = dict(base_manifest)
    manifest.update({"updated_at": result_document["updated_at"], "results": [public_result(row) for row in ordered], "gate": gate})
    atomic_write_json(run_dir / "results.json", result_document)
    atomic_write_json(run_dir / "MANIFEST.json", manifest)


def sanitized_failure_row(task_id: str, arm: str, error: Exception) -> dict[str, Any]:
    row = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task_id": task_id,
        "arm": arm,
        "valid": False,
        "correctness": None,
        "failure_kind": error.__class__.__name__,
        # HarnessError messages are harness-authored strings (operation names),
        # never provider output - safe to expose for diagnosability
        "failure_detail": str(error)[:200] if isinstance(error, HarnessError) else None,
        "worktree_cleaned": True,
        "completed_at": utc_now(),
    }
    if isinstance(error, SetupCommandError):
        row["setup"] = error.summary
    return row


def build_base_manifest(
    *,
    run_id: str,
    task_path: pathlib.Path,
    task_document: dict[str, Any],
    tasks: list[dict[str, Any]],
    pi_bin: pathlib.Path,
    greppy_bin: pathlib.Path,
    warm_greppy: bool,
) -> dict[str, Any]:
    explorer_system = system_prompt("explorer", greppy_bin)
    greppy_system = system_prompt("greppy", greppy_bin)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "run_id": run_id,
        "created_at": utc_now(),
        "publishable": True,
        "contains_raw_traces": False,
        "model": {"provider": DEFAULT_PROVIDER, "id": DEFAULT_MODEL, "thinking": DEFAULT_THINKING},
        "tools_per_arm": {arm: ARM_TOOLS[arm].split(",") for arm in ARMS},
        "provider_extension": {
            "repository_path": "bench/agent_efficiency/minimax-provider.js",
            "sha256": sha256_bytes(PROVIDER_EXTENSION.read_bytes()),
            "explicit_registration_probe": True,
        },
        "executables": {
            "pi": {
                "sha256": sha256_bytes(pi_bin.read_bytes()),
                "version": executable_version(pi_bin),
            },
            "greppy": {
                "sha256": sha256_bytes(greppy_bin.read_bytes()),
                "version": executable_version(greppy_bin),
            },
        },
        "greppy_source": greppy_source_identity(),
        "platform": {
            "operating_system": platform.system(),
            "os_release": platform.release(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
        },
        "task_file": {
            "name": task_path.name,
            "sha256": sha256_bytes(task_path.read_bytes()),
            "canonical_content_sha256": sha256_bytes(canonical_json_bytes(task_document)),
        },
        "tasks": [task_manifest_entry(task) for task in tasks],
        "prompt_contract": {
            "shared_system_sha256": sha256_text(SHARED_SYSTEM_PROMPT),
            "explorer_treatment_sha256": sha256_text(EXPLORER_POLICY),
            "greppy_treatment_sha256": sha256_text(GREPPY_POLICY_TEMPLATE),
            "explorer_full_system_sha256": sha256_text(explorer_system),
            "greppy_full_system_sha256": sha256_text(greppy_system),
            "same_user_prompt_per_pair": True,
            "only_intended_prompt_delta": "navigation treatment",
        },
        "isolation": {
            "temporary_git_worktree_per_arm": True,
            "greppy_store_per_arm": True,
            "pi_config_per_arm": True,
            "worktree_cleanup_in_finally": True,
        },
        "setup_contract": {
            "required_task_field": True,
            "direct_argv_without_shell": True,
            "runs_in_each_fresh_preflight_and_arm_worktree": True,
            "runs_before_mutation": True,
            "provider_key_removed": True,
            "excluded_from_agent_wall": True,
        },
        "warm_greppy_outside_measurement": warm_greppy,
        "gate_preregistration": {
            "correctness": (
                "Greppy paired correctness wins >= losses, plus one-sided exact "
                "McNemar regression alarm at p < 0.05"
            ),
            "efficiency_population": "pairs where both arms pass the independent test",
            "minimum_sample": "at least 30 complete pairs and at least 20 both-solved pairs",
            "efficiency": "sum ratio <= 0.80 for tool calls AND source opens AND input tokens",
            "wall_time": "reported only for solved pairs; never a gate metric",
            "failed_test_speed_credit": False,
        },
    }


RESUME_IDENTITY_FIELDS = (
    "schema_version",
    "harness_version",
    "model",
    "tools_per_arm",
    "provider_extension",
    "executables",
    "greppy_source",
    "platform",
    "task_file",
    "tasks",
    "prompt_contract",
    "isolation",
    "setup_contract",
    "warm_greppy_outside_measurement",
    "gate_preregistration",
)


def validate_resume_identity(previous: dict[str, Any], current: dict[str, Any]) -> None:
    if not isinstance(previous, dict):
        raise HarnessError("resume manifest must be an object")
    mismatches = [field for field in RESUME_IDENTITY_FIELDS if previous.get(field) != current.get(field)]
    if mismatches:
        raise HarnessError(f"resume identity mismatch: {', '.join(mismatches)}")


def validate_resume_rows(rows: Any, expected_task_ids: Sequence[str]) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise HarnessError("resume results must be an array")
    expected = set(expected_task_ids)
    seen: set[tuple[str, str]] = set()
    validated: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise HarnessError("resume result has an invalid schema")
        task_id = row.get("task_id")
        arm = row.get("arm")
        if task_id not in expected or arm not in ARMS:
            raise HarnessError("resume result does not belong to the selected task set")
        key = (task_id, arm)
        if key in seen:
            raise HarnessError(f"duplicate resume result: {task_id}/{arm}")
        seen.add(key)
        validated.append(row)
    return validated


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=pathlib.Path, required=True, help="task JSON matching task.schema.json")
    parser.add_argument("--task", action="append", dest="task_ids", help="run one task id (repeatable)")
    parser.add_argument(
        "--arms",
        help="comma-separated subset of arms to run (default: all). "
        "Other arms are neither run nor marked failed, so an existing baseline stays intact.",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, help="checkpoint/manifest directory")
    parser.add_argument("--run-id", help="stable run id; defaults to UTC timestamp plus task hash")
    parser.add_argument("--resume", action="store_true", help="resume completed arms from output-dir/results.json")
    parser.add_argument("--pi-bin", default=os.environ.get("PI_BIN", "pi"))
    parser.add_argument("--greppy-bin", default=os.environ.get("GREPPY_BENCH_BIN", str(REPO_ROOT / "target" / "release" / "greppy")))
    parser.add_argument("--warm-greppy", action="store_true", help="index the Greppy arm before measured agent time")
    parser.add_argument("--validate-only", action="store_true", help="validate tasks without cloning or invoking Pi")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    install_cleanup_signal_handlers()
    args = parse_args(argv)
    task_path = args.tasks.resolve()
    task_document, tasks = load_tasks(task_path, set(args.task_ids or []) or None)
    if args.validate_only:
        print(f"validated {len(tasks)} task(s)")
        return 0

    api_key = os.environ.get("MINIMAX_API_KEY", "")
    if not api_key:
        raise HarnessError("MINIMAX_API_KEY is required in the environment")
    if not PROVIDER_EXTENSION.is_file():
        raise HarnessError("existing MiniMax provider extension is missing")
    pi_bin = resolve_executable(args.pi_bin)
    greppy_bin = resolve_executable(args.greppy_bin)
    verify_provider_registration(pi_bin)
    task_hash = sha256_bytes(canonical_json_bytes(task_document))[:10]
    run_id = args.run_id or f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{task_hash}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", run_id):
        raise HarnessError("run-id is invalid")
    run_dir = (args.output_dir or (HERE / "runs" / run_id)).resolve()
    raw_run_dir = RAW_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_run_dir.mkdir(parents=True, exist_ok=True)

    existing_document: dict[str, Any] | None = None
    previous_manifest: dict[str, Any] = {}
    result_path = run_dir / "results.json"
    if result_path.exists():
        if not args.resume:
            raise HarnessError("output already contains results; pass --resume")
        existing_document = json.loads(result_path.read_text(encoding="utf-8"))
        manifest_path = run_dir / "MANIFEST.json"
        if not manifest_path.is_file():
            raise HarnessError("resume manifest is missing")
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected_ids = [task["id"] for task in tasks]
    base_manifest = build_base_manifest(
        run_id=run_id,
        task_path=task_path,
        task_document=task_document,
        tasks=tasks,
        pi_bin=pi_bin,
        greppy_bin=greppy_bin,
        warm_greppy=args.warm_greppy,
    )
    existing_rows = validate_resume_rows(
        existing_document.get("results") if existing_document is not None else [],
        expected_ids,
    )
    if previous_manifest:
        validate_resume_identity(previous_manifest, base_manifest)
        base_manifest["created_at"] = previous_manifest.get("created_at", base_manifest["created_at"])
        if previous_manifest.get("mutation_preflight"):
            base_manifest["mutation_preflight"] = previous_manifest["mutation_preflight"]
    rows = existing_rows
    save_checkpoint(run_dir=run_dir, run_id=run_id, rows=rows, base_manifest=base_manifest, expected_task_ids=expected_ids)
    completed = {(row["task_id"], row["arm"]) for row in rows}
    secrets = [api_key]

    if args.arms:
        requested = [a.strip() for a in args.arms.split(",") if a.strip()]
        unknown = [a for a in requested if a not in ARMS]
        if unknown:
            raise SystemExit(f"unknown arm(s): {unknown}; valid arms are {list(ARMS)}")
        selected_arms = tuple(a for a in ARMS if a in requested)
    else:
        selected_arms = ARMS

    for task in tasks:
        if all((task["id"], arm) in completed for arm in selected_arms):
            continue
        print(f"[{task['id']}] preparing pinned repository", flush=True)
        # ignore_cleanup_errors: the greppy daemon may still be flushing into
        # the worktree when the context exits; a leaked temp dir on an
        # ephemeral runner is harmless, a crashed 2.5h benchmark run is not.
        with tempfile.TemporaryDirectory(
            prefix=f"greppy-agent-coding-{task['id']}-", ignore_cleanup_errors=True
        ) as tmp_name:
            task_tmp = pathlib.Path(tmp_name)
            try:
                backing = clone_pinned_repository(task, task_tmp)
                preflight = run_mutation_preflight(task, backing, task_tmp, raw_run_dir / task["id"], secrets)
                base_manifest.setdefault("mutation_preflight", {})[task["id"]] = preflight
                if not preflight["valid"]:
                    if task.get("task_bank") == "v2":
                        detail = preflight.get("patched_failure_classification") or {}
                        raise HarnessError(
                            "v2 preflight requires setup success and a patched-test failure "
                            "with test-framework evidence; "
                            f"verdict={preflight.get('failure_kind')} "
                            f"proof={detail.get('proof_line')!r}"
                        )
                    raise HarnessError(
                        "mutation preflight requires setup success, a clean-source test pass, "
                        "and a mutated-source test failure without timeout"
                    )
                for arm in deterministic_arm_order(task["id"]):
                    if arm not in selected_arms:
                        continue
                    if (task["id"], arm) in completed:
                        continue
                    print(f"[{task['id']}] {arm}", flush=True)
                    try:
                        # Provider-reported stream errors (rate limits, upstream
                        # 5xx) invalidate an attempt without measuring anything;
                        # retry those up to two times so infrastructure noise
                        # does not consume task validity. Timeouts and harness
                        # failures are not retried.
                        for attempt in range(1, 6):
                            # each attempt needs untouched worktree/store dirs;
                            # reusing the previous attempt's task_tmp fails on
                            # the existing worktree path
                            attempt_tmp = task_tmp if attempt == 1 else task_tmp / f"retry{attempt}"
                            try:
                                row = run_arm(
                                    arm=arm,
                                    task=task,
                                    backing=backing,
                                    task_tmp=attempt_tmp,
                                    raw_dir=raw_run_dir / task["id"] / arm,
                                    pi_bin=pi_bin,
                                    greppy_bin=greppy_bin,
                                    warm_greppy=args.warm_greppy,
                                    expected_mutation_hash=preflight["mutation_diff_sha256"],
                                    secrets=secrets,
                                )
                            except HarnessError as harness_error:
                                # warmup/worktree failures are runner-environment
                                # noise, not measurements - retry them like
                                # provider errors instead of consuming the task
                                if attempt >= 5:
                                    raise
                                print(f"[{task['id']}] {arm}: {harness_error}, retry {attempt}/4", flush=True)
                                time.sleep(30 * attempt)
                                continue
                            row["agent"]["provider_attempts"] = attempt
                            provider_flake = (
                                not row["valid"]
                                and row["agent"].get("reported_error")
                                and not row["agent"].get("timed_out")
                            )
                            if not provider_flake:
                                break
                            print(f"[{task['id']}] {arm}: provider error, retry {attempt}/4", flush=True)
                            time.sleep(30 * attempt)
                    except Exception as error:  # checkpoint setup failures without exposing stderr/source
                        row = sanitized_failure_row(task["id"], arm, error)
                    rows.append(row)
                    completed.add((task["id"], arm))
                    save_checkpoint(run_dir=run_dir, run_id=run_id, rows=rows, base_manifest=base_manifest, expected_task_ids=expected_ids)
            except Exception as error:
                for arm in selected_arms:
                    if (task["id"], arm) not in completed:
                        rows.append(sanitized_failure_row(task["id"], arm, error))
                        completed.add((task["id"], arm))
                save_checkpoint(run_dir=run_dir, run_id=run_id, rows=rows, base_manifest=base_manifest, expected_task_ids=expected_ids)

    gate = grade_results(rows, expected_ids)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HarnessError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
