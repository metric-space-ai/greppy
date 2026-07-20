#!/usr/bin/env python3
"""Validate harvested v2 candidates with the parent-fail/full-commit-pass proof.

For every candidate this script creates (or reuses) a disposable clone below the
scratchpad, extracts the test-only and non-test portions of the real commit,
then runs the narrowest repository-native test command it can derive:

  A. parent + test patch (must fail, including a compile failure)
  B. parent + complete commit diff (must pass)

Results are appended and fsynced one JSON object at a time.  Reruns resume by
(class, repository, commit).  Harvest clones are only read; all checkouts,
patches, dependency trees, build products, and logs live below --scratch-root.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
DEFAULT_HARVEST = Path(
    "/private/tmp/claude-501/-Users-michaelwelsch-Documents-greppy/"
    "703e5fe0-4911-4d49-b220-03fd41c3e4dd/scratchpad/harvest-repos"
)
DEFAULT_SERIOUS = HERE / "harvest_candidates_v2_serious.jsonl"
DEFAULT_MECHANICAL = HERE / "harvest_candidates_v2.jsonl"
DEFAULT_OUTPUT = HERE / "validated_v2.jsonl"
DEFAULT_NOTES = HERE / "VALIDATION-NOTES-v2.md"

REPO_NAMES = {
    "https://github.com/pallets/flask": "flask",
    "https://github.com/gohugoio/hugo": "hugo",
    "https://github.com/google/gson": "gson",
    "https://github.com/colinhacks/zod": "zod",
    "https://github.com/serde-rs/serde": "serde",
    "https://github.com/tokio-rs/tokio": "tokio",
}

FAIL_PATTERNS = re.compile(
    r"(?:\bFAILED\b|failures?:|assertionerror|assertion failed|expected:|"
    r"panicked at|test result: FAILED|there (?:was|were) \d+ failure)", re.I
)
COMPILE_PATTERNS = re.compile(
    r"(?:could not compile|compilation failure|compile error|error\[E\d+\]|"
    r"cannot find (?:symbol|name|function|method|module|crate)|unresolved import|"
    r"mismatched types|syntaxerror|typeerror:|ts\(\d+\)|BUILD FAILURE)", re.I
)
TOOLCHAIN_PATTERNS = re.compile(
    r"(?:command not found|no such file or directory|requires? (?:python|rust|rustc|node|java|jdk|nightly)|"
    r"unsupported class file|invalid target release|source option \d+ is no longer supported|"
    r"failed to download|network is unreachable|could not resolve host|connection timed out|"
    r"no matching distribution found|resolution impossible|lockfile is broken|"
    r"ERR_PNPM_UNSUPPORTED_ENGINE|unsupported engine|unable to locate a java runtime)", re.I
)


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int | None
    seconds: float
    output: str
    timed_out: bool = False
    missing: str | None = None
    log_path: Path | None = None


@dataclass
class CommandPlan:
    commands: list[list[str]]
    env_overrides: list[dict[str, str]]
    compile_only: bool = False

    def display(self) -> list[str]:
        if len(self.commands) == 1 and not self.env_overrides[0]:
            return self.commands[0]
        chunks = []
        for argv, env in zip(self.commands, self.env_overrides):
            prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(env.items()))
            command = shlex.join(argv)
            chunks.append((prefix + " " + command).strip())
        return ["bash", "-lc", " && ".join(chunks)]


@dataclass
class ProofResult:
    state: str
    summary: str
    test_result: CommandResult | None
    setup_results: list[CommandResult]


class DeadlineExpired(Exception):
    pass


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}")
    return rows


def repair_and_load_output(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = path.read_bytes()
    rows: list[dict[str, Any]] = []
    good_end = 0
    pos = 0
    lines = data.splitlines(keepends=True)
    for i, raw in enumerate(lines):
        pos += len(raw)
        if not raw.strip():
            good_end = pos
            continue
        try:
            rows.append(json.loads(raw))
            good_end = pos
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise SystemExit(f"{path}: malformed non-final JSONL line {i + 1}")
            break
    if good_end != len(data):
        with path.open("r+b") as fh:
            fh.truncate(good_end)
            fh.flush()
            os.fsync(fh.fileno())
        print(f"repaired incomplete final line in {path}", file=sys.stderr)
    return rows


def candidate_key(candidate_class: str, row: dict[str, Any]) -> tuple[str, str, str]:
    return candidate_class, row["repo"], row["commit"]


def row_class(row: dict[str, Any]) -> str:
    return row.get("candidate_class") or ("S" if "type" in row else "M")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        written = 0
        while written < len(payload):
            written += os.write(fd, payload[written:])
        os.fsync(fd)
    finally:
        os.close(fd)


def git(repo: Path, args: list[str], *, check: bool = True, text: bool = False) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=text,
        errors="replace" if text else None,
    )
    if check and proc.returncode:
        err = proc.stderr if text else proc.stderr.decode("utf-8", "replace")
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {err[-1000:]}")
    return proc


def git_bytes(repo: Path, args: list[str]) -> bytes:
    return git(repo, args).stdout


def git_text(repo: Path, args: list[str]) -> str:
    return git(repo, args, text=True).stdout


def object_exists(repo: Path, spec: str) -> bool:
    return git(repo, ["cat-file", "-e", spec], check=False).returncode == 0


def show_file(repo: Path, commit: str, path: str) -> str | None:
    proc = git(repo, ["show", f"{commit}:{path}"], check=False)
    if proc.returncode:
        return None
    return proc.stdout.decode("utf-8", "replace")


def list_changed_paths(repo: Path, parent: str, commit: str) -> list[str]:
    out = git_text(repo, ["diff", "--name-only", "--no-renames", parent, commit])
    return [line for line in out.splitlines() if line]


def extract_patches(
    source: Path, candidate_class: str, index: int, row: dict[str, Any], scratch: Path
) -> tuple[bytes, bytes, bytes, Path]:
    parent, commit = row["parent"], row["commit"]
    tests = list(dict.fromkeys(row.get("tests_touched", [])))
    all_paths = list_changed_paths(source, parent, commit)
    test_set = set(tests)
    code_paths = [p for p in all_paths if p not in test_set]
    common = ["diff", "--binary", "--full-index", "--no-renames", parent, commit, "--"]
    test_patch = git_bytes(source, [*common, *tests]) if tests else b""
    code_patch = git_bytes(source, [*common, *code_paths]) if code_paths else b""
    full_patch = git_bytes(source, [*common, *all_paths]) if all_paths else b""

    patch_dir = scratch / "patches" / candidate_class / f"{index:03d}-{commit[:12]}"
    patch_dir.mkdir(parents=True, exist_ok=True)
    (patch_dir / "test.patch").write_bytes(test_patch)
    (patch_dir / "code.patch").write_bytes(code_patch)
    (patch_dir / "full.patch").write_bytes(full_patch)
    (patch_dir / "manifest.json").write_text(
        json.dumps({"tests": tests, "code": code_paths, "all": all_paths}, indent=2) + "\n",
        encoding="utf-8",
    )
    return test_patch, code_patch, full_patch, patch_dir


def ensure_runner(source: Path, runner: Path) -> None:
    if runner.exists():
        if not (runner / ".git").exists():
            raise RuntimeError(f"scratch runner exists but is not a Git clone: {runner}")
        return
    runner.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", "clone", "--quiet", "--shared", "--no-checkout", str(source), str(runner)],
        capture_output=True,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-1000:])


def reset_runner(runner: Path, parent: str) -> None:
    git(runner, ["reset", "--hard", "--quiet"], check=False)
    git(runner, ["clean", "-fd", "--quiet"], check=False)
    git(runner, ["checkout", "--detach", "--force", "--quiet", parent])
    git(runner, ["reset", "--hard", "--quiet", parent])
    git(runner, ["clean", "-fd", "--quiet"])


def apply_patch(runner: Path, patch: bytes, patch_path: Path) -> tuple[bool, str]:
    proc = subprocess.run(
        ["git", "apply", "--index", "--binary", "--whitespace=nowarn", str(patch_path)],
        cwd=runner, capture_output=True,
    )
    if proc.returncode:
        return False, proc.stderr.decode("utf-8", "replace")[-1200:]
    return True, "applied"


def changed_new_lines(source: Path, parent: str, commit: str, path: str) -> list[int]:
    diff = git_text(source, ["diff", "--unified=0", "--no-renames", parent, commit, "--", path])
    changed: list[int] = []
    new_lineno: int | None = None
    for line in diff.splitlines():
        m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if m:
            new_lineno = int(m.group(1))
            continue
        if new_lineno is None or line.startswith(("diff --git", "---", "+++")):
            continue
        if line.startswith("+"):
            # Added blank separators and import-only lines do not identify a
            # test. Nonblank additions inside a function do.
            if line[1:].strip():
                changed.append(new_lineno)
            new_lineno += 1
        elif line.startswith(" "):
            new_lineno += 1
        elif line.startswith("-") or line.startswith("\\"):
            pass
    return changed


def python_nodes(content: str, changed: list[int], path: str) -> list[str]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    found: set[str] = set()
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test")
    ]
    for lineno in changed:
        matches = [f for f in functions if f.lineno <= lineno <= getattr(f, "end_lineno", f.lineno)]
        if not matches:
            continue
        fn = min(matches, key=lambda n: getattr(n, "end_lineno", n.lineno) - n.lineno)
        parent = parents.get(fn)
        if isinstance(parent, ast.ClassDef):
            found.add(f"{path}::{parent.name}::{fn.name}")
        else:
            found.add(f"{path}::{fn.name}")
    return sorted(found)


def function_spans(content: str, language: str) -> list[tuple[int, int, str, str]]:
    lines = content.splitlines()
    # Keep every top-level declaration as a boundary, not only tests. Otherwise
    # a changed helper following TestA would be misattributed to TestA until the
    # next test declaration.
    declarations: list[tuple[int, str, str]] = []
    if language == "go":
        pattern = re.compile(r"^func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(")
        for i, line in enumerate(lines, 1):
            m = pattern.match(line)
            if m:
                name = m.group(1)
                if name.startswith("Benchmark"):
                    kind = "benchmark"
                elif name.startswith("Test") or name.startswith("Example"):
                    kind = "test"
                else:
                    kind = "other"
                declarations.append((i, name, kind))
    elif language == "rust":
        # These harvested test files declare tests at module top level. Exclude
        # indented nested helpers so an outer #[test] keeps its complete span.
        pattern = re.compile(r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(")
        for i, line in enumerate(lines, 1):
            m = pattern.match(line)
            if not m:
                continue
            attrs = "\n".join(lines[max(0, i - 5): i - 1])
            kind = "test" if (
                "#[test]" in attrs or "#[tokio::test" in attrs or "#[test_case" in attrs
            ) else "other"
            declarations.append((i, m.group(1), kind))
    elif language == "java":
        pattern = re.compile(
            r"^\s*(?:public|protected|private)?\s*(?:static\s+)?(?:final\s+)?"
            r"[\w<>?,.\[\] ]+\s+(\w+)\s*\([^;]*\)\s*(?:throws [^{]+)?\s*\{"
        )
        for i, line in enumerate(lines, 1):
            m = pattern.match(line)
            if not m:
                continue
            attrs = "\n".join(lines[max(0, i - 5): i - 1])
            kind = "test" if "@Test" in attrs or m.group(1).startswith("test") else "other"
            declarations.append((i, m.group(1), kind))
    spans = []
    for idx, (start, name, kind) in enumerate(declarations):
        end = declarations[idx + 1][0] - 1 if idx + 1 < len(declarations) else len(lines)
        spans.append((start, end, name, kind))
    return spans


def names_at_lines(spans: list[tuple[int, int, str, str]], changed: list[int], kind: str) -> list[str]:
    out = set()
    for lineno in changed:
        for start, end, name, item_kind in spans:
            if item_kind == kind and start <= lineno <= end:
                out.add(name)
    return sorted(out)


def nearest_cargo_package(source: Path, commit: str, path: str) -> tuple[str, str] | None:
    parent = Path(path).parent
    while str(parent) not in ("", "."):
        manifest = (parent / "Cargo.toml").as_posix()
        text = show_file(source, commit, manifest)
        if text is not None:
            try:
                package = tomllib.loads(text).get("package", {}).get("name")
            except tomllib.TOMLDecodeError:
                package = None
            if package:
                return parent.as_posix(), package
        parent = parent.parent
    text = show_file(source, commit, "Cargo.toml")
    if text is not None:
        try:
            package = tomllib.loads(text).get("package", {}).get("name")
        except tomllib.TOMLDecodeError:
            package = None
        if package:
            return ".", package
    return None


def derive_flask_plan(
    source: Path, row: dict[str, Any], venv: Path
) -> CommandPlan:
    scopes: list[str] = []
    for path in row["tests_touched"]:
        content = show_file(source, row["commit"], path)
        if content is None or not path.endswith(".py"):
            scopes.append(path)
            continue
        nodes = python_nodes(content, changed_new_lines(source, row["parent"], row["commit"], path), path)
        scopes.extend(nodes or [path])
    scopes = list(dict.fromkeys(scopes))
    return CommandPlan([[str(venv / "bin" / "python3"), "-m", "pytest", "-q", *scopes]], [{}])


def re2_escape(value: str) -> str:
    return re.sub(r"([\\.^$|?*+(){}\[\]])", r"\\\1", value)


def derive_go_plan(source: Path, row: dict[str, Any]) -> CommandPlan:
    packages: set[str] = set()
    tests: set[str] = set()
    benches: set[str] = set()
    script_names: set[str] = set()
    for path in row["tests_touched"]:
        if path.startswith("testscripts/"):
            packages.add(".")
            if path.endswith(".txt"):
                script_names.add(Path(path).stem)
            continue
        if path.endswith(".go"):
            dirname = Path(path).parent.as_posix()
            packages.add("." if dirname == "." else f"./{dirname}")
            content = show_file(source, row["commit"], path) or ""
            spans = function_spans(content, "go")
            changed = changed_new_lines(source, row["parent"], row["commit"], path)
            tests.update(names_at_lines(spans, changed, "test"))
            benches.update(names_at_lines(spans, changed, "benchmark"))
        else:
            packages.add("." if Path(path).parent.as_posix() == "." else f"./{Path(path).parent.as_posix()}")
    # The proof is the repository test, not `go test`'s implicit host-version
    # vet pass. Historical Hugo commits can trigger a Go 1.26 vet panic after
    # all selected tests pass, so disable vet explicitly and retain compilation.
    if script_names:
        commands: list[list[str]] = []
        envs: list[dict[str, str]] = []
        normal_packages = sorted(p for p in packages if p != ".")
        if normal_packages:
            normal = ["go", "test", "-vet=off", *normal_packages, "-count=1", "-v"]
            if tests:
                normal += ["-run", "^(" + "|".join(sorted(tests)) + ")$"]
            commands.append(normal)
            envs.append({})
        script_alt = "|".join(re2_escape(x) for x in sorted(script_names))
        commands.append([
            "go", "test", "-vet=off", ".", "-count=1", "-v",
            "-run", f"^TestCommands$/^({script_alt})$",
        ])
        envs.append({})
        return CommandPlan(commands, envs)

    argv = ["go", "test", "-vet=off", *sorted(packages), "-count=1", "-v"]
    if tests:
        argv += ["-run", "^(" + "|".join(sorted(tests)) + ")$"]
    elif benches:
        argv += ["-run", "^$"]
    if benches:
        argv += ["-bench", "^(" + "|".join(sorted(benches)) + ")$", "-benchtime=1x"]
    return CommandPlan([argv], [{}])


def derive_java_plan(source: Path, row: dict[str, Any]) -> CommandPlan:
    by_module: dict[str, list[str]] = defaultdict(list)
    for path in row["tests_touched"]:
        if not path.endswith(".java"):
            continue
        module = path.split("/", 1)[0] if "/src/test/" in path else "."
        cls = Path(path).stem
        content = show_file(source, row["commit"], path) or ""
        spans = function_spans(content, "java")
        methods = names_at_lines(
            spans, changed_new_lines(source, row["parent"], row["commit"], path), "test"
        )
        spec = cls + ("#" + "+".join(methods) if methods else "")
        by_module[module].append(spec)
    modules = sorted(by_module) or ["gson"]
    specs = [spec for module in modules for spec in by_module[module]]
    argv = ["mvn", "-q"]
    if modules != ["."]:
        argv += ["-pl", ",".join(modules)]
    if specs:
        argv += [f"-Dtest={','.join(specs)}"]
    argv += ["test"]
    return CommandPlan([argv], [{}])


def derive_zod_plan(source: Path, row: dict[str, Any]) -> CommandPlan:
    paths = [p for p in row["tests_touched"] if p.endswith((".ts", ".tsx", ".js"))]
    if object_exists(source, f"{row['commit']}:pnpm-lock.yaml"):
        return CommandPlan([["pnpm", "exec", "vitest", "run", *paths]], [{}])
    # The historical Yarn repository's Deno copy is generated from src. Running
    # both through Jest would duplicate the same case and Jest does not collect
    # deno/lib by default, so scope to the real src test file(s).
    src_paths = [p for p in paths if p.startswith("src/")] or paths
    return CommandPlan(
        [["yarn", "run", "test:ts-jest", "--runTestsByPath", *src_paths, "--runInBand"]],
        [{}],
    )


def rust_direct_targets(
    source: Path, row: dict[str, Any], crate_root: str, paths: Iterable[str]
) -> tuple[list[str], bool]:
    targets: list[str] = []
    needs_lib = False
    prefix = "" if crate_root == "." else crate_root.rstrip("/") + "/"
    for path in paths:
        rel = path[len(prefix):] if path.startswith(prefix) else path
        parts = Path(rel).parts
        if len(parts) >= 2 and parts[0] == "tests" and len(parts) == 2 and path.endswith(".rs"):
            targets += ["--test", Path(path).stem]
        elif parts and parts[0] == "src":
            needs_lib = True
    if needs_lib:
        targets.insert(0, "--lib")
    return targets, needs_lib


def derive_rust_plan(source: Path, row: dict[str, Any], repo_name: str) -> CommandPlan:
    paths = row["tests_touched"]
    # Compile-fail fixtures have a real repository harness; name it explicitly.
    if any("/tests/ui/" in p for p in paths):
        return CommandPlan(
            [["cargo", "+nightly", "test", "-p", "serde_test_suite", "--test", "compiletest", "ui", "--", "--exact"]],
            [{}],
        )
    if repo_name == "tokio" and any(p.startswith("tests-build/tests/fail/") for p in paths):
        return CommandPlan(
            [["cargo", "test", "-p", "tests-build", "--test", "macros", "compile_fail_full", "--features", "full", "--", "--exact"]],
            [{}],
        )
    if repo_name == "tokio" and any("/loom_" in p or "/loom/" in p for p in paths):
        # Loom cfg is not suitable for ordinary integration targets. Keep the
        # changed loom unit and changed integration test in separate commands;
        # deletion-only test files contribute no runnable test on the patched tree.
        return CommandPlan(
            [
                ["cargo", "test", "-p", "tokio", "--features", "full", "--lib", "multi_stealer"],
                ["cargo", "test", "-p", "tokio", "--features", "full", "--test", "rt_unstable_metrics", "worker_local_queue_depth", "--", "--exact"],
            ],
            [{"RUSTFLAGS": "--cfg loom"}, {}],
        )

    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in paths:
        pkg = nearest_cargo_package(source, row["commit"], path)
        if pkg:
            grouped[pkg].append(path)
    commands: list[list[str]] = []
    envs: list[dict[str, str]] = []
    compile_only = True
    for (crate_root, package), crate_paths in sorted(grouped.items()):
        targets, needs_lib = rust_direct_targets(source, row, crate_root, crate_paths)
        test_names: set[str] = set()
        has_test_attribute = False
        for path in crate_paths:
            if not path.endswith(".rs"):
                continue
            content = show_file(source, row["commit"], path) or ""
            if "#[test" in content or "#[tokio::test" in content:
                has_test_attribute = True
            spans = function_spans(content, "rust")
            test_names.update(names_at_lines(
                spans, changed_new_lines(source, row["parent"], row["commit"], path), "test"
            ))
        compile_only = compile_only and not has_test_attribute
        base = ["cargo", "test", "-p", package]
        if repo_name == "tokio":
            # Tokio's `full` is the supported test feature set. `--all-features`
            # additionally enables Linux-only io-uring/taskdump features and
            # fails on macOS before the touched test can run.
            base += ["--features", "full"]
        base += targets
        # Cargo accepts one substring filter, not a regex. Use it only when the
        # patch maps to exactly one changed test; otherwise touched test binaries
        # are the tight reliable scope.
        if len(test_names) == 1:
            base += [next(iter(test_names)), "--", "--exact"]
        commands.append(base)
        env = {}
        if needs_lib and any("/loom_" in p or "/loom/" in p for p in crate_paths):
            env["RUSTFLAGS"] = "--cfg loom"
        envs.append(env)
    if not commands:
        # Keep a concrete, guaranteed-failing command in the record rather than
        # silently broadening to the whole workspace.
        commands = [["cargo", "test", "--no-run", "--", "--exact"]]
        envs = [{}]
        compile_only = True
    return CommandPlan(commands, envs, compile_only=compile_only)


def derive_plan(
    source: Path, row: dict[str, Any], repo_name: str, venv: Path
) -> CommandPlan:
    if repo_name == "flask":
        return derive_flask_plan(source, row, venv)
    if repo_name == "hugo":
        return derive_go_plan(source, row)
    if repo_name == "gson":
        return derive_java_plan(source, row)
    if repo_name == "zod":
        return derive_zod_plan(source, row)
    if repo_name in ("serde", "tokio"):
        return derive_rust_plan(source, row, repo_name)
    raise RuntimeError(f"unsupported repository: {repo_name}")


def setup_commands(
    source: Path, row: dict[str, Any], repo_name: str, venv: Path, phase: str
) -> list[list[str]]:
    if repo_name == "flask":
        commands = []
        if phase == "A":
            commands.append(["python3", "-m", "venv", "--clear", str(venv)])
        commands.append([
            str(venv / "bin" / "python3"), "-m", "pip", "install", "-e", ".",
            "pytest==8.4.2", "asgiref==3.11.1", "python-dotenv==1.2.2",
        ])
        return commands
    if repo_name == "zod":
        if object_exists(source, f"{row['commit']}:pnpm-lock.yaml"):
            return [["pnpm", "install", "--frozen-lockfile", "--prefer-offline"]]
        return [["yarn", "install", "--frozen-lockfile", "--prefer-offline"]]
    return []


def executable_missing(argv: list[str], env: dict[str, str]) -> str | None:
    if not argv:
        return "empty command"
    if argv[0] == "cargo" and len(argv) > 1 and argv[1] == "+nightly":
        proc = subprocess.run(["rustup", "toolchain", "list"], capture_output=True, text=True)
        if proc.returncode or not any(line.startswith("nightly") for line in proc.stdout.splitlines()):
            return "Rust nightly toolchain (required by this repository's UI test harness)"
    path = env.get("PATH", os.environ.get("PATH", ""))
    if os.path.sep not in argv[0] and shutil.which(argv[0], path=path) is None:
        return argv[0]
    if os.path.sep in argv[0] and not Path(argv[0]).exists():
        return argv[0]
    return None


def run_command(
    argv: list[str], cwd: Path, env: dict[str, str], timeout: float, log_path: Path
) -> CommandResult:
    missing = executable_missing(argv, env)
    if missing:
        result = CommandResult(argv, None, 0.0, f"missing executable/toolchain: {missing}", missing=missing)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(result.output + "\n", encoding="utf-8")
        result.log_path = log_path
        return result
    start = time.monotonic()
    try:
        proc = subprocess.run(
            argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=max(0.1, timeout),
        )
        output = proc.stdout.decode("utf-8", "replace")
        result = CommandResult(argv, proc.returncode, time.monotonic() - start, output)
    except subprocess.TimeoutExpired as exc:
        raw = exc.stdout or b""
        if isinstance(raw, str):
            output = raw
        else:
            output = raw.decode("utf-8", "replace")
        result = CommandResult(argv, None, time.monotonic() - start, output, timed_out=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.output, encoding="utf-8", errors="replace")
    result.log_path = log_path
    return result


def has_executed_test(output: str, repo_name: str, plan: CommandPlan) -> bool:
    if plan.compile_only:
        return True
    if repo_name == "flask":
        return not re.search(r"(?:collected 0 items|no tests ran)", output, re.I)
    if repo_name == "hugo":
        if "Benchmark" in " ".join(plan.display()):
            return bool(re.search(r"^Benchmark\S+", output, re.M))
        return bool(re.search(r"=== RUN\s+", output))
    if repo_name == "gson":
        return not re.search(r"(?:No tests (?:were )?executed|Tests run: 0)", output, re.I)
    if repo_name == "zod":
        return not re.search(r"(?:No test files found|no tests found|Tests\s+0 passed)", output, re.I)
    if repo_name in ("serde", "tokio"):
        passed = [int(x) for x in re.findall(r"test result: ok\. (\d+) passed", output)]
        failed = [int(x) for x in re.findall(r"test result: FAILED\. (\d+) passed; (\d+) failed", output)]
        return any(x > 0 for x in passed) or any(a + b > 0 for a, b in failed)
    return True


def classify_failure(output: str) -> str:
    if TOOLCHAIN_PATTERNS.search(output):
        return "toolchain-failure"
    if COMPILE_PATTERNS.search(output):
        return "compile-failure"
    if FAIL_PATTERNS.search(output):
        return "assertion/test-failure"
    return "command-failure"


def excerpt(output: str, limit: int = 900) -> str:
    clean = "\n".join(line.rstrip() for line in output.strip().splitlines() if line.strip())
    if len(clean) <= limit:
        return clean
    return clean[-limit:]


def result_description(label: str, result: CommandResult) -> str:
    log = str(result.log_path) if result.log_path else "(no log)"
    if result.timed_out:
        state = "timeout"
    elif result.missing:
        state = f"missing {result.missing}"
    else:
        state = f"exit {result.returncode}"
    tail = excerpt(result.output)
    return f"{label}: {state} after {result.seconds:.1f}s; log={log}; tail={tail!r}"


def run_proof(
    runner: Path,
    source: Path,
    row: dict[str, Any],
    repo_name: str,
    plan: CommandPlan,
    venv: Path,
    phase: str,
    timeout_seconds: int,
    log_dir: Path,
    base_env: dict[str, str],
) -> ProofResult:
    deadline = time.monotonic() + timeout_seconds
    setup_results: list[CommandResult] = []
    for idx, argv in enumerate(setup_commands(source, row, repo_name, venv, phase), 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return ProofResult("timeout", f"proof {phase}: timeout before setup completed", None, setup_results)
        result = run_command(argv, runner, base_env, remaining, log_dir / f"{phase}-setup-{idx}.log")
        setup_results.append(result)
        if result.timed_out:
            return ProofResult("timeout", result_description(f"proof {phase} setup", result), None, setup_results)
        if result.returncode != 0:
            return ProofResult("setup-failure", result_description(f"proof {phase} setup", result), None, setup_results)

    combined_output = []
    total_seconds = 0.0
    last_result: CommandResult | None = None
    for idx, (argv, overrides) in enumerate(zip(plan.commands, plan.env_overrides), 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return ProofResult("timeout", f"proof {phase}: timeout before all test commands completed", last_result, setup_results)
        env = base_env.copy()
        env.update(overrides)
        result = run_command(argv, runner, env, remaining, log_dir / f"{phase}-test-{idx}.log")
        last_result = result
        total_seconds += result.seconds
        combined_output.append(result.output)
        if result.timed_out:
            return ProofResult("timeout", result_description(f"proof {phase} test", result), result, setup_results)
        if result.returncode != 0:
            kind = classify_failure(result.output)
            return ProofResult(kind, result_description(f"proof {phase} test ({kind})", result), result, setup_results)
    output = "\n".join(combined_output)
    if not has_executed_test(output, repo_name, plan):
        summary = (
            f"proof {phase}: commands exited 0 but no touched test/benchmark was observed; "
            f"logs={log_dir}"
        )
        return ProofResult("zero-tests", summary, last_result, setup_results)
    command_text = shlex.join(plan.display())
    return ProofResult(
        "pass",
        f"proof {phase}: pass (exit 0, {total_seconds:.1f}s), command={command_text!r}, logs={log_dir}",
        last_result,
        setup_results,
    )


def verify_full_tree(runner: Path, commit: str) -> tuple[bool, str]:
    proc = git(runner, ["diff", "--quiet", commit, "--"], check=False)
    if proc.returncode == 0:
        return True, "full diff reproduces commit tree"
    names = git_text(runner, ["diff", "--name-only", commit, "--"])
    return False, f"full diff tree mismatch: {names.strip()}"


def tool_versions() -> dict[str, str]:
    commands = {
        "python": ["python3", "--version"],
        "go": ["go", "version"],
        "rustc": ["rustc", "--version"],
        "cargo": ["cargo", "--version"],
        "node": ["node", "--version"],
        "pnpm": ["pnpm", "--version"],
        "maven": ["mvn", "--version"],
    }
    versions = {}
    for name, argv in commands.items():
        if shutil.which(argv[0]) is None:
            versions[name] = "missing"
            continue
        proc = subprocess.run(argv, capture_output=True, text=True)
        text = (proc.stdout or proc.stderr).strip().splitlines()
        versions[name] = text[0] if text else f"exit {proc.returncode}"
    versions["yarn"] = "missing" if shutil.which("yarn") is None else subprocess.run(
        ["yarn", "--version"], capture_output=True, text=True
    ).stdout.strip()
    nightly = subprocess.run(["rustup", "toolchain", "list"], capture_output=True, text=True)
    versions["rust-nightly"] = "installed" if any(
        x.startswith("nightly") for x in nightly.stdout.splitlines()
    ) else "missing"
    return versions


def notes_markdown(
    all_candidates: list[tuple[str, dict[str, Any]]], rows: list[dict[str, Any]], versions: dict[str, str]
) -> str:
    by_key = {(row_class(r), r["repo"], r["commit"]): r for r in rows}
    types: dict[str, list[str]] = defaultdict(list)
    for cls, candidate in all_candidates:
        typ = candidate.get("type") or candidate.get("category") or "unknown"
        if typ not in types[cls]:
            types[cls].append(typ)

    lines = [
        "# VALIDATION-NOTES v2",
        "",
        f"Stand: {time.strftime('%Y-%m-%d %H:%M:%S %z')}. Die JSONL wird pro Kandidat "
        "angehängt und per `fsync` gesichert.",
        "",
        "## Ausbeute pro Klasse und Typ",
        "",
        "| Klasse | Typ | valid | not-failing | toolchain | test-patch-empty | offen | gesamt |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cls in ("S", "M"):
        for typ in types.get(cls, []):
            candidates = [c for ccls, c in all_candidates if ccls == cls and (c.get("type") or c.get("category")) == typ]
            counts = Counter()
            for c in candidates:
                result = by_key.get((cls, c["repo"], c["commit"]))
                counts[result["verdict"] if result else "open"] += 1
            lines.append(
                f"| {cls} | {typ} | {counts['valid']} | {counts['test-not-failing']} | "
                f"{counts['toolchain-issue']} | {counts['test-patch-empty']} | {counts['open']} | {len(candidates)} |"
            )
    lines += ["", "## Ausbeute pro Repository", "",
              "| Klasse | Repository | valid | not-failing | toolchain | leer | offen | gesamt |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for cls in ("S", "M"):
        repos = []
        for ccls, c in all_candidates:
            repo = REPO_NAMES.get(c["repo"], c["repo"])
            if ccls == cls and repo not in repos:
                repos.append(repo)
        for repo in repos:
            candidates = [c for ccls, c in all_candidates if ccls == cls and REPO_NAMES.get(c["repo"], c["repo"]) == repo]
            counts = Counter()
            for c in candidates:
                result = by_key.get((cls, c["repo"], c["commit"]))
                counts[result["verdict"] if result else "open"] += 1
            lines.append(
                f"| {cls} | {repo} | {counts['valid']} | {counts['test-not-failing']} | "
                f"{counts['toolchain-issue']} | {counts['test-patch-empty']} | {counts['open']} | {len(candidates)} |"
            )

    valid = [r for r in rows if r.get("verdict") == "valid"]
    lines += ["", "## Drei valide Beispiele", ""]
    if not valid:
        lines.append("Noch keine validierten Beispiele.")
    for row in valid[:3]:
        typ = row.get("type") or row.get("category")
        lines += [
            f"- **{row_class(row)}/{typ} · {REPO_NAMES.get(row['repo'], row['repo'])} "
            f"`{row['commit'][:12]}`** — A: {row['proof_a']} B: {row['proof_b']}"
        ]

    blockers = [r for r in rows if r.get("verdict") == "toolchain-issue"]
    lines += ["", "## Toolchain-Blocker", ""]
    if not blockers:
        lines.append("Keine bislang festgestellten Toolchain-Blocker.")
    for row in blockers:
        detail = row["proof_a"] if "toolchain" in row["proof_a"].lower() or "setup" in row["proof_a"].lower() else row["proof_b"]
        lines.append(
            f"- {row_class(row)}/{row.get('type') or row.get('category')} · "
            f"{REPO_NAMES.get(row['repo'], row['repo'])} `{row['commit'][:12]}`: {detail}"
        )
    lines += ["", "### Vorhandene Versionen", ""]
    for name, value in versions.items():
        lines.append(f"- `{name}`: {value}")

    lines += [
        "", "## Erzeugte Pfade", "",
        f"- `{DEFAULT_OUTPUT.relative_to(REPO_ROOT)}` — append-only Kandidatenergebnisse",
        f"- `{DEFAULT_NOTES.relative_to(REPO_ROOT)}` — dieser fortlaufend neu erzeugte Bericht",
        "- Scratchpad `validation-v2/` — Runner-Klone, Patches, Logs, Venvs und Build-Caches",
        "", "## Offene Bedenken", "",
    ]
    open_count = sum(
        1 for cls, c in all_candidates if (cls, c["repo"], c["commit"]) not in by_key
    )
    lines.append(f"- **Offen:** {open_count} Kandidaten wurden noch nicht ausgeführt.")
    lines.append(
        "- `valid` wird ausschließlich vergeben, wenn A mit einem beobachteten Nicht-Timeout-Fehler "
        "endet und B dieselbe Scope-Ausführung mit Exit 0 besteht. Setup-Fehler, Timeouts und "
        "Null-Test-Läufe werden nie als Parent-Fail-Beweis gewertet."
    )
    if any(r.get("verdict") == "test-not-failing" for r in rows):
        lines.append(
            "- `test-not-failing` bedeutet, dass der echte Test-Patch auf dem Parent bereits grün war; "
            "diese Kandidaten sind trotz eines grünen Full-Commit-Laufs nicht testentscheidend."
        )
    lines.append(
        "- Historische Projekte können mit den heute vorhandenen Python-/Rust-/Node-/Java-Versionen "
        "inkompatibel sein. Solche Fälle bleiben `toolchain-issue`; es werden keine Versionen geraten."
    )
    return "\n".join(lines) + "\n"


def write_notes(
    path: Path,
    scratch: Path,
    all_candidates: list[tuple[str, dict[str, Any]]],
    rows: list[dict[str, Any]],
    versions: dict[str, str],
) -> None:
    text = notes_markdown(all_candidates, rows, versions)
    temp = scratch / "notes-next.md"
    temp.parent.mkdir(parents=True, exist_ok=True)
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def validate_one(
    candidate_class: str,
    index: int,
    row: dict[str, Any],
    harvest_root: Path,
    scratch: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    repo_name = REPO_NAMES.get(row["repo"])
    if repo_name is None:
        raise RuntimeError(f"unknown repository URL: {row['repo']}")
    source = harvest_root / repo_name
    if not source.exists():
        raise RuntimeError(f"harvest clone missing: {source}")
    for oid in (row["parent"], row["commit"]):
        if not object_exists(source, f"{oid}^{{commit}}"):
            raise RuntimeError(f"commit {oid} missing from {source}")

    test_patch, code_patch, full_patch, patch_dir = extract_patches(
        source, candidate_class, index, row, scratch
    )
    result = dict(row)
    result["candidate_class"] = candidate_class
    result["scoped_test_command"] = []
    result["test_patch_sha256"] = hashlib.sha256(test_patch).hexdigest()
    result["proof_a"] = "not run"
    result["proof_b"] = "not run"

    if not test_patch.strip():
        result.update({
            "verdict": "test-patch-empty",
            "proof_a": "test patch is empty after restricting the commit diff to tests_touched",
            "proof_b": "not run: no test patch to validate",
        })
        return result

    runner = scratch / "runners" / repo_name
    ensure_runner(source, runner)
    venv = scratch / "venvs" / repo_name
    plan = derive_plan(source, row, repo_name, venv)
    result["scoped_test_command"] = plan.display()
    log_dir = scratch / "logs" / candidate_class / f"{index:03d}-{repo_name}-{row['commit'][:12]}"
    base_env = os.environ.copy()
    base_env["CARGO_TARGET_DIR"] = str(scratch / "caches" / "cargo-target" / repo_name)
    base_env["GOTOOLCHAIN"] = "local"  # never auto-download/guess a Go toolchain
    base_env["CI"] = "1"

    try:
        reset_runner(runner, row["parent"])
        ok, why = apply_patch(runner, test_patch, patch_dir / "test.patch")
        if not ok:
            result.update({
                "verdict": "toolchain-issue",
                "proof_a": f"test patch apply failed: {why}",
                "proof_b": "not run: test patch could not be constructed on the recorded parent",
            })
            return result
        proof_a = run_proof(
            runner, source, row, repo_name, plan, venv, "A", timeout_seconds,
            log_dir, base_env,
        )
        result["proof_a"] = proof_a.summary
        if proof_a.state in ("timeout", "setup-failure", "zero-tests"):
            result.update({
                "verdict": "toolchain-issue",
                "proof_b": f"not run: proof A ended as {proof_a.state}; timeout/setup/null-test failures are not guessed",
            })
            return result

        reset_runner(runner, row["parent"])
        ok, why = apply_patch(runner, full_patch, patch_dir / "full.patch")
        if not ok:
            result.update({
                "verdict": "toolchain-issue",
                "proof_b": f"full commit diff apply failed: {why}",
            })
            return result
        tree_ok, tree_why = verify_full_tree(runner, row["commit"])
        if not tree_ok:
            result.update({"verdict": "toolchain-issue", "proof_b": tree_why})
            return result
        proof_b = run_proof(
            runner, source, row, repo_name, plan, venv, "B", timeout_seconds,
            log_dir, base_env,
        )
        result["proof_b"] = f"{tree_why}; {proof_b.summary}"
        if proof_b.state != "pass":
            result["verdict"] = "toolchain-issue"
        elif proof_a.state == "pass":
            result["verdict"] = "test-not-failing"
        else:
            result["verdict"] = "valid"
        return result
    except Exception as exc:
        result.update({
            "verdict": "toolchain-issue",
            "proof_a": result.get("proof_a", "not run"),
            "proof_b": f"validator exception: {type(exc).__name__}: {exc}",
        })
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serious", type=Path, default=DEFAULT_SERIOUS)
    parser.add_argument("--mechanical", type=Path, default=DEFAULT_MECHANICAL)
    parser.add_argument("--harvest-repos", type=Path, default=DEFAULT_HARVEST)
    parser.add_argument("--scratch-root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES)
    parser.add_argument("--classes", default="S,M", help="comma-separated S,M; order is always S then M")
    parser.add_argument("--limit", type=int, default=0, help="maximum new candidates this invocation")
    parser.add_argument("--timeout", type=int, default=900, help="seconds per proof, including setup")
    parser.add_argument("--dry-run", action="store_true", help="derive and print test commands without patches/tests")
    args = parser.parse_args()

    harvest_root = args.harvest_repos.resolve()
    scratch = (args.scratch_root or (harvest_root.parent / "validation-v2")).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    serious = load_jsonl(args.serious)
    mechanical = load_jsonl(args.mechanical)
    all_candidates = [("S", r) for r in serious] + [("M", r) for r in mechanical]
    selected = {x.strip().upper() for x in args.classes.split(",") if x.strip()}
    ordered = [(cls, r) for cls, r in all_candidates if cls in selected]
    versions = tool_versions()

    existing = repair_and_load_output(args.output)
    done = {candidate_key(row_class(r), r) for r in existing}
    write_notes(args.notes, scratch, all_candidates, existing, versions)

    if args.dry_run:
        for index, (cls, row) in enumerate(ordered, 1):
            repo_name = REPO_NAMES[row["repo"]]
            source = harvest_root / repo_name
            venv = scratch / "venvs" / repo_name
            plan = derive_plan(source, row, repo_name, venv)
            print(json.dumps({
                "class": cls, "index": index, "repo": repo_name, "commit": row["commit"],
                "command": plan.display(), "compile_only": plan.compile_only,
            }, ensure_ascii=False))
        return 0

    new_count = 0
    class_indices = Counter()
    for cls, row in all_candidates:
        class_indices[cls] += 1
        if cls not in selected:
            continue
        key = candidate_key(cls, row)
        if key in done:
            continue
        if args.limit and new_count >= args.limit:
            break
        index = class_indices[cls]
        repo_name = REPO_NAMES.get(row["repo"], row["repo"])
        typ = row.get("type") or row.get("category")
        print(
            f"[{cls} {index:02d}] {repo_name:5s} {typ:24s} {row['commit'][:12]}",
            file=sys.stderr, flush=True,
        )
        result = validate_one(cls, index, row, harvest_root, scratch, args.timeout)
        append_jsonl(args.output, result)
        existing.append(result)
        done.add(key)
        new_count += 1
        write_notes(args.notes, scratch, all_candidates, existing, versions)
        print(
            f"          -> {result['verdict']}: A={result['proof_a'][:100]} B={result['proof_b'][:100]}",
            file=sys.stderr, flush=True,
        )

    # Final parseability and mandatory-field check.
    final_rows = repair_and_load_output(args.output)
    for lineno, row in enumerate(final_rows, 1):
        for field in ("verdict", "proof_a", "proof_b", "scoped_test_command", "test_patch_sha256"):
            if field not in row:
                raise SystemExit(f"{args.output}:{lineno}: missing {field}")
        if row["verdict"] == "valid" and not (
            "proof A test" in row["proof_a"] and "proof B: pass" in row["proof_b"]
        ):
            raise SystemExit(f"{args.output}:{lineno}: invalid proof invariant for valid candidate")
    write_notes(args.notes, scratch, all_candidates, final_rows, versions)
    print(f"wrote {new_count} new rows; total={len(final_rows)} -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
