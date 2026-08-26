#!/usr/bin/env python3
"""Create the deterministic, representative portable-CoW performance fixture."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess


SCHEMA = "greppy.portable-cow-fixture.v3"
FIXED_GIT_DATE = "2000-01-01T00:00:00Z"


def run(*args: str, cwd: pathlib.Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(args, cwd=cwd, env=env, check=True)


def rust_sources(modules: int) -> dict[str, str]:
    sources = {
        f"rust/module_{index:03}.rs": (
            f"pub fn transform_{index:03}(input: u64) -> u64 {{\n"
            f"    input.rotate_left({index % 31 + 1}).wrapping_mul({index * 2 + 3})\n"
            "}\n"
        )
        for index in range(modules)
    }
    declarations = "\n".join(f"mod module_{index:03};" for index in range(modules))
    calls = "\n".join(
        f"        value = module_{index:03}::transform_{index:03}(value);"
        for index in range(modules)
    )
    sources["rust/main.rs"] = (
        f"{declarations}\n\n"
        "fn main() {\n"
        "    let mut value = 1_u64;\n"
        "    for _ in 0..256 {\n"
        f"{calls}\n"
        "    }\n"
        "    assert_ne!(value, 0);\n"
        "}\n"
    )
    return sources


def python_sources(modules: int) -> dict[str, str]:
    sources = {
        f"python/module_{index:03}.py": (
            f"def transform_{index:03}(value):\n"
            f"    return ((value << {index % 7 + 1}) ^ (value + {index * 2 + 3}))"
            " & 0xFFFFFFFF\n"
        )
        for index in range(modules)
    }
    imports = "\n".join(
        f"from module_{index:03} import transform_{index:03}" for index in range(modules)
    )
    calls = "\n".join(
        f"        value = transform_{index:03}(value)" for index in range(modules)
    )
    sources["python/test_sample.py"] = (
        f"{imports}\n\n"
        "value = 1\n"
        "for _ in range(100_000):\n"
        f"{calls}\n"
        "assert isinstance(value, int) and 0 <= value <= 0xFFFFFFFF\n"
    )
    return sources


def node_sources(modules: int) -> dict[str, str]:
    sources = {
        f"node/module_{index:03}.js": (
            f"module.exports = value => "
            f"(((value << {index % 7 + 1}) ^ (value + {index * 2 + 3})) >>> 0);\n"
        )
        for index in range(modules)
    }
    imports = "\n".join(
        f"const transform{index:03} = require('./module_{index:03}');"
        for index in range(modules)
    )
    calls = "\n".join(
        f"    value = transform{index:03}(value);" for index in range(modules)
    )
    sources["node/test.js"] = (
        f"{imports}\n\n"
        "let value = 1;\n"
        "for (let round = 0; round < 500_000; round += 1) {\n"
        f"{calls}\n"
        "}\n"
        "if (!Number.isInteger(value) || value < 0) process.exit(1);\n"
    )
    sources["node/package.json"] = json.dumps(
        {
            "name": "greppy-portable-cow-performance-fixture",
            "private": True,
            "scripts": {"test": "node test.js"},
            "version": "1.0.0",
        },
        indent=2,
        sort_keys=True,
    ) + "\n"
    return sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--files", type=int, default=300_000)
    parser.add_argument("--modules", type=int, default=24)
    args = parser.parse_args()
    if args.root.exists():
        raise SystemExit(f"refusing to replace existing fixture: {args.root}")
    if args.modules < 16:
        raise SystemExit("representative fixture requires at least 16 modules per toolchain")

    projects: dict[str, str] = {}
    projects.update(rust_sources(args.modules))
    projects.update(python_sources(args.modules))
    projects.update(node_sources(args.modules))
    manifest_path = ".greppy-portable-cow-fixture.json"
    base_files = args.files - len(projects) - 1
    if base_files < 1:
        raise SystemExit("requested fixture is too small for its toolchain projects")

    args.root.mkdir()
    run("git", "init", "-q", "-b", "main", cwd=args.root)
    run("git", "config", "user.email", "perf@greppy.invalid", cwd=args.root)
    run("git", "config", "user.name", "Greppy Performance Gate", cwd=args.root)
    process = subprocess.Popen(
        ["git", "fast-import", "--quiet"],
        cwd=args.root,
        stdin=subprocess.PIPE,
    )
    stream = process.stdin
    assert stream is not None
    stream.write(b"blob\nmark :1\ndata 5\nbase\n")
    stream.write(b"commit refs/heads/main\nmark :2\n")
    stream.write(b"committer Greppy Gate <perf@greppy.invalid> 946684800 +0000\n")
    stream.write(b"data 7\nfixture\n")
    for index in range(base_files):
        stream.write(
            f"M 100644 :1 files/{index // 1000:03}/{index:06}.txt\n".encode()
        )
    stream.write(b"\n")
    stream.close()
    if process.wait() != 0:
        raise SystemExit("git fast-import failed")
    run("git", "reset", "--hard", "-q", "main", cwd=args.root)

    for relative, content in projects.items():
        path = args.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    manifest = {
        "schema": SCHEMA,
        "tracked_files": args.files,
        "modules_per_toolchain": args.modules,
        "rust_sources": args.modules + 1,
        "python_sources": args.modules + 1,
        "node_sources": args.modules + 2,
    }
    (args.root / manifest_path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    run("git", "add", "rust", "python", "node", manifest_path, cwd=args.root)
    git_env = os.environ.copy()
    git_env.update(
        {
            "GIT_AUTHOR_DATE": FIXED_GIT_DATE,
            "GIT_COMMITTER_DATE": FIXED_GIT_DATE,
        }
    )
    run(
        "git",
        "commit",
        "-qm",
        "add representative toolchain fixtures",
        cwd=args.root,
        env=git_env,
    )
    count = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=args.root
    ).count(b"\0")
    if count != args.files:
        raise SystemExit(f"fixture count is {count}, expected {args.files}")
    print(
        json.dumps(
            {
                "commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=args.root, text=True
                ).strip(),
                **manifest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
