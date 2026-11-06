#!/usr/bin/env python3
"""Prove CLI scratch-cache setup against real operator-cache symlinks."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "crates/cli/src/agent.rs"
FILTER = "toolchain_cache_roots_preserve_operator_links_and_refuse_run_escape"
TEST = "agent::tests::" + FILTER
CACHE_FILTER = "confined_child_home_and_cache_paths_are_inside_run_scratch"
FIXED = """    // Confined tool children use CARGO_HOME=<run scratch>/cargo and
    // CARGO_TARGET_DIR=<run scratch>/target (prepare_tool_cache_env). Those
    // paths are already below scratch_dir. Do not create or grant the unused
    // operator caches: existing storage symlinks are not writable run roots.
"""
LEGACY = """    // Mutation: restore the old, unused operator cache grants.
    let cargo = std::env::var_os("CARGO_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".cargo"));
    let cargo_registry = cargo.join("registry");
    let cargo_git = cargo.join("git");
    let _ = std::fs::create_dir_all(&cargo_registry);
    let _ = std::fs::create_dir_all(&cargo_git);
    roots.push(cargo_registry);
    roots.push(cargo_git);
"""


def run(package, name, red=False):
    args = ["cargo", "test", "--locked", "--jobs", "2", "-p", package, "--lib"]
    if package == "greppy":
        args += ["--features", "ci-test-assets"]
    env = os.environ.copy()
    env["RUST_MIN_STACK"] = "8388608"
    result = subprocess.run(
        [*args, name, "--", "--test-threads=1"], cwd=ROOT, env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600,
    )
    if red:
        if not (result.returncode == 101 and TEST + " ... FAILED" in result.stdout
                and "test result: FAILED" in result.stdout
                and "operator cache links must not block private scratch setup" in result.stdout
                and "symlink component" in result.stdout):
            raise AssertionError("Wrong mutation cause or unexpected success:\n" + result.stdout)
    elif not (result.returncode == 0 and name + " ... ok" in result.stdout):
        raise AssertionError("Contract did not execute successfully:\n" + result.stdout)


run("greppy", FILTER)
run("greppy-agent", CACHE_FILTER)
original = CLI.read_text()
if original.count(FIXED) != 1:
    raise AssertionError("Legacy root mutation anchor must be unique")
try:
    CLI.write_text(original.replace(FIXED, LEGACY, 1))
    run("greppy", FILTER, red=True)
    print("mutation: restoring operator Cargo grants makes real symlink-cache setup red (exit101)", flush=True)
finally:
    CLI.write_text(original)
run("greppy", FILTER)
run("greppy-agent", CACHE_FILTER)
print("mutation: restored narrow roots preserve cache aliases, refuse run escape and execute scratch Cargo/target writes", flush=True)
