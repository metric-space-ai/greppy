#!/usr/bin/env python3
"""Prove live-child FD inheritance and reject a completed lifetime fixture."""
from pathlib import Path
import os
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/cli/src/inference_daemon.rs"
TEST = "inference_daemon::tests::concurrent_ordinary_spawn_keeps_lease_while_detached_child_drops_it"
COMMAND = [
    "cargo", "test", "--locked", "--jobs", "2", "-p", "greppy",
    "--features", "ci-test-assets", "--lib", TEST,
    "--", "--exact", "--test-threads=1",
]


def run(cause=None):
    env = os.environ.copy()
    env["RUST_MIN_STACK"] = "8388608"
    result = subprocess.run(
        COMMAND, cwd=ROOT, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600,
    )
    if cause:
        valid = (result.returncode == 101 and TEST + " ... FAILED" in result.stdout
                 and cause in result.stdout)
    else:
        valid = result.returncode == 0 and TEST + " ... ok" in result.stdout
    if not valid:
        raise AssertionError(result.stdout)


original = SOURCE.read_text()
run()
print("mutation: concurrently spawned live ordinary child retains the lease; live detached child releases it", flush=True)
start = original.index("pub(crate) fn seal_detached_child_fds(command:")
end = original.index("\n#[cfg(not(unix))]", start)
no_seal = """pub(crate) fn seal_detached_child_fds(_command: &mut std::process::Command) -> std::io::Result<()> {
    Ok(())
}
"""
try:
    SOURCE.write_text(original[:start] + no_seal + original[end:])
    run("detached child retained the lease")
    print("mutation: bypassing production child FD sealing is red at the real inherited kernel lease", flush=True)
finally:
    SOURCE.write_text(original)

join = "        let mut ordinary = ordinary.join().unwrap();"
assert original.count(join) == 1
try:
    SOURCE.write_text(original.replace(join, join + "\n        ordinary.0.stdin.take();\n        ordinary.0.wait().unwrap();", 1))
    run("ordinary lease probe exited before the inheritance assertion")
    print("mutation: completing the ordinary child's lifetime is red before checking lease inheritance", flush=True)
finally:
    SOURCE.write_text(original)
run()
assert SOURCE.read_text() == original, "production source not restored byte-for-byte"
print("mutation: source restored byte-for-byte; live-child kernel lease contract green", flush=True)
