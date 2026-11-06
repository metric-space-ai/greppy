#!/usr/bin/env python3
"""Exercise real kernel FDs on a failed production worker bootstrap, then lose ownership."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/web-runtime/runtime/src/supervisor.rs"
TEST = "supervisor::tests::failed_worker_image_preparation_releases_kernel_descriptors"
DIAGNOSTIC = "failed worker image preparation leaked kernel descriptors"
COMMAND = [
    "cargo", "test", "--locked", "--jobs", "2",
    "--manifest-path", "crates/web-runtime/Cargo.toml",
    "-p", "web-runtime", "--no-default-features", "--lib", TEST,
    "--", "--exact", "--test-threads=1",
]


def run(mutant=False):
    result = subprocess.run(
        COMMAND, cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=600,
    )
    if mutant:
        valid = (result.returncode == 101
                 and TEST + " ... FAILED" in result.stdout
                 and DIAGNOSTIC in result.stdout)
    else:
        valid = (result.returncode == 0
                 and TEST + " ... ok" in result.stdout)
    if not valid:
        raise AssertionError(result.stdout)
    print("mutation: " + ("raw descriptor ownership loss is red at the kernel FD assertion"
                          if mutant else "failed bootstrap returns to the original kernel FD count"),
          flush=True)


original = SOURCE.read_text()
run()
mutated = original
for owner in ("cap_read", "proto_child"):
    old = f"let {owner}_fd = {owner}.as_raw_fd();"
    assert mutated.count(old) == 1
    mutated = mutated.replace(
        old, f"let {owner}_fd = std::os::fd::IntoRawFd::into_raw_fd({owner});")
    old = f"        drop({owner});"
    assert mutated.count(old) == 1
    mutated = mutated.replace(old, "")
try:
    SOURCE.write_text(mutated)
    run(mutant=True)
finally:
    SOURCE.write_text(original)
run()
subprocess.run(["git", "diff", "--exit-code", "--", str(SOURCE.relative_to(ROOT))],
               cwd=ROOT, check=True)
