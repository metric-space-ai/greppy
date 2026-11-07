#!/usr/bin/env python3
"""Exercise the production render-pool code with real Rayon workers and faults."""
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/web-runtime/vendor/servo-paint/shared_render_workers.rs"
PREFIX = "shared_render_workers_contract::tests::"
CONTRACTS = (
    "overlapping_renderers_share_real_workers_and_release_the_last_owner",
    "restart_drains_running_render_work_before_starting_new_workers",
    "retiring_worker_refuses_registry_reentry_before_the_embedder_lock",
    "registry_drop_does_not_join_an_externally_owned_pool",
)
FAULTS = (
    ("reuse", "return Ok(pool);",
     "return rayon::ThreadPoolBuilder::new().num_threads(count.max(1)).build().map(std::sync::Arc::new).map_err(|error| std::io::Error::other(error.to_string()));",
     CONTRACTS[0], "overlapping renderers started duplicate worker threads"),
    ("drain", "join_workers(previous.workers)?;", "drop(previous.workers);",
     CONTRACTS[1], "new render workers overlapped retiring generation"),
    ("retention", "*current = Some(Generation {",
     "std::mem::forget(pool.clone());\n                *current = Some(Generation {",
     CONTRACTS[0], "renderer pool survived its last owner"),
    ("worker-lock", "if IN_RENDER_WORKER.with(Cell::get) {", "if false {",
     CONTRACTS[2], "renderer worker blocked on the retiring pool registry"),
)


def run(case=None):
    command = ["cargo", "test", "--locked", "--manifest-path", "crates/web-runtime/Cargo.toml",
               "-p", "web-runtime", "--no-default-features", "--lib",
               PREFIX + case if case else "shared_render_workers_contract",
               "--", "--test-threads=1", "--nocapture"]
    if case:
        command.append("--exact")
    process = subprocess.Popen(command, cwd=ROOT,
        env=dict(os.environ, CARGO_BUILD_JOBS="2", RUST_TEST_THREADS="1"),
        start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=240)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        raise AssertionError("render pool contract process exceeded its finite deadline:\n" +
                             stdout + stderr)
    return process.returncode, stdout + stderr


def positive():
    code, output = run()
    if code != 0 or any(PREFIX + name not in output for name in CONTRACTS):
        raise AssertionError("production render pool contract failed:\n" + output)
    for line in output.splitlines():
        if "render pool:" in line:
            print(line, flush=True)


def main():
    positive()
    original = SOURCE.read_bytes()
    text = original.decode()
    for label, old, fault, case, incident in FAULTS:
        if text.count(old) != 1:
            raise AssertionError("render pool source target is not unique: " + label)
        try:
            SOURCE.write_text(text.replace(old, fault))
            code, output = run(case)
            print("render-pool fault: " + label + "\n" + output, flush=True)
            if code != 101 or incident not in output:
                raise AssertionError("wrong render pool source-fault outcome: " + label)
            print("render-pool source counter rejected: " + incident, flush=True)
        finally:
            SOURCE.write_bytes(original)
        if SOURCE.read_bytes() != original:
            raise AssertionError("render pool source did not restore byte-exactly")
        positive()
    subprocess.run(["git", "diff", "--exit-code", "--", str(SOURCE)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
