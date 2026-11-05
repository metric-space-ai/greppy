#!/usr/bin/env python3
"""Check real daemon watchdog contracts in the isolated CI checkout."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/cli/src/inference_daemon.rs"
CASES = [
    (
        "started.max(status.state_started)",
        "started",
        "cold_model_load_does_not_consume_ready_watchdog_budget",
        "cold model load consumed the ready inference budget",
    ),
    (
        "Some(load_budget) if state == LifecycleState::Loading => state_elapsed >= load_budget,",
        "Some(_load_budget) if state == LifecycleState::Loading => state_elapsed >= hard_timeout,",
        "summary_loading_watchdog_uses_model_load_budget",
        "summary model load ended before its load budget",
    ),
]


def run(case=None):
    command = [
        "cargo", "test", "--locked", "--jobs", "2", "-p", "greppy",
        "--features", "ci-test-assets", "--lib",
    ]
    if case:
        command += ["inference_daemon::tests::" + case, "--", "--exact"]
    else:
        command += ["inference_daemon::tests::", "--"]
    command += ["--nocapture", "--test-threads=1"]
    return subprocess.run(
        command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=600, check=False,
    )


def main():
    original = SOURCE.read_bytes()
    green = run()
    if green.returncode != 0:
        print(green.stdout)
        raise SystemExit("fixed daemon contracts did not pass")
    for old, mutant, case, expected in CASES:
        if original.count(old.encode()) != 1:
            raise SystemExit(f"mutation target must match once: {case}")
        try:
            SOURCE.write_bytes(original.replace(old.encode(), mutant.encode(), 1))
            red = run(case)
            if red.returncode != 101 or expected not in red.stdout or "test result: FAILED" not in red.stdout:
                print(red.stdout)
                raise SystemExit(f"watchdog mutation did not fail at {case}")
            print(f"mutation: watchdog regression turns {case} red", flush=True)
        finally:
            SOURCE.write_bytes(original)
    restored = run()
    if restored.returncode != 0:
        print(restored.stdout)
        raise SystemExit("restored daemon contracts did not pass")
    print("mutation: restored daemon phase and load-budget contracts green", flush=True)
    subprocess.run(["git", "diff", "--exit-code", "--", str(SOURCE)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
