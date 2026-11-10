#!/usr/bin/env python3
"""Exercise the real Native action path with and without its empty-ledger fast path."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/web-runtime/runtime/src/content_worker.rs"
TEST = "locator_click_waits_for_actionable_target"
MARKER = "empty pump ledger performed a redundant window property walk"
COMMAND = [
    "cargo", "test", "--locked",
    "--manifest-path", "crates/web-runtime/Cargo.toml",
    "--package", "web-runtime", "--test", "session-daemon", TEST,
    "--", "--exact", "--test-threads=1", "--nocapture",
]
GUARD = """    fn settle_pump_tokens(&self, webview: &WebView) {
        // Each confirmed per-token cleanup advances the floor and removes its
        // ledger entry. Avoid another JS roundtrip and two whole-window walks
        // when nothing remains; unconfirmed cleanup still requires the sweep.
        if self.pump_pending.borrow().is_empty() {
            return;
        }
"""
UNCONDITIONAL = """    fn settle_pump_tokens(&self, webview: &WebView) {
"""


def exercise():
    result = subprocess.run(
        COMMAND, cwd=ROOT, capture_output=True, text=True, timeout=900,
    )
    output = result.stdout + result.stderr
    print(output, flush=True)
    return result.returncode, output


def require_positive():
    code, output = exercise()
    assert code == 0 and f"test {TEST} ..." in output, (
        "real Native actionability positive control did not pass", code
    )
    assert "test result: ok. 1 passed" in output, "positive selected no real test"


def main():
    original = SOURCE.read_text()
    assert original.count(GUARD) == 1, "empty-ledger guard must be unique"
    require_positive()
    try:
        SOURCE.write_text(original.replace(GUARD, UNCONDITIONAL))
        code, output = exercise()
        assert code == 101 and MARKER in output, (
            "unconditional cleanup did not fail its real window-walk contract", code
        )
        print("confirmed empty-ledger guard omission: " + MARKER, flush=True)
    finally:
        SOURCE.write_text(original)
    require_positive()
    subprocess.run(["git", "diff", "--exit-code", "--", str(SOURCE)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
