#!/usr/bin/env python3
"""Real native fill focus/input effects and an unwanted-pointer source counter."""
from pathlib import Path
import os
import subprocess
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'crates/web-runtime/runtime/src/content_worker.rs'
GOOD = "            \"locator.fill\" => {\n                self.resolve_actionable(&params)?;\n                let page_id = required_str(&params, \"page\")?;\n                let value = required_str(&params, \"value\")?;\n                let (webview, _) = self.page(&page_id)?.clone();\n                self.present_exclusively(&webview);\n                // fill_script focuses the field. Pointer input here is both\n                // redundant work and an observable, unintended click.\n"
BAD = "            \"locator.fill\" => {\n                let resolved = self.resolve_actionable(&params)?;\n                let page_id = required_str(&params, \"page\")?;\n                let value = required_str(&params, \"value\")?;\n                let (webview, delegate) = self.page(&page_id)?.clone();\n                self.present_exclusively(&webview);\n                click_at(\n                    &webview,\n                    &delegate,\n                    resolved.x,\n                    resolved.y,\n                    resolved.width,\n                    resolved.height,\n                    || self.spin_all_event_loops(),\n                )?;\n                self.spin_all_event_loops();\n"
COMMAND = ['cargo', 'test', '--locked', '--jobs', '2',
           '--manifest-path', 'crates/web-runtime/Cargo.toml',
           '--package', 'web-runtime', '--test', 'session-daemon',
           'web_click_and_fill_drive_a_fixture',
           '--', '--exact', '--test-threads=1', '--nocapture']
def run():
    result = subprocess.run(COMMAND, cwd=ROOT, env=dict(os.environ, CARGO_BUILD_JOBS='2', RUST_TEST_THREADS='1'),
                            capture_output=True, text=True, timeout=600)
    print(result.stdout, result.stderr)
    return result
def positive():
    result = run()
    assert result.returncode == 0 and 'web_click_and_fill_drive_a_fixture ... ok' in result.stdout
def main():
    original = SOURCE.read_text()
    assert original.count(GOOD) == 1, 'native fill source target is not unique'
    positive()
    try:
        SOURCE.write_text(original.replace(GOOD, BAD, 1))
        result = run()
        assert result.returncode == 101 and 'fill unexpectedly dispatched pointer events' in result.stdout + result.stderr
        print('mutation: native fill rejected unintended pointer dispatch', flush=True)
    finally:
        SOURCE.write_text(original)
    positive()
    subprocess.run(['git', 'diff', '--exit-code', '--', str(SOURCE)], cwd=ROOT, check=True)
if __name__ == '__main__':
    main()
