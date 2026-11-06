#!/usr/bin/env python3
"""Controlled mutations of the actual web resource lifecycle paths."""
import argparse
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
def prove(path, old, new, command, marker):
    original = path.read_text()
    assert original.count(old) == 1, (path, old)
    try:
        path.write_text(original.replace(old, new))
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=120)
        print(result.stdout, result.stderr)
        assert result.returncode != 0 and marker in result.stdout + result.stderr, "mutation escaped its contract"
        print("MUTATION_RED", path, marker, flush=True)
    finally:
        path.write_text(original)
    subprocess.run(command, cwd=ROOT, check=True, timeout=120)
    subprocess.run(["git", "diff", "--exit-code", "--", str(path)], cwd=ROOT, check=True)
    print("RESTORED_GREEN", path, flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", action="store_true")
    parser.add_argument("--slots", action="store_true")
    args = parser.parse_args()
    if args.slots:
        prove(ROOT / "crates/web-runtime/runtime/src/object_usage.rs",
              "session.objects.forget_page(&page);", "// omitted live-page release",
              ["cargo", "test", "--locked", "--manifest-path", "crates/web-runtime/Cargo.toml",
               "-p", "web-runtime", "--lib", "--no-default-features",
               "live_object_slots_follow_acknowledged_identity_not_historical_creations", "--", "--test-threads=1"],
              "live_object_slots_follow_acknowledged_identity_not_historical_creations")
    elif args.records:
        prove(ROOT / "crates/web-runtime/runtime/src/retained_records.rs",
              "if self.budget.reserve(size)", "if true",
              ["cargo", "test", "--locked", "--manifest-path", "crates/web-runtime/Cargo.toml",
               "-p", "web-runtime", "--lib", "--no-default-features",
               "retained_records_bound_bytes_before_retaining_and_account_updates", "--", "--test-threads=1"],
              "retained_records_bound_bytes_before_retaining_and_account_updates")
    else:
        prove(ROOT / "crates/web-runtime/runtime/js/playwright.mjs",
              "page._ownedContext = context;", "// omitted implicit context ownership",
              ["node", "--test", "crates/web-runtime/runtime/tests/playwright-resource-lifecycle.cjs"],
              "browser.newPage owns its implicit context")
        prove(ROOT / "crates/web-runtime/runtime/js/playwright.mjs",
              "this._context._pages = this._context._pages.filter(page => page !== this)",
              "this._context._pages = this._context._pages",
              ["node", "--test", "crates/web-runtime/runtime/tests/playwright-resource-lifecycle.cjs"],
              "closed pages and contexts release wrapper ownership")
        prove(ROOT / "crates/web-runtime/runtime/js/playwright.mjs",
              "  _primeNetworkHistory() {", "  _primeNetworkHistory() { return;",
              ["node", "--test", "crates/web-runtime/runtime/tests/playwright-resource-lifecycle.cjs"],
              "a late network observer does not replay")
