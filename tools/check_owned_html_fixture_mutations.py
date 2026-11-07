#!/usr/bin/env python3
"""Qualify scope-owned HTML fixtures with real TCP sockets and source faults."""
from pathlib import Path
import json
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "crates/web-runtime/runtime/tests/support/owned_html_fixture.rs"


def qualify_http_scenarios(scratch):
    support = ROOT / "crates/web-runtime/runtime/tests/support"
    source = support / "recorded_http_fixture.rs"
    original = source.read_text()
    wrapper = Path(scratch) / "http_scenarios.rs"
    wrapper.write_text(
        "#[path = " + json.dumps(str(source), ensure_ascii=False) + "]\n"
        "mod recorded_http_fixture;\n"
        "#[path = " + json.dumps(str(support / "owned_http_scenarios.rs"), ensure_ascii=False) + "]\n"
        "mod owned_http_scenarios;\n"
    )
    binary = str(Path(scratch) / "owned-http-scenarios")

    def run():
        subprocess.run(["rustc", "--edition=2021", "--test", "-D", "warnings",
                        str(wrapper), "-o", binary], cwd=ROOT, check=True, timeout=60)
        result = subprocess.run([binary, "owned_http_scenarios", "--test-threads=1", "--nocapture"],
                                cwd=ROOT, capture_output=True, text=True, timeout=30)
        print(result.stdout, result.stderr, flush=True)
        return result

    def positive():
        result = run()
        assert result.returncode == 0 and "test result: ok." in result.stdout

    positive()
    old = """        self.stop.store(true, Ordering::Relaxed);
        self.worker.take().unwrap().join().unwrap();"""
    fault = """        let _ = &self.stop;
        let _ = self.worker.take();"""
    scenarios = support / "owned_http_scenarios.rs"
    saved = {source: original, scenarios: scenarios.read_text()}
    cases = [
        (source, old, fault, "scenario fixture listener survived guard drop"),
        (source, "std::borrow::Cow::Borrowed(&body[..content_length])",
         "std::borrow::Cow::Borrowed(&[])",
         "POST fixture discarded or changed the real body"),
        (scenarios, 'request.header("cookie")', 'None::<&str>',
         "cookie fixture discarded or changed the real header"),
    ]
    for target, old, fault, marker in cases:
        original_target = saved[target]
        assert original_target.count(old) == 1, "HTTP scenario source target is not unique"
        try:
            target.write_text(original_target.replace(old, fault))
            result = run()
            assert result.returncode == 101 and marker in result.stdout + result.stderr
            print("mutation: rejected " + marker, flush=True)
        finally:
            target.write_text(original_target)
        positive()
    assert all(target.read_text() == original_target for target, original_target in saved.items())


def main():
    original = SOURCE.read_text()
    cases = [
        (
            """        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            // All socket operations are nonblocking; even a silent/slow client
            // cannot keep the fixture alive after its owning test has finished.
            let _ = worker.join();
        }""",
            """        // Deliberately restore the old detached fixture lifetime.
        let _ = &self.stop;
        let _ = self.worker.take();""",
            "fixture listener survived guard drop",
        ),
        ("    drop(html);", "    std::mem::forget(html);",
         "fixture body survived guard drop"),
    ]
    with tempfile.TemporaryDirectory(prefix="greppy-owned-html-fixture-",
                                     dir=os.environ.get("RUNNER_TEMP") or os.environ.get("TMPDIR")) as scratch:
        binary = str(Path(scratch) / "owned-html-fixture")

        def run():
            subprocess.run(["rustc", "--edition=2021", "--test", "-D", "warnings",
                            str(SOURCE), "-o", binary], cwd=ROOT, check=True, timeout=60)
            result = subprocess.run([binary, "owned_html_fixture", "--test-threads=1", "--nocapture"],
                                    cwd=ROOT, capture_output=True, text=True, timeout=30)
            print(result.stdout, result.stderr, flush=True)
            return result

        def positive():
            result = run()
            assert result.returncode == 0 and "test result: ok." in result.stdout

        positive()
        for old, fault, marker in cases:
            assert original.count(old) == 1, "owned HTML fixture mutation target is not unique"
            try:
                SOURCE.write_text(original.replace(old, fault))
                result = run()
                assert result.returncode == 101 and marker in result.stdout + result.stderr
                print("mutation: rejected " + marker, flush=True)
            finally:
                SOURCE.write_text(original)
            positive()
        assert SOURCE.read_text() == original
        qualify_http_scenarios(scratch)


if __name__ == "__main__":
    main()
