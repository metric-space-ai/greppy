#!/usr/bin/env python3
"""Counterproofs for truthful pre-DOM deadlines; never change the action budget."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "crates/web-runtime/runtime/src/locator_diagnostics.rs"
CONTENT = ROOT / "crates/web-runtime/runtime/src/content_worker.rs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    command = ["cargo", "test", "--locked", "--jobs", "2",
        "--manifest-path", "crates/web-runtime/Cargo.toml", "--package", "web-runtime"]
    if args.native:
        command += ["--test", "session-daemon", "native_actions_return_current_state"]
        cases = [(CONTENT,
            'let mut last = String::from("failed_check=unobserved");',
            'let mut last = String::from("failed_check=stable");',
            "unobserved DOM deadline fabricated a failed check")]
        no_match_marker = "unobserved DOM deadline fabricated NO_MATCH"
    else:
        command += ["--lib", "--no-default-features",
            "unobserved_actionability_deadline_never_fabricates_a_failed_check"]
        cases = [(POLICY,
            'format!("failed_check=unobserved; {last}")',
            'format!("failed_check=stable; {last}")',
            "unobserved actionability deadline fabricated a failed check")]
        no_match_marker = "unobserved actionability deadline fabricated NO_MATCH"
    cases.append((POLICY,
        'return ("TIMEOUT", UNOBSERVED_DOM_RECOVERY);',
        'return ("NO_MATCH", UNOBSERVED_DOM_RECOVERY);',
        no_match_marker))
    command += ["--", "--test-threads=1", "--nocapture"]

    def run():
        result = subprocess.run(command, cwd=ROOT, capture_output=True,
                                text=True, timeout=600 if args.native else 180)
        print(result.stdout, result.stderr)
        return result

    def positive():
        result = run()
        assert result.returncode == 0 and "test result: ok." in result.stdout

    positive()
    for source, old, fault, marker in cases:
        original = source.read_text()
        assert original.count(old) == 1, "unobserved-deadline source target is not unique"
        try:
            source.write_text(original.replace(old, fault))
            result = run()
            assert result.returncode == 101 and marker in result.stdout + result.stderr
            print("mutation: rejected " + marker, flush=True)
        finally:
            source.write_text(original)
        positive()
        subprocess.run(["git", "diff", "--exit-code", "--", str(source)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
