#!/usr/bin/env python3
"""Execute agent browser lifecycle contracts and precise source mutations."""
import argparse
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run_case(native):
    name = ("confined_web_uses_parent_runtime_and_reads_real_page" if native
            else "coding_task_does_not_start_a_browser")
    command = ["cargo", "test", "--locked", "--package", "greppy",
               "--features", "ci-test-assets", "--jobs", "2", "--test", "agent_p", name,
               "--", "--exact", "--test-threads=1", "--nocapture"]
    if native:
        command.append("--ignored")
    env = dict(os.environ, CARGO_BUILD_JOBS="2", RUST_TEST_THREADS="1")
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    print(result.stdout, result.stderr)
    return result, name


def require_positive(native):
    result, name = run_case(native)
    assert result.returncode == 0 and name in result.stdout and "test result: ok." in result.stdout


def mutation(source, old, new, expected, native):
    path = ROOT / source
    original = path.read_text()
    assert original.count(old) == 1, (source, old)
    try:
        path.write_text(original.replace(old, new))
        result, _ = run_case(native)
        assert result.returncode == 101 and expected in result.stdout + result.stderr
        print("mutation: expected incident detected:", expected, flush=True)
    finally:
        path.write_text(original)
    require_positive(native)
    subprocess.run(["git", "diff", "--exit-code", "--", str(path)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    if args.native:
        runtime = Path(os.environ["GREPPY_WEB_RUNTIME"])
        assert runtime.is_file(), runtime
    require_positive(args.native)
    if args.native:
        source = "crates/cli/src/web/common.rs"
        mutation(source,
                 "let (run_id, identity) = runtime_run_id(identity_root);",
                 "let _ = identity_root;\n    let (run_id, identity) = runtime_run_id(root);",
                 "confined web tool did not read the genuine native page", True)
        mutation(source,
                 "let owned = std::mem::take(\n        &mut *agent_runtimes()\n            .lock()\n            .unwrap_or_else(|error| error.into_inner()),\n    );",
                 "let owned: AgentRuntimeScopes = Default::default();",
                 "agent exit retained its authenticated runtime socket", True)
    else:
        anchor = "// The web-only attach hook starts the runtime lazily in this parent."
        mutation("crates/cli/src/agent.rs", anchor,
                 anchor + '\n                let _ = crate::web::prepare_agent_runtime(&["web".to_owned(), "open".to_owned(), "http://127.0.0.1/".to_owned()], workspace.worktree_path());',
                 "coding-only task started a browser runtime", False)


if __name__ == "__main__":
    main()
