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


def mutation(source, old, new, expected, native, *, companion=None):
    changes = [(ROOT / source, old, new)]
    if companion is not None:
        changes.append((ROOT / companion[0], companion[1], companion[2]))
    originals = {}
    for path, old, _ in changes:
        original = path.read_text()
        assert original.count(old) == 1, (str(path), old)
        originals[path] = original
    try:
        for path, old, new in changes:
            path.write_text(originals[path].replace(old, new))
        result, _ = run_case(native)
        assert result.returncode == 101 and expected in result.stdout + result.stderr
        print("mutation: expected incident detected:", expected, flush=True)
    finally:
        for path, original in originals.items():
            path.write_text(original)
    require_positive(native)
    subprocess.run(["git", "diff", "--exit-code", "--",
                    *(str(path) for path in originals)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    if args.native:
        runtime = Path(os.environ["GREPPY_WEB_RUNTIME"])
        assert runtime.is_file(), runtime
    require_positive(args.native)
    if args.native:
        # Keep mode=Enforce and its JSON label, but omit real tool confinement.
        # Only the actual outside-root write oracle may reject this fault.
        mutation("crates/agent/src/sandbox.rs",
                 "SandboxMode::Enforce(spec) => apply_enforce(cmd, bin, args, spec),",
                 "SandboxMode::Enforce(_) => { *cmd = Command::new(bin); cmd.args(args.iter().map(AsRef::as_ref)); Ok(()) },",
                 "confined tool changed outside-write canary", True)
        source = "crates/cli/src/web/common.rs"
        mutation(source,
                 "let (run_id, identity) = runtime_run_id(identity_root);",
                 "let _ = identity_root;\n    let (run_id, identity) = runtime_run_id(root);",
                 "confined web tool did not read the genuine native page", True)
        mutation(source,
                 "let owned = std::mem::take(\n        &mut *agent_runtimes()\n            .lock()\n            .unwrap_or_else(|error| error.into_inner()),\n    );",
                 "let owned: AgentRuntimeScopes = Default::default();",
                 "agent exit retained its authenticated runtime socket", True,
                 # Only this deliberate omission adopts the test parent's
                 # official inherited capability before the normal claim.
                 # Recovery then uses the same in-memory token over FD4;
                 # no reconnect-cookie assumption or product-auth change.
                 companion=(
                     "crates/cli/src/agent.rs",
                     "        match crate::web_attach::claim_persistent_parent() {",
                     '        assert!(crate::web_attach::current_token().is_some(), "mutant cleanup control FD unavailable");\n'
                     "        match crate::web_attach::claim_persistent_parent() {"))
        mutation("crates/cli/tests/agent_p.rs",
                 "remaining = &candidate[documents.byte_offset()..];",
                 "break; // fault: ignore later runtime-status documents",
                 "web tool calls did not reuse one authenticated runtime", True)
    else:
        anchor = "// The web-only attach hook starts the runtime lazily in this parent."
        mutation("crates/cli/src/agent.rs", anchor,
                 anchor + '\n                let _ = crate::web::prepare_agent_runtime(&["web".to_owned(), "open".to_owned(), "http://127.0.0.1/".to_owned()], workspace.worktree_path());',
                 "coding-only task started a browser runtime", False)


if __name__ == "__main__":
    main()
