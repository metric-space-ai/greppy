//! Diagnostics never reach stdout: agents and scripts parse stdout as the
//! command result, so a tracing line there corrupts it (0.4.1: a WARN from
//! retained-capture cleanup appeared in sandboxed tool output).

use std::path::PathBuf;
use std::process::Command;

fn scratch(tag: &str) -> PathBuf {
    // Canonical: the CLI records workspaces under their canonical path and the
    // macOS default temp dir is a symlink into /private/var.
    let dir = std::env::temp_dir().canonicalize().unwrap().join(format!(
        "greppy-logs-on-stderr-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("repo/.git")).unwrap();
    std::fs::write(dir.join("repo/lib.rs"), "pub fn marker() -> i32 { 1 }\n").unwrap();
    dir
}

#[test]
fn trace_level_logging_stays_off_stdout() {
    let dir = scratch("trace");
    let output = Command::new(env!("CARGO_BIN_EXE_greppy"))
        .args(["where-am-i"])
        .current_dir(dir.join("repo"))
        .env("GREPPY_STORE_DIR", dir.join("store"))
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env("GREPPY_LOG", "trace")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "stdout={stdout} stderr={stderr}");
    for line in stdout.lines() {
        assert!(
            !(line.contains("\"level\"") && line.contains("\"target\"")),
            "tracing line leaked to stdout: {line}"
        );
    }
    assert!(
        stdout.contains("lib.rs") || stdout.contains("rust"),
        "where-am-i result missing from stdout: {stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
