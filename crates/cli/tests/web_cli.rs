//! `greppy web` CLI wiring. Does not require JavaScript engines.

use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn run(args: &[&str]) -> (i32, String, String) {
    let output = Command::new(bin())
        .args(args)
        .env_remove("GREPPY_WEB_RUNTIME_SUPERVISOR")
        .env_remove("GREPPY_WEB_CONTROLLER_WORKER")
        .env_remove("GREPPY_WEB_CONTENT_WORKER")
        .env("PATH", "/usr/bin:/bin")
        .output()
        .expect("greppy web");
    (
        output.status.code().unwrap_or(1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn web_is_not_grep_passthrough() {
    let (code, stdout, stderr) = run(&["web", "status", "--json"]);
    assert_ne!(code, 2, "web must not be unknown clap usage: {stderr}");
    assert!(
        stdout.contains("runtime_unavailable")
            || stdout.contains("experimental")
            || stdout.contains("greppy.web-runtime.v1"),
        "stdout={stdout} stderr={stderr}"
    );
}

#[test]
fn web_status_json_is_unavailable_without_runtime_bins() {
    let (code, stdout, _stderr) = run(&["web", "status", "--json"]);
    assert_eq!(code, 31, "stdout={stdout}");
    assert!(
        stdout.contains("\"schema\":\"greppy.web-runtime.v1\"")
            || stdout.contains("greppy.web-runtime.v1")
    );
    assert!(stdout.contains("runtime_unavailable") || stdout.contains("error"));
}

#[test]
fn web_run_requires_session() {
    let (code, stdout, _stderr) = run(&["web", "run", "--script-file", "x.js", "--json"]);
    assert_eq!(code, 30, "stdout={stdout}");
    assert!(stdout.contains("session"));
}
