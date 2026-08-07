//! Integration tests for the ripgrep-style passthrough.
//!
//! Agents routinely emit `rg`-flavoured invocations. The shipped binary
//! must (1) delegate byte-exactly to a real ripgrep when one exists,
//! (2) translate the common flag subset to real grep when none exists
//! (forced here via `GREPPY_REAL_RG=""`), and (3) refuse loudly — never
//! search wrongly — for untranslatable flags.

use std::path::PathBuf;
use std::process::{Command, Stdio};

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_greppy"))
}

fn unique_tempdir(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "greppy-rg-passthrough-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// Run greppy with ripgrep discovery disabled (translation path).
fn run_translated(args: &[&str], cwd: &PathBuf) -> std::process::Output {
    let mut cmd = Command::new(binary_path());
    cmd.args(args)
        .current_dir(cwd)
        .env("GREPPY_REAL_RG", "")
        .env("GREPPY_STORE_DIR", unique_tempdir("store"))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    cmd.output().expect("spawn greppy")
}

fn fixture_dir() -> PathBuf {
    let d = unique_tempdir("fixture");
    std::fs::write(d.join("a.txt"), "alpha\nBeta gamma\n").unwrap();
    std::fs::write(d.join("lib.rs"), "fn alpha() {}\n").unwrap();
    std::fs::create_dir_all(d.join("target")).unwrap();
    std::fs::write(d.join("target").join("gen.rs"), "fn alpha_generated() {}\n").unwrap();
    d
}

#[test]
fn smart_case_lowercase_matches_uppercase_line() {
    let d = fixture_dir();
    let out = run_translated(&["--smart-case", "beta", "."], &d);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Beta gamma"), "stdout: {stdout}");
    assert_eq!(out.status.code(), Some(0));
}

#[test]
fn smart_case_uppercase_stays_sensitive() {
    let d = fixture_dir();
    let out = run_translated(&["-S", "ALPHA", "."], &d);
    assert_eq!(out.status.code(), Some(1), "must not match lowercase alpha");
}

#[test]
fn type_filter_limits_to_rust_files() {
    let d = fixture_dir();
    let out = run_translated(&["-trust", "alpha", "."], &d);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("lib.rs"), "stdout: {stdout}");
    assert!(!stdout.contains("a.txt"), "type filter leaked: {stdout}");
}

#[test]
fn negated_glob_excludes_directory() {
    let d = fixture_dir();
    let out = run_translated(&["-g", "!target", "alpha", "."], &d);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("lib.rs"), "stdout: {stdout}");
    assert!(!stdout.contains("gen.rs"), "excluded dir leaked: {stdout}");
}

#[test]
fn untranslatable_flag_refuses_loudly() {
    let d = fixture_dir();
    let out = run_translated(&["--files"], &d);
    assert_ne!(out.status.code(), Some(0));
    // Refusals go to STDOUT: agents habitually append 2>/dev/null, and a
    // lesson they never see teaches nothing. The nonzero exit code still
    // marks the failure for scripts.
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("--files"), "stdout: {stdout}");
    assert!(stdout.contains("find PATH -type f"), "stdout: {stdout}");
}

#[test]
fn replace_flag_names_the_edit_alternative() {
    let d = fixture_dir();
    let out = run_translated(&["alpha", "--replace", "omega", "."], &d);
    assert_ne!(out.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("greppy edit regex-cas"), "stdout: {stdout}");
}

#[test]
fn rg_placeholder_token_routes_to_rg_mode() {
    let d = fixture_dir();
    let out = run_translated(&["rg", "-S", "beta", "."], &d);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Beta gamma"), "stdout: {stdout}");
}

#[test]
fn rg_without_path_returns_guidance_on_idle_stdin() {
    let d = fixture_dir();
    let mut command = Command::new(binary_path());
    command
        .args(["rg", "alpha"])
        .current_dir(&d)
        .env("GREPPY_STORE_DIR", unique_tempdir("idle-rg-store"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().expect("spawn greppy");
    let stdin = child.stdin.take().expect("open child stdin");
    let (send, receive) = std::sync::mpsc::channel();
    std::thread::spawn(move || send.send(child.wait_with_output()).unwrap());
    let output = match receive.recv_timeout(std::time::Duration::from_secs(15)) {
        Ok(output) => output.expect("collect greppy output"),
        Err(_) => {
            drop(stdin);
            let _ = receive.recv_timeout(std::time::Duration::from_secs(1));
            panic!("greppy rg alpha waited indefinitely on an idle stdin pipe");
        }
    };
    drop(stdin);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(output.status.code(), Some(64), "stdout: {stdout}");
    assert!(
        stdout.contains("path argument or data on stdin"),
        "{stdout}"
    );
}

#[test]
fn plain_grep_invocation_is_untouched_by_rg_routing() {
    let d = fixture_dir();
    // No rg-only flags: must stay a literal grep passthrough (BRE, no
    // implicit recursion — explicit file argument).
    let out = run_translated(&["-n", "alpha", "a.txt"], &d);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(stdout, "1:alpha\n");
    assert_eq!(out.status.code(), Some(0));
}

#[test]
fn byte_exact_delegation_when_real_ripgrep_exists() {
    let real_rg = std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|p| p.join("rg"))
            .find(|c| c.is_file())
    });
    let Some(real_rg) = real_rg else {
        eprintln!("skipping: no real ripgrep on PATH");
        return;
    };
    let d = fixture_dir();
    let ours = {
        let mut cmd = Command::new(binary_path());
        cmd.args(["--smart-case", "beta"])
            .current_dir(&d)
            .env("GREPPY_REAL_RG", &real_rg)
            .env("GREPPY_STORE_DIR", unique_tempdir("store"))
            .stdin(Stdio::null());
        cmd.output().expect("spawn greppy")
    };
    let theirs = {
        let mut cmd = Command::new(&real_rg);
        cmd.args(["--smart-case", "beta"])
            .current_dir(&d)
            .stdin(Stdio::null());
        cmd.output().expect("spawn rg")
    };
    assert_eq!(ours.stdout, theirs.stdout);
    assert_eq!(ours.status.code(), theirs.status.code());
}
