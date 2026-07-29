//! Regression coverage for agent-friendly grep directory operands.

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fresh_workspace() -> (PathBuf, PathBuf) {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let base = std::env::temp_dir().join(format!(
        "greppy-cli-grep-postel-{}-{n}",
        std::process::id()
    ));
    let repo = base.join("repo");
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(repo.join("tree/nested")).unwrap();
    std::fs::write(repo.join("tree/top.txt"), "miss\n").unwrap();
    std::fs::write(repo.join("tree/nested/hit.txt"), "needle here\n").unwrap();
    (repo, base.join("store"))
}

fn run_greppy(repo: &Path, store: &Path, args: &[&str]) -> Output {
    Command::new(bin())
        .args(args)
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .stdin(Stdio::null())
        .output()
        .expect("run greppy")
}

fn run_real_grep(repo: &Path, args: &[&str]) -> Output {
    Command::new("/usr/bin/grep")
        .args(args)
        .current_dir(repo)
        .stdin(Stdio::null())
        .output()
        .expect("run real grep")
}

fn assert_implicit_recursive_matches_real_grep(args: &[&str], expected_args: &[&str]) {
    let (repo, store) = fresh_workspace();
    let actual = run_greppy(&repo, &store, args);
    let expected = run_real_grep(&repo, expected_args);
    assert_eq!(actual.status.code(), expected.status.code());
    assert_eq!(actual.stdout, expected.stdout);
    assert_eq!(actual.stderr, expected.stderr);
    let stdout = String::from_utf8_lossy(&actual.stdout);
    assert!(stdout.contains("tree/nested/hit.txt"), "{stdout}");
    assert!(!String::from_utf8_lossy(&actual.stderr).contains("Is a directory"));
}

#[test]
fn named_grep_directory_operand_defaults_to_recursive() {
    assert_implicit_recursive_matches_real_grep(
        &["grep", "needle", "tree"],
        &["-r", "needle", "tree"],
    );
}

#[test]
fn named_grep_explicit_regexp_form_defaults_to_recursive() {
    assert_implicit_recursive_matches_real_grep(
        &["grep", "-e", "needle", "tree"],
        &["-r", "-e", "needle", "tree"],
    );
}
