//! Explanatory empty-output coverage for literal code search.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fresh_workspace(tag: &str) -> (PathBuf, PathBuf) {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let base = std::env::temp_dir().join(format!(
        "greppy-cli-search-empty-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    let repo = base.join("repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    (repo, base.join("store"))
}

fn run(repo: &Path, store: &Path, args: &[&str]) -> (i32, String, String) {
    let output = Command::new(bin())
        .args(args)
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .output()
        .expect("run greppy");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn empty_literal_search_names_interpretation_and_path_filters() {
    let (repo, store) = fresh_workspace("literal");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn present() {}\n").unwrap();

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &["search-code", "absent_value", "src"],
    );

    assert_eq!(code, 1, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("(no matches)"), "{stdout}");
    assert!(stdout.contains("query_interpreted_as: literal"), "{stdout}");
    assert!(stdout.contains("path_filters: src"), "{stdout}");
}

#[test]
fn empty_metacharacter_search_teaches_the_regex_retry() {
    let (repo, store) = fresh_workspace("metacharacters");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn present() {}\n").unwrap();

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &["search-code", "absent.*value", "src"],
    );

    assert_eq!(code, 1, "stdout={stdout}\nstderr={stderr}");
    assert!(
        stdout.contains("regex metacharacters are literal in search-code"),
        "{stdout}"
    );
    assert!(
        stdout.contains("try: greppy rg 'absent.*value' src"),
        "{stdout}"
    );
}
