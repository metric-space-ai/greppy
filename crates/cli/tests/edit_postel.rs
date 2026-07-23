//! Regression coverage for conventional agent-guessed edit flags.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fresh_workspace() -> (PathBuf, PathBuf) {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let base = std::env::temp_dir().join(format!(
        "greppy-cli-edit-postel-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    let repo = base.join("repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(
        repo.join("src/lib.rs"),
        r#"pub const ANSWER: i32 = 42;

pub fn first() -> i32 {
    1
}

pub fn second() -> i32 {
    2
}

pub fn third() -> i32 {
    3
}
"#,
    )
    .unwrap();
    (repo, base.join("store"))
}

fn run(repo: &Path, store: &Path, args: &[&str]) -> (i32, String, String) {
    let output = Command::new(bin())
        .args(args)
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .stdin(Stdio::null())
        .output()
        .expect("run greppy");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn assert_applied(result: &(i32, String, String), label: &str) {
    assert_eq!(
        result.0, 0,
        "{label}\nstdout={}\nstderr={}",
        result.1, result.2
    );
    assert!(result.1.contains("\"status\": \"applied\""), "{label}: {}", result.1);
}

#[test]
fn misspelled_nested_edit_flag_suggests_complete_source_invocation() {
    let (repo, store) = fresh_workspace();
    let body = repo.join("body.rs");
    std::fs::write(&body, "{ 11 }").unwrap();

    let result = run(
        &repo,
        &store,
        &[
            "edit",
            "replace-body",
            "--symbol",
            "first",
            "--sorce",
            body.to_str().unwrap(),
            "--dry-run",
        ],
    );

    assert_eq!(result.0, 64, "stdout={}\nstderr={}", result.1, result.2);
    assert!(
        result.1.contains(&format!(
            " edit replace-body --symbol first --source {} --dry-run",
            body.display()
        )),
        "{}",
        result.1
    );
}

#[test]
fn source_and_regex_old_new_aliases_are_accepted_end_to_end() {
    let (repo, store) = fresh_workspace();
    let indexed = run(&repo, &store, &["index", "."]);
    assert_eq!(indexed.0, 0, "stdout={}\nstderr={}", indexed.1, indexed.2);

    let body = repo.join("body.rs");
    std::fs::write(&body, "{ 11 }").unwrap();
    let replaced_body = run(
        &repo,
        &store,
        &[
            "edit",
            "replace-body",
            "--symbol",
            "first",
            "--source",
            body.to_str().unwrap(),
            "--dry-run",
        ],
    );
    assert_applied(&replaced_body, "replace-body --source");

    let read = run(&repo, &store, &["read", "second", "--handle", "--json"]);
    assert_eq!(read.0, 0, "stdout={}\nstderr={}", read.1, read.2);
    let read_json: serde_json::Value = serde_json::from_str(&read.1).unwrap();
    let handle = read_json["handle"].as_str().expect("read handle");
    let span = repo.join("span.rs");
    std::fs::write(&span, "pub fn second() -> i32 { 22 }\n").unwrap();
    let replaced_span = run(
        &repo,
        &store,
        &[
            "edit",
            "replace-span",
            "--target",
            handle,
            "--source",
            span.to_str().unwrap(),
            "--dry-run",
        ],
    );
    assert_applied(&replaced_span, "replace-span --source");

    let adjacent = repo.join("adjacent.rs");
    std::fs::write(&adjacent, "pub fn inserted() -> i32 { 33 }\n").unwrap();
    let inserted_after = run(
        &repo,
        &store,
        &[
            "edit",
            "insert-after",
            "--symbol",
            "third",
            "--source",
            adjacent.to_str().unwrap(),
            "--dry-run",
        ],
    );
    assert_applied(&inserted_after, "insert-after --source");

    let before = repo.join("before.rs");
    std::fs::write(&before, "pub fn before_third() -> i32 { 30 }\n").unwrap();
    let inserted_before = run(
        &repo,
        &store,
        &[
            "edit",
            "insert-before",
            "--symbol",
            "third",
            "--source",
            before.to_str().unwrap(),
            "--dry-run",
        ],
    );
    assert_applied(&inserted_before, "insert-before --source");

    let regex = run(
        &repo,
        &store,
        &[
            "edit",
            "regex-cas",
            "--file",
            "src/lib.rs",
            "--old",
            "ANSWER: i32 = [0-9]+",
            "--new",
            "ANSWER: i32 = 43",
            "--dry-run",
        ],
    );
    assert_applied(&regex, "regex-cas --old/--new");

    let unchanged = std::fs::read_to_string(repo.join("src/lib.rs")).unwrap();
    assert!(unchanged.contains("pub const ANSWER: i32 = 42;"), "{unchanged}");
    assert!(unchanged.contains("pub fn first() -> i32 {\n    1\n}"), "{unchanged}");
    assert!(unchanged.contains("pub fn second() -> i32 {\n    2\n}"), "{unchanged}");
    assert!(!unchanged.contains("pub fn inserted()"), "{unchanged}");
    assert!(!unchanged.contains("pub fn before_third()"), "{unchanged}");
}
