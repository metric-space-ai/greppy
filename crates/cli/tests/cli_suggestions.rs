//! Ordered missing-symbol suggestions and question-kind correction.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fresh_repo(tag: &str) -> (PathBuf, PathBuf) {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let base = std::env::temp_dir().join(format!(
        "greppy-cli-suggestions-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    let repo = base.join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    git(&repo, &["init", "-q"]);
    git(
        &repo,
        &["config", "user.email", "suggestions@example.invalid"],
    );
    git(&repo, &["config", "user.name", "Suggestion Tests"]);
    (repo, base.join("store"))
}

fn git(repo: &Path, args: &[&str]) {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .expect("run git");
    assert!(
        output.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn commit_all(repo: &Path, message: &str) {
    git(repo, &["add", "."]);
    git(repo, &["commit", "-qm", message]);
}

fn run(repo: &Path, store: &Path, args: &[&str], envs: &[(&str, &Path)]) -> (i32, String, String) {
    let mut command = Command::new(bin());
    command
        .args(args)
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1");
    for (key, value) in envs {
        command.env(key, value);
    }
    let output = command.output().expect("run greppy");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn index(repo: &Path, store: &Path) {
    let result = run(repo, store, &["index", "."], &[]);
    assert_eq!(result.0, 0, "stdout={}\nstderr={}", result.1, result.2);
}

fn combined(result: &(i32, String, String)) -> String {
    format!("{}{}", result.1, result.2)
}

fn assert_factual_only(text: &str) {
    let lower = text.to_ascii_lowercase();
    assert!(!lower.contains("next:"), "{text}");
    assert!(!lower.contains("did you mean"), "{text}");
    assert!(!lower.contains("\ntry:"), "{text}");
    assert!(!lower.contains("\nretry:"), "{text}");
}

fn suggestion_rows(text: &str) -> Vec<&str> {
    text.lines().filter(|line| line.starts_with("  ")).collect()
}

#[test]
fn stage_1_normalizes_spelling_without_semantic_search() {
    let (repo, store) = fresh_repo("normalized");
    std::fs::write(
        repo.join("src/lib.rs"),
        "pub fn parseConfig() -> usize { 1 }\n",
    )
    .unwrap();
    commit_all(&repo, "base");
    index(&repo, &store);
    let marker = repo.parent().unwrap().join("semantic-marker");

    let result = run(
        &repo,
        &store,
        &["who-calls", "parse_config"],
        &[("GREPPY_TEST_SUGGESTION_SEMANTIC_MARKER", &marker)],
    );
    let text = combined(&result);
    assert_eq!(result.0, 1, "{text}");
    assert!(text.contains("no symbol `parse_config`"), "{text}");
    assert!(text.contains("parseConfig"), "{text}");
    assert_eq!(suggestion_rows(&text).len(), 1, "{text}");
    assert!(!marker.exists(), "semantic stage ran for a stage-1 hit");
    assert_factual_only(&text);
}

#[test]
fn stage_2_reports_graph_proven_rename_and_commit() {
    let (repo, store) = fresh_repo("rename");
    std::fs::write(repo.join("src/lib.rs"), "pub fn fooBar() -> usize { 7 }\n").unwrap();
    commit_all(&repo, "old name");
    std::fs::write(repo.join("src/lib.rs"), "pub fn fooBaz() -> usize { 7 }\n").unwrap();
    commit_all(&repo, "rename symbol");
    let head = Command::new("git")
        .args(["rev-parse", "--short=7", "HEAD"])
        .current_dir(&repo)
        .output()
        .unwrap();
    let head = String::from_utf8_lossy(&head.stdout).trim().to_string();
    index(&repo, &store);

    let result = run(&repo, &store, &["callees", "fooBar"], &[]);
    let text = combined(&result);
    assert_eq!(result.0, 1, "{text}");
    assert!(
        text.contains(&format!("`fooBar` renamed to `fooBaz` in {head}")),
        "{text}"
    );
    assert_factual_only(&text);
}

#[test]
fn stage_3_returns_the_unique_corrected_address() {
    let (repo, store) = fresh_repo("qualification");
    std::fs::write(
        repo.join("src/lib.rs"),
        "pub mod store { pub fn bar() -> usize { 1 } }\n",
    )
    .unwrap();
    commit_all(&repo, "base");
    index(&repo, &store);

    let result = run(&repo, &store, &["callees", "Foo::bar"], &[]);
    let text = combined(&result);
    assert_eq!(result.0, 1, "{text}");
    assert!(text.contains("bar"), "{text}");
    assert_eq!(suggestion_rows(&text).len(), 1, "{text}");
    assert_factual_only(&text);
}

#[test]
fn stage_4_accepts_a_two_edit_transposition() {
    let (repo, store) = fresh_repo("edit-distance");
    std::fs::write(
        repo.join("src/lib.rs"),
        "pub fn loadConfig() -> usize { 1 }\n",
    )
    .unwrap();
    commit_all(&repo, "base");
    index(&repo, &store);

    let result = run(&repo, &store, &["who-calls", "laodConfig"], &[]);
    let text = combined(&result);
    assert_eq!(result.0, 1, "{text}");
    assert!(text.contains("loadConfig"), "{text}");
    assert_factual_only(&text);
}

#[test]
fn stage_5_matches_name_fragments() {
    let (repo, store) = fresh_repo("fragment");
    std::fs::write(
        repo.join("src/lib.rs"),
        "pub fn ConfigLoaderFactory() -> usize { 1 }\n",
    )
    .unwrap();
    commit_all(&repo, "base");
    index(&repo, &store);

    let result = run(&repo, &store, &["brief", "Config"], &[]);
    let text = combined(&result);
    assert_eq!(result.0, 1, "{text}");
    assert!(text.contains("no symbol `Config`"), "{text}");
    assert_factual_only(&text);
}

#[test]
fn stage_6_runs_only_after_name_stages_are_empty() {
    let (repo, store) = fresh_repo("semantic");
    std::fs::write(
        repo.join("src/lib.rs"),
        "/// Persists authenticated session state.\npub fn write_login_record() -> usize { 1 }\n",
    )
    .unwrap();
    commit_all(&repo, "base");
    index(&repo, &store);
    let marker = repo.parent().unwrap().join("semantic-marker");

    let result = run(
        &repo,
        &store,
        &["who-calls", "write login"],
        &[("GREPPY_TEST_SUGGESTION_SEMANTIC_MARKER", &marker)],
    );
    let text = combined(&result);
    assert_eq!(result.0, 1, "{text}");
    assert!(marker.exists(), "semantic stage did not run");
    assert!(text.contains("write_login_record"), "{text}");
    assert_factual_only(&text);
}

#[test]
fn stage_7_reports_the_actual_kind_for_who_calls() {
    let (repo, store) = fresh_repo("kind");
    std::fs::write(repo.join("src/lib.rs"), "pub const MAX_SIZE: usize = 64;\n").unwrap();
    commit_all(&repo, "base");
    index(&repo, &store);

    let result = run(&repo, &store, &["who-calls", "MAX_SIZE"], &[]);
    let text = combined(&result);
    assert_eq!(result.0, 0, "{text}");
    assert!(text.contains("MAX_SIZE"), "{text}");
    assert!(text.contains("const"), "{text}");
    assert!(text.contains("src/lib.rs:1"), "{text}");
    assert!(!text.contains("not found"), "{text}");
    assert!(!text.contains("no callers"), "{text}");
    assert_factual_only(&text);
}

#[test]
fn suggestions_are_capped_at_three() {
    let (repo, store) = fresh_repo("cap");
    let mut source = String::new();
    for index in 0..20 {
        source.push_str(&format!(
            "pub fn ConfigLoaderFactory{index}() -> usize {{ {index} }}\n"
        ));
    }
    std::fs::write(repo.join("src/lib.rs"), source).unwrap();
    commit_all(&repo, "base");
    index(&repo, &store);

    let result = run(&repo, &store, &["who-calls", "Config"], &[]);
    let text = combined(&result);
    assert_eq!(result.0, 1, "{text}");
    assert_eq!(suggestion_rows(&text).len(), 3, "{text}");
    assert_factual_only(&text);
}
