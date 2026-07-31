//! The 0.3.0 missing-symbol answer: a bare `no symbol` line, then a
//! `similar names:` block when (and only when) a name tier qualifies —
//! no instructions, no meaning fallback on the navigation verbs.

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

/// The rows of the `similar names:` block — every non-empty line after the
/// header. An absent block yields no rows: the absence is the statement.
fn similar_name_rows(text: &str) -> Vec<&str> {
    match text.split("similar names:\n").nth(1) {
        Some(block) => block.lines().filter(|line| !line.is_empty()).collect(),
        None => Vec::new(),
    }
}

#[test]
fn miss_normalizes_case_and_underscores() {
    let (repo, store) = fresh_repo("normalized");
    std::fs::write(
        repo.join("src/lib.rs"),
        "pub fn parseConfig() -> usize { 1 }\n",
    )
    .unwrap();
    commit_all(&repo, "base");
    index(&repo, &store);

    let result = run(&repo, &store, &["who-calls", "parse_config"], &[]);
    let text = combined(&result);
    assert_eq!(result.0, 1, "{text}");
    // Equal after lowercasing and removing underscores — the near-certain
    // tier, one exact row and nothing else.
    assert_eq!(
        text, "no symbol `parse_config`\n\nsimilar names:\nsrc/lib.rs:1  parseConfig\n",
        "{text}"
    );
    assert_eq!(similar_name_rows(&text).len(), 1, "{text}");
    assert_factual_only(&text);
}

#[test]
fn miss_after_rename_offers_the_new_name() {
    let (repo, store) = fresh_repo("rename");
    std::fs::write(repo.join("src/lib.rs"), "pub fn fooBar() -> usize { 7 }\n").unwrap();
    commit_all(&repo, "old name");
    std::fs::write(repo.join("src/lib.rs"), "pub fn fooBaz() -> usize { 7 }\n").unwrap();
    commit_all(&repo, "rename symbol");
    index(&repo, &store);

    // No git provenance in the answer: the renamed-away name is a plain
    // miss, the new name is an edit-distance-≤-2 similar name.
    let result = run(&repo, &store, &["callees", "fooBar"], &[]);
    let text = combined(&result);
    assert_eq!(result.0, 1, "{text}");
    assert_eq!(
        text, "no symbol `fooBar`\n\nsimilar names:\nsrc/lib.rs:1  fooBaz\n",
        "{text}"
    );
    assert_eq!(similar_name_rows(&text).len(), 1, "{text}");
    assert_factual_only(&text);
}

#[test]
fn miss_on_qualified_name_offers_the_bare_definition() {
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
    assert_eq!(
        text, "no symbol `Foo::bar`\n\nsimilar names:\nsrc/lib.rs:1  bar\n",
        "{text}"
    );
    assert_eq!(similar_name_rows(&text).len(), 1, "{text}");
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
fn nav_miss_has_no_meaning_fallback() {
    let (repo, store) = fresh_repo("semantic");
    std::fs::write(
        repo.join("src/lib.rs"),
        "/// Persists authenticated session state.\npub fn write_login_record() -> usize { 1 }\n",
    )
    .unwrap();
    commit_all(&repo, "base");
    index(&repo, &store);

    // A prose query on a navigation verb is a bare miss: `search` IS the
    // meaning stage now, so who-calls never falls through to it — when no
    // name tier qualifies, no block follows and the absence is the statement.
    let result = run(&repo, &store, &["who-calls", "write login"], &[]);
    let text = combined(&result);
    assert_eq!(result.0, 1, "{text}");
    assert_eq!(text, "no symbol `write login`\n", "{text}");
    assert!(similar_name_rows(&text).is_empty(), "{text}");
    assert_factual_only(&text);
}

#[test]
fn who_calls_a_const_reports_no_callers() {
    let (repo, store) = fresh_repo("kind");
    std::fs::write(repo.join("src/lib.rs"), "pub const MAX_SIZE: usize = 64;\n").unwrap();
    commit_all(&repo, "base");
    index(&repo, &store);

    // who-calls walks CALLS and USAGE, so the wrong-kind refusal no longer
    // applies to it: an unreferenced const is a lawful query with an empty
    // answer — said, without parentheses, at exit 0.
    let result = run(&repo, &store, &["who-calls", "MAX_SIZE"], &[]);
    let text = combined(&result);
    assert_eq!(result.0, 0, "{text}");
    assert_eq!(text, "no callers\n", "{text}");
    assert!(!text.contains("not a function"), "{text}");
    assert!(!text.contains("is a const"), "{text}");
    assert_factual_only(&text);
}

#[test]
fn substring_fragments_are_never_suggested() {
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

    // Twenty names contain the fragment, and none qualifies: substring
    // matches are never candidates (`data` must not suggest `metadata`).
    // No cap is needed because no block follows — the absence is the
    // statement.
    let result = run(&repo, &store, &["who-calls", "Config"], &[]);
    let text = combined(&result);
    assert_eq!(result.0, 1, "{text}");
    assert_eq!(text, "no symbol `Config`\n", "{text}");
    assert!(similar_name_rows(&text).is_empty(), "{text}");
    assert_factual_only(&text);
}
