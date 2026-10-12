//! Real CLI coverage: stale symbol lookup -> publication -> useful literal read.
use std::path::Path;
use std::process::{Command, Output};

fn run(repo: &Path, store: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_greppy"))
        .args(args)
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_CONTEXT_SCOPE", "context-transition-cli")
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env_remove("GREPPY_CONTEXT_ENVELOPE")
        .env("GREPPY_AUTO_REINDEX", "0")
        .output()
        .unwrap()
}

#[test]
fn stale_symbol_lookup_announces_published_graph_once_on_literal_read() {
    let temp = tempfile::tempdir().unwrap();
    let repo = temp.path().join("repo");
    let store = temp.path().join("store");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    let file = repo.join("value.rs");
    std::fs::write(&file, "pub fn first_value() -> i32 { 1 }\n").unwrap();
    let initial = run(&repo, &store, &["index", "."]);
    assert!(initial.status.success(), "{initial:?}");
    let quiet = run(&repo, &store, &["read-file", "value.rs"]);
    assert!(quiet.status.success());
    assert!(quiet.stderr.is_empty(), "{quiet:?}");

    std::fs::write(&file, "pub fn second_value() -> i32 { 2 }\n").unwrap();
    let stale = run(&repo, &store, &["search-symbol", "second_value"]);
    assert_eq!(stale.status.code(), Some(75), "{stale:?}");
    let pending = run(&repo, &store, &["read-file", "value.rs"]);
    assert!(pending.status.success());
    assert!(pending.stderr.is_empty(), "{pending:?}");
    let publication = run(&repo, &store, &["index", "."]);
    assert!(publication.status.success(), "{publication:?}");
    // Failed provider-policy validation and a no-match lookup must not consume
    // the pending hint merely because graph freshness was established.
    let provider_refusal = Command::new(env!("CARGO_BIN_EXE_greppy"))
        .args(["search-symbol", "second_value"])
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", &store)
        .env("GREPPY_CONTEXT_SCOPE", "context-transition-cli")
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env("GREPPY_PROVIDER_POLICY", "invalid-fixture-policy")
        .output()
        .unwrap();
    assert!(!provider_refusal.status.success(), "{provider_refusal:?}");
    assert!(String::from_utf8_lossy(&provider_refusal.stderr).contains("GREPPY_PROVIDER_POLICY"));
    let missing = run(&repo, &store, &["search-symbol", "absent_fixture_symbol"]);
    assert_eq!(missing.status.code(), Some(1), "{missing:?}");
    let ready = run(&repo, &store, &["read-file", "value.rs"]);

    assert!(ready.status.success());
    let hint = String::from_utf8(ready.stderr).unwrap();
    assert_eq!(hint.matches("graph preparation completed").count(), 1);
    // The fixture has no embedding publication and must not recommend search.
    assert!(!hint.contains("semantic embedding"), "{hint}");
    let repeated = run(&repo, &store, &["read-file", "value.rs"]);
    assert!(repeated.status.success());
    assert!(repeated.stderr.is_empty(), "{repeated:?}");

    // Successful symbol use acknowledges the next cycle without a redundant hint.
    std::fs::write(&file, "pub fn third_value() -> i32 { 3 }\n").unwrap();
    assert_eq!(
        run(&repo, &store, &["search-symbol", "third_value"])
            .status
            .code(),
        Some(75)
    );
    assert!(run(&repo, &store, &["index", "."]).status.success());
    assert!(run(&repo, &store, &["search-symbol", "third_value"])
        .status
        .success());
    let already_used = run(&repo, &store, &["read-file", "value.rs"]);
    assert!(already_used.status.success());
    assert!(already_used.stderr.is_empty(), "{already_used:?}");
}
