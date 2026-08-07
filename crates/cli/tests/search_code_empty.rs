//! Explanatory empty-output coverage for `search-pattern`, plus the
//! retirement pin for the pre-0.3.0 name `search-code` whose literal-search
//! contract this file used to pin.
//!
//! 0.3.0 contract (dev/SEARCH-OUTPUT-SPEC.md, normative):
//! * `search-code` is dead vocabulary: refused as an unknown subcommand
//!   (exit 64) before grep passthrough — never grepped, never answered.
//! * `search-pattern` is regex-native: a pattern with metacharacters is
//!   simply a pattern. Zero hits are a successful bounded status: they name
//!   the empty scope, distinguish a path-filter miss when possible, and give
//!   concrete broader-search and refresh actions.

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

/// The pre-0.3.0 `search-code` subcommand is dead-listed vocabulary: like
/// `edit change-signature` (edit_m4) it must be REFUSED as an unknown
/// subcommand — an agent with a stale habit learns immediately instead of
/// getting garbage grep matches for `search-code` as a pattern.
#[test]
fn retired_search_code_is_refused_not_grepped() {
    let (repo, store) = fresh_workspace("retired");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn present() {}\n").unwrap();

    let (code, stdout, stderr) = run(&repo, &store, &["search-code", "absent_value", "src"]);

    let text = format!("{stdout}{stderr}");
    assert_eq!(
        code, 64,
        "`greppy search-code` must refuse as invalid vocabulary; stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        text.contains("unrecognized subcommand 'search-code'"),
        "the refusal names the dead verb; got: {text}"
    );
    assert!(
        !text.contains("src/lib.rs") && !text.contains("no matches"),
        "the refusal neither greps nor answers; got: {text}"
    );
}

/// Zero hits bind the active path filter to concrete next actions.
#[test]
fn empty_search_pattern_names_the_path_filter_and_next_actions() {
    let (repo, store) = fresh_workspace("empty-filter");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn present() {}\n").unwrap();

    let (code, stdout, _stderr) = run(
        &repo,
        &store,
        &["search-pattern", "absent_value", "--path", "src"],
    );

    assert_eq!(
        code, 0,
        "bounded no-match status is successful; stdout={stdout}"
    );
    assert!(stdout.contains("status: no_matches"), "{stdout}");
    assert!(
        stdout.contains("message: no matches under path filter: src"),
        "{stdout}"
    );
    assert!(
        stdout.contains("next: retry without the path filter"),
        "{stdout}"
    );
    assert!(stdout.contains("greppy index ."), "{stdout}");
}

/// `search-pattern` is regex-native: `absent.*value` is simply a pattern that
/// matches nothing. The pre-0.3.0 teaching ("regex metacharacters are literal
/// in search-code" + "try: greppy rg ...") is dead with the command.
#[test]
fn metacharacter_pattern_is_just_a_pattern_without_teaching() {
    let (repo, store) = fresh_workspace("metacharacters");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn present() {}\n").unwrap();

    let (code, stdout, _stderr) = run(
        &repo,
        &store,
        &["search-pattern", "absent.*value", "--path", "src"],
    );

    assert_eq!(
        code, 0,
        "bounded no-match status is successful; stdout={stdout}"
    );
    assert!(stdout.contains("status: no_matches"), "{stdout}");
    assert!(
        stdout.contains("message: no matches under path filter: src"),
        "{stdout}"
    );
    assert!(
        stdout.contains("greppy search-pattern 'absent.*value'"),
        "the retry preserves the regex instead of reinterpreting it; got: {stdout:?}"
    );
}

/// A computed case-insensitive count remains attached to the bounded status.
#[test]
fn empty_search_pattern_reports_the_case_insensitive_fact() {
    let (repo, store) = fresh_workspace("case-fact");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn present() {}\n").unwrap();

    let (code, stdout, _stderr) = run(
        &repo,
        &store,
        &["search-pattern", "PRESENT", "--fixed", "--path", "src"],
    );

    assert_eq!(
        code, 0,
        "bounded no-match status is successful; stdout={stdout}"
    );
    assert!(stdout.contains("status: no_matches"), "{stdout}");
    assert!(
        stdout.contains("case-insensitive: 1 matches"),
        "the empty answer carries the computed case-insensitive fact; got: {stdout:?}"
    );
}
