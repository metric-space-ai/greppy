//! Every agent-facing invocation either returns evidence or a bounded status
//! that names the next useful action. These regressions come from measured
//! agent traces where a bare failure immediately caused a fallback to shell
//! search.

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
        "greppy-invocation-outcomes-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    let repo = base.join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(
        repo.join("src/lexer.rs"),
        "pub struct CommentIndentation;\n\npub fn block_tokens(input: &str) -> usize {\n    input.len()\n}\n",
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
        .output()
        .expect("run greppy");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn indexed_workspace(tag: &str) -> (PathBuf, PathBuf) {
    let (repo, store) = fresh_workspace(tag);
    let (code, stdout, stderr) = run(&repo, &store, &["index", "."]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    (repo, store)
}

#[test]
fn read_code_flag_returns_source_and_names_the_noop() {
    let (repo, store) = indexed_workspace("read-code");

    let (code, stdout, stderr) = run(&repo, &store, &["read", "block_tokens", "--code"]);

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("`--code` is ignored"), "{stdout}");
    assert!(
        stdout.contains("src/lexer.rs:3-5  block_tokens"),
        "{stdout}"
    );
    assert!(stdout.contains("pub fn block_tokens"), "{stdout}");
}

#[test]
fn read_path_positional_returns_the_file_and_names_the_degradation() {
    let (repo, store) = indexed_workspace("read-path");

    let (code, stdout, stderr) = run(&repo, &store, &["read", "src/lexer.rs"]);

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(
        stdout.contains("`src/lexer.rs` is a path; reading it as a file"),
        "{stdout}"
    );
    assert!(stdout.contains("src/lexer.rs:1-5"), "{stdout}");
    assert!(stdout.contains("pub struct CommentIndentation"), "{stdout}");
}

#[test]
fn unknown_kind_searches_without_the_filter_and_lists_valid_values() {
    let (repo, store) = indexed_workspace("kind");

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &["search-symbol", "CommentIndentation", "--kind", "cop"],
    );

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(
        stdout.contains("ignored unknown --kind value `cop`; searching without a kind filter"),
        "{stdout}"
    );
    assert!(
        stdout.contains("function, method, class, struct, enum, trait"),
        "{stdout}"
    );
    assert!(stdout.contains("src/lexer.rs:1"), "{stdout}");
    assert!(stdout.contains("CommentIndentation"), "{stdout}");
}

#[test]
fn empty_searches_return_bounded_statuses_and_next_actions() {
    let (repo, store) = indexed_workspace("empty");

    let (symbol_code, symbol_out, symbol_err) = run(&repo, &store, &["search-symbol", "respond"]);
    assert_eq!(symbol_code, 0, "stdout={symbol_out}\nstderr={symbol_err}");
    assert!(symbol_out.contains("status: no_matches"), "{symbol_out}");
    assert!(
        symbol_out.contains("message: no definition named `respond`"),
        "{symbol_out}"
    );
    assert!(
        symbol_out.contains("next: search source text: greppy search-pattern respond --fixed"),
        "{symbol_out}"
    );
    assert!(
        symbol_out.contains("next: refresh definitions"),
        "{symbol_out}"
    );

    let (path_code, path_out, path_err) = run(
        &repo,
        &store,
        &[
            "search-pattern",
            "block_tokens",
            "--path",
            "missing/subtree",
        ],
    );
    assert_eq!(path_code, 0, "stdout={path_out}\nstderr={path_err}");
    assert!(path_out.contains("status: no_matches"), "{path_out}");
    assert!(
        path_out.contains("message: no matches under path filter: missing/subtree"),
        "{path_out}"
    );
    assert!(
        path_out.contains("source match(es) exist outside the path filter"),
        "{path_out}"
    );
    assert!(
        path_out.contains("next: retry without the path filter"),
        "{path_out}"
    );
}

#[test]
fn unknown_option_is_a_greppy_status_and_never_a_grep_error() {
    let (repo, store) = fresh_workspace("unknown-option");

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &["--future-greppy-option", "block_tokens", "src/lexer.rs"],
    );

    assert_eq!(code, 64, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("status: invalid_invocation"), "{stdout}");
    assert!(
        stdout.contains("argument: `--future-greppy-option`"),
        "{stdout}"
    );
    assert!(stdout.contains("nothing was passed to grep"), "{stdout}");
    assert!(stdout.contains("next: run `greppy --help`"), "{stdout}");
    assert!(!stdout.contains("Usage: grep"), "{stdout}");
    assert!(!stderr.contains("grep"), "{stderr}");

    let (typo_code, typo_out, typo_err) =
        run(&repo, &store, &["serach-symbol", "CommentIndentation"]);
    assert_eq!(typo_code, 64, "stdout={typo_out}\nstderr={typo_err}");
    assert!(
        typo_out.contains("status: invalid_invocation"),
        "{typo_out}"
    );
    assert!(
        typo_out.contains("nothing was passed to grep"),
        "{typo_out}"
    );
    assert!(
        typo_out.contains("did you mean `greppy search-symbol ...`"),
        "{typo_out}"
    );
    assert!(!typo_out.contains("Usage: grep"), "{typo_out}");
}
