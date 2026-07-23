//! File-oriented `greppy read` hardening coverage.

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
        "greppy-cli-read-file-{tag}-{}-{n}",
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
fn read_path_prints_numbered_file_lines_without_an_index() {
    let (repo, store) = fresh_workspace("numbered");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "alpha\nbeta\ngamma\ndelta\n").unwrap();

    let (code, stdout, stderr) = run(&repo, &store, &["read", "src/lib.rs"]);

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.starts_with("src/lib.rs:1-4\n"), "{stdout}");
    assert!(stdout.contains("1 | alpha"), "{stdout}");
    assert!(stdout.contains("4 | delta"), "{stdout}");
}

#[test]
fn read_path_lines_selects_an_inclusive_range() {
    let (repo, store) = fresh_workspace("range");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "alpha\nbeta\ngamma\ndelta\n").unwrap();

    let (code, stdout, stderr) = run(&repo, &store, &["read", "src/lib.rs", "--lines", "2:3"]);

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.starts_with("src/lib.rs:2-3\n"), "{stdout}");
    assert!(stdout.contains("2 | beta"), "{stdout}");
    assert!(stdout.contains("3 | gamma"), "{stdout}");
    assert!(!stdout.contains("alpha"), "{stdout}");
    assert!(!stdout.contains("delta"), "{stdout}");
}

#[test]
fn read_path_flag_and_singular_line_return_a_replaceable_range_handle() {
    let (repo, store) = fresh_workspace("path-line-handle");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "alpha\nbeta\ngamma\n").unwrap();

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &[
            "read",
            "--path",
            "src/lib.rs",
            "--line",
            "2",
            "--handle",
            "--json",
        ],
    );

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    let read: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(read["path"], "src/lib.rs");
    assert_eq!(read["start_line"], 2);
    assert_eq!(read["end_line"], 2);
    assert_eq!(read["lines"][0]["text"], "beta");
    let handle = read["handle"].as_str().expect("range handle");

    let replacement = repo.join("replacement.txt");
    std::fs::write(&replacement, "changed\n").unwrap();
    let (edit_code, edit_stdout, edit_stderr) = run(
        &repo,
        &store,
        &[
            "edit",
            "replace-span",
            "--target",
            handle,
            "--source-file",
            replacement.to_str().unwrap(),
        ],
    );
    assert_eq!(edit_code, 0, "stdout={edit_stdout}\nstderr={edit_stderr}");
    assert_eq!(
        std::fs::read_to_string(repo.join("src/lib.rs")).unwrap(),
        "alpha\nchanged\ngamma\n"
    );
}

#[test]
fn read_path_qualified_symbol_never_leaks_same_named_foreign_definition() {
    let (repo, store) = fresh_workspace("path-qualified-symbol");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(
        repo.join("src/a.rs"),
        "pub fn target() -> &'static str { \"from_a\" }\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("src/b.rs"),
        "pub fn target() -> &'static str { \"from_b\" }\n",
    )
    .unwrap();
    let (index_code, index_stdout, index_stderr) = run(&repo, &store, &["index", "."]);
    assert_eq!(
        index_code, 0,
        "stdout={index_stdout}\nstderr={index_stderr}"
    );

    let (code, stdout, stderr) = run(&repo, &store, &["read", "src/a.rs::target"]);

    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("src/a.rs"), "{stdout}");
    assert!(stdout.contains("from_a"), "{stdout}");
    assert!(!stdout.contains("src/b.rs"), "{stdout}");
    assert!(!stdout.contains("from_b"), "{stdout}");
}

#[test]
fn misspelled_read_path_suggests_paths_not_symbols() {
    let (repo, store) = fresh_workspace("suggestion");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(repo.join("src/lib.rs"), "pub fn target() {}\n").unwrap();

    let (code, stdout, stderr) = run(&repo, &store, &["read", "src/lbi.rs"]);

    assert_eq!(code, 10, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("closest paths"), "{stdout}");
    assert!(stdout.contains("src/lib.rs"), "{stdout}");
    assert!(stdout.contains("try: greppy read src/lib.rs"), "{stdout}");
    assert!(!stdout.contains("closest definitions"), "{stdout}");
}

fn handle_from(stdout: &str) -> &str {
    stdout
        .lines()
        .find_map(|line| line.strip_prefix("handle: "))
        .expect("read output contains a handle")
}

#[test]
fn read_path_range_handle_drives_replace_span_and_certificate() {
    let (repo, store) = fresh_workspace("range-handle-replace");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    let original = "fn one() {}\nfn two() {}\nfn three() {}\n";
    std::fs::write(repo.join("src/lib.rs"), original).unwrap();

    let (read_code, read_stdout, read_stderr) = run(
        &repo,
        &store,
        &["read", "src/lib.rs", "--lines", "2:2", "--handle"],
    );
    assert_eq!(read_code, 0, "stdout={read_stdout}\nstderr={read_stderr}");
    let handle = handle_from(&read_stdout).to_string();
    let replacement = repo.join("replacement.rs");
    std::fs::write(&replacement, "fn changed() {}\n").unwrap();

    let (edit_code, edit_stdout, edit_stderr) = run(
        &repo,
        &store,
        &[
            "edit",
            "replace-span",
            "--target",
            &handle,
            "--source-file",
            replacement.to_str().unwrap(),
        ],
    );

    assert_eq!(edit_code, 0, "stdout={edit_stdout}\nstderr={edit_stderr}");
    let certificate: serde_json::Value = serde_json::from_str(&edit_stdout).unwrap();
    assert_eq!(certificate["status"], "applied");
    assert_eq!(certificate["exit_code"], 0);
    assert_eq!(certificate["published"], true);
    assert_eq!(
        std::fs::read_to_string(repo.join("src/lib.rs")).unwrap(),
        "fn one() {}\nfn changed() {}\nfn three() {}\n"
    );
}

#[test]
fn read_path_handle_is_stale_after_any_file_change() {
    let (repo, store) = fresh_workspace("range-handle-stale");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    let original = "fn one() {}\nfn two() {}\nfn three() {}\n";
    let changed = "// changed elsewhere\nfn one() {}\nfn two() {}\nfn three() {}\n";
    std::fs::write(repo.join("src/lib.rs"), original).unwrap();

    let (read_code, read_stdout, read_stderr) = run(
        &repo,
        &store,
        &["read", "src/lib.rs", "--lines", "2:2", "--handle"],
    );
    assert_eq!(read_code, 0, "stdout={read_stdout}\nstderr={read_stderr}");
    let handle = handle_from(&read_stdout).to_string();
    std::fs::write(repo.join("src/lib.rs"), changed).unwrap();
    let replacement = repo.join("replacement.rs");
    std::fs::write(&replacement, "fn changed() {}\n").unwrap();

    let (edit_code, edit_stdout, edit_stderr) = run(
        &repo,
        &store,
        &[
            "edit",
            "replace-span",
            "--target",
            &handle,
            "--source-file",
            replacement.to_str().unwrap(),
        ],
    );

    assert_eq!(edit_code, 12, "stdout={edit_stdout}\nstderr={edit_stderr}");
    let certificate: serde_json::Value = serde_json::from_str(&edit_stdout).unwrap();
    assert_eq!(certificate["status"], "stale");
    assert_eq!(certificate["exit_code"], 12);
    assert_eq!(certificate["published"], false);
    assert_eq!(
        std::fs::read_to_string(repo.join("src/lib.rs")).unwrap(),
        changed
    );
}

#[test]
fn read_whole_file_handle_covers_and_replaces_every_byte() {
    let (repo, store) = fresh_workspace("whole-file-handle");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    let original = "fn one() {}\nfn two() {}\n";
    std::fs::write(repo.join("src/lib.rs"), original).unwrap();

    let (read_code, read_stdout, read_stderr) =
        run(&repo, &store, &["read", "src/lib.rs", "--handle", "--json"]);
    assert_eq!(read_code, 0, "stdout={read_stdout}\nstderr={read_stderr}");
    let read_json: serde_json::Value = serde_json::from_str(&read_stdout).unwrap();
    assert_eq!(read_json["byte_start"], 0);
    assert_eq!(read_json["byte_end"], original.len());
    let handle = read_json["handle"].as_str().expect("JSON handle");
    let replacement = repo.join("replacement.rs");
    let replacement_text = "fn replacement() {}\n";
    std::fs::write(&replacement, replacement_text).unwrap();

    let (edit_code, edit_stdout, edit_stderr) = run(
        &repo,
        &store,
        &[
            "edit",
            "replace-span",
            "--target",
            handle,
            "--source-file",
            replacement.to_str().unwrap(),
        ],
    );

    assert_eq!(edit_code, 0, "stdout={edit_stdout}\nstderr={edit_stderr}");
    let certificate: serde_json::Value = serde_json::from_str(&edit_stdout).unwrap();
    assert_eq!(certificate["status"], "applied");
    assert_eq!(
        std::fs::read_to_string(repo.join("src/lib.rs")).unwrap(),
        replacement_text
    );
}
