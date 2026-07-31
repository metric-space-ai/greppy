//! Contract coverage for read, read-smart, and read-file.

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
        "greppy-cli-read-family-{tag}-{}-{n}",
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
        .arg("--root")
        .arg(repo)
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

fn index(repo: &Path, store: &Path) {
    let (code, stdout, stderr) = run(repo, store, &["index", repo.to_str().unwrap()]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
}

#[test]
fn read_is_symbols_only_whole_and_doc_extended() {
    let (repo, store) = fresh_workspace("whole");
    std::fs::write(
        repo.join("lib.rs"),
        "/// Authored docs.\n#[inline]\npub fn target() {\n    let x = 1;\n    println!(\"{x}\");\n}\n",
    )
    .unwrap();
    index(&repo, &store);

    let (code, stdout, stderr) = run(&repo, &store, &["read", "target"]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert_eq!(
        stdout,
        "lib.rs:1-6  target\n/// Authored docs.\n#[inline]\npub fn target() {\n    let x = 1;\n    println!(\"{x}\");\n}\n"
    );

    let (path_code, path_out, _) = run(&repo, &store, &["read", "lib.rs"]);
    assert_eq!(path_code, 1, "{path_out}");
    assert!(path_out.starts_with("no symbol `lib.rs`"), "{path_out}");
}

#[test]
fn read_head_and_tail_have_truthful_adjacent_headers() {
    let (repo, store) = fresh_workspace("head-tail");
    std::fs::write(
        repo.join("lib.rs"),
        "fn target() {\n    one();\n    two();\n    three();\n}\n",
    )
    .unwrap();
    index(&repo, &store);

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &["read", "target", "--head", "2", "--tail", "2"],
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert_eq!(
        stdout,
        "lib.rs:1-2  target\nfn target() {\n    one();\nlib.rs:4-5  target\n    three();\n}\n"
    );
}

#[test]
fn read_multi_delivers_successes_and_nav_failures() {
    let (repo, store) = fresh_workspace("partial");
    std::fs::write(repo.join("lib.rs"), "fn first() {}\nfn second() {}\n").unwrap();
    index(&repo, &store);

    let (code, stdout, stderr) = run(&repo, &store, &["read", "first", "missing", "second"]);
    assert_eq!(code, 1, "stdout={stdout}\nstderr={stderr}");
    assert!(
        stdout.contains("lib.rs:1-1  first\nfn first() {}"),
        "{stdout}"
    );
    assert!(stdout.contains("\n\nno symbol `missing`\n"), "{stdout}");
    assert!(
        stdout.contains("\n\nlib.rs:2-2  second\nfn second() {}"),
        "{stdout}"
    );
    assert!(!stdout.contains("read:"), "{stdout}");
}

#[test]
fn read_smart_folds_by_structure_and_expand_chains() {
    let (repo, store) = fresh_workspace("smart");
    std::fs::write(
        repo.join("lib.rs"),
        "fn target(xs: &[i32]) {\n    let mut n = 0;\n    for x in xs {\n        if *x > 0 {\n            n += x;\n        }\n    }\n    println!(\"{n}\");\n}\n",
    )
    .unwrap();
    index(&repo, &store);

    let (code, stdout, stderr) = run(&repo, &store, &["read-smart", "target"]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("    let mut n = 0;\n"), "{stdout}");
    assert!(
        stdout.contains("    … 3-7 folded source block — greppy expand "),
        "{stdout}"
    );
    assert!(!stdout.contains("        if *x > 0"), "{stdout}");
    let id = stdout
        .lines()
        .find_map(|line| line.split("greppy expand ").nth(1))
        .expect("gap id");

    let (expand_code, expanded, expand_stderr) = run(&repo, &store, &["expand", id]);
    assert_eq!(expand_code, 0, "stdout={expanded}\nstderr={expand_stderr}");
    assert!(expanded.starts_with("    for x in xs {\n"), "{expanded}");
    assert!(
        expanded.contains("        … 4-6 folded source block — greppy expand "),
        "{expanded}"
    );
    assert!(expanded.ends_with("    }\n"), "{expanded}");
}

#[test]
fn read_smart_applies_path_filters_before_ambiguity_resolution() {
    let (repo, store) = fresh_workspace("smart-path");
    std::fs::create_dir_all(repo.join("a")).unwrap();
    std::fs::create_dir_all(repo.join("b")).unwrap();
    std::fs::write(repo.join("a/lib.rs"), "fn target() {\n    a();\n}\n").unwrap();
    std::fs::write(repo.join("b/lib.rs"), "fn target() {\n    b();\n}\n").unwrap();
    index(&repo, &store);

    let (code, stdout, stderr) = run(&repo, &store, &["read-smart", "target", "--path", "a"]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.starts_with("a/lib.rs:1-3  target\n"), "{stdout}");
    assert!(!stdout.contains("b/lib.rs"), "{stdout}");
}

#[test]
fn read_file_pages_and_expand_continues_at_the_named_line() {
    let (repo, store) = fresh_workspace("pages");
    let content = (1..=805)
        .map(|line| format!("line {line}\n"))
        .collect::<String>();
    std::fs::write(repo.join("long.txt"), &content).unwrap();

    let (code, stdout, stderr) = run(&repo, &store, &["read-file", "long.txt"]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.starts_with("long.txt:1-400\nline 1\n"), "{stdout}");
    assert!(
        stdout.contains("line 400\n405 more lines — greppy expand "),
        "{stdout}"
    );
    assert!(stdout.ends_with(" continues at 401\n"), "{stdout}");
    let id = stdout
        .lines()
        .last()
        .and_then(|line| line.split("greppy expand ").nth(1))
        .and_then(|tail| tail.split_whitespace().next())
        .expect("continuation id");

    let (expand_code, expanded, expand_stderr) = run(&repo, &store, &["expand", id]);
    assert_eq!(expand_code, 0, "stdout={expanded}\nstderr={expand_stderr}");
    assert!(
        expanded.starts_with("long.txt:401-800\nline 401\n"),
        "{expanded}"
    );
    assert!(
        expanded.contains("5 more lines — greppy expand "),
        "{expanded}"
    );
    assert!(expanded.ends_with(" continues at 801\n"), "{expanded}");
}

#[test]
fn read_file_range_and_all_bypass_pagination() {
    let (repo, store) = fresh_workspace("range-all");
    std::fs::write(repo.join("config.json"), "a\nb\nc\nd\n").unwrap();

    let (code, stdout, stderr) = run(
        &repo,
        &store,
        &["read-file", "config.json", "--lines", "2:3"],
    );
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    assert_eq!(stdout, "config.json:2-3\nb\nc\n");

    let (all_code, all_out, all_err) = run(&repo, &store, &["read-file", "config.json", "--all"]);
    assert_eq!(all_code, 0, "stdout={all_out}\nstderr={all_err}");
    assert_eq!(all_out, "config.json:1-4\na\nb\nc\nd\n");
}

#[test]
fn read_handle_is_compact_and_existing_json_shape_survives() {
    let (repo, store) = fresh_workspace("handle");
    std::fs::write(repo.join("lib.rs"), "pub fn target() {}\n").unwrap();
    index(&repo, &store);

    let (code, stdout, stderr) = run(&repo, &store, &["read", "target", "--handle", "--json"]);
    assert_eq!(code, 0, "stdout={stdout}\nstderr={stderr}");
    let value: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(value["schema_version"], "greppy.read.v1");
    assert_eq!(value["status"], "ok");
    let handle = value["handle"].as_str().expect("handle");
    assert!(handle.starts_with("geh2:"), "{handle}");
    assert!(handle.len() <= 70, "{}: {handle}", handle.len());
}

#[test]
fn compact_read_handle_still_drives_replace_span() {
    let (repo, store) = fresh_workspace("handle-replace");
    let original = "pub fn target() {}\n";
    std::fs::write(repo.join("lib.rs"), original).unwrap();
    index(&repo, &store);

    let (read_code, read_out, read_err) = run(&repo, &store, &["read", "target", "--handle"]);
    assert_eq!(read_code, 0, "stdout={read_out}\nstderr={read_err}");
    let handle = read_out
        .lines()
        .find_map(|line| line.strip_prefix("handle: "))
        .expect("compact read handle");

    let replacement = "pub fn target() { println!(\"changed\"); }\n";
    let (edit_code, edit_out, edit_err) = run(
        &repo,
        &store,
        &["replace-span", handle, replacement, "--dry-run"],
    );
    assert_eq!(edit_code, 0, "stdout={edit_out}\nstderr={edit_err}");
    assert_eq!(
        std::fs::read_to_string(repo.join("lib.rs")).unwrap(),
        original,
        "dry-run must not publish the replacement"
    );
}
