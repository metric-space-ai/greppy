//! Regression coverage for Postel-style navigation inputs and miss guidance.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fresh_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "greppy-cli-nav-postel-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

fn make_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::create_dir_all(repo.join("src/inside")).unwrap();
    std::fs::create_dir_all(repo.join("tests")).unwrap();

    std::fs::write(
        repo.join("src/api.rs"),
        r#"
pub fn target() {}

#[allow(non_snake_case)]
pub fn startsWith() {}

pub trait Encode {
    fn serialize(&self) -> u32;
}

pub struct Option;

impl Encode for Option {
    fn serialize(&self) -> u32 {
        7
    }
}
"#,
    )
    .unwrap();
    std::fs::write(
        repo.join("src/inside/caller.rs"),
        "pub fn caller_inside() { crate::api::target(); }\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("tests/outside.rs"),
        "pub fn caller_outside() { crate::api::target(); }\n",
    )
    .unwrap();

    (repo, root.join("store"))
}

fn run(args: &[&str], cwd: &Path, store: &Path) -> (i32, String, String) {
    let out = Command::new(bin())
        .args(args)
        .current_dir(cwd)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .output()
        .expect("spawn greppy");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn indexed_repo(tag: &str) -> (PathBuf, PathBuf) {
    let (repo, store) = make_repo(tag);
    let (code, stdout, stderr) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index failed\nstdout={stdout}\nstderr={stderr}");
    (repo, store)
}

fn without_expand_id(output: (i32, String, String)) -> (i32, String, String) {
    let stdout = output
        .1
        .lines()
        .filter(|line| !line.starts_with("Expand: greppy expand "))
        .collect::<Vec<_>>()
        .join("\n");
    (output.0, stdout, output.2)
}

#[test]
fn who_calls_positional_directory_filter_returns_subset_and_explains_empty_scope() {
    let (repo, store) = indexed_repo("path-filter");

    let (code, stdout, stderr) = run(&["who-calls", "target", "src/inside"], &repo, &store);
    assert_eq!(code, 0, "stderr={stderr}\nstdout={stdout}");
    assert!(stdout.contains("caller_inside"), "stdout={stdout}");
    assert!(!stdout.contains("caller_outside"), "stdout={stdout}");

    let (code, stdout, stderr) = run(&["who-calls", "target", "does/not/exist"], &repo, &store);
    assert_eq!(code, 0, "stderr={stderr}\nstdout={stdout}");
    assert!(
        stdout.contains("no callers under path filter: does/not/exist"),
        "stdout={stdout}"
    );
}

#[test]
fn root_file_and_subdirectory_misuse_teaches_the_real_root_and_corrected_command() {
    let (repo, store) = indexed_repo("root-guidance");
    let real_root = repo.canonicalize().unwrap().to_string_lossy().into_owned();

    for wrong_root in ["src/api.rs", "src/inside"] {
        let (code, stdout, stderr) = run(
            &["who-calls", "target", "--root", wrong_root],
            &repo,
            &store,
        );
        assert_ne!(code, 0, "stdout={stdout}\nstderr={stderr}");
        let combined = format!("{stdout}\n{stderr}");
        assert!(
            combined.contains("--root selects the indexed repository root"),
            "combined={combined}"
        );
        assert!(combined.contains(&real_root), "combined={combined}");
        assert!(
            combined.contains(&format!(
                "greppy who-calls target {wrong_root} --root {real_root}"
            )),
            "combined={combined}"
        );
    }
}

#[test]
fn symbol_miss_suggests_case_insensitive_near_match_and_discovery_commands() {
    let (repo, store) = indexed_repo("miss-guidance");
    let (code, stdout, stderr) = run(&["who-calls", "startswith"], &repo, &store);
    assert_eq!(code, 1, "stderr={stderr}\nstdout={stdout}");
    assert!(
        stdout.contains("suggestion: `startsWith`"),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains("try: greppy search-symbols"),
        "stdout={stdout}"
    );
    assert!(
        stdout.contains("try: greppy semantic-search"),
        "stdout={stdout}"
    );
}

#[test]
fn type_method_query_resolves_rust_trait_impl_method() {
    let (repo, store) = indexed_repo("trait-impl-method");

    let (code, stdout, stderr) = run(&["brief", "Option"], &repo, &store);
    assert_eq!(code, 0, "stderr={stderr}\nstdout={stdout}");
    assert!(stdout.contains("Option"), "stdout={stdout}");

    let (code, stdout, stderr) = run(&["read", "Option::serialize"], &repo, &store);
    assert_eq!(code, 0, "stderr={stderr}\nstdout={stdout}");
    assert!(stdout.contains("fn serialize(&self)"), "stdout={stdout}");
    assert!(stdout.contains("src/api.rs"), "stdout={stdout}");
}

#[test]
fn limit_max_path_and_read_symbol_aliases_are_output_identical() {
    let (repo, store) = indexed_repo("aliases");

    let limit = run(&["search-code", "pub", "--limit", "1"], &repo, &store);
    let max = run(&["search-code", "pub", "--max", "1"], &repo, &store);
    assert_eq!(limit, max, "--limit and --max must be exact aliases");

    for (command, query, path) in [
        ("who-calls", "target", "src/inside"),
        ("callees", "caller_inside", "src/api.rs"),
        ("find-usages", "target", "src/inside"),
        ("search-code", "target", "src/inside"),
        ("search-symbols", "target", "src/api.rs"),
    ] {
        let positional = without_expand_id(run(&[command, query, path], &repo, &store));
        let flagged = without_expand_id(run(
            &[command, query, "--path", path],
            &repo,
            &store,
        ));
        assert_eq!(
            positional, flagged,
            "{command}: positional PATH and --path differ"
        );
    }

    let positional = run(&["read", "target"], &repo, &store);
    let flagged = run(&["read", "--symbol", "target"], &repo, &store);
    assert_eq!(positional, flagged, "positional SYMBOL and --symbol differ");
}

#[test]
fn global_output_flags_work_before_and_after_subcommand() {
    let (repo, store) = indexed_repo("global-flags");

    for flag in ["--json", "--code", "--all"] {
        let before = run(&[flag, "who-calls", "target"], &repo, &store);
        let after = run(&["who-calls", "target", flag], &repo, &store);
        if flag == "--json" {
            assert_eq!(before.0, after.0);
            assert_eq!(before.2, after.2);
            let before: serde_json::Value = serde_json::from_str(&before.1).unwrap();
            let after: serde_json::Value = serde_json::from_str(&after.1).unwrap();
            assert_eq!(before["command"], after["command"]);
            assert_eq!(before["all"], after["all"]);
            assert_eq!(before["hits"], after["hits"]);
            assert_eq!(before["shown"], after["shown"]);
        } else {
            assert_eq!(before, after, "global flag ordering differs for {flag}");
        }
    }

    let before = run(
        &["--root", ".", "search-code", "target", "--limit", "1"],
        &repo,
        &store,
    );
    let after = run(
        &["search-code", "target", "--limit", "1", "--root", "."],
        &repo,
        &store,
    );
    assert_eq!(before, after, "--root ordering differs");
}

#[test]
fn unknown_flag_suggests_a_complete_corrected_invocation() {
    let (repo, store) = indexed_repo("unknown-flag");
    let (code, stdout, stderr) = run(
        &["search-code", "target", "--jsoon", "--limit", "1"],
        &repo,
        &store,
    );
    assert_eq!(code, 64, "stdout={stdout}\nstderr={stderr}");
    assert!(
        stdout.contains(" search-code target --json --limit 1"),
        "stdout={stdout}"
    );
}
