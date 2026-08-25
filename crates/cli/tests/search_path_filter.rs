//! `--path P` on the search family: hits are filtered to files under P
//! BEFORE any count is taken, an empty filtered set says so the way the nav
//! commands do, and an empty filtered set is a successful bounded status.

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
        "greppy-cli-search-path-{tag}-{}-{n}",
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

/// Two files defining like-named symbols, one under `src/`, one under
/// `tools/` — enough to see the filter keep one side and drop the other.
fn indexed_two_tree_repo(tag: &str) -> (PathBuf, PathBuf) {
    let (repo, store) = fresh_workspace(tag);
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::create_dir_all(repo.join("tools")).unwrap();
    std::fs::write(
        repo.join("src/lib.rs"),
        "pub fn parse_widget(input: &str) -> &str { input }\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("tools/lib.rs"),
        "pub fn parse_widget_tools(input: &str) -> &str { input }\n",
    )
    .unwrap();
    let (code, _out, err) = run(&repo, &store, &["index"]);
    assert_eq!(code, 0, "index failed; stderr={err}");
    (repo, store)
}

#[test]
fn search_symbol_path_keeps_only_hits_under_the_filter() {
    let (repo, store) = indexed_two_tree_repo("symbol-keep");

    let (code, out, err) = run(
        &repo,
        &store,
        &["search-symbol", "parse_widget", "--path", "src"],
    );

    assert_eq!(code, 0, "stdout={out}\nstderr={err}");
    assert!(out.contains("src/lib.rs"), "{out}");
    assert!(
        !out.contains("tools/lib.rs"),
        "--path src must drop the tools hit; got: {out}"
    );
}

#[test]
fn search_symbol_path_with_no_hit_returns_bounded_status() {
    let (repo, store) = indexed_two_tree_repo("symbol-empty");

    let (code, out, err) = run(
        &repo,
        &store,
        &["search-symbol", "parse_widget", "--path", "no-such-dir"],
    );

    // grep's convention: 0 for a hit, 1 for none. The bounded status
    // below carries the guidance; the exit code does not repeat it.
    assert_eq!(code, 1, "stdout={out}\nstderr={err}");
    assert!(
        out.contains("no definition named `parse_widget` under path filter: no-such-dir"),
        "{out}"
    );
    assert!(!out.contains("lib.rs:"), "no hit rows may leak: {out}");
}

#[test]
fn search_symbol_path_json_counts_the_filtered_set() {
    let (repo, store) = indexed_two_tree_repo("symbol-json");

    let (code, out, err) = run(
        &repo,
        &store,
        &[
            "search-symbol",
            "parse_widget",
            "--path",
            "src",
            "--json",
            "--diagnostics",
        ],
    );

    assert_eq!(code, 0, "stdout={out}\nstderr={err}");
    let value: serde_json::Value = serde_json::from_str(&out).expect("json output");
    assert_eq!(value["path_filters"], serde_json::json!(["src"]));
    let hits = value["hits"].as_array().expect("hits array");
    assert!(!hits.is_empty(), "{out}");
    for hit in hits {
        let file = hit["file"].as_str().expect("hit file");
        assert!(file.starts_with("src/"), "filtered hit escaped: {file}");
    }
    assert_eq!(
        value["total_exact"].as_i64().expect("total_exact"),
        hits.len() as i64,
        "total_exact must count the filtered set: {out}"
    );
}

#[test]
fn search_pattern_path_filters_rows_before_counting() {
    let (repo, store) = indexed_two_tree_repo("pattern-json");

    let (code, out, err) = run(
        &repo,
        &store,
        &[
            "search-pattern",
            "fn parse_widget",
            "--path",
            "src",
            "--json",
            "--diagnostics",
        ],
    );

    assert_eq!(code, 0, "stdout={out}\nstderr={err}");
    let value: serde_json::Value = serde_json::from_str(&out).expect("json output");
    assert_eq!(value["path_filters"], serde_json::json!(["src"]));
    let hits = value["hits"].as_array().expect("hits array");
    assert!(!hits.is_empty(), "{out}");
    for hit in hits {
        let file = hit["file"].as_str().expect("hit file");
        assert!(file.starts_with("src/"), "filtered hit escaped: {file}");
    }
    assert_eq!(
        value["total_exact"].as_i64().expect("total_exact"),
        hits.len() as i64,
        "total_exact must count the filtered rows: {out}"
    );
}

#[test]
fn search_pattern_path_with_no_hit_returns_bounded_status() {
    let (repo, store) = indexed_two_tree_repo("pattern-empty");

    let (code, out, err) = run(
        &repo,
        &store,
        &["search-pattern", "fn parse_widget", "--path", "no-such-dir"],
    );

    // grep's convention: 0 for a hit, 1 for none. The bounded status
    // below carries the guidance; the exit code does not repeat it.
    assert_eq!(code, 1, "stdout={out}\nstderr={err}");
    assert!(
        out.contains("no matches under path filter: no-such-dir"),
        "{out}"
    );
}

#[test]
fn search_pattern_without_path_still_sees_both_trees() {
    let (repo, store) = indexed_two_tree_repo("pattern-unfiltered");

    let (code, out, err) = run(&repo, &store, &["search-pattern", "fn parse_widget"]);

    assert_eq!(code, 0, "stdout={out}\nstderr={err}");
    assert!(out.contains("src/lib.rs"), "{out}");
    assert!(out.contains("tools/lib.rs"), "{out}");
}
