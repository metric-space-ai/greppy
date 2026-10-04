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
fn existing_absolute_outside_paths_are_rejected_before_index_preparation() {
    let (repo, store) = fresh_workspace("outside-absolute");
    let external = repo.parent().unwrap().join("terminal_report.py");
    std::fs::write(&external, "def record(): pass\n").unwrap();
    std::fs::write(repo.join("lib.rs"), "pub fn record() {}\n").unwrap();
    for command in ["search-symbol", "search-pattern", "search"] {
        let (code, out, err) = run(
            &repo,
            &store,
            &[command, "record", "--path", external.to_str().unwrap()],
        );
        assert_eq!(code, 64, "{command}: stdout={out}; stderr={err}");
        assert!(err.contains("outside the repository"), "{err}");
        assert!(err.contains("--root") && err.contains("read-file"), "{err}");
        assert!(!out.contains("greppy index"), "{out}");
        assert!(
            !store.join("workspaces").exists(),
            "wrong-root search started an index"
        );
    }
    let source = repo.join("lib.rs");
    let (code, out, err) = run(
        &repo,
        &store,
        &[
            "search-symbol",
            "record",
            "--path",
            source.to_str().unwrap(),
        ],
    );
    assert_eq!(code, 0, "inside path: stdout={out}; stderr={err}");
    assert!(out.contains("lib.rs"), "{out}");
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
}

#[test]
fn search_pattern_relative_filter_uses_selected_root_from_another_checkout() {
    let (repo, store) = fresh_workspace("external-cwd-filter");
    let caller = repo.parent().unwrap().join("other-checkout");
    for root in [&repo, &caller] {
        std::fs::create_dir_all(root.join("src")).unwrap();
    }
    std::fs::write(
        repo.join("src/lib.rs"),
        "fn selected() { /* SELECTED_ONLY */ }\n",
    )
    .unwrap();
    std::fs::write(caller.join("src/lib.rs"), "fn foreign() {}\n").unwrap();
    let root_arg = repo.to_str().unwrap();
    let nested = repo.join("src");
    for (cwd, filter) in [
        (caller.as_path(), "src"),
        (caller.as_path(), "src/lib.rs"),
        (nested.as_path(), "lib.rs"),
    ] {
        let (code, out, err) = run(
            cwd,
            &store,
            &[
                "search-pattern",
                "SELECTED_ONLY",
                "--fixed",
                "--root",
                root_arg,
                "--path",
                filter,
                "--json",
            ],
        );
        assert_eq!(code, 0, "filter={filter}; stdout={out}; stderr={err}");
        let result: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(result["total_exact"], 1, "{out}");
        assert_eq!(result["hits"][0]["file"], "src/lib.rs", "{out}");
    }
    let (code, out, err) = run(
        &caller,
        &store,
        &[
            "search-pattern",
            "foreign",
            "--root",
            root_arg,
            "--path",
            "../other-checkout/src",
            "--json",
        ],
    );
    assert_eq!(code, 1, "stdout={out}; stderr={err}");
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
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
fn search_symbol_accepts_the_qualified_method_name_it_displays() {
    let (repo, store) = fresh_workspace("qualified-method");
    std::fs::write(
        repo.join("lib.rs"),
        "struct EmbeddingGemma;\nimpl EmbeddingGemma {\n    fn states_for_chunk(&self) {}\n}\nstruct Other;\nimpl Other {\n    fn states_for_chunk(&self) {}\n}\n",
    ).unwrap();
    let (code, out, err) = run(&repo, &store, &["index"]);
    assert_eq!(code, 0, "stdout={out}\nstderr={err}");
    let (code, out, err) = run(
        &repo,
        &store,
        &[
            "search-symbol",
            "EmbeddingGemma::states_for_chunk",
            "--code",
        ],
    );
    assert_eq!(code, 0, "stdout={out}\nstderr={err}");
    assert!(!out.contains("no_matches"), "{out}");
    assert!(!out.contains("similar names:"), "{out}");
    assert!(out.contains("EmbeddingGemma::states_for_chunk"), "{out}");
    assert!(out.contains("fn states_for_chunk"), "{out}");
    assert!(!out.contains("Other::states_for_chunk"), "{out}");

    let (code, out, err) = run(
        &repo,
        &store,
        &[
            "search-symbol",
            "EmbeddingGemma::states_for_chunk",
            "--json",
            "--diagnostics",
        ],
    );
    assert_eq!(code, 0, "stdout={out}\nstderr={err}");
    let value: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(value["status"], "ok", "{out}");
    assert_eq!(value["total_exact"], 1, "{out}");
    let hits = value["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 1, "{out}");
    assert!(
        hits[0]["qualified_name"]
            .as_str()
            .unwrap()
            .contains("EmbeddingGemma"),
        "{out}"
    );

    let (code, out, err) = run(
        &repo,
        &store,
        &[
            "search-symbol",
            "EmbeddingGemma::states_for_chunk",
            "--path",
            "absent",
        ],
    );
    assert_eq!(code, 1, "stdout={out}\nstderr={err}");
    assert!(!out.contains("lib.rs:"), "{out}");
    std::fs::remove_dir_all(repo.parent().unwrap()).unwrap();
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
