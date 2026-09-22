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

#[cfg(unix)]
#[test]
fn search_pattern_limits_content_scans_to_the_path_filter() {
    use std::os::unix::fs::PermissionsExt;

    let (repo, store) = indexed_two_tree_repo("pattern-content-scope");
    let base = repo.parent().unwrap();
    let shim_dir = base.join("bin");
    std::fs::create_dir_all(&shim_dir).unwrap();
    let trace = base.join("grep-args");
    let shim = shim_dir.join("grep");
    std::fs::write(
        &shim,
        "#!/bin/sh\nprintf '<call>\\n' >> \"$GREPPY_TEST_GREP_TRACE\"\nprintf '%s\\n' \"$@\" >> \"$GREPPY_TEST_GREP_TRACE\"\nexec /usr/bin/grep \"$@\"\n",
    )
    .unwrap();
    std::fs::set_permissions(&shim, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut search_path = vec![shim_dir];
    search_path.extend(std::env::split_paths(
        &std::env::var_os("PATH").unwrap_or_default(),
    ));
    let search_path = std::env::join_paths(search_path).unwrap();

    // Check both file and directory scopes, regex and literal searches, and
    // the second content scan used to explain a case-sensitive miss.
    for (query, scope, fixed, expected_scans) in [
        ("parse_widget", "src/lib.rs", false, 1),
        ("PARSE_WIDGET", "src", false, 2),
        ("PARSE_WIDGET", "src/lib.rs", true, 2),
    ] {
        std::fs::write(&trace, "").unwrap();
        let mut command = Command::new(bin());
        command
            .args(["search-pattern", query, "--path", scope])
            .current_dir(&repo)
            .env("GREPPY_STORE_DIR", &store)
            .env("GREPPY_TEST_SKIP_INFERENCE", "1")
            .env("GREPPY_AUTO_REINDEX", "0")
            .env("GREPPY_TEST_GREP_TRACE", &trace)
            .env("PATH", &search_path);
        if fixed {
            command.arg("--fixed");
        }
        let output = command.output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "stdout={stdout}\nstderr={}",
            String::from_utf8_lossy(&output.stderr)
        );
        if expected_scans == 2 {
            assert!(stdout.contains("case-insensitive: 1 matches"), "{stdout}");
        } else {
            assert!(stdout.contains("src/lib.rs"), "{stdout}");
        }
        let recorded = std::fs::read_to_string(&trace).unwrap();
        let scans: Vec<_> = recorded
            .split("<call>\n")
            .filter(|s| !s.is_empty())
            .collect();
        assert_eq!(scans.len(), expected_scans, "{recorded}");
        for scan in scans {
            let args: Vec<_> = scan.lines().collect();
            let separator = args.iter().position(|arg| *arg == "--").unwrap();
            assert_eq!(
                &args[separator + 2..],
                &["src/lib.rs"],
                "content scan escaped the requested scope: {recorded}"
            );
        }
    }
    std::fs::remove_dir_all(base).unwrap();
}
