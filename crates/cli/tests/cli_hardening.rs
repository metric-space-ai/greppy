//! Integration tests for the Track 1 CLI-hardening fixes
//! (RV-003, RV-006, RV-007, RV-011).
//!
//! These spawn the real `greppy` binary as a subprocess so the cwd /
//! repo-root / store-path resolution is exercised end-to-end (the
//! relevant dispatch helpers are private to the crate, and cwd-sensitive
//! behaviour cannot be tested by mutating the shared process cwd under
//! cargo's parallel test runner). Each test gets an isolated
//! `GREPPY_STORE_DIR` so they never collide.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

/// Path to the binary under test (provided by cargo for integration tests).
fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

/// Create a unique, fresh scratch directory under the system temp dir.
fn fresh_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("greppy-cli-it-{tag}-{pid}-{n}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Build a minimal git-rooted repo with one Rust file containing
/// `marker`, plus an empty `sub/` directory. Returns (repo_root,
/// store_dir).
fn make_repo(tag: &str, marker: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join("sub")).unwrap();
    // `.git` is the repo-root marker that resolve_root walks up to find.
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(
        repo.join("lib.rs"),
        format!("pub fn {marker}() -> i32 {{ 7 }}\n"),
    )
    .unwrap();
    let store = root.join("store");
    (repo, store)
}

/// Run the binary with the given args, cwd, and store dir. Returns
/// (exit_code, stdout, stderr).
fn run(args: &[&str], cwd: &Path, store_dir: &Path) -> (i32, String, String) {
    run_with_env(args, cwd, store_dir, &[])
}

fn run_with_env(
    args: &[&str],
    cwd: &Path,
    store_dir: &Path,
    envs: &[(&str, &str)],
) -> (i32, String, String) {
    let mut cmd = Command::new(bin());
    cmd.args(args)
        .current_dir(cwd)
        .env("GREPPY_STORE_DIR", store_dir)
        .env_remove("GREPPY_DISCOVER_INCLUDE")
        .env_remove("GREPPY_DISCOVER_EXCLUDE");
    for (key, value) in envs {
        cmd.env(key, value);
    }
    let out = cmd
        // Keep the child from inheriting an unexpected store override.
        .output()
        .expect("spawn greppy");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn git(repo: &Path, args: &[&str]) {
    let out = Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .expect("spawn git");
    assert!(
        out.status.success(),
        "git {args:?} failed\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

fn make_real_git_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join("src")).unwrap();
    std::fs::write(
        repo.join("src/lib.rs"),
        "pub fn clean_committed_marker() -> i32 { 1 }\n",
    )
    .unwrap();
    git(&repo, &["init"]);
    git(&repo, &["config", "user.email", "greppy@example.invalid"]);
    git(&repo, &["config", "user.name", "greppy test"]);
    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-m", "baseline"]);
    let store = root.join("store");
    (repo, store)
}

/// Locate the single `graph.db` created beneath `store_dir`.
fn find_graph_db(store_dir: &Path) -> Option<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.file_name().and_then(|s| s.to_str()) == Some("graph.db") {
                    out.push(p);
                }
            }
        }
    }
    let mut found = Vec::new();
    walk(store_dir, &mut found);
    found.into_iter().next()
}

fn backup_path_for_db(db: &Path) -> PathBuf {
    let file_name = db.file_name().unwrap().to_string_lossy();
    db.with_file_name(format!("{file_name}.prev"))
}

fn next_snapshot_paths_for_db(db: &Path) -> Vec<PathBuf> {
    let Some(parent) = db.parent() else {
        return Vec::new();
    };
    let Some(file_name) = db.file_name().and_then(|s| s.to_str()) else {
        return Vec::new();
    };
    let prefix = format!("{file_name}.next.");
    let mut paths = std::fs::read_dir(parent)
        .ok()
        .into_iter()
        .flat_map(|rd| rd.flatten())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|name| name.starts_with(&prefix))
        })
        .collect::<Vec<_>>();
    paths.sort();
    paths
}

fn corrupt_snapshot_for_db(db: &Path) -> Option<PathBuf> {
    let parent = db.parent()?;
    std::fs::read_dir(parent)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|name| name.starts_with("graph.db.corrupt."))
        })
}

#[cfg(unix)]
fn mode_of(path: &Path) -> u32 {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(path).unwrap().mode() & 0o777
}

// ---------------------------------------------------------------------------
// RV-011 — index . then search-code finds content (same project identity).
// RV-006 — searching from a subdirectory resolves the SAME store.
// ---------------------------------------------------------------------------

#[test]
fn index_dot_then_search_from_root_and_subdir() {
    let (repo, store) = make_repo("casedot", "alpha_unique_marker");

    // `greppy index .` from the repo root.
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("project: repo"),
        "index should key project on the repo-root basename; got: {out}"
    );

    // RV-011: search-code from the repo root finds the indexed content.
    let (code, out, err) = run(&["search-code", "alpha_unique_marker"], &repo, &store);
    assert_eq!(code, 0, "search-code from root should exit 0; stderr={err}");
    assert!(
        out.contains("alpha_unique_marker"),
        "search-code from root must find indexed content (RV-011); got: {out:?}"
    );

    // RV-006: search-code from a SUBDIRECTORY must resolve the same store
    // (walk up to the .git root) and still find the content — not exit 73.
    let sub = repo.join("sub");
    let (code, out, err) = run(&["search-code", "alpha_unique_marker"], &sub, &store);
    assert_eq!(
        code, 0,
        "search-code from subdir must exit 0, not 73 (RV-006); stderr={err}"
    );
    assert!(
        out.contains("alpha_unique_marker"),
        "search-code from subdir must find content via the shared store (RV-006); got: {out:?}"
    );
    assert!(
        !out.contains("(no matches)"),
        "subdir search must not report (no matches); got: {out:?}"
    );
}

#[test]
fn search_code_json_reports_exact_counts_and_truncation_metadata() {
    let root = fresh_dir("search-json");
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::create_dir_all(repo.join("src")).unwrap();
    let lines = (0..25)
        .map(|i| format!("pub fn json_marker_{i}() {{ let json_unique_marker = {i}; }}"))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(repo.join("src/lib.rs"), format!("{lines}\n")).unwrap();
    let store = root.join("store");

    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );

    let (code, out, err) = run(
        &["search-code", "--json", "json_unique_marker"],
        &repo,
        &store,
    );
    assert_eq!(code, 0, "search-code --json should exit 0; stderr={err}");
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "search-code");
    assert_eq!(v["status"], "ok");
    assert_eq!(v["fresh"], true);
    assert_eq!(v["query"], "json_unique_marker");
    assert_eq!(v["project"], "repo");
    assert_eq!(v["provider_complete"], false);
    assert!(
        v["incomplete_provider_count"].as_u64().unwrap_or(0) >= 1,
        "search-code JSON must expose provider incompleteness: {v:?}"
    );
    assert!(
        v["incomplete_providers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["language"] == "rust"),
        "rust provider incompleteness must be visible: {v:?}"
    );
    assert_eq!(v["total_exact"], 25);
    assert_eq!(v["shown"], 20);
    assert_eq!(v["omitted"], 5);
    assert_eq!(v["truncated"], true);
    assert_eq!(v["hits"].as_array().unwrap().len(), 20);
    assert!(
        v["hits"][0]["location"]
            .as_str()
            .unwrap_or("")
            .starts_with("src/lib.rs:"),
        "hit must carry grep-like location, got {v:?}"
    );
}

/// D2 fail-open, small drift: the one-file edit is auto-reindexed
/// inline, so search-code answers FRESH about the CURRENT tree — the
/// removed marker is honestly gone, the new one is found. (The old
/// contract refused with `skipped_stale_index` + exit 1.)
#[test]
fn search_code_json_auto_reindexes_and_reports_current_state() {
    let (repo, store) = make_repo("search-json-stale", "old_json_stale_marker");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );
    std::fs::write(
        repo.join("lib.rs"),
        "pub fn new_json_stale_marker() -> i32 { 8 }\n",
    )
    .unwrap();

    let (code, out, err) = run(
        &["search-code", "--json", "old_json_stale_marker"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 1,
        "healed index: the OLD marker no longer exists anywhere; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "search-code");
    assert_eq!(v["status"], "ok");
    assert_eq!(
        v["fresh"], true,
        "auto-reindex must yield a fresh answer: {v:?}"
    );
    assert_eq!(v["total_exact"], 0);
    assert_eq!(v["hits"].as_array().unwrap().len(), 0);

    let (code, out, err) = run(
        &["search-code", "--json", "new_json_stale_marker"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "healed index must find the NEW marker; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["status"], "ok");
    assert!(
        !v["hits"].as_array().unwrap().is_empty(),
        "healed index must serve the current content: {v:?}"
    );
}

/// D2: with the auto-reindex kill switch set, stale search-code --json
/// serves the OLD indexed rows, labeled `fresh: false`, plus a stderr
/// warning — never exit-1-with-nothing.
#[test]
fn search_code_json_serves_labeled_stale_hits_when_auto_reindex_disabled() {
    let (repo, store) = make_repo("search-json-stale-label", "old_labeled_stale_marker");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );
    std::fs::write(
        repo.join("lib.rs"),
        "pub fn replacement_marker() -> i32 { 8 }\n",
    )
    .unwrap();

    let (code, out, err) = run_with_env(
        &["search-code", "--json", "old_labeled_stale_marker"],
        &repo,
        &store,
        &[("GREPPY_AUTO_REINDEX", "0")],
    );
    assert_eq!(
        code, 0,
        "labeled-stale search-code must serve the indexed rows; stderr={err}\nstdout={out}"
    );
    assert!(
        err.contains("index may be stale") && err.contains("run 'grep index'"),
        "labeled-stale search-code must warn on stderr; stderr={err:?}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "search-code");
    assert_eq!(v["status"], "ok");
    assert_eq!(v["fresh"], false, "result must be labeled stale: {v:?}");
    assert_eq!(v["freshness"]["state"], "stale");
    assert_eq!(v["freshness"]["stale_file_count"], 1);
    assert!(
        !v["hits"].as_array().unwrap().is_empty(),
        "labeled-stale search-code must serve rows from the existing index: {v:?}"
    );
}

#[test]
fn provider_policy_require_complete_does_not_block_search_code_json() {
    let (repo, store) = make_repo("provider-policy-search-code", "provider_policy_code_marker");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );

    let (code, out, err) = run_with_env(
        &["search-code", "--json", "provider_policy_code_marker"],
        &repo,
        &store,
        &[("GREPPY_PROVIDER_POLICY", "require_complete")],
    );
    assert_eq!(
        code, 0,
        "strict provider policy must not block literal search-code; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "search-code JSON should remain machine-readable; stderr={err:?}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "search-code");
    assert_eq!(v["status"], "ok");
    assert_eq!(v["provider_complete"], false);
    assert_eq!(v["shown"], 1);
    assert_eq!(v["hits"].as_array().unwrap().len(), 1);
}

#[test]
fn provider_policy_require_complete_blocks_search_symbols_json() {
    let (repo, store) = make_repo(
        "provider-policy-search-symbols",
        "provider_policy_symbol_marker",
    );
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );

    let (code, out, err) = run_with_env(
        &["search-symbols", "--json", "provider_policy_symbol_marker"],
        &repo,
        &store,
        &[("GREPPY_PROVIDER_POLICY", "require_complete")],
    );
    assert_eq!(
        code, 1,
        "strict provider policy should block provider-dependent symbol output; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "strict search-symbols JSON should not require stderr parsing; stderr={err:?}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "search-symbols");
    assert_eq!(v["status"], "skipped_incomplete_provider");
    assert_eq!(v["provider_complete"], false);
    assert!(
        v["incomplete_provider_count"].as_u64().unwrap_or(0) >= 1,
        "strict provider policy must expose the incomplete providers: {v:?}"
    );
    assert_eq!(v["total_exact"], 0);
    assert_eq!(v["shown"], 0);
    assert_eq!(v["hits"].as_array().unwrap().len(), 0);
}

#[test]
fn provider_policy_require_complete_blocks_context_json() {
    let (repo, store) = make_repo("provider-policy-context", "provider_policy_context_marker");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );

    let (code, out, err) = run_with_env(
        &["context", "--json", "provider_policy_context_marker"],
        &repo,
        &store,
        &[("GREPPY_PROVIDER_POLICY", "require_complete")],
    );
    assert_eq!(
        code, 1,
        "strict provider policy should block context spans from partial providers; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "strict context JSON should not require stderr parsing; stderr={err:?}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "context");
    assert_eq!(v["status"], "skipped_incomplete_provider");
    assert_eq!(v["provider_complete"], false);
    assert_eq!(v["shown"], 0);
    assert_eq!(v["spans"].as_array().unwrap().len(), 0);
}

#[test]
fn provider_policy_require_complete_blocks_semantic_vectors_before_model_config() {
    let (repo, store) = make_repo(
        "provider-policy-semantic-vector",
        "provider_policy_semantic_vector_marker",
    );
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );

    let (code, out, err) = run_with_env(
        &[
            "semantic-search",
            "--vectors",
            "--json",
            "--embedding-gguf",
            "/missing/embeddinggemma.gguf",
            "--embedding-tokenizer",
            "/missing/tokenizer.json",
            "find provider policy semantic vector marker",
        ],
        &repo,
        &store,
        &[("GREPPY_PROVIDER_POLICY", "require_complete")],
    );
    assert_eq!(
        code, 1,
        "strict provider policy should block semantic vectors before model config/load; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "strict semantic vector JSON should not surface model-load/config errors; stderr={err:?}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "semantic-search");
    assert_eq!(v["mode"], "vector");
    assert_eq!(v["status"], "skipped_incomplete_provider");
    assert_eq!(v["provider_complete"], false);
    assert_eq!(v["shown"], 0);
    assert_eq!(v["hits"].as_array().unwrap().len(), 0);
}

#[test]
fn provider_policy_require_complete_blocks_plus_vectors_before_model_config() {
    let (repo, store) = make_repo("provider-policy-plus-vector", "provider_policy_plus_marker");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );

    let (code, out, err) = run_with_env(
        &[
            "plus",
            "provider_policy_plus_marker",
            "--vectors",
            "--json",
            "--embedding-gguf",
            "/missing/embeddinggemma.gguf",
            "--embedding-tokenizer",
            "/missing/tokenizer.json",
        ],
        &repo,
        &store,
        &[("GREPPY_PROVIDER_POLICY", "require_complete")],
    );
    assert_eq!(
        code, 1,
        "strict provider policy should block plus vectors before model config/load; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "strict plus vector JSON should not surface model-load/config errors; stderr={err:?}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "plus");
    assert_eq!(v["status"], "skipped_incomplete_provider");
    assert_eq!(v["provider_complete"], false);
    assert_eq!(v["vectors"], true);
    assert_eq!(v["vector_status"], "skipped_incomplete_provider");
    assert_eq!(v["shown"], 0);
    assert_eq!(v["hits"].as_array().unwrap().len(), 0);
}

#[test]
fn search_code_stale_text_falls_back_to_live_grep() {
    let (repo, store) = make_repo("search-text-stale", "old_text_stale_marker");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );
    std::fs::write(
        repo.join("lib.rs"),
        "pub fn new_text_stale_marker() -> i32 { 8 }\n",
    )
    .unwrap();

    // Kill the inline auto-reindex so the stale text path (live-grep
    // fallback) is actually exercised; with the default policy this
    // small drift would be healed and served from the index instead.
    let (code, out, err) = run_with_env(
        &["search-code", "new_text_stale_marker"],
        &repo,
        &store,
        &[("GREPPY_AUTO_REINDEX", "0")],
    );
    assert_eq!(
        code, 0,
        "stale search-code text should live-grep current files; stderr={err}\nstdout={out}"
    );
    assert!(
        err.contains("falling back to live grep"),
        "stale search-code text should explain live fallback; stderr={err:?}"
    );
    assert!(
        out.contains("new_text_stale_marker"),
        "live fallback must find the current marker; got: {out:?}"
    );
    assert!(
        !out.contains("old_text_stale_marker"),
        "live fallback must not emit stale indexed snippets; got: {out:?}"
    );
}

#[test]
fn search_code_changed_text_live_greps_only_git_changes_without_index() {
    let (repo, store) = make_real_git_repo("search-code-changed-text");
    std::fs::write(
        repo.join("src/lib.rs"),
        "pub fn changed_text_marker() -> i32 { 2 }\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("src/new.rs"),
        "pub fn changed_text_marker_untracked() -> i32 { 3 }\n",
    )
    .unwrap();

    let (code, out, err) = run(
        &["search-code", "--changed", "changed_text_marker"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "search-code --changed text should not require an index; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "search-code --changed text should be a clean grep-like query; stderr={err:?}"
    );
    assert!(
        out.contains("src/lib.rs:1"),
        "modified tracked file must be searched; got: {out:?}"
    );
    assert!(
        out.contains("src/new.rs:1"),
        "untracked file must be searched; got: {out:?}"
    );
    assert!(
        !out.contains("clean_committed_marker"),
        "clean committed files must not be searched by --changed; got: {out:?}"
    );
}

#[test]
fn search_code_changed_json_reports_live_scope_and_exact_counts() {
    let (repo, store) = make_real_git_repo("search-code-changed-json");
    std::fs::write(
        repo.join("src/lib.rs"),
        "pub fn changed_json_marker() -> i32 { 2 }\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("src/new.rs"),
        "pub fn changed_json_marker_untracked() -> i32 { 3 }\n",
    )
    .unwrap();

    let (code, out, err) = run(
        &["search-code", "--changed", "--json", "changed_json_marker"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "search-code --changed --json should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "machine-readable changed search-code JSON should not require stderr parsing; stderr={err:?}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "search-code");
    assert_eq!(v["status"], "ok");
    assert_eq!(v["scope"], "changed");
    assert_eq!(v["backend"], "live_grep");
    assert_eq!(v["fresh"], true);
    assert_eq!(v["freshness"], serde_json::Value::Null);
    assert_eq!(v["changed_files_total"], 2);
    assert_eq!(v["total_exact"], 2);
    assert_eq!(v["shown"], 2);
    assert_eq!(v["omitted"], 0);
    assert_eq!(v["truncated"], false);
    assert_eq!(v["hits"].as_array().unwrap().len(), 2);
}

#[test]
fn search_code_staged_text_greps_git_index_blob_not_worktree() {
    let (repo, store) = make_real_git_repo("search-code-staged-text");
    std::fs::write(
        repo.join("src/lib.rs"),
        "pub fn staged_text_marker() -> i32 { 2 }\n",
    )
    .unwrap();
    git(&repo, &["add", "src/lib.rs"]);
    std::fs::write(
        repo.join("src/lib.rs"),
        "pub fn unstaged_after_add_marker() -> i32 { 3 }\n",
    )
    .unwrap();

    let (code, out, err) = run(
        &["search-code", "--staged", "staged_text_marker"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "search-code --staged text should search staged blobs without requiring an index; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "search-code --staged text should be a clean grep-like query; stderr={err:?}"
    );
    assert!(
        out.contains("src/lib.rs:1"),
        "staged blob must be searched; got: {out:?}"
    );
    assert!(
        out.contains("staged_text_marker"),
        "staged blob content must be visible; got: {out:?}"
    );

    let (code, out, err) = run(
        &["search-code", "--staged", "unstaged_after_add_marker"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "text no-match keeps existing search-code no-match behavior; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("(no matches)"),
        "--staged must not read unstaged worktree-only content; got: {out:?}"
    );
}

#[test]
fn search_code_staged_json_reports_git_blob_scope_and_exact_counts() {
    let (repo, store) = make_real_git_repo("search-code-staged-json");
    std::fs::write(
        repo.join("src/lib.rs"),
        "pub fn staged_json_marker() -> i32 { 2 }\n",
    )
    .unwrap();
    git(&repo, &["add", "src/lib.rs"]);
    std::fs::write(
        repo.join("src/lib.rs"),
        "pub fn unstaged_json_after_add_marker() -> i32 { 3 }\n",
    )
    .unwrap();

    let (code, out, err) = run(
        &["search-code", "--staged", "--json", "staged_json_marker"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "search-code --staged --json should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "machine-readable staged search-code JSON should not require stderr parsing; stderr={err:?}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "search-code");
    assert_eq!(v["status"], "ok");
    assert_eq!(v["scope"], "staged");
    assert_eq!(v["backend"], "git_blob_grep");
    assert_eq!(v["fresh"], true);
    assert_eq!(v["freshness"], serde_json::Value::Null);
    assert_eq!(v["staged_files_total"], 1);
    assert_eq!(v["total_exact"], 1);
    assert_eq!(v["shown"], 1);
    assert_eq!(v["omitted"], 0);
    assert_eq!(v["truncated"], false);
    assert_eq!(v["hits"].as_array().unwrap().len(), 1);
    assert!(
        v["hits"][0]["snippet"]
            .as_str()
            .unwrap_or("")
            .contains("staged_json_marker"),
        "JSON hit must come from staged blob, got {v:?}"
    );

    let (code, out, err) = run(
        &[
            "search-code",
            "--staged",
            "--json",
            "unstaged_json_after_add_marker",
        ],
        &repo,
        &store,
    );
    assert_eq!(
        code, 1,
        "staged JSON no-match should use the existing JSON no-match code; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["status"], "no_matches");
    assert_eq!(v["total_exact"], 0);
    assert!(err.is_empty());
}

#[test]
fn search_code_since_text_and_json_live_grep_rev_diff_without_index() {
    let (repo, store) = make_real_git_repo("search-code-since");
    std::fs::write(
        repo.join("src/lib.rs"),
        "pub fn since_diff_marker() -> i32 { 4 }\n",
    )
    .unwrap();
    git(&repo, &["add", "src/lib.rs"]);
    git(&repo, &["commit", "-m", "feature since"]);

    let (code, out, err) = run(
        &["search-code", "--since", "HEAD~1", "since_diff_marker"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "search-code --since text should search rev-diff files without an index; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "search-code --since text should be a clean grep-like query; stderr={err:?}"
    );
    assert!(
        out.contains("src/lib.rs:1"),
        "rev-diff file must be searched; got: {out:?}"
    );
    assert!(
        out.contains("since_diff_marker"),
        "rev-diff content must be visible; got: {out:?}"
    );

    let (code, out, err) = run(
        &[
            "search-code",
            "--since",
            "HEAD~1",
            "--json",
            "since_diff_marker",
        ],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "search-code --since --json should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "machine-readable since search-code JSON should not require stderr parsing; stderr={err:?}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "search-code");
    assert_eq!(v["status"], "ok");
    assert_eq!(v["scope"], "since");
    assert_eq!(v["backend"], "git_diff_live_grep");
    assert_eq!(v["fresh"], true);
    assert_eq!(v["freshness"], serde_json::Value::Null);
    assert_eq!(v["merge_base"], serde_json::Value::Null);
    assert_eq!(v["diff_files_total"], 1);
    assert_eq!(v["total_exact"], 1);
    assert_eq!(v["shown"], 1);
    assert_eq!(v["omitted"], 0);
    assert_eq!(v["truncated"], false);
    assert_eq!(v["hits"].as_array().unwrap().len(), 1);
    assert_eq!(v["diff_rev"].as_str().unwrap_or("").len(), 40);
}

#[test]
fn search_code_base_text_and_json_live_grep_merge_base_diff_without_index() {
    let (repo, store) = make_real_git_repo("search-code-base");
    git(&repo, &["branch", "basepoint"]);
    std::fs::write(
        repo.join("src/lib.rs"),
        "pub fn base_diff_marker() -> i32 { 5 }\n",
    )
    .unwrap();
    git(&repo, &["add", "src/lib.rs"]);
    git(&repo, &["commit", "-m", "feature base"]);

    let (code, out, err) = run(
        &["search-code", "--base", "basepoint", "base_diff_marker"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "search-code --base text should search merge-base diff files without an index; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "search-code --base text should be a clean grep-like query; stderr={err:?}"
    );
    assert!(
        out.contains("src/lib.rs:1"),
        "merge-base diff file must be searched; got: {out:?}"
    );
    assert!(
        out.contains("base_diff_marker"),
        "merge-base diff content must be visible; got: {out:?}"
    );

    let (code, out, err) = run(
        &[
            "search-code",
            "--base",
            "basepoint",
            "--json",
            "base_diff_marker",
        ],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "search-code --base --json should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "machine-readable base search-code JSON should not require stderr parsing; stderr={err:?}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "search-code");
    assert_eq!(v["status"], "ok");
    assert_eq!(v["scope"], "base");
    assert_eq!(v["backend"], "git_diff_live_grep");
    assert_eq!(v["fresh"], true);
    assert_eq!(v["freshness"], serde_json::Value::Null);
    assert_eq!(v["diff_files_total"], 1);
    assert_eq!(v["total_exact"], 1);
    assert_eq!(v["shown"], 1);
    assert_eq!(v["omitted"], 0);
    assert_eq!(v["truncated"], false);
    assert_eq!(v["hits"].as_array().unwrap().len(), 1);
    assert_eq!(v["diff_rev"].as_str().unwrap_or("").len(), 40);
    assert_eq!(v["merge_base"].as_str().unwrap_or("").len(), 40);
}

fn insert_default_model_vectors(store_dir: &Path, count: usize) {
    let db = find_graph_db(store_dir).expect("graph.db exists after index");
    let mut store = greppy_store::Store::open(&db).expect("open graph store");
    let generation = store
        .list_workspace_states()
        .expect("workspace state lookup")
        .into_iter()
        .next()
        .expect("workspace state present")
        .graph_generation;

    for i in 0..count {
        store
            .upsert_vector_embedding(&greppy_store::NewVectorEmbedding {
                project: "repo".into(),
                model_id: "google/embeddinggemma-300m".into(),
                prompt_version: greppy_embed_native::PROMPT_VERSION.into(),
                task: greppy_search::EMBEDDINGGEMMA_CODE_RETRIEVAL_PROFILE.into(),
                node_id: None,
                chunk_idx: 0,
                qualified_name: format!("repo.vector_guard_{i}"),
                file_path: "lib.rs".into(),
                start_line: 1,
                end_line: 1,
                content_sha256: format!("{:064x}", i + 1),
                graph_generation: generation,
                vector: vec![1.0, 0.0],
            })
            .expect("insert vector embedding");
    }
}

#[test]
fn semantic_vectors_guard_skips_before_model_load_when_over_budget() {
    let (repo, store_dir) = make_repo("semantic-vector-guard", "vector_guard_marker");
    let (code, out, err) = run(&["index", "."], &repo, &store_dir);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );
    insert_default_model_vectors(&store_dir, 2);

    let (code, out, err) = run_with_env(
        &[
            "semantic-search",
            "--vectors",
            "--json",
            "--embedding-gguf",
            "/missing/embeddinggemma.gguf",
            "--embedding-tokenizer",
            "/missing/tokenizer.json",
            "find vector guard marker",
        ],
        &repo,
        &store_dir,
        &[("GREPPY_VECTOR_EXACT_CANDIDATE_LIMIT", "1")],
    );
    assert_eq!(
        code, 1,
        "over-budget vector search should return no-hit code without trying to load the missing model; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "guard path should be a controlled semantic result, not a model-load error; stderr={err:?}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["status"], "skipped_exact_scan_candidate_limit");
    assert_eq!(v["backend"], "exact_cosine");
    assert_eq!(v["project"], "repo");
    assert_eq!(v["provider_complete"], false);
    assert!(
        v["incomplete_provider_count"].as_u64().unwrap_or(0) >= 1,
        "vector semantic JSON must expose provider incompleteness: {v:?}"
    );
    assert!(
        v["incomplete_providers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["language"] == "rust"),
        "rust provider incompleteness must be visible: {v:?}"
    );
    assert_eq!(v["candidate_limit"], 1);
    assert_eq!(v["total_exact"], 2);
    assert_eq!(v["shown"], 0);
    assert_eq!(v["truncated"], true);
    assert_eq!(v["hits"].as_array().unwrap().len(), 0);
}

#[test]
fn semantic_vectors_stale_index_skips_before_model_load() {
    let (repo, store_dir) = make_repo("semantic-vector-stale", "vector_stale_marker");
    let (code, out, err) = run(&["index", "."], &repo, &store_dir);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );
    insert_default_model_vectors(&store_dir, 1);
    std::fs::write(
        repo.join("lib.rs"),
        "pub fn vector_stale_marker_changed() -> i32 { 8 }\n",
    )
    .unwrap();

    let (code, out, err) = run(
        &[
            "semantic-search",
            "--vectors",
            "--json",
            "--embedding-gguf",
            "/missing/embeddinggemma.gguf",
            "--embedding-tokenizer",
            "/missing/tokenizer.json",
            "find vector stale marker",
        ],
        &repo,
        &store_dir,
    );
    assert_eq!(
        code, 1,
        "stale vector search should return no-hit code without trying to load the missing model; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "stale guard path should be controlled JSON, not a model-load error; stderr={err:?}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["status"], "skipped_stale_index");
    assert_eq!(v["fresh"], false);
    assert_eq!(v["freshness"]["state"], "stale");
    assert_eq!(v["total_exact"], 1);
    assert_eq!(v["shown"], 0);
    assert_eq!(v["hits"].as_array().unwrap().len(), 0);
}

/// D2 fail-open: stale algorithmic semantic serves the OLD indexed
/// hits, labeled `fresh: false` + stderr warning, instead of refusing
/// (kill switch pins the labeled-stale path; the default policy would
/// auto-heal this one-file drift and answer fresh).
#[test]
fn semantic_algorithmic_stale_index_serves_labeled_hits() {
    let (repo, store_dir) = make_repo("semantic-stale", "semantic_stale_marker");
    let (code, out, err) = run(&["index", "."], &repo, &store_dir);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );
    std::fs::write(
        repo.join("lib.rs"),
        "pub fn semantic_stale_marker_changed() -> i32 { 9 }\n",
    )
    .unwrap();

    let (code, out, err) = run_with_env(
        &["semantic-search", "--json", "semantic_stale_marker"],
        &repo,
        &store_dir,
        &[("GREPPY_AUTO_REINDEX", "0")],
    );
    assert_eq!(
        code, 0,
        "labeled-stale semantic must serve the indexed hits; stderr={err}\nstdout={out}"
    );
    assert!(
        err.contains("index may be stale") && err.contains("run 'grep index'"),
        "labeled-stale semantic must warn on stderr; stderr={err:?}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["mode"], "algorithmic");
    assert_eq!(v["status"], "ok");
    assert_eq!(v["fresh"], false, "result must be labeled stale: {v:?}");
    assert_eq!(v["freshness"]["state"], "stale");
    assert_eq!(v["freshness"]["stale_file_count"], 1);
    assert!(
        !v["hits"].as_array().unwrap().is_empty(),
        "labeled-stale semantic must serve hits from the existing index: {v:?}"
    );
}

#[test]
fn diagnostics_json_exposes_provider_incompleteness() {
    let (repo, store) = make_repo("diag", "diagnostics_unique_marker");
    std::fs::write(repo.join("notes.txt"), "not indexed as code\n").unwrap();
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );

    let (code, out, err) = run(&["diagnostics", "--json"], &repo, &store);
    assert_eq!(
        code, 73,
        "diagnostics must be non-zero while providers are incomplete; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["schema_current"], true);
    assert_eq!(v["integrity_ok"], true);
    let providers = v["projects"][0]["provider_states"]
        .as_array()
        .expect("provider_states array");
    let rust = providers
        .iter()
        .find(|p| p["language"] == "rust")
        .expect("rust provider diagnostics");
    assert_eq!(rust["status"], "partial");
    assert!(
        rust["unsupported_edge_classes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|edge| edge == "tests"),
        "rust provider must expose missing edge classes: {rust:?}"
    );
    let skips = v["projects"][0]["index_skips"]
        .as_array()
        .expect("index_skips array");
    let txt = skips
        .iter()
        .find(|s| s["rel_path"] == "notes.txt")
        .expect("unsupported notes.txt skip metadata");
    assert_eq!(txt["reason"], "unsupported_language");
    assert_eq!(txt["language"], "file extension .txt");
    assert!(
        v["projects"][0]["skip_counts_by_reason"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["reason"] == "unsupported_language" && row["count"] == 1),
        "diagnostics must expose skip counts by reason: {v:?}"
    );
}

#[test]
fn doctor_json_reports_missing_index_as_structured_status() {
    let root = fresh_dir("doctor-no-index");
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    let store = root.join("store");

    let (code, out, err) = run(&["doctor", "--json"], &repo, &store);
    assert_eq!(
        code, 1,
        "doctor --json without an index should return status code 1; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "doctor --json should report missing index in JSON, not stderr; stderr={err:?}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "doctor");
    assert_eq!(v["status"], "no_index");
    assert_eq!(v["healthy"], false);
    assert_eq!(v["store_exists"], false);
    assert_eq!(v["project"], "repo");
    assert_eq!(v["project_present"], false);
    assert_eq!(v["fresh"], false);
}

#[test]
fn index_status_json_reports_freshness_stats_and_provider_health() {
    let (repo, store) = make_repo("index-status", "status_unique_marker");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );

    let (code, out, err) = run(&["index", "status", "--json"], &repo, &store);
    assert_eq!(
        code, 73,
        "index status --json should be non-zero while providers are incomplete; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "index-status");
    assert_eq!(v["status"], "unhealthy");
    assert_eq!(v["healthy"], false);
    assert_eq!(v["store_exists"], true);
    assert_eq!(v["project"], "repo");
    assert_eq!(v["project_present"], true);
    assert_eq!(v["fresh"], true);
    assert_eq!(v["schema_current"], true);
    assert_eq!(v["integrity_ok"], true);
    assert!(v["graph_generation"].as_u64().unwrap_or(0) >= 1);
    assert!(v["stats"]["nodes"].as_u64().unwrap_or(0) >= 1);
    assert!(v["incomplete_provider_count"].as_u64().unwrap_or(0) >= 1);
}

#[test]
fn index_status_json_exposes_dirty_overlay_breakdown() {
    let (repo, store) = make_real_git_repo("dirty-overlay-status");
    std::fs::write(repo.join(".gitignore"), "ignored.log\n").unwrap();
    std::fs::write(
        repo.join("src/delete_me.rs"),
        "pub fn dirty_delete_marker() -> i32 { 1 }\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("src/rename_me.rs"),
        "pub fn dirty_rename_marker() -> i32 { 2 }\n",
    )
    .unwrap();
    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-m", "dirty overlay fixtures"]);

    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed before dirty overlay; stderr={err}\nstdout={out}"
    );

    std::fs::write(
        repo.join("src/lib.rs"),
        "pub fn dirty_modified_marker() -> i32 { 3 }\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("src/staged.rs"),
        "pub fn dirty_staged_marker() -> i32 { 4 }\n",
    )
    .unwrap();
    git(&repo, &["add", "src/staged.rs"]);
    std::fs::remove_file(repo.join("src/delete_me.rs")).unwrap();
    git(&repo, &["mv", "src/rename_me.rs", "src/renamed.rs"]);
    std::fs::write(
        repo.join("src/untracked.rs"),
        "pub fn dirty_untracked_marker() -> i32 { 5 }\n",
    )
    .unwrap();
    std::fs::write(repo.join("ignored.log"), "generated\n").unwrap();

    let (code, out, err) = run(&["index", "status", "--json"], &repo, &store);
    assert_eq!(
        code, 73,
        "dirty index status should be unhealthy; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "index status --json should not require stderr parsing; stderr={err:?}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "index-status");
    assert_eq!(v["fresh"], false);
    let overlay = &v["dirty_overlay"];
    assert_eq!(overlay["git_available"], true);
    assert_eq!(overlay["clean"], false);
    assert!(overlay["total"].as_u64().unwrap_or(0) >= 6, "{overlay:?}");
    assert!(
        overlay["staged_count"].as_u64().unwrap_or(0) >= 2,
        "{overlay:?}"
    );
    assert!(
        overlay["unstaged_count"].as_u64().unwrap_or(0) >= 2,
        "{overlay:?}"
    );
    assert_eq!(overlay["untracked_count"], 1);
    assert_eq!(overlay["ignored_count"], 1);
    assert!(
        overlay["deleted_count"].as_u64().unwrap_or(0) >= 1,
        "{overlay:?}"
    );
    assert!(
        overlay["renamed_count"].as_u64().unwrap_or(0) >= 1,
        "{overlay:?}"
    );
    let files = overlay["files"].as_array().expect("dirty overlay files");
    assert!(
        files
            .iter()
            .any(|f| f["path"] == "src/staged.rs" && f["staged"] == true),
        "staged file must be represented: {overlay:?}"
    );
    assert!(
        files
            .iter()
            .any(|f| f["path"] == "src/lib.rs" && f["unstaged"] == true),
        "unstaged modified file must be represented: {overlay:?}"
    );
    assert!(
        files
            .iter()
            .any(|f| f["path"] == "src/untracked.rs" && f["untracked"] == true),
        "untracked file must be represented: {overlay:?}"
    );
    assert!(
        files
            .iter()
            .any(|f| f["path"] == "ignored.log" && f["ignored"] == true),
        "ignored file must be represented: {overlay:?}"
    );
    assert!(
        files.iter().any(|f| f["deleted"] == true),
        "deleted file must be represented: {overlay:?}"
    );
    assert!(
        files
            .iter()
            .any(|f| f["path"] == "src/renamed.rs" && f["renamed"] == true),
        "renamed file must be represented: {overlay:?}"
    );
}

#[test]
fn r3_large_repo_file_limit_cli_reports_and_persists_truncation() {
    let root = fresh_dir("r3-large-limit");
    let repo = root.join("repo");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::create_dir_all(repo.join("src")).unwrap();
    for i in 0..5 {
        std::fs::write(
            repo.join("src").join(format!("f{i}.rs")),
            format!("pub fn large_limit_marker_{i}() -> i32 {{ {i} }}\n"),
        )
        .unwrap();
    }
    let store = root.join("store");

    let (code, out, err) = run_with_env(
        &["index", "."],
        &repo,
        &store,
        &[("GREPPY_MAX_FILES", "2")],
    );
    assert_eq!(
        code, 0,
        "file-limited clean index should still succeed; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("3 file-limit"),
        "CLI report must expose file-limit truncation count; stdout={out}"
    );

    let (code, out, err) = run(&["diagnostics", "--json"], &repo, &store);
    assert_eq!(
        code, 73,
        "diagnostics should be non-zero for partial providers, not hidden truncation; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    let skips = v["projects"][0]["index_skips"]
        .as_array()
        .expect("index_skips array");
    assert_eq!(skips.len(), 3, "three skipped files must be persisted");
    assert!(skips.iter().all(|s| s["reason"] == "file_limit"));
    assert!(
        v["projects"][0]["skip_counts_by_reason"]
            .as_array()
            .unwrap()
            .iter()
            .any(|row| row["reason"] == "file_limit" && row["count"] == 3),
        "diagnostics must expose file_limit count: {v:?}"
    );

    for skip in skips {
        let rel = skip["rel_path"].as_str().expect("rel_path");
        let n = rel
            .strip_prefix("src/f")
            .and_then(|s| s.strip_suffix(".rs"))
            .expect("fixture path shape");
        let marker = format!("large_limit_marker_{n}");
        // Pin the auto-reindex off: this query runs WITHOUT the
        // GREPPY_MAX_FILES limit, so the D2 inline heal would
        // (correctly) index the skipped files and surface the marker.
        // Here we assert the truncated graph itself never leaks the
        // skipped symbols.
        let (_code, out, _err) = run_with_env(
            &["search-symbols", &marker],
            &repo,
            &store,
            &[("GREPPY_AUTO_REINDEX", "0")],
        );
        assert!(
            !out.contains(&marker),
            "file-limit skipped symbol {marker} must not leak from stale graph rows; got {out:?}"
        );
    }
}

#[test]
fn discover_scope_env_controls_index_and_query_freshness() {
    let (repo, store) = make_real_git_repo("discover-scope-env");
    std::fs::create_dir_all(repo.join("tests")).unwrap();
    std::fs::write(
        repo.join("tests/integration.rs"),
        "pub fn outside_scope_marker() -> i32 { 9 }\n",
    )
    .unwrap();
    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-m", "add outside scope"]);

    let scope_env = [("GREPPY_DISCOVER_INCLUDE", "src/*.rs")];
    let (code, out, err) = run_with_env(&["index", "."], &repo, &store, &scope_env);
    assert_eq!(
        code, 0,
        "scoped index should succeed; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("discover scope: v1;I8:src/*.rs"),
        "index output must expose non-default discover scope; stdout={out}"
    );

    let (code, out, err) = run_with_env(
        &["search-symbols", "clean_committed_marker", "--json"],
        &repo,
        &store,
        &scope_env,
    );
    assert_eq!(
        code, 0,
        "matching scoped query should be fresh; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["status"], "ok");
    assert_eq!(v["fresh"], true);
    assert_eq!(v["freshness"]["discover_scope"], "v1;I8:src/*.rs");
    assert!(
        v["hits"]
            .as_array()
            .unwrap()
            .iter()
            .any(|h| h["name"] == "clean_committed_marker"),
        "scoped query must return the indexed symbol: {v:?}"
    );

    let (code, out, err) = run(
        &["search-symbols", "clean_committed_marker", "--json"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 1,
        "default query must reject a scoped index instead of emitting stale hits; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["status"], "skipped_stale_index");
    assert_eq!(v["fresh"], false);
    assert_eq!(v["freshness"]["discover_scope"], "default");
    assert!(
        v["freshness"]["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r.as_str().unwrap_or("").contains("indexer version/scope")),
        "default query must report scope mismatch: {v:?}"
    );
    assert_eq!(v["hits"].as_array().unwrap().len(), 0);
}

// ---------------------------------------------------------------------------
// RV-006 — explicit global `--root` targets the same store from anywhere.
// ---------------------------------------------------------------------------

#[test]
fn global_root_flag_resolves_same_store_from_outside() {
    let (repo, store) = make_repo("caseroot", "beta_unique_marker");

    // Index using an explicit --root, run from an unrelated cwd.
    let outside = fresh_dir("caseroot-outside");
    let repo_s = repo.to_str().unwrap();
    let (code, out, err) = run(&["--root", repo_s, "index", repo_s], &outside, &store);
    assert_eq!(code, 0, "index --root should succeed; stderr={err}\n{out}");

    // search-code with `--root` after the subcommand (global flag) from
    // the same unrelated cwd must hit the same store.
    let (code, out, err) = run(
        &["search-code", "--root", repo_s, "beta_unique_marker"],
        &outside,
        &store,
    );
    assert_eq!(code, 0, "search-code --root should exit 0; stderr={err}");
    assert!(
        out.contains("beta_unique_marker"),
        "global --root must target the indexed store (RV-006); got: {out:?}"
    );

    // And `--root` before the subcommand must work identically.
    let (code, out, _err) = run(
        &["--root", repo_s, "search-code", "beta_unique_marker"],
        &outside,
        &store,
    );
    assert_eq!(code, 0, "global --root before subcommand must work");
    assert!(out.contains("beta_unique_marker"), "got: {out:?}");
}

// ---------------------------------------------------------------------------
// RV-007 — store dir is 0700 and graph.db is 0600 (not world-readable).
// ---------------------------------------------------------------------------

#[cfg(unix)]
#[test]
fn store_dir_700_and_db_600() {
    let (repo, store) = make_repo("caseperm", "gamma_unique_marker");
    let (code, _out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index should succeed; stderr={err}");

    let db = find_graph_db(&store).expect("graph.db must exist after index");
    assert_eq!(
        mode_of(&db),
        0o600,
        "graph.db must be mode 0600, not world-readable (RV-007)"
    );

    // The workspace-hash directory that holds the db must be 0700.
    let db_dir = db.parent().unwrap();
    assert_eq!(
        mode_of(db_dir),
        0o700,
        "store hash dir must be mode 0700 (RV-007)"
    );
}

#[test]
fn r3_atomic_snapshot_second_success_keeps_previous_known_good_backup() {
    let (repo, store) = make_repo("r3backup", "old_atomic_marker");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "first index should succeed; stderr={err}\n{out}");

    let db = find_graph_db(&store).expect("graph.db must exist after first index");
    let backup = backup_path_for_db(&db);
    assert!(
        !backup.exists(),
        "first publish has no previous snapshot to keep"
    );

    std::fs::write(
        repo.join("lib.rs"),
        "pub fn new_atomic_marker() -> i32 { 9 }\n",
    )
    .unwrap();
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "second index should succeed; stderr={err}\n{out}");
    assert!(
        backup.exists(),
        "second publish must keep previous known-good snapshot at {}",
        backup.display()
    );
    assert!(
        std::fs::metadata(&backup).unwrap().len() > 0,
        "backup snapshot must not be empty"
    );

    let (code, out, err) = run(&["search-symbols", "new_atomic_marker"], &repo, &store);
    assert_eq!(code, 0, "new active snapshot should query; stderr={err}");
    assert!(
        out.contains("new_atomic_marker"),
        "active index must be the second snapshot; got {out:?}"
    );
    let (_code, out, _err) = run(&["search-symbols", "old_atomic_marker"], &repo, &store);
    assert!(
        !out.contains("old_atomic_marker"),
        "old symbol must not leak from active snapshot after publish; got {out:?}"
    );
}

#[test]
fn r3_cli_atomic_snapshot_uses_incremental_seed_from_active_index() {
    let (repo, store) = make_repo("r3-incremental-cli", "old_incremental_marker");
    std::fs::write(
        repo.join("helper.rs"),
        "pub fn untouched_incremental_helper() -> i32 { 1 }\n",
    )
    .unwrap();

    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "first index should succeed; stderr={err}\n{out}");
    assert!(
        out.contains("indexed 2 files"),
        "first run must index both supported files; stdout={out}"
    );

    std::fs::write(
        repo.join("lib.rs"),
        "pub fn new_incremental_marker() -> i32 { 9 }\n",
    )
    .unwrap();

    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "second index should succeed through seeded atomic temp snapshot; stderr={err}\n{out}"
    );
    assert!(
        out.contains("indexed 1 files"),
        "seeded production snapshot must take the incremental path and only re-index the changed file; stdout={out}"
    );

    let (code, out, err) = run(&["search-symbols", "new_incremental_marker"], &repo, &store);
    assert_eq!(
        code, 0,
        "incremental active snapshot should query new marker; stderr={err}"
    );
    assert!(
        out.contains("new_incremental_marker"),
        "new symbol must be visible after seeded incremental publish; got {out:?}"
    );
    let (code, out, err) = run(
        &["search-symbols", "untouched_incremental_helper"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "unchanged file's symbol must survive seeded incremental publish; stderr={err}"
    );
    assert!(
        out.contains("untouched_incremental_helper"),
        "unchanged file's graph rows must be preserved by incremental temp snapshot; got {out:?}"
    );
    let (_code, out, _err) = run(&["search-symbols", "old_incremental_marker"], &repo, &store);
    assert!(
        !out.contains("old_incremental_marker"),
        "changed file's old symbol must not leak after incremental publish; got {out:?}"
    );
}

#[test]
fn r3_failed_snapshot_does_not_replace_active_index() {
    let (repo, store) = make_repo("r3fail", "old_failure_marker");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "first index should succeed; stderr={err}\n{out}");
    let db = find_graph_db(&store).expect("graph.db must exist after first index");
    let active_before = std::fs::read(&db).unwrap();

    std::fs::write(
        repo.join("lib.rs"),
        "pub fn new_failure_marker() -> i32 { 9 }\n",
    )
    .unwrap();

    let (code, out, err) = run_with_env(
        &["index", "."],
        &repo,
        &store,
        &[(
            "GREPPY_TEST_INDEX_FAILPOINT",
            "error-after-temp-before-publish",
        )],
    );
    assert_eq!(
        code, 73,
        "test failpoint after temp build must fail before publish; stdout={out} stderr={err}"
    );
    assert_eq!(
        std::fs::read(&db).unwrap(),
        active_before,
        "failed temp index must leave previous active graph.db bytes unchanged"
    );

    // D2 fail-open (auto-reindex pinned off so the PRESERVED active
    // graph is what answers): the stale-but-intact active index serves
    // its rows, honestly labeled via stderr.
    let (code, out, err) = run_with_env(
        &["search-symbols", "old_failure_marker"],
        &repo,
        &store,
        &[("GREPPY_AUTO_REINDEX", "0")],
    );
    assert_eq!(
        code, 0,
        "preserved active index must serve labeled results; stderr={err}\nstdout={out}"
    );
    assert!(
        err.contains("index may be stale"),
        "labeled-stale serving must warn on stderr; stderr={err:?}"
    );
    assert!(
        out.contains("old_failure_marker"),
        "preserved active graph rows must be served (labeled); got {out:?}"
    );

    // The failed temp graph must never become visible: its new symbol
    // is absent from the preserved active index.
    let (_code, out, _err) = run_with_env(
        &["search-symbols", "new_failure_marker"],
        &repo,
        &store,
        &[("GREPPY_AUTO_REINDEX", "0")],
    );
    assert!(
        !out.contains("new_failure_marker"),
        "failed publish must not expose symbols from the failed temp graph; got {out:?}"
    );
}

#[test]
fn r3_corrupt_active_snapshot_is_quarantined_and_replaced() {
    let (repo, store) = make_repo("r3corrupt", "old_corrupt_marker");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "first index should succeed; stderr={err}\n{out}");

    let db = find_graph_db(&store).expect("graph.db must exist after first index");
    std::fs::write(&db, b"not a sqlite database").unwrap();
    std::fs::write(
        repo.join("lib.rs"),
        "pub fn new_corrupt_marker() -> i32 { 11 }\n",
    )
    .unwrap();

    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "valid new snapshot should replace corrupt active DB; stdout={out} stderr={err}"
    );
    assert!(
        err.contains("quarantined"),
        "corrupt active DB should be reported as quarantined; stderr={err}"
    );
    let corrupt = corrupt_snapshot_for_db(&db).expect("corrupt active DB must be quarantined");
    assert_eq!(std::fs::read(&corrupt).unwrap(), b"not a sqlite database");

    let (code, out, err) = run(&["search-symbols", "new_corrupt_marker"], &repo, &store);
    assert_eq!(
        code, 0,
        "replacement active snapshot should query; stderr={err}"
    );
    assert!(
        out.contains("new_corrupt_marker"),
        "new symbol must be visible after corrupt-active recovery; got {out:?}"
    );

    let (code, out, err) = run(&["diagnostics", "--json"], &repo, &store);
    assert_eq!(
        code, 73,
        "diagnostics should still report provider incompleteness, not store corruption; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["integrity_ok"], true);
}

#[cfg(unix)]
#[test]
fn r3_killed_index_before_publish_preserves_active_and_recovers() {
    let (repo, store) = make_repo("r3-kill-before-publish", "old_kill_marker");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "first index should succeed; stderr={err}\n{out}");

    let db = find_graph_db(&store).expect("graph.db must exist after first index");
    let lock_path = db.with_file_name(format!(
        "{}.lock",
        db.file_name().unwrap().to_str().unwrap()
    ));
    let active_before = std::fs::read(&db).unwrap();
    assert!(
        next_snapshot_paths_for_db(&db).is_empty(),
        "clean store should not start with temp next snapshots"
    );

    std::fs::write(
        repo.join("lib.rs"),
        "pub fn new_kill_marker() -> i32 { 13 }\n",
    )
    .unwrap();
    let ready = store.join("failpoint-ready");
    let mut child = Command::new(bin())
        .args(["index", "."])
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", &store)
        .env("GREPPY_TEST_INDEX_FAILPOINT", "after-temp-before-publish")
        .env("GREPPY_TEST_INDEX_FAILPOINT_READY", &ready)
        .env("GREPPY_TEST_INDEX_FAILPOINT_HOLD_MS", "120000")
        .env_remove("GREPPY_DISCOVER_INCLUDE")
        .env_remove("GREPPY_DISCOVER_EXCLUDE")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn failpoint greppy index");

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    while !ready.exists() {
        if let Some(status) = child.try_wait().expect("poll failpoint child") {
            panic!("failpoint child exited before ready marker: {status}");
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("timeout waiting for failpoint ready marker");
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    assert_eq!(
        std::fs::read(&db).unwrap(),
        active_before,
        "active graph.db must remain unchanged while temp snapshot is paused before publish"
    );
    assert!(
        lock_path.exists(),
        "paused indexer must hold the writer lock before it is killed"
    );
    let temp_paths = next_snapshot_paths_for_db(&db);
    assert!(
        !temp_paths.is_empty(),
        "paused indexer must leave a temp snapshot to simulate crash cleanup; db={}",
        db.display()
    );

    child.kill().expect("kill failpoint child");
    let killed = child.wait().expect("wait for killed failpoint child");
    assert!(
        !killed.success(),
        "killed failpoint child must not report success"
    );
    assert_eq!(
        std::fs::read(&db).unwrap(),
        active_before,
        "killing before publish must preserve the previous active graph.db bytes"
    );
    assert!(
        lock_path.exists(),
        "SIGKILL simulation should leave a stale lock for the next indexer to recover"
    );
    assert!(
        !next_snapshot_paths_for_db(&db).is_empty(),
        "SIGKILL simulation should leave stale graph.db.next.* files before recovery"
    );

    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "next index should take over the dead lock, clean stale temp snapshots and publish; stdout={out} stderr={err}"
    );
    assert!(
        !lock_path.exists(),
        "successful recovery index must remove the stale/taken-over lock"
    );
    assert!(
        next_snapshot_paths_for_db(&db).is_empty(),
        "successful recovery index must remove stale graph.db.next.* snapshots"
    );

    let (code, out, err) = run(&["search-symbols", "new_kill_marker"], &repo, &store);
    assert_eq!(
        code, 0,
        "recovered active snapshot should query new marker; stderr={err}"
    );
    assert!(
        out.contains("new_kill_marker"),
        "new symbol must be visible after recovery index; got {out:?}"
    );
    let (_code, out, _err) = run(&["search-symbols", "old_kill_marker"], &repo, &store);
    assert!(
        !out.contains("old_kill_marker"),
        "old symbol must not leak after recovery publish; got {out:?}"
    );
}

// ---------------------------------------------------------------------------
// RV-003 — a pre-held (live) lock makes a second index exit 75 without
// running the indexer / writing.
// ---------------------------------------------------------------------------

#[test]
fn held_lock_makes_second_index_exit_75_without_writing() {
    let (repo, store) = make_repo("caselock", "delta_unique_marker");

    // First index establishes the store and its directory.
    let (code, _out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "first index should succeed; stderr={err}");

    let db = find_graph_db(&store).expect("graph.db must exist");
    let lock_path = db.with_file_name(format!(
        "{}.lock",
        db.file_name().unwrap().to_str().unwrap()
    ));

    // Forge a *live* lock: our own (running) PID and a current
    // timestamp so the stale-recovery path cannot take it over.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let live_pid = std::process::id();
    std::fs::write(&lock_path, format!("{live_pid}\n{now}\n")).unwrap();
    assert!(lock_path.exists(), "lock file should be present");

    // Capture a fingerprint of the db before the contended index attempt
    // so we can prove it was NOT modified.
    let before = std::fs::metadata(&db).unwrap();
    let before_len = before.len();
    #[cfg(unix)]
    let before_mtime = {
        use std::os::unix::fs::MetadataExt;
        (before.mtime(), before.mtime_nsec())
    };

    // Second index must hit the held lock and bail with EX_TEMPFAIL (75).
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 75,
        "second index under a held lock must exit 75 (RV-003); stdout={out} stderr={err}"
    );
    assert!(
        !out.contains("indexed"),
        "indexer must NOT run while the lock is held (RV-003); stdout={out}"
    );
    assert!(
        err.contains("lock held"),
        "should report the held lock on stderr; stderr={err}"
    );

    // The db must be byte-identical: the indexer did not write.
    let after = std::fs::metadata(&db).unwrap();
    assert_eq!(after.len(), before_len, "db length must be unchanged");
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        assert_eq!(
            (after.mtime(), after.mtime_nsec()),
            before_mtime,
            "db mtime must be unchanged: the indexer must not have run (RV-003)"
        );
    }

    // The held lock must still belong to us — the contended run must NOT
    // have deleted the live holder's lock (the old `_ => None` bug let the
    // RAII drop remove it).
    assert!(
        lock_path.exists(),
        "the live holder's lock must survive a contended index attempt"
    );
    let body = std::fs::read_to_string(&lock_path).unwrap();
    assert!(
        body.starts_with(&live_pid.to_string()),
        "lock body must still be ours (pid {live_pid}); got: {body:?}"
    );

    let _ = std::fs::remove_file(&lock_path);
}

#[test]
fn r3_stale_lock_is_taken_over_by_cli_index() {
    let (repo, store) = make_repo("r3-stale-lock", "stale_lock_marker");

    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "first index should succeed; stderr={err}\n{out}");

    let db = find_graph_db(&store).expect("graph.db must exist");
    let lock_path = db.with_file_name(format!(
        "{}.lock",
        db.file_name().unwrap().to_str().unwrap()
    ));
    let stale_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
        .saturating_sub(10 * 60);
    std::fs::write(
        &lock_path,
        format!("{}\n{stale_secs}\n", std::process::id()),
    )
    .unwrap();

    std::fs::write(
        repo.join("lib.rs"),
        "pub fn stale_lock_marker_after_takeover() -> i32 { 12 }\n",
    )
    .unwrap();

    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "stale lock should be taken over, not reported as contention; stdout={out} stderr={err}"
    );
    assert!(
        out.contains("indexed"),
        "indexer must run after stale-lock takeover; stdout={out}"
    );
    assert!(
        !lock_path.exists(),
        "RAII lock cleanup must remove the stale/taken-over lock file"
    );

    let (code, out, err) = run(
        &["search-symbols", "stale_lock_marker_after_takeover"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "new active snapshot should query after stale-lock takeover; stderr={err}"
    );
    assert!(
        out.contains("stale_lock_marker_after_takeover"),
        "second index under stale-lock takeover must publish the new graph; got {out:?}"
    );
}
