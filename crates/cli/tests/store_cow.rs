//! End-to-end differential tests for the 0.3.2 Store-CoW path.
//!
//! The Base is built with the real CLI at a committed tree. The same dirty
//! worktree is then indexed once as Base+Delta and once as a full private
//! snapshot. User-visible query JSON must agree.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn git(repo: &Path, args: &[&str]) -> String {
    let output = Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .expect("spawn git");
    assert!(
        output.status.success(),
        "git {args:?} failed\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

fn run(
    repo: &Path,
    store: &Path,
    args: &[&str],
    overlay: Option<(&Path, &str)>,
) -> (i32, String, String) {
    run_with_env(repo, store, args, overlay, &[])
}

fn run_with_env(
    repo: &Path,
    store: &Path,
    args: &[&str],
    overlay: Option<(&Path, &str)>,
    extra_env: &[(&str, &str)],
) -> (i32, String, String) {
    let mut command = Command::new(bin());
    command
        .args(args)
        .current_dir(repo)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env_remove("GREPPY_AGENT_STORE_MODE")
        .env_remove("GREPPY_AGENT_BASE_STORE")
        .env_remove("GREPPY_AGENT_BASE_COMMIT")
        .env_remove("GREPPY_PROJECT_IDENTITY")
        .env_remove("GREPPY_DISCOVER_INCLUDE")
        .env_remove("GREPPY_DISCOVER_EXCLUDE");
    if let Some((base, commit)) = overlay {
        command
            .env("GREPPY_AGENT_STORE_MODE", "overlay")
            .env("GREPPY_AGENT_BASE_STORE", base)
            .env("GREPPY_AGENT_BASE_COMMIT", commit);
    }
    for (key, value) in extra_env {
        command.env(key, value);
    }
    let output = command.output().expect("spawn greppy");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn index(repo: &Path, store: &Path, overlay: Option<(&Path, &str)>) {
    let (code, stdout, stderr) = run(repo, store, &["index", "."], overlay);
    assert_eq!(code, 0, "index failed\nstdout={stdout}\nstderr={stderr}");
}

fn graph_db_under(root: &Path) -> PathBuf {
    fn visit(path: &Path, found: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            let child = entry.path();
            if child.is_dir() {
                visit(&child, found);
            } else if child.file_name().is_some_and(|name| name == "graph.db") {
                found.push(child);
            }
        }
    }

    let mut found = Vec::new();
    visit(root, &mut found);
    assert_eq!(
        found.len(),
        1,
        "expected one graph.db below {}, found {found:?}",
        root.display()
    );
    found.pop().unwrap()
}

fn query_json(
    repo: &Path,
    store: &Path,
    args: &[&str],
    overlay: Option<(&Path, &str)>,
) -> serde_json::Value {
    let mut value = query_json_raw(repo, store, args, overlay);
    normalize_ephemeral_fields(&mut value);
    value
}

fn query_json_raw(
    repo: &Path,
    store: &Path,
    args: &[&str],
    overlay: Option<(&Path, &str)>,
) -> serde_json::Value {
    let mut argv = args.to_vec();
    argv.push("--json");
    let (code, stdout, stderr) = run(repo, store, &argv, overlay);
    assert_eq!(
        code, 0,
        "query {args:?} failed\nstdout={stdout}\nstderr={stderr}"
    );
    serde_json::from_str(&stdout)
        .unwrap_or_else(|error| panic!("invalid JSON for {args:?}: {error}; stdout={stdout:?}"))
}

fn query_text(repo: &Path, store: &Path, args: &[&str], overlay: Option<(&Path, &str)>) -> String {
    let (code, stdout, stderr) = run(repo, store, args, overlay);
    assert_eq!(
        code, 0,
        "query {args:?} failed\nstdout={stdout}\nstderr={stderr}"
    );
    stdout
}

fn normalize_ephemeral_fields(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(object) => {
            // Expand handles identify a store-local evidence pack. Its
            // answer summary is deterministic, but the opaque id correctly
            // differs between independently built oracle stores.
            object.remove("expand_id");
            // Freshness timing is observational telemetry, not a query
            // result; two separately opened stores cannot have identical
            // wall-clock duration even when their visible state is equal.
            object.remove("elapsed_ms");
            if object.contains_key("edge_type")
                && object.contains_key("source_id")
                && object.contains_key("target_id")
            {
                object.remove("id");
                object.remove("source_id");
                object.remove("target_id");
            }
            if object.contains_key("payload_json") && object.contains_key("payload_text") {
                object.remove("id");
                object.remove("created_at");
                object.remove("expires_at");
                object.remove("graph_generation");
            }
            if let Some(expand) = object.get_mut("expand").and_then(|v| v.as_object_mut()) {
                expand.remove("id");
            }
            for child in object.values_mut() {
                normalize_ephemeral_fields(child);
            }
        }
        serde_json::Value::Array(values) => {
            for child in values {
                normalize_ephemeral_fields(child);
            }
        }
        _ => {}
    }
}

#[test]
fn committed_task_delta_status_uses_base_union_across_worktrees() {
    let scratch = tempfile::tempdir().unwrap();
    let base_repo = scratch.path().join("base-repo");
    let task_repo = scratch.path().join("task-repo");
    let base_store = scratch.path().join("base-store");
    let delta_store = scratch.path().join("delta-store");
    std::fs::create_dir_all(base_repo.join("src")).unwrap();
    std::fs::write(
        base_repo.join("src/changed.rs"),
        "pub fn changed_at_base() -> usize { 1 }\n",
    )
    .unwrap();
    std::fs::write(
        base_repo.join("src/other.rs"),
        "pub fn unchanged_other() -> usize { 2 }\n",
    )
    .unwrap();
    for index in 0..100 {
        std::fs::write(
            base_repo.join(format!("src/unchanged_{index:03}.rs")),
            format!("pub fn unchanged_{index:03}() -> usize {{ {index} }}\n"),
        )
        .unwrap();
    }
    git(&base_repo, &["init", "-q"]);
    git(&base_repo, &["config", "user.email", "cow@test.invalid"]);
    git(&base_repo, &["config", "user.name", "Store CoW"]);
    git(&base_repo, &["add", "."]);
    git(&base_repo, &["commit", "-q", "-m", "base"]);
    let base_commit = git(&base_repo, &["rev-parse", "HEAD"]);

    let (base_code, base_out, base_err) = run_with_env(
        &base_repo,
        &base_store,
        &["index", "."],
        None,
        &[("GREPPY_PROJECT_IDENTITY", "source")],
    );
    assert_eq!(
        base_code, 0,
        "base index failed\nstdout={base_out}\nstderr={base_err}"
    );
    let base_source_graph = graph_db_under(&base_store);
    let base_identity = greppy_store::BaseStoreIdentity {
        format_version: greppy_store::BASE_STORE_FORMAT_VERSION,
        canonical_repository_identity: "fixture:WALinuxAgent".into(),
        git_object_format: "sha1".into(),
        base_tree_oid: git(&base_repo, &["rev-parse", "HEAD^{tree}"]),
        store_schema_version: greppy_store::migrate::CURRENT_VERSION,
        indexer_version: greppy_core::INDEXER_VERSION_BASE.into(),
        parser_and_extractor_versions: "fixture-parser-v1".into(),
        summary_model_and_prompt_version: "fixture-summary-v1".into(),
        embedding_model: "fixture-embedding-v1".into(),
        embedding_prompt_version: "fixture-prompt-v1".into(),
        embedding_dimensions: 768,
        embedding_encoding: "f32+i8-v1".into(),
    };
    let published_root = scratch.path().join("published-base-root");
    let base_layout = greppy_store::BaseStoreLayout::new(&published_root, &base_identity).unwrap();
    let _builder = base_layout.acquire_builder(false).unwrap().unwrap();
    let summary_dir = scratch.path().join("published-base-summary");
    drop(greppy_store::SummaryCache::open(&summary_dir).unwrap());
    base_layout
        .publish_graph_with_summary(
            base_identity,
            &base_source_graph,
            &summary_dir.join(greppy_store::SUMMARY_CACHE_DB_FILE),
        )
        .unwrap();

    let task_repo_string = task_repo.to_string_lossy().into_owned();
    git(
        &base_repo,
        &["worktree", "add", "-q", task_repo_string.as_str()],
    );
    git(&task_repo, &["config", "user.email", "cow@test.invalid"]);
    git(&task_repo, &["config", "user.name", "Store CoW"]);
    std::fs::write(
        task_repo.join("src/changed.rs"),
        "pub fn changed_in_task_commit() -> usize { 3 }\n",
    )
    .unwrap();
    git(&task_repo, &["add", "src/changed.rs"]);
    git(&task_repo, &["commit", "-q", "-m", "task change"]);

    let overlay = Some((base_layout.graph.as_path(), base_commit.as_str()));
    let overlay_env = [
        ("GREPPY_PROJECT_IDENTITY", "Azure__WALinuxAgent"),
        ("GREPPY_AGENT_BASE_REUSED", "1"),
    ];
    let (delta_code, delta_out, delta_err) = run_with_env(
        &task_repo,
        &delta_store,
        &["index", "."],
        overlay,
        &overlay_env,
    );
    assert_eq!(
        delta_code, 0,
        "Delta index failed\nstdout={delta_out}\nstderr={delta_err}"
    );
    let (status_code, status_out, status_err) = run_with_env(
        &task_repo,
        &delta_store,
        &["index", "status", "--json"],
        overlay,
        &overlay_env,
    );
    assert_eq!(
        status_code, 0,
        "a committed one-file Delta over a complete Base must be healthy; \
         stderr={status_err}\nstdout={status_out}"
    );
    let status: serde_json::Value = serde_json::from_str(&status_out).unwrap();
    assert_eq!(status["healthy"], true, "{status:#}");
    assert_eq!(status["fresh"], true, "{status:#}");
    assert_eq!(status["git_tracked_files"], 102, "{status:#}");
    assert_eq!(status["stats"]["files"], 102, "{status:#}");
    assert_eq!(
        status["coverage_warning"],
        serde_json::Value::Null,
        "{status:#}"
    );
    assert_eq!(status["store_cow"]["mode"], "overlay", "{status:#}");
    assert_eq!(status["store_cow"]["base_cache_hit"], true, "{status:#}");
    assert_eq!(status["store_cow"]["base_complete"], true, "{status:#}");
    assert_eq!(status["store_cow"]["dirty_file_count"], 1, "{status:#}");
    assert_eq!(status["store_cow"]["deleted_file_count"], 0, "{status:#}");
}

#[test]
fn overlay_matches_full_private_index_for_dirty_deleted_renamed_and_untracked_files() {
    let scratch = tempfile::tempdir().unwrap();
    let repo = scratch.path().join("repo");
    let base_store = scratch.path().join("base-store");
    let delta_store = scratch.path().join("delta-store");
    let full_store = scratch.path().join("full-store");
    std::fs::create_dir_all(repo.join("src")).unwrap();

    std::fs::write(
        repo.join("src/clean.rs"),
        "pub fn clean_base() -> i32 { 11 }\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("src/changed.rs"),
        "use crate::clean::clean_base;\npub fn old_changed() -> i32 { clean_base() }\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("src/deleted.rs"),
        "pub fn deleted_symbol() -> i32 { 13 }\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("src/renamed_old.rs"),
        "pub fn renamed_symbol_old() -> i32 { 17 }\n",
    )
    .unwrap();
    git(&repo, &["init", "-q"]);
    git(&repo, &["config", "user.email", "cow@test.invalid"]);
    git(&repo, &["config", "user.name", "Store CoW"]);
    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-q", "-m", "base"]);
    let base_commit = git(&repo, &["rev-parse", "HEAD"]);

    index(&repo, &base_store, None);
    let base_source_graph = graph_db_under(&base_store);
    let published_root = scratch.path().join("published-base-root");
    let base_identity = greppy_store::BaseStoreIdentity {
        format_version: greppy_store::BASE_STORE_FORMAT_VERSION,
        canonical_repository_identity: format!("fixture:{}", repo.display()),
        git_object_format: "sha1".into(),
        base_tree_oid: git(&repo, &["rev-parse", "HEAD^{tree}"]),
        store_schema_version: greppy_store::migrate::CURRENT_VERSION,
        indexer_version: greppy_core::INDEXER_VERSION_BASE.into(),
        parser_and_extractor_versions: "fixture-parser-v1".into(),
        summary_model_and_prompt_version: "fixture-summary-v1".into(),
        embedding_model: "fixture-embedding-v1".into(),
        embedding_prompt_version: "fixture-prompt-v1".into(),
        embedding_dimensions: 768,
        embedding_encoding: "f32+i8-v1".into(),
    };
    let base_layout = greppy_store::BaseStoreLayout::new(&published_root, &base_identity).unwrap();
    let _builder = base_layout.acquire_builder(false).unwrap().unwrap();
    let summary_dir = scratch.path().join("published-base-summary");
    drop(greppy_store::SummaryCache::open(&summary_dir).unwrap());
    base_layout
        .publish_graph_with_summary(
            base_identity,
            &base_source_graph,
            &summary_dir.join(greppy_store::SUMMARY_CACHE_DB_FILE),
        )
        .unwrap();
    let base_graph = base_layout.graph.clone();

    std::fs::write(
        repo.join("src/changed.rs"),
        "use crate::clean::clean_base;\npub fn new_changed() -> i32 { clean_base() + 1 }\n",
    )
    .unwrap();
    std::fs::remove_file(repo.join("src/deleted.rs")).unwrap();
    std::fs::rename(
        repo.join("src/renamed_old.rs"),
        repo.join("src/renamed_new.rs"),
    )
    .unwrap();
    std::fs::write(
        repo.join("src/renamed_new.rs"),
        "pub fn renamed_symbol_new() -> i32 { 19 }\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("src/untracked.rs"),
        "pub fn untracked_symbol() -> i32 { 23 }\n",
    )
    .unwrap();

    let overlay = Some((base_graph.as_path(), base_commit.as_str()));
    index(&repo, &delta_store, overlay);
    let (diagnostics_code, diagnostics_out, diagnostics_err) =
        run(&repo, &delta_store, &["diagnostics", "--json"], overlay);
    assert!(
        matches!(diagnostics_code, 0 | 73),
        "overlay diagnostics must be queryable; stderr={diagnostics_err}"
    );
    let diagnostics: serde_json::Value = serde_json::from_str(&diagnostics_out).unwrap();
    assert_eq!(diagnostics["store_cow"]["mode"], "overlay");
    assert_eq!(diagnostics["store_cow"]["base_complete"], true);
    assert_eq!(diagnostics["store_cow"]["dirty_file_count"], 3);
    assert_eq!(diagnostics["store_cow"]["deleted_file_count"], 2);
    assert!(diagnostics["store_cow"]["base_identity"]
        .as_str()
        .is_some_and(|identity| identity.len() == 64));
    assert!(diagnostics["store_cow"]["delta_identity"]
        .as_str()
        .is_some_and(|identity| identity.len() == 64));

    let (status_code, status_out, status_err) =
        run(&repo, &delta_store, &["index", "status", "--json"], overlay);
    assert_eq!(
        status_code, 0,
        "a complete Base plus its indexed Delta must be healthy; \
         stderr={status_err}\nstdout={status_out}"
    );
    let status: serde_json::Value = serde_json::from_str(&status_out).unwrap();
    assert_eq!(status["healthy"], true, "{status:#}");
    assert_eq!(status["fresh"], true, "{status:#}");
    assert_eq!(
        status["coverage_warning"],
        serde_json::Value::Null,
        "{status:#}"
    );
    assert_eq!(status["git_tracked_files"], 4, "{status:#}");
    assert_eq!(status["stats"]["files"], 4, "{status:#}");
    assert_eq!(
        status["store_cow"]["base_cache_hit"],
        serde_json::Value::Null
    );
    assert_eq!(status["store_cow"]["base_complete"], true, "{status:#}");
    index(&repo, &full_store, None);

    let delta_graph_before_failure = graph_db_under(&delta_store);
    let delta_bytes_before_failure = std::fs::read(&delta_graph_before_failure).unwrap();
    let (failed_code, failed_out, failed_err) = run_with_env(
        &repo,
        &delta_store,
        &["index", "."],
        overlay,
        &[(
            "GREPPY_TEST_INDEX_FAILPOINT",
            "error-after-temp-before-publish",
        )],
    );
    assert_ne!(
        failed_code, 0,
        "pre-publication failpoint must fail; stdout={failed_out}\nstderr={failed_err}"
    );
    assert_eq!(
        std::fs::read(&delta_graph_before_failure).unwrap(),
        delta_bytes_before_failure,
        "failed Delta generation must leave the prior active snapshot byte-identical"
    );
    index(&repo, &delta_store, overlay);

    for symbol in [
        "clean_base",
        "new_changed",
        "renamed_symbol_new",
        "untracked_symbol",
        "old_changed",
        "deleted_symbol",
        "renamed_symbol_old",
    ] {
        let overlay_result = query_json(&repo, &delta_store, &["search-symbol", symbol], overlay);
        let full_result = query_json(&repo, &full_store, &["search-symbol", symbol], None);
        assert_eq!(
            overlay_result, full_result,
            "search-symbol parity failed for {symbol}"
        );
    }

    let query_matrix: Vec<Vec<&str>> = vec![
        vec!["who-calls", "clean_base"],
        vec!["callees", "new_changed"],
        vec!["brief", "new_changed"],
        vec!["brief", "clean_base", "new_changed"],
        vec!["read", "clean_base"],
        vec!["read", "clean_base", "new_changed"],
        vec!["search-graph", "--name", "clean_base"],
        vec![
            "trace",
            "--symbol",
            "new_changed",
            "--direction",
            "outgoing",
            "--depth",
            "3",
        ],
        vec!["impact", "new_changed", "--direction", "outgoing", "--all"],
        vec!["fan-in"],
        vec!["fan-out"],
        vec!["graph-locate", "src/changed.rs:2"],
        vec!["path", "--from", "new_changed", "--to", "clean_base"],
        vec!["search-pattern", "clean_base", "--fixed", "--all"],
        vec!["plus", "clean_base", "--k", "10"],
        vec!["context", "clean_base", "--k", "6"],
        vec!["where-am-i"],
    ];
    for args in &query_matrix {
        let overlay_result = query_json(&repo, &delta_store, args, overlay);
        let full_result = query_json(&repo, &full_store, args, None);
        assert_eq!(
            overlay_result, full_result,
            "query parity failed for {args:?}"
        );
    }
    for args in [
        &["stats"][..],
        &["read-smart", "clean_base", "new_changed", "--depth", "1"][..],
    ] {
        assert_eq!(
            query_text(&repo, &delta_store, args, overlay),
            query_text(&repo, &full_store, args, None),
            "text query parity failed for {args:?}"
        );
    }
    let overlay_brief = query_json_raw(&repo, &delta_store, &["brief", "new_changed"], overlay);
    let full_brief = query_json_raw(&repo, &full_store, &["brief", "new_changed"], None);
    let overlay_expand = overlay_brief["expand"]["id"]
        .as_str()
        .expect("overlay brief expand id");
    let full_expand = full_brief["expand"]["id"]
        .as_str()
        .expect("full brief expand id");
    assert_eq!(
        query_json(&repo, &delta_store, &["expand", overlay_expand], overlay),
        query_json(&repo, &full_store, &["expand", full_expand], None),
        "expand payload parity failed"
    );

    // Query opens may reuse the visibility pinned into the active Delta, but
    // automatic refresh must rescan Git live before publishing its next
    // generation. An edit made outside greppy therefore becomes visible on
    // the first structural query without an explicit `index` command.
    std::fs::write(
        repo.join("src/clean.rs"),
        "pub fn clean_after_external_edit() -> i32 { 29 }\n",
    )
    .unwrap();
    let auto_overlay = query_json(
        &repo,
        &delta_store,
        &["search-symbol", "clean_after_external_edit"],
        overlay,
    );
    let auto_full_store = scratch.path().join("auto-full-store");
    index(&repo, &auto_full_store, None);
    assert_eq!(
        auto_overlay,
        query_json(
            &repo,
            &auto_full_store,
            &["search-symbol", "clean_after_external_edit"],
            None,
        ),
        "cached visibility must not hide an external edit from auto-refresh"
    );

    // A complete revert must collapse the private graph back to an empty
    // Delta while the composed result becomes the original Base again.
    git(&repo, &["reset", "--hard", "-q", "HEAD"]);
    git(&repo, &["clean", "-fdq"]);
    let clean_full_store = scratch.path().join("clean-full-store");
    index(&repo, &delta_store, overlay);
    index(&repo, &clean_full_store, None);
    let (_, diagnostics_out, _) = run(&repo, &delta_store, &["doctor", "--json"], overlay);
    let diagnostics: serde_json::Value = serde_json::from_str(&diagnostics_out).unwrap();
    assert_eq!(diagnostics["store_cow"]["dirty_file_count"], 0);
    assert_eq!(diagnostics["store_cow"]["deleted_file_count"], 0);
    for symbol in [
        "clean_base",
        "old_changed",
        "deleted_symbol",
        "renamed_symbol_old",
        "new_changed",
        "renamed_symbol_new",
        "untracked_symbol",
    ] {
        assert_eq!(
            query_json(&repo, &delta_store, &["search-symbol", symbol], overlay),
            query_json(&repo, &clean_full_store, &["search-symbol", symbol], None,),
            "exact-revert parity failed for {symbol}"
        );
    }
    let delta_graph = graph_db_under(&delta_store);
    let delta =
        greppy_store::Store::open_with(&delta_graph, greppy_store::OpenOptions::read_only())
            .unwrap();
    for table in [
        "nodes",
        "raw_edges",
        "edges",
        "overlay_edges",
        "file_state",
        "file_content",
        "vector_embeddings",
    ] {
        let count: i64 = delta
            .conn()
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 0, "exact revert must empty Delta table {table}");
    }
}

fn timed_query(
    repo: &Path,
    store: &Path,
    args: &[&str],
    overlay: Option<(&Path, &str)>,
) -> Duration {
    let started = Instant::now();
    let (code, stdout, stderr) = run(repo, store, args, overlay);
    let elapsed = started.elapsed();
    assert_eq!(
        code, 0,
        "timed query {args:?} failed\nstdout={stdout}\nstderr={stderr}"
    );
    elapsed
}

fn percentile_micros(samples: &[Duration], percentile: usize) -> u128 {
    let mut values = samples.iter().map(Duration::as_micros).collect::<Vec<_>>();
    values.sort_unstable();
    let rank = (values.len() * percentile).div_ceil(100).saturating_sub(1);
    values[rank]
}

fn regression_percent(candidate: u128, baseline: u128) -> f64 {
    if baseline == 0 {
        return 0.0;
    }
    (candidate as f64 - baseline as f64) * 100.0 / baseline as f64
}

/// Manual 0.3.2 release gate. It is ignored in ordinary CI because timing
/// assertions require an otherwise idle release host. Run with the internal
/// `store-cow-release-perf` feature, one test thread, and `--nocapture` to retain
/// the versioned JSON evidence record without model-inference variance.
fn require_release_profile() {
    #[cfg(debug_assertions)]
    panic!("the 0.3.2 performance gate must run with cargo test --release");
}

#[test]
#[ignore = "0.3.2 release performance gate"]
fn store_cow_release_performance_gate() {
    const FILES: usize = 1_200;
    const DIRTY_FILES: usize = 6;
    const QUERY_WARMUP_ROUNDS: usize = 4;
    const QUERY_ROUNDS: usize = 40;

    require_release_profile();

    let scratch = tempfile::tempdir().unwrap();
    let repo = scratch.path().join("repo");
    let source = repo.join("src");
    let base_store = scratch.path().join("base-store");
    let clean_delta_store = scratch.path().join("clean-delta-store");
    let dirty_delta_store = scratch.path().join("dirty-delta-store");
    let full_store = scratch.path().join("full-store");
    std::fs::create_dir_all(&source).unwrap();
    for index in 0..FILES {
        let previous = index.saturating_sub(1);
        let body = if index == 0 {
            "pub fn cow_gate_000() -> usize { 0 }\n".to_string()
        } else {
            format!(
                "use crate::file_{previous:03}::cow_gate_{previous:03};\n\
                 pub fn cow_gate_{index:03}() -> usize {{ cow_gate_{previous:03}() + 1 }}\n"
            )
        };
        std::fs::write(source.join(format!("file_{index:03}.rs")), body).unwrap();
    }
    git(&repo, &["init", "-q"]);
    git(&repo, &["config", "user.email", "cow-perf@test.invalid"]);
    git(&repo, &["config", "user.name", "Store CoW Perf"]);
    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-q", "-m", "base"]);
    let base_commit = git(&repo, &["rev-parse", "HEAD"]);

    index(&repo, &base_store, None);
    let base_source_graph = graph_db_under(&base_store);
    let base_identity = greppy_store::BaseStoreIdentity {
        format_version: greppy_store::BASE_STORE_FORMAT_VERSION,
        canonical_repository_identity: format!("perf-fixture:{}", repo.display()),
        git_object_format: "sha1".into(),
        base_tree_oid: git(&repo, &["rev-parse", "HEAD^{tree}"]),
        store_schema_version: greppy_store::migrate::CURRENT_VERSION,
        indexer_version: greppy_core::INDEXER_VERSION_BASE.into(),
        parser_and_extractor_versions: "perf-parser-v1".into(),
        summary_model_and_prompt_version: "perf-summary-v1".into(),
        embedding_model: "perf-embedding-v1".into(),
        embedding_prompt_version: "perf-prompt-v1".into(),
        embedding_dimensions: 768,
        embedding_encoding: "f32+i8-v1".into(),
    };
    let published_root = scratch.path().join("published-base-root");
    let base_layout = greppy_store::BaseStoreLayout::new(&published_root, &base_identity).unwrap();
    let _builder = base_layout.acquire_builder(false).unwrap().unwrap();
    let summary_dir = scratch.path().join("published-base-summary");
    drop(greppy_store::SummaryCache::open(&summary_dir).unwrap());
    base_layout
        .publish_graph_with_summary(
            base_identity,
            &base_source_graph,
            &summary_dir.join(greppy_store::SUMMARY_CACHE_DB_FILE),
        )
        .unwrap();
    let base_graph = base_layout.graph.clone();
    let overlay = Some((base_graph.as_path(), base_commit.as_str()));

    index(&repo, &clean_delta_store, overlay);
    let query_args = ["search-symbol", "cow_gate_180", "--json"];
    for _ in 0..QUERY_WARMUP_ROUNDS {
        timed_query(&repo, &base_store, &query_args, None);
        timed_query(&repo, &clean_delta_store, &query_args, overlay);
    }
    let mut clean_single = Vec::new();
    let mut clean_overlay = Vec::new();
    for round in 0..QUERY_ROUNDS {
        if round % 2 == 0 {
            clean_single.push(timed_query(&repo, &base_store, &query_args, None));
            clean_overlay.push(timed_query(&repo, &clean_delta_store, &query_args, overlay));
        } else {
            clean_overlay.push(timed_query(&repo, &clean_delta_store, &query_args, overlay));
            clean_single.push(timed_query(&repo, &base_store, &query_args, None));
        }
    }

    for index in 0..DIRTY_FILES {
        let file_index = 180 + index;
        std::fs::write(
            source.join(format!("file_{file_index:03}.rs")),
            format!("pub fn cow_gate_{file_index:03}() -> usize {{ {file_index} + 7 }}\n"),
        )
        .unwrap();
    }
    let overlay_index_started = Instant::now();
    index(&repo, &dirty_delta_store, overlay);
    let overlay_index = overlay_index_started.elapsed();
    let full_index_started = Instant::now();
    index(&repo, &full_store, None);
    let full_index = full_index_started.elapsed();

    for _ in 0..QUERY_WARMUP_ROUNDS {
        timed_query(&repo, &full_store, &query_args, None);
        timed_query(&repo, &dirty_delta_store, &query_args, overlay);
    }
    let mut dirty_single = Vec::new();
    let mut dirty_overlay = Vec::new();
    for round in 0..QUERY_ROUNDS {
        if round % 2 == 0 {
            dirty_single.push(timed_query(&repo, &full_store, &query_args, None));
            dirty_overlay.push(timed_query(&repo, &dirty_delta_store, &query_args, overlay));
        } else {
            dirty_overlay.push(timed_query(&repo, &dirty_delta_store, &query_args, overlay));
            dirty_single.push(timed_query(&repo, &full_store, &query_args, None));
        }
    }

    let clean_baseline_median_us = percentile_micros(&clean_single, 50);
    let clean_overlay_median_us = percentile_micros(&clean_overlay, 50);
    let baseline_median_us = percentile_micros(&dirty_single, 50);
    let overlay_median_us = percentile_micros(&dirty_overlay, 50);
    let baseline_p95_us = percentile_micros(&dirty_single, 95);
    let overlay_p95_us = percentile_micros(&dirty_overlay, 95);
    let clean_median = regression_percent(clean_overlay_median_us, clean_baseline_median_us);
    let dirty_median = regression_percent(overlay_median_us, baseline_median_us);
    let dirty_p95 = regression_percent(overlay_p95_us, baseline_p95_us);
    let warm_base_improvement =
        (full_index.as_secs_f64() - overlay_index.as_secs_f64()) * 100.0 / full_index.as_secs_f64();
    let source_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let source_commit = git(source_root, &["rev-parse", "HEAD"]);
    let source_tracked_worktree_dirty = !git(
        source_root,
        &["status", "--porcelain", "--untracked-files=no"],
    )
    .is_empty();
    let feature_flags = [
        ("bash-smart", cfg!(feature = "bash-smart")),
        ("ci-test-assets", cfg!(feature = "ci-test-assets")),
        (
            "store-cow-release-perf",
            cfg!(feature = "store-cow-release-perf"),
        ),
        ("metal", cfg!(feature = "metal")),
        ("accelerate", cfg!(feature = "accelerate")),
        ("cuda", cfg!(feature = "cuda")),
        ("cpu-only", cfg!(feature = "cpu-only")),
    ]
    .into_iter()
    .filter_map(|(name, enabled)| enabled.then_some(name))
    .collect::<Vec<_>>();
    let evidence = serde_json::json!({
        "schema_version": "greppy.store-cow-performance.v1",
        "greppy_version": env!("CARGO_PKG_VERSION"),
        "binary": bin(),
        "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "source_commit": source_commit,
        "source_tracked_worktree_dirty": source_tracked_worktree_dirty,
        "feature_flags": feature_flags,
        "store_schema": greppy_store::migrate::CURRENT_VERSION,
        "indexer_version": greppy_core::INDEXER_VERSION_BASE,
        "fixture_files": FILES,
        "dirty_files": DIRTY_FILES,
        "query_warmup_rounds": QUERY_WARMUP_ROUNDS,
        "query_rounds": QUERY_ROUNDS,
        "baseline": "0.3.1-compatible single Store query/full private graph index",
        "warm_base": {
            "overlay_index_ms": overlay_index.as_secs_f64() * 1000.0,
            "full_private_index_ms": full_index.as_secs_f64() * 1000.0,
            "improvement_percent": warm_base_improvement,
            "gate_percent": 50.0,
        },
        "query_regression": {
            "baseline_median_us": baseline_median_us,
            "overlay_median_us": overlay_median_us,
            "baseline_p95_us": baseline_p95_us,
            "overlay_p95_us": overlay_p95_us,
            "median_percent": dirty_median,
            "p95_percent": dirty_p95,
            "median_gate_percent": 10.0,
            "p95_gate_percent": 20.0,
        },
        "warm_serial_clean_regression": {
            "baseline_median_us": clean_baseline_median_us,
            "overlay_median_us": clean_overlay_median_us,
            "median_percent": clean_median,
            "gate_percent": 5.0,
        },
    });
    println!("{}", serde_json::to_string_pretty(&evidence).unwrap());

    assert!(
        !source_tracked_worktree_dirty,
        "release performance evidence must be measured from a tracked-clean source commit"
    );
    assert!(
        Path::new(bin()).is_absolute(),
        "release performance evidence must name an absolute binary path"
    );
    assert!(
        warm_base_improvement >= 50.0,
        "warm Base improvement {warm_base_improvement:.2}% misses 50% gate"
    );
    assert!(
        dirty_median <= 10.0,
        "overlay median regression {dirty_median:.2}% exceeds 10% gate"
    );
    assert!(
        dirty_p95 <= 20.0,
        "overlay p95 regression {dirty_p95:.2}% exceeds 20% gate"
    );
    assert!(
        clean_median <= 5.0,
        "warm serial regression {clean_median:.2}% exceeds 5% gate"
    );
}
