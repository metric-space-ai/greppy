use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fixture(tag: &str, source: &str) -> (PathBuf, PathBuf, Scratch) {
    let unique = COUNTER.fetch_add(1, Ordering::SeqCst);
    let scratch = std::env::temp_dir().join(format!(
        "greppy-semantic-fallback-{tag}-{}-{unique}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&scratch);
    let repo = scratch.join("repo");
    let store = scratch.join("store");
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(repo.join("lib.rs"), source).unwrap();
    (repo, store, Scratch(scratch))
}

fn run(
    args: &[&str],
    cwd: &Path,
    store: &Path,
    extra_env: &[(&str, &str)],
) -> (i32, String, String) {
    let mut command = Command::new(bin());
    command
        .args(args)
        .current_dir(cwd)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env_remove("GREPPY_TEST_EMBED_ASSET_MISSING")
        .stdin(Stdio::null());
    for (key, value) in extra_env {
        command.env(key, value);
    }
    let output = command.output().expect("run greppy");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn graph_db(store: &Path) -> PathBuf {
    let mut stack = vec![store.to_path_buf()];
    while let Some(directory) = stack.pop() {
        for entry in std::fs::read_dir(&directory).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.file_name().and_then(|name| name.to_str()) == Some("graph.db") {
                return path;
            }
        }
    }
    panic!("graph.db not found under {}", store.display());
}

fn index_graph(repo: &Path, store: &Path) {
    let (code, stdout, stderr) = run(&["index", "."], repo, store, &[]);
    assert_eq!(
        code, 0,
        "graph index failed; stdout={stdout}\nstderr={stderr}"
    );
}

#[test]
fn semantic_building_status_has_eta_and_copyable_query_fallback() {
    let (repo, store, _scratch) = fixture(
        "building",
        "pub fn semantic_progress_marker() -> i32 { 7 }\n",
    );
    index_graph(&repo, &store);

    // Publish a deterministic live embedding job so semantic-search reports
    // build progress without spawning a model process during this test.
    let job = graph_db(&store).parent().unwrap().join("index.job");
    std::fs::write(
        &job,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": "greppy.background-job.v2",
            "kind": "embedding",
            "pid": std::process::id(),
            "state": "embedding",
            "backend": "cuda",
            "completed_spans": 3,
            "total_spans": 12,
            "eta_seconds": 9,
            "last_error": serde_json::Value::Null,
        }))
        .unwrap(),
    )
    .unwrap();

    let (code, stdout, stderr) = run(
        &["semantic-search", "find semantic progress marker"],
        &repo,
        &store,
        &[],
    );
    assert_eq!(
        code, 75,
        "building semantic index must be retryable; stdout={stdout}\nstderr={stderr}"
    );
    assert!(stderr.is_empty(), "status belongs on stdout: {stderr:?}");
    assert_eq!(
        stdout.lines().next(),
        Some("semantic index building — 3/12 spans, ETA ~9s (backend cuda)")
    );
    assert!(
        stdout.contains("try: greppy search-symbols marker"),
        "symbol fallback must use a real query token: {stdout:?}"
    );
    assert!(
        stdout.contains("try: greppy grep -rnE 'semantic|progress|marker' ."),
        "grep fallback must be copyable and contain the query tokens: {stdout:?}"
    );
}

#[test]
fn semantic_missing_asset_is_explicit_and_nonempty() {
    let (repo, store, _scratch) = fixture(
        "asset-missing",
        "pub fn asset_fallback_marker() -> i32 { 11 }\n",
    );
    index_graph(&repo, &store);

    let (code, stdout, stderr) = run(
        &["semantic-search", "find asset fallback marker"],
        &repo,
        &store,
        &[("GREPPY_TEST_EMBED_ASSET_MISSING", "1")],
    );
    assert_eq!(
        code, 69,
        "missing backend must be distinguishable from zero hits; stdout={stdout}\nstderr={stderr}"
    );
    assert!(stderr.is_empty(), "controlled status belongs on stdout");
    assert!(
        stdout
            .lines()
            .next()
            .is_some_and(|line| line.starts_with("semantic backend unavailable (asset missing)")),
        "first line must name the missing backend asset: {stdout:?}"
    );
    assert!(
        stdout.contains("try: greppy search-symbols marker")
            && stdout.contains("try: greppy grep -rnE 'asset|fallback|marker' ."),
        "missing-asset output needs exact non-semantic retries: {stdout:?}"
    );
}

#[test]
fn brief_missing_asset_keeps_definition_callers_and_callees() {
    let (repo, store, _scratch) = fixture(
        "brief-graph-only",
        r#"pub fn helper() -> i32 {
    leaf()
}

pub fn leaf() -> i32 {
    7
}

pub fn caller() -> i32 {
    helper()
}
"#,
    );
    index_graph(&repo, &store);

    let (code, stdout, stderr) = run(
        &["brief", "helper"],
        &repo,
        &store,
        &[("GREPPY_TEST_EMBED_ASSET_MISSING", "1")],
    );
    assert_eq!(
        code, 0,
        "graph-only brief remains successful; stdout={stdout}\nstderr={stderr}"
    );
    assert!(stderr.is_empty(), "brief degradation belongs on stdout");
    assert_eq!(
        stdout.lines().next(),
        Some(
            "semantic backend unavailable (asset missing) — brief continuing with graph-only definition/callers/callees"
        )
    );
    assert!(
        stdout.contains("pub fn helper()")
            && stdout.contains("-- CALLERS (1) --")
            && stdout.contains("caller")
            && stdout.contains("-- CALLS (1) --")
            && stdout.contains("leaf"),
        "brief must preserve graph evidence without EmbeddingGemma: {stdout:?}"
    );
}
