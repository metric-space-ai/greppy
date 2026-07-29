//! Missing-asset / still-building coverage for the 0.3.0 meaning-search
//! command `search` (the replacement for the retired `semantic-search`)
//! and for `brief`, plus the retirement pin for the dead verb.
//!
//! 0.3.0 contract (dev/SEARCH-OUTPUT-SPEC.md + dev/NAV-OUTPUT-SPEC-BRIEF.md,
//! normative; NAV law 1: no justification, no instruction):
//! * While the embedding index is still building, `search` prints ONE status
//!   line with progress and ETA and exits 1 (grep's convention). Never
//!   partial hits, never `try:` fallback instructions.
//! * When the embedding assets cannot be resolved, `search` prints ONE line
//!   naming the unavailable semantic index and exits 1 — a message, not a
//!   different exit code, distinguishes it from zero hits.
//! * `brief` needs no embedding model: definition head, sketch and the
//!   aggregated caller line are graph evidence and are printed without any
//!   degradation notice.
//! * `semantic-search` is dead-listed vocabulary: refused as an unknown
//!   subcommand (exit 64), never grepped.

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
fn search_building_status_is_one_line_with_progress_and_eta() {
    let (repo, store, _scratch) = fixture(
        "building",
        "pub fn semantic_progress_marker() -> i32 { 7 }\n",
    );
    index_graph(&repo, &store);

    // Publish a deterministic live embedding job so `search` reports build
    // progress without spawning a model process during this test.
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
        &["search", "find semantic progress marker"],
        &repo,
        &store,
        &[],
    );
    assert_eq!(
        code, 1,
        "a building semantic index is not an answer: grep's exit 1; stdout={stdout}\nstderr={stderr}"
    );
    assert!(stderr.is_empty(), "status belongs on stdout: {stderr:?}");
    assert_eq!(
        stdout, "semantic index building — 3/12 spans, ETA ~9s (backend cuda)\n",
        "exactly ONE status line with progress and ETA — never partial hits, \
         never try: instructions (NAV law 1); got: {stdout:?}"
    );
}

#[test]
fn search_missing_asset_names_the_unavailable_backend() {
    let (repo, store, _scratch) = fixture(
        "asset-missing",
        "pub fn asset_fallback_marker() -> i32 { 11 }\n",
    );
    index_graph(&repo, &store);

    let (code, stdout, stderr) = run(
        &["search", "find asset fallback marker"],
        &repo,
        &store,
        &[("GREPPY_TEST_EMBED_ASSET_MISSING", "1")],
    );
    assert_eq!(
        code, 1,
        "an unavailable backend is not an answer: grep's exit 1; stdout={stdout}\nstderr={stderr}"
    );
    assert!(stderr.is_empty(), "controlled status belongs on stdout");
    assert_eq!(
        stdout, "semantic index unavailable — embedding assets could not be resolved\n",
        "exactly ONE line names the unavailable semantic index — the message, \
         not the exit code, distinguishes it from zero hits; got: {stdout:?}"
    );
    assert!(
        !stdout.contains("lib.rs:"),
        "no hits may be served without the embedding backend: {stdout:?}"
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
    assert!(stderr.is_empty(), "brief prints no degradation notice");
    // The 0.3.0 brief shape (dev/NAV-OUTPUT-SPEC-BRIEF.md): address, the
    // verbatim definition head, the one-line-per-step sketch naming the
    // callee, the closing brace, and the aggregated caller line. No
    // "semantic backend unavailable" notice (NAV law 1), no 0.2.x
    // `-- CALLERS (n) --` ASCII bars.
    assert_eq!(
        stdout, "lib.rs:1\npub fn helper() -> i32 {\n  2  leaf\n}\n\ncalled by caller\n",
        "brief keeps the definition head, the callee sketch and the caller \
         line without the embedding model; got: {stdout:?}"
    );
    assert!(
        !stdout.contains("semantic backend unavailable") && !stdout.contains("-- CALLERS"),
        "no degradation notice and no retired section bars; got: {stdout:?}"
    );
}

/// The pre-0.3.0 `semantic-search` subcommand is dead-listed vocabulary
/// (dev/SEARCH-OUTPUT-SPEC.md: "`greppy semantic-search X` must be an unknown
/// subcommand, not a grep for 'semantic-search' in a file called X").
#[test]
fn retired_semantic_search_is_refused_not_grepped() {
    let (repo, store, _scratch) = fixture(
        "retired",
        "pub fn semantic_progress_marker() -> i32 { 7 }\n",
    );

    let (code, stdout, stderr) = run(
        &["semantic-search", "find semantic progress marker"],
        &repo,
        &store,
        &[],
    );
    let text = format!("{stdout}{stderr}");
    assert_eq!(
        code, 64,
        "`greppy semantic-search` must refuse as invalid vocabulary; stdout={stdout}\nstderr={stderr}"
    );
    assert!(
        text.contains("unrecognized subcommand 'semantic-search'"),
        "the refusal names the dead verb; got: {text}"
    );
    assert!(
        !text.contains("lib.rs:"),
        "the refusal never greps for the dead verb; got: {text}"
    );
}
