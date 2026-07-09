//! Integration tests for the Track-A graph-navigation commands:
//! `who-calls`, `find-usages`, and the extended `trace` (incoming
//! direction + edge filter + depth).
//!
//! These spawn the real `greppy` binary against a multi-file fixture
//! indexed end-to-end, so the cross-file CALLS / USES / TYPE_REF edges
//! resolved by the indexer/resolver are exercised exactly as an agent
//! would see them. Each test gets an isolated `GREPPY_STORE_DIR` so
//! parallel runs never collide.
//!
//! The fixture shapes mirror the proven cross-file edge tests in
//! `crates/indexer/src/lib.rs` (`cross_file_calls_edge_is_persisted_*`,
//! `cross_file_type_ref_edge_is_persisted`,
//! `cross_file_uses_edge_is_persisted`) so we know the indexer really
//! produces the edges these commands read back.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fresh_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("greppy-cli-graphnav-{tag}-{pid}-{n}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Build a git-rooted repo whose `src/lib.rs` exercises all three
/// cross-file reference edges into `src/helper.rs` / `src/types.rs`:
///
/// * `caller()`  --CALLS-->    `do_it()`       (helper.rs)
/// * `render(w: Widget)` --TYPE_REF--> `Widget` (types.rs)
/// * `build()`   --USES-->     `Marker`        (types.rs)
///
/// Returns (repo_root, store_dir).
fn make_graph_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).unwrap();
    // `.git` is the repo-root marker resolve_root walks up to find.
    std::fs::create_dir_all(repo.join(".git")).unwrap();

    // lib.rs references symbols defined in the two sibling modules.
    std::fs::write(
        src.join("lib.rs"),
        r#"
mod helper;
mod types;

fn caller() {
    helper::do_it();
}

fn render(w: types::Widget) -> u32 { w.w }

fn build() {
    let _m = make(types::Marker);
}

fn make(_x: types::Marker) {}
"#,
    )
    .unwrap();

    std::fs::write(
        src.join("helper.rs"),
        "pub fn do_it() -> u32 {\n    let answer = 42;\n    answer\n}\n",
    )
    .unwrap();

    std::fs::write(
        src.join("types.rs"),
        "pub struct Widget { pub w: u32 }\npub struct Marker;\n",
    )
    .unwrap();

    let store = root.join("store");
    (repo, store)
}

fn make_python_class_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    let checkov = repo.join("checkov");
    let cloudformation = checkov.join("cloudformation");
    std::fs::create_dir_all(&cloudformation).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();

    std::fs::write(
        checkov.join("runner_filter.py"),
        r#"def setup_filter():
    return 1

def should_run_check():
    return 2

class RunnerFilter:
    def __init__(self):
        setup_filter()

    def apply(self):
        should_run_check()
        setup_filter()
"#,
    )
    .unwrap();

    std::fs::write(
        cloudformation.join("runner.py"),
        r#"from checkov.runner_filter import RunnerFilter

def build_filter():
    return RunnerFilter()

def use_filter():
    f = RunnerFilter()
    return f
"#,
    )
    .unwrap();

    let store = root.join("store");
    (repo, store)
}

fn index_python_class_fixture(tag: &str) -> (PathBuf, PathBuf) {
    let (repo, store) = make_python_class_repo(tag);
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );
    (repo, store)
}

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
        .env("GREPPY_TEST_SKIP_INFERENCE", "1");
    for (key, value) in envs {
        cmd.env(key, value);
    }
    let out = cmd.output().expect("spawn greppy");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn expand_id_from_stdout(out: &str) -> Option<String> {
    out.lines()
        .find(|line| line.starts_with("Expand: greppy expand "))
        .and_then(|line| line.split_whitespace().nth(3))
        .map(str::to_string)
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

/// Index the fixture once and assert it succeeded; shared setup.
fn index_fixture(tag: &str) -> (PathBuf, PathBuf) {
    let (repo, store) = make_graph_repo(tag);
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );
    (repo, store)
}

fn make_real_git_diff_impact_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("lib.rs"), "mod hub;\nmod callers;\n").unwrap();
    std::fs::write(src.join("hub.rs"), "pub fn hub() -> u32 { 7 }\n").unwrap();
    std::fs::write(
        src.join("callers.rs"),
        r#"
pub fn caller_a() -> u32 { crate::hub::hub() }
pub fn caller_b() -> u32 { crate::hub::hub() }
pub fn caller_c() -> u32 { crate::hub::hub() }
"#,
    )
    .unwrap();

    git(&repo, &["init"]);
    git(&repo, &["config", "user.email", "greppy@example.invalid"]);
    git(&repo, &["config", "user.name", "greppy test"]);
    git(&repo, &["add", "."]);
    git(&repo, &["commit", "-m", "baseline"]);
    git(&repo, &["branch", "basepoint"]);

    std::fs::write(src.join("hub.rs"), "pub fn hub() -> u32 { 8 }\n").unwrap();
    git(&repo, &["add", "src/hub.rs"]);
    git(&repo, &["commit", "-m", "change hub"]);

    let store = root.join("store");
    (repo, store)
}

// ---------------------------------------------------------------------------
// who-calls — incoming CALLS edges resolve to the cross-file caller.
// ---------------------------------------------------------------------------

#[test]
fn who_calls_lists_cross_file_caller_with_file_line() {
    let (repo, store) = index_fixture("whocalls");

    // `do_it` is defined in helper.rs and called by `caller` in lib.rs.
    let (code, out, err) = run(&["who-calls", "do_it"], &repo, &store);
    assert_eq!(
        code, 0,
        "who-calls should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("caller"),
        "who-calls do_it must list the caller `caller`; got: {out:?}"
    );
    assert!(
        out.contains("src/lib.rs:"),
        "who-calls must print the caller's file:line (src/lib.rs); got: {out:?}"
    );
    // The callee itself must NOT appear as its own caller.
    assert!(
        !out.contains("(no callers)"),
        "who-calls must find at least one caller; got: {out:?}"
    );
}

#[test]
fn who_calls_prints_line_span_and_expand_pack_round_trips() {
    let (repo, store) = index_fixture("whocalls-expand");

    let (code, out, err) = run(&["who-calls", "do_it"], &repo, &store);
    assert_eq!(
        code, 0,
        "who-calls should exit 0; stderr={err}\nstdout={out}"
    );
    let caller_line = out
        .lines()
        .find(|line| line.contains("caller") && line.contains("src/lib.rs:"))
        .unwrap_or_else(|| panic!("missing caller line in stdout: {out:?}"));
    assert!(
        caller_line.contains('-'),
        "caller line must include start-end span, got: {caller_line:?}"
    );
    let id = expand_id_from_stdout(&out)
        .unwrap_or_else(|| panic!("missing Expand line in stdout: {out:?}"));

    let (code, expanded, err) = run(&["expand", &id], &repo, &store);
    assert_eq!(
        code, 0,
        "expand should exit 0; stderr={err}\nstdout={expanded}"
    );
    assert!(
        expanded.contains("helper::do_it()") && expanded.contains("source:"),
        "expand pack must include prepared source evidence, got: {expanded:?}"
    );
}

#[test]
fn expand_missing_id_reports_clear_message() {
    let (repo, store) = index_fixture("expand-missing");

    let (code, out, err) = run(&["expand", "does-not-exist"], &repo, &store);
    assert_eq!(code, 1, "missing expand id should exit 1; stderr={err}");
    assert!(
        out.contains("expand: id not found or expired: does-not-exist"),
        "missing expand id must be visible on stdout; got: {out:?}"
    );
}

#[test]
fn who_calls_reports_no_callers_for_uncalled_symbol() {
    let (repo, store) = index_fixture("whocalls-none");
    // `Widget` is a struct — nothing CALLS it.
    let (code, out, _err) = run(&["who-calls", "Widget"], &repo, &store);
    assert_eq!(code, 0);
    assert!(
        out.contains("(no callers)"),
        "an uncalled symbol must report no callers; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// find-usages — incoming USES + TYPE_REF edges, with the edge kind shown.
// ---------------------------------------------------------------------------

#[test]
fn find_usages_lists_type_ref_into_struct() {
    let (repo, store) = index_fixture("usages-typeref");

    // `Widget`'s type is used by `render`'s parameter (TYPE_REF).
    let (code, out, err) = run(&["find-usages", "Widget"], &repo, &store);
    assert_eq!(
        code, 0,
        "find-usages should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        // Type references are persisted under the unified C-reference USAGE
        // label (formerly the separate TYPE_REF pass).
        out.contains("USAGE"),
        "find-usages Widget must label the edge kind USAGE; got: {out:?}"
    );
    assert!(
        out.contains("render"),
        "find-usages Widget must list `render` as the referrer; got: {out:?}"
    );
    assert!(
        out.contains("src/lib.rs:"),
        "find-usages must print the referrer's file:line; got: {out:?}"
    );
}

#[test]
fn find_usages_lists_uses_into_struct() {
    let (repo, store) = index_fixture("usages-uses");

    // `Marker` is referenced (USES) from `build`/`make` in lib.rs.
    let (code, out, err) = run(&["find-usages", "Marker"], &repo, &store);
    assert_eq!(
        code, 0,
        "find-usages should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        // Value references are persisted under the unified C-reference USAGE
        // label (formerly the separate USES pass).
        out.contains("USAGE"),
        "find-usages Marker must show a USAGE reference edge; got: {out:?}"
    );
    assert!(
        out.contains("src/lib.rs:"),
        "find-usages must print the referrer's file:line; got: {out:?}"
    );
    assert!(
        !out.contains("(no usages)"),
        "Marker is referenced cross-file, so usages must be non-empty; got: {out:?}"
    );
}

#[test]
fn references_lists_calls_and_usages_with_edge_kind() {
    let (repo, store) = index_fixture("references");

    let (code, out, err) = run(&["references", "do_it"], &repo, &store);
    assert_eq!(
        code, 0,
        "references should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("CALLS"),
        "references do_it must show the CALLS edge kind; got: {out:?}"
    );
    assert!(
        out.contains("caller"),
        "references do_it must list the caller `caller`; got: {out:?}"
    );

    let (code, out, err) = run(&["references", "Widget"], &repo, &store);
    assert_eq!(
        code, 0,
        "references Widget should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("USAGE") && out.contains("render"),
        "references Widget must include the USAGE referrer `render`; got: {out:?}"
    );
}

#[test]
fn direct_navigation_json_reports_exact_counts() {
    let (repo, store) = index_fixture("nav-json");

    let cases = [
        ("who-calls", "do_it", "caller", None),
        ("callees", "caller", "do_it", None),
        ("find-usages", "Widget", "render", Some("USAGE")),
        ("references", "do_it", "caller", Some("CALLS")),
    ];

    for (cmd, symbol, expected_qname, expected_edge) in cases {
        let (code, out, err) = run(&[cmd, symbol, "--json"], &repo, &store);
        assert_eq!(
            code, 0,
            "{cmd} --json should exit 0; stderr={err}\nstdout={out}"
        );
        let v: serde_json::Value = serde_json::from_str(&out)
            .unwrap_or_else(|e| panic!("invalid {cmd} json: {e}; stdout={out:?}"));
        assert_eq!(v["command"], cmd);
        assert_eq!(v["symbol"], symbol);
        assert_eq!(v["project"], "repo");
        assert_eq!(v["symbol_found"], true);
        assert_eq!(v["fresh"], true);
        assert_eq!(v["freshness"]["state"], "fresh");
        assert!(
            v["freshness"]["reasons"].as_array().unwrap().is_empty(),
            "fresh graph must not report stale reasons: {v:?}"
        );
        assert_eq!(v["provider_complete"], false);
        assert!(
            v["incomplete_provider_count"].as_u64().unwrap_or(0) >= 1,
            "nav JSON must expose provider incompleteness: {v:?}"
        );
        assert!(
            v["incomplete_providers"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| p["language"] == "rust"),
            "rust provider incompleteness must be visible: {v:?}"
        );
        assert_eq!(v["total_exact"], 1);
        assert_eq!(v["shown"], 1);
        assert_eq!(v["omitted"], 0);
        assert_eq!(v["truncated"], false);
        assert_eq!(v["expand"]["available"], true);
        assert_eq!(v["expand"]["kind"], "evidence_pack");
        assert!(
            v["expand"]["id"].as_str().is_some_and(|id| !id.is_empty()),
            "{cmd} JSON must expose expand id: {v:?}"
        );
        let hits = v["hits"].as_array().expect("hits array");
        assert_eq!(hits.len(), 1);
        assert!(
            hits[0]["qualified_name"]
                .as_str()
                .unwrap_or("")
                .contains(expected_qname),
            "{cmd} hit should include {expected_qname}, got {v:?}"
        );
        assert!(
            hits[0]["file_path"]
                .as_str()
                .unwrap_or("")
                .starts_with("src/"),
            "{cmd} hit must carry a repo-relative file path, got {v:?}"
        );
        if let Some(edge) = expected_edge {
            assert_eq!(hits[0]["edge_type"], edge);
        }
    }
}

/// D2 fail-open, small drift: a single edited file is auto-reindexed
/// inline and the query answers FRESH from the healed graph (the old
/// contract failed closed with `skipped_stale_index` here).
#[test]
fn direct_navigation_json_auto_reindexes_small_stale_drift() {
    let (repo, store) = index_fixture("nav-json-stale");
    std::fs::write(
        repo.join("src/lib.rs"),
        r#"
mod helper;
mod types;

fn caller() {
    helper::do_it();
}

fn render(w: types::Widget) -> u32 { w.w + 1 }
"#,
    )
    .unwrap();

    let (code, out, err) = run(&["who-calls", "do_it", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "small stale drift must be auto-healed, not refused; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "auto-healed query must not warn about staleness; stderr={err:?}"
    );
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid nav json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "who-calls");
    assert_eq!(
        v["status"],
        serde_json::Value::Null,
        "healed query must not be skipped: {v:?}"
    );
    assert_eq!(
        v["fresh"], true,
        "auto-reindex must yield a fresh answer: {v:?}"
    );
    assert_eq!(v["freshness"]["state"], "fresh");
    let hits = v["hits"].as_array().expect("hits array");
    assert!(
        hits.iter().any(|h| h["qualified_name"]
            .as_str()
            .unwrap_or("")
            .contains("caller")),
        "healed graph must resolve the CURRENT caller of do_it: {v:?}"
    );
}

/// D2 fail-open, large drift: with more files changed than the inline
/// auto-reindex cap (10), the commands serve rows FROM THE EXISTING
/// INDEX, honestly labeled (`fresh: false` + stderr warning), instead
/// of refusing with exit 1 as the old fail-closed contract did.
#[test]
fn graph_commands_serve_labeled_rows_on_large_stale_drift() {
    let (repo, store) = index_fixture("graph-stale-gate");
    std::fs::write(
        repo.join("src/helper.rs"),
        "pub fn do_it_renamed() -> u32 { 42 }\n",
    )
    .unwrap();
    // Push the drift past AUTO_REINDEX_MAX_FILES (10) so the inline
    // heal is skipped and the labeled-stale path is exercised.
    for i in 0..11 {
        std::fs::write(
            repo.join(format!("src/extra_{i}.rs")),
            format!("pub fn extra_{i}() -> u32 {{ {i} }}\n"),
        )
        .unwrap();
    }

    let json_cases: Vec<(Vec<&str>, &str, &str)> = vec![
        (
            vec!["search-graph", "--name", "do_it", "--json"],
            "search-graph",
            "hits",
        ),
        (
            vec!["trace", "--symbol", "caller", "--json"],
            "trace",
            "steps",
        ),
        (
            vec!["graph-locate", "src/helper.rs:1", "--json"],
            "graph-locate",
            "hits",
        ),
        (vec!["impact", "do_it", "--json"], "impact", "hits"),
        (vec!["fan-in", "--json"], "fan-in", "hits"),
    ];
    for (args, command, collection_field) in json_cases {
        let (code, out, err) = run(&args, &repo, &store);
        assert_eq!(
            code, 0,
            "labeled-stale {command} must serve results, not refuse; stderr={err}\nstdout={out}"
        );
        assert!(
            err.contains("index may be stale"),
            "labeled-stale {command} must warn on stderr; stderr={err:?}"
        );
        assert!(
            err.contains("run 'grep index'"),
            "stale warning must tell the agent the fix; stderr={err:?}"
        );
        let v: serde_json::Value = serde_json::from_str(&out).unwrap_or_else(|e| {
            panic!("invalid labeled-stale {command} json: {e}; stdout={out:?}")
        });
        assert_eq!(v["command"], command);
        assert_eq!(
            v["status"],
            serde_json::Value::Null,
            "labeled-stale {command} must not be skipped: {v:?}"
        );
        assert_eq!(
            v["fresh"], false,
            "{command} must label the result stale: {v:?}"
        );
        assert_eq!(v["freshness"]["state"], "stale");
        assert_eq!(
            v["freshness"]["stale_file_count"], 12,
            "{command} must report the drift extent: {v:?}"
        );
        assert!(
            !v[collection_field].as_array().unwrap().is_empty(),
            "labeled-stale {command} must serve rows from the existing index: {v:?}"
        );
    }

    let (code, out, err) = run(&["brief", "do_it"], &repo, &store);
    assert_eq!(
        code, 0,
        "labeled-stale brief must serve the indexed brief; stderr={err}\nstdout={out}"
    );
    assert!(
        err.contains("index may be stale"),
        "labeled-stale brief must warn on stderr; stderr={err:?}"
    );
    assert!(
        out.contains("do_it"),
        "labeled-stale brief must serve rows from the existing index; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// trace --direction incoming — walks back from the callee to the caller.
// ---------------------------------------------------------------------------

#[test]
fn trace_incoming_walks_back_to_caller() {
    let (repo, store) = index_fixture("trace-in");

    // Incoming CALLS trace from `do_it` must include the caller `caller`.
    let (code, out, err) = run(
        &[
            "trace",
            "--symbol",
            "do_it",
            "--direction",
            "incoming",
            "--edge",
            "CALLS",
        ],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "trace incoming should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("do_it"),
        "trace must include the start symbol do_it; got: {out:?}"
    );
    assert!(
        out.contains("caller"),
        "incoming trace from do_it must reach `caller`; got: {out:?}"
    );
    // Actionable output: qualified_name + file:line span.
    assert!(
        out.contains("src/lib.rs:"),
        "trace must print actionable file:line for the caller; got: {out:?}"
    );
}

#[test]
fn trace_outgoing_default_walks_to_callee() {
    let (repo, store) = index_fixture("trace-out");

    // Default direction (outgoing) from `caller` must reach `do_it`.
    let (code, out, err) = run(&["trace", "--symbol", "caller"], &repo, &store);
    assert_eq!(
        code, 0,
        "trace outgoing should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("do_it"),
        "outgoing trace from caller must reach do_it; got: {out:?}"
    );
    assert!(
        out.contains("src/helper.rs:"),
        "outgoing trace must print the callee's file:line (helper.rs); got: {out:?}"
    );
}

#[test]
fn trace_json_reports_steps_counts_and_metadata() {
    let (repo, store) = index_fixture("trace-json");

    let (code, out, err) = run(&["trace", "--symbol", "caller", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "trace --json should exit 0; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid trace json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "trace");
    assert_eq!(v["symbol"], "caller");
    assert_eq!(v["project"], "repo");
    assert_eq!(v["symbol_found"], true);
    assert_eq!(v["fresh"], true);
    assert_eq!(v["provider_complete"], false);
    assert_eq!(v["scope"], "bounded_bfs");
    assert_eq!(v["direction"], "outgoing");
    assert_eq!(v["edge_type"], "CALLS");
    assert_eq!(v["max_depth"], 4);
    assert_eq!(v["total_exact"], 2);
    assert_eq!(v["shown"], 2);
    assert_eq!(v["omitted"], 0);
    assert_eq!(v["truncated"], false);
    let steps = v["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0]["depth"], 0);
    assert!(steps[0]["qualified_name"]
        .as_str()
        .unwrap_or("")
        .contains("caller"));
    assert!(steps[0]["via_edge"].is_null());
    assert_eq!(steps[1]["depth"], 1);
    assert!(steps[1]["qualified_name"]
        .as_str()
        .unwrap_or("")
        .contains("do_it"));
    assert_eq!(steps[1]["via_edge"]["edge_type"], "CALLS");
}

#[test]
fn trace_depth_zero_returns_only_start() {
    let (repo, store) = index_fixture("trace-depth0");

    // depth 0 means: emit only the start node, no neighbours.
    let (code, out, err) = run(
        &["trace", "--symbol", "caller", "--depth", "0"],
        &repo,
        &store,
    );
    assert_eq!(code, 0, "trace --depth 0 should exit 0; stderr={err}");
    assert!(
        out.contains("caller") && out.contains("depth=0"),
        "depth 0 must emit the start node; got: {out:?}"
    );
    assert!(
        !out.contains("do_it"),
        "depth 0 must NOT walk to the callee do_it; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// search-symbols — improved output carries label + qualified_name + file:line.
// ---------------------------------------------------------------------------

#[test]
fn search_symbols_prints_label_and_file_line() {
    let (repo, store) = index_fixture("symbols");

    let (code, out, err) = run(&["search-symbols", "Widget"], &repo, &store);
    assert_eq!(
        code, 0,
        "search-symbols should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("Widget"),
        "search-symbols must find the Widget symbol; got: {out:?}"
    );
    assert!(
        out.contains("src/types.rs:"),
        "search-symbols must print the symbol's file:line; got: {out:?}"
    );
    assert!(
        // Rust struct defs are labeled `Class` (C-reference parity).
        out.contains("Class"),
        "search-symbols must print the node label (Class); got: {out:?}"
    );
}

#[test]
fn search_symbols_json_reports_exact_counts_and_metadata() {
    let (repo, store) = index_fixture("symbols-json");

    let (code, out, err) = run(&["search-symbols", "Widget", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "search-symbols --json should exit 0; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "search-symbols");
    assert_eq!(v["status"], "ok");
    assert_eq!(v["fresh"], true);
    assert_eq!(v["provider_complete"], false);
    assert!(
        v["incomplete_provider_count"].as_u64().unwrap_or(0) >= 1,
        "search-symbols JSON must expose provider incompleteness: {v:?}"
    );
    assert!(
        v["incomplete_providers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["language"] == "rust"),
        "rust provider incompleteness must be visible: {v:?}"
    );
    let hits = v["hits"].as_array().expect("hits array");
    assert!(!hits.is_empty());
    assert_eq!(v["total_exact"].as_i64().unwrap(), hits.len() as i64);
    assert_eq!(v["shown"].as_i64().unwrap(), hits.len() as i64);
    assert_eq!(v["omitted"], 0);
    assert_eq!(v["truncated"], false);
    assert!(
        hits.iter().any(|h| h["label"] == "Class"
            && h["file_path"] == "src/types.rs"
            && h["qualified_name"]
                .as_str()
                .unwrap_or("")
                .contains("Widget")),
        "search-symbols JSON must expose the matched symbol; got: {v:?}"
    );
}

/// D2 fail-open, small drift: renaming one symbol's file is healed by
/// the inline auto-reindex, so the query reflects the CURRENT tree —
/// the renamed-away symbol is gone, the new one is findable — instead
/// of the old fail-closed `skipped_stale_index` refusal.
#[test]
fn search_symbols_json_auto_reindexes_and_reports_current_state() {
    let (repo, store) = index_fixture("symbols-json-stale");
    std::fs::write(
        repo.join("src/types.rs"),
        "pub struct WidgetRenamed { pub w: u32 }\npub struct Marker;\n",
    )
    .unwrap();

    let (code, out, err) = run(
        &["search-symbols", "WidgetRenamed", "--json"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "healed search-symbols must find the CURRENT symbol; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "search-symbols");
    assert_eq!(v["status"], "ok");
    assert_eq!(
        v["fresh"], true,
        "auto-reindex must yield a fresh answer: {v:?}"
    );
    assert!(
        v["hits"]
            .as_array()
            .unwrap()
            .iter()
            .any(|h| h["qualified_name"]
                .as_str()
                .unwrap_or("")
                .contains("WidgetRenamed")),
        "healed index must expose the renamed symbol: {v:?}"
    );
}

#[test]
fn search_graph_json_reports_exact_counts_and_metadata() {
    let (repo, store) = index_fixture("search-graph-json");

    let (code, out, err) = run(
        &["search-graph", "--name", "Widget", "--json"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "search-graph --json should exit 0; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid search-graph json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "search-graph");
    assert_eq!(v["project"], "repo");
    assert_eq!(v["filters"]["name"], "Widget");
    assert_eq!(v["fresh"], true);
    assert_eq!(v["provider_complete"], false);
    assert_eq!(v["scope"], "node_search");
    assert_eq!(v["limit"], 50);
    assert_eq!(v["total_exact"], 1);
    assert_eq!(v["shown"], 1);
    assert_eq!(v["omitted"], 0);
    assert_eq!(v["truncated"], false);
    let hits = v["hits"].as_array().expect("hits array");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0]["name"], "Widget");
    assert_eq!(hits[0]["file_path"], "src/types.rs");
    assert_eq!(hits[0]["label"], "Class");
}

#[test]
fn provider_policy_require_complete_blocks_graph_commands_json_and_brief_text() {
    let (repo, store) = index_fixture("provider-policy-graph");
    let env = [("GREPPY_PROVIDER_POLICY", "require_complete")];

    let cases: Vec<(Vec<&str>, &str, &str)> = vec![
        (
            vec!["search-graph", "--name", "Widget", "--json"],
            "search-graph",
            "hits",
        ),
        (
            vec!["trace", "--symbol", "caller", "--json"],
            "trace",
            "steps",
        ),
        (vec!["who-calls", "do_it", "--json"], "who-calls", "hits"),
        (
            vec!["find-usages", "Widget", "--json"],
            "find-usages",
            "hits",
        ),
        (vec!["references", "Widget", "--json"], "references", "hits"),
        (
            vec!["graph-locate", "src/lib.rs:6", "--json"],
            "graph-locate",
            "hits",
        ),
        (vec!["impact", "do_it", "--json"], "impact", "hits"),
        (vec!["fan-in", "--json"], "fan-in", "hits"),
    ];

    for (args, command, empty_field) in cases {
        let (code, out, err) = run_with_env(&args, &repo, &store, &env);
        assert_eq!(
            code, 1,
            "strict provider policy should block {command}; stderr={err}\nstdout={out}"
        );
        assert!(
            err.is_empty(),
            "strict graph JSON should not require stderr parsing for {command}; stderr={err:?}"
        );
        let v: serde_json::Value = serde_json::from_str(&out)
            .unwrap_or_else(|e| panic!("invalid {command} json: {e}; stdout={out:?}"));
        assert_eq!(v["command"], command);
        assert_eq!(v["status"], "skipped_incomplete_provider");
        assert_eq!(v["provider_complete"], false);
        assert!(
            v["incomplete_provider_count"].as_u64().unwrap_or(0) >= 1,
            "strict graph JSON must expose incomplete providers for {command}: {v:?}"
        );
        assert_eq!(v["total_exact"], 0);
        assert_eq!(v["shown"], 0);
        assert_eq!(v[empty_field].as_array().unwrap().len(), 0);
    }

    let (code, out, err) = run_with_env(&["brief", "do_it"], &repo, &store, &env);
    assert_eq!(
        code, 1,
        "strict provider policy should block brief text; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "strict brief text skip should stay on stdout; stderr={err:?}"
    );
    assert!(
        out.contains("brief: skipped indexed provider-dependent output"),
        "brief strict skip must be explicit; got: {out:?}"
    );
}

#[test]
fn graph_locate_maps_grep_line_to_enclosing_symbol() {
    let (repo, store) = index_fixture("graph-locate");

    let (code, out, err) = run(&["graph-locate", "src/lib.rs:6"], &repo, &store);
    assert_eq!(
        code, 0,
        "graph-locate should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("caller") && out.contains("src/lib.rs:"),
        "graph-locate src/lib.rs:6 must locate the enclosing caller function; got: {out:?}"
    );
    assert!(
        out.contains("match=nearest_preceding"),
        "body-line fallback must be explicit in text output; got: {out:?}"
    );
    assert!(
        out.contains("Function"),
        "graph-locate text output must include the node label; got: {out:?}"
    );
}

#[test]
fn graph_locate_json_reports_metadata_and_no_match() {
    let (repo, store) = index_fixture("graph-locate-json");

    let (code, out, err) = run(
        &[
            "graph-locate",
            "--file",
            "./src/lib.rs",
            "--line",
            "9",
            "--json",
        ],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "graph-locate --json should exit 0; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid graph-locate json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "graph-locate");
    assert_eq!(v["project"], "repo");
    assert_eq!(v["file_path"], "src/lib.rs");
    assert_eq!(v["line"], 9);
    assert_eq!(v["location_found"], true);
    assert_eq!(v["match_kind"], "enclosing");
    assert_eq!(v["fresh"], true);
    assert_eq!(v["provider_complete"], false);
    assert_eq!(v["scope"], "file_line_innermost_symbol");
    assert_eq!(v["total_exact"], 1);
    assert_eq!(v["shown"], 1);
    assert_eq!(v["omitted"], 0);
    assert_eq!(v["truncated"], false);
    let hits = v["hits"].as_array().expect("hits array");
    assert_eq!(hits.len(), 1);
    assert!(hits[0]["qualified_name"]
        .as_str()
        .unwrap_or("")
        .contains("render"));

    let (code, out, err) = run(&["graph-locate", "src/lib.rs:4", "--json"], &repo, &store);
    assert_eq!(
        code, 1,
        "graph-locate no-match should exit 1; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid graph-locate no-match json: {e}; stdout={out:?}"));
    assert_eq!(v["location_found"], false);
    assert!(v["match_kind"].is_null());
    assert_eq!(v["total_exact"], 0);
    assert!(v["hits"].as_array().unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// Unknown symbol — the navigation commands exit 1 with a clear message,
// they do not panic or report a bogus result.
// ---------------------------------------------------------------------------

#[test]
fn navigation_commands_report_missing_symbol() {
    let (repo, store) = index_fixture("missing");

    for cmd in [
        vec!["who-calls", "does_not_exist_xyz"],
        vec!["find-usages", "does_not_exist_xyz"],
        vec!["references", "does_not_exist_xyz"],
        vec!["trace", "--symbol", "does_not_exist_xyz"],
    ] {
        let (code, out, _err) = run(&cmd, &repo, &store);
        assert_eq!(
            code, 1,
            "missing symbol must exit 1 for {cmd:?}; got out={out:?}"
        );
        assert!(
            out.contains("symbol not found"),
            "missing symbol must report not-found for {cmd:?}; got: {out:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// F1 — who-calls caps its output (token-bomb guard) and `--all` lifts it.
// ---------------------------------------------------------------------------

/// Build a repo where `hub()` is called from 60 distinct functions across
/// many files, so `who-calls hub` resolves far more than the NAV_LIMIT (40)
/// callers.
fn make_hot_symbol_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();

    // hub.rs defines the hot target.
    std::fs::write(src.join("hub.rs"), "pub fn hub() -> u32 { 7 }\n").unwrap();

    // 60 caller functions, 10 per module file, each calling hub().
    let mut lib = String::from("mod hub;\n");
    for f in 0..6 {
        lib.push_str(&format!("mod callers{f};\n"));
        let mut m = String::new();
        for i in 0..10 {
            m.push_str(&format!(
                "pub fn caller_{f}_{i}() {{ let _ = crate::hub::hub(); }}\n"
            ));
        }
        std::fs::write(src.join(format!("callers{f}.rs")), m).unwrap();
    }
    std::fs::write(src.join("lib.rs"), lib).unwrap();

    let store = root.join("store");
    (repo, store)
}

#[test]
fn who_calls_caps_output_and_all_lifts_it() {
    let (repo, store) = make_hot_symbol_repo("hot");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index must succeed; stderr={err}\nstdout={out}");

    // Default: capped at NAV_LIMIT (40) result rows + a "more" footer.
    let (code, out, err) = run(&["who-calls", "hub"], &repo, &store);
    assert_eq!(code, 0, "who-calls should exit 0; stderr={err}");
    // Row lines start at column 0; the grep-shaped call-site evidence
    // lines (P4) are indented and do not count against the row cap.
    let caller_lines = out
        .lines()
        .filter(|l| l.contains("caller_") && !l.starts_with(' '))
        .count();
    assert_eq!(
        caller_lines, 40,
        "default who-calls must cap at 40 caller rows; got {caller_lines}\n{out}"
    );
    assert!(
        out.contains("40 shown of 60 total"),
        "capped output must carry the 'N more' footer; got: {out}"
    );

    // `--code` uses the much tighter CODE_NAV_LIMIT (6): each row carries a
    // ~25-line body, so the default 40 would be a ~1000-line token bomb.
    let (code, out, err) = run(&["who-calls", "hub", "--code"], &repo, &store);
    assert_eq!(code, 0, "who-calls --code should exit 0; stderr={err}");
    // With --code each row is a header line + a body line, both mention the
    // caller name; count only the `qualified_name file:line` header rows.
    let header_rows = out
        .lines()
        .filter(|l| l.contains("::Function::caller_"))
        .count();
    assert_eq!(
        header_rows, 6,
        "who-calls --code must cap at 6 rows (bodies are large); got {header_rows}\n{out}"
    );
    assert!(
        out.contains("6 shown of 60 total"),
        "--code capped output must still carry the footer; got: {out}"
    );

    // `--all` lifts the cap: all 60 callers, no footer.
    let (code, out, err) = run(&["who-calls", "hub", "--all"], &repo, &store);
    assert_eq!(code, 0, "who-calls --all should exit 0; stderr={err}");
    let caller_lines = out
        .lines()
        .filter(|l| l.contains("caller_") && !l.starts_with(' '))
        .count();
    assert_eq!(
        caller_lines, 60,
        "--all must print every caller; got {caller_lines}\n{out}"
    );
    assert!(
        !out.contains("shown of"),
        "--all must not print a truncation footer; got: {out}"
    );
    assert!(
        !out.contains("Expand:"),
        "--all already emits the full result set and must not advertise expand; got: {out}"
    );
}

#[test]
fn fan_in_and_fan_out_rank_call_graph_degrees() {
    let (repo, store) = make_hot_symbol_repo("fan-degree");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index must succeed; stderr={err}\nstdout={out}");

    let (code, out, err) = run(&["fan-in", "--limit", "3"], &repo, &store);
    assert_eq!(code, 0, "fan-in should exit 0; stderr={err}");
    let first = out.lines().next().unwrap_or("");
    assert!(
        first.starts_with("60 ") && first.contains("hub"),
        "fan-in must rank hub first with 60 incoming CALLS; got: {out:?}"
    );

    let (code, out, err) = run(&["fan-out", "--limit", "3"], &repo, &store);
    assert_eq!(code, 0, "fan-out should exit 0; stderr={err}");
    let caller_rows = out.lines().filter(|l| l.contains("caller_")).count();
    assert_eq!(
        caller_rows, 3,
        "fan-out --limit 3 must print three caller rows; got: {out:?}"
    );
    assert!(
        out.lines().take(3).all(|l| l.starts_with("1 ")),
        "each hot caller has outgoing degree 1; got: {out:?}"
    );
    assert!(
        out.contains("3 shown of 60 total"),
        "fan-out must carry exact truncation footer; got: {out:?}"
    );
}

#[test]
fn fan_degree_json_reports_exact_counts_and_metadata() {
    let (repo, store) = make_hot_symbol_repo("fan-degree-json");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index must succeed; stderr={err}\nstdout={out}");

    let (code, out, err) = run(&["fan-in", "--limit", "3", "--json"], &repo, &store);
    assert_eq!(code, 0, "fan-in --json should exit 0; stderr={err}");
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid fan-in json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "fan-in");
    assert_eq!(v["project"], "repo");
    assert_eq!(v["scope"], "degree_rank");
    assert_eq!(v["direction"], "incoming");
    assert_eq!(v["edge_type"], "CALLS");
    assert_eq!(v["total_exact"], 1);
    assert_eq!(v["shown"], 1);
    assert_eq!(v["omitted"], 0);
    assert_eq!(v["truncated"], false);
    let hits = v["hits"].as_array().expect("fan-in hits");
    assert_eq!(hits[0]["degree"], 60);
    assert!(hits[0]["qualified_name"]
        .as_str()
        .unwrap_or("")
        .contains("hub"));

    let (code, out, err) = run(&["fan-out", "--limit", "3", "--json"], &repo, &store);
    assert_eq!(code, 0, "fan-out --json should exit 0; stderr={err}");
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid fan-out json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "fan-out");
    assert_eq!(v["scope"], "degree_rank");
    assert_eq!(v["direction"], "outgoing");
    assert_eq!(v["edge_type"], "CALLS");
    assert_eq!(v["requested_limit"], 3);
    assert_eq!(v["limit"], 3);
    assert_eq!(v["total_exact"], 60);
    assert_eq!(v["shown"], 3);
    assert_eq!(v["omitted"], 57);
    assert_eq!(v["truncated"], true);
    let hits = v["hits"].as_array().expect("fan-out hits");
    assert_eq!(hits.len(), 3);
    assert!(hits.iter().all(|hit| hit["degree"] == 1));
}

// ---------------------------------------------------------------------------
// impact — the transitive blast radius in ONE call.
// ---------------------------------------------------------------------------

#[test]
fn impact_incoming_reports_transitive_callers_in_one_call() {
    // hub() is called by 60 functions (all hop 1). `impact hub` must report
    // them as the transitive caller set, capped at NAV_LIMIT (40) + footer —
    // the single-command answer to "what breaks if I change hub?".
    let (repo, store) = make_hot_symbol_repo("impact");
    let (code, _out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index must succeed; stderr={err}");

    let (code, out, err) = run(&["impact", "hub"], &repo, &store);
    assert_eq!(code, 0, "impact should exit 0; stderr={err}");
    let hop_rows = out.lines().filter(|l| l.starts_with("hop ")).count();
    assert_eq!(
        hop_rows, 40,
        "impact must cap at NAV_LIMIT rows; got {hop_rows}\n{out}"
    );
    assert!(
        out.contains("hop 1") && out.contains("caller_"),
        "impact must report transitive callers at their hop distance; got: {out}"
    );
    assert!(
        out.contains("40 shown of 60 total"),
        "impact must carry the truncation footer with the true total; got: {out}"
    );
}

#[test]
fn impact_json_reports_exact_scope_counts_and_metadata() {
    let (repo, store) = make_hot_symbol_repo("impact-json");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index must succeed; stderr={err}\nstdout={out}");

    let (code, out, err) = run(&["impact", "hub", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "impact --json should exit 0; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid impact json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "impact");
    assert_eq!(v["symbol"], "hub");
    assert_eq!(v["project"], "repo");
    assert_eq!(v["symbol_found"], true);
    assert_eq!(v["fresh"], true);
    assert_eq!(v["provider_complete"], false);
    assert_eq!(v["scope"], "transitive");
    assert_eq!(v["direction"], "incoming");
    assert_eq!(v["edge_type"], "all_references");
    assert_eq!(
        v["edge_types"],
        serde_json::json!(["CALLS", "USAGE", "USES", "TYPE_REF", "IMPORTS"])
    );
    assert_eq!(v["max_hops"], 6);
    assert_eq!(v["total_exact"], 60);
    assert_eq!(v["shown"], 40);
    assert_eq!(v["omitted"], 20);
    assert_eq!(v["truncated"], true);
    let hits = v["hits"].as_array().expect("hits array");
    assert_eq!(hits.len(), 40);
    assert!(
        hits.iter().all(|hit| hit["hops"] == 1),
        "all hot callers are direct hop-1 callers: {v:?}"
    );
    assert!(
        hits.iter().all(|hit| hit["qualified_name"]
            .as_str()
            .unwrap_or("")
            .contains("caller_")),
        "impact hits should be caller functions: {v:?}"
    );
}

#[test]
fn impact_json_explicit_calls_edge_is_not_remapped_to_all_references() {
    let (repo, store) = make_hot_symbol_repo("impact-json-explicit-calls");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index must succeed; stderr={err}\nstdout={out}");

    let (code, out, err) = run(
        &["impact", "hub", "--edge", "CALLS", "--json"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "impact --edge CALLS --json should exit 0; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid impact json: {e}; stdout={out:?}"));
    assert_eq!(v["edge_type"], "CALLS");
    assert_eq!(v["edge_types"], serde_json::json!(["CALLS"]));
}

#[test]
fn impact_since_json_maps_changed_hunk_to_symbol_and_transitive_callers() {
    let (repo, store) = make_real_git_diff_impact_repo("impact-since-diff");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index must succeed; stderr={err}\nstdout={out}");

    let (code, out, err) = run(&["impact", "--since", "HEAD~1", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "impact --since --json should exit 0; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid impact diff json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "impact");
    assert_eq!(v["status"], "ok");
    assert_eq!(v["scope"], "diff");
    assert_eq!(v["diff_scope"], "since");
    assert_eq!(v["backend"], "git_diff_graph");
    assert_eq!(v["fresh"], true);
    assert_eq!(v["direction"], "incoming");
    assert_eq!(v["edge_type"], "all_references");
    assert_eq!(
        v["edge_types"],
        serde_json::json!(["CALLS", "USAGE", "USES", "TYPE_REF", "IMPORTS"])
    );
    assert_eq!(v["diff_files_total"], 1);
    assert_eq!(v["source_total"], 1);
    assert_eq!(v["source_symbols"].as_array().unwrap().len(), 1);
    assert!(
        v["source_symbols"][0]["qualified_name"]
            .as_str()
            .unwrap_or("")
            .contains("hub"),
        "changed hunk should map to hub source symbol: {v:?}"
    );
    assert_eq!(v["total_exact"], 3);
    let hits = v["hits"].as_array().expect("impact diff hits");
    assert_eq!(hits.len(), 3);
    assert!(
        hits.iter().all(|hit| hit["qualified_name"]
            .as_str()
            .unwrap_or("")
            .contains("caller_")),
        "impact diff hits should be caller functions: {v:?}"
    );
    assert!(hits.iter().all(|hit| hit["source_count"] == 1));
}

#[test]
fn impact_base_text_and_json_use_merge_base_diff_sources() {
    let (repo, store) = make_real_git_diff_impact_repo("impact-base-diff");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index must succeed; stderr={err}\nstdout={out}");

    let (code, out, err) = run(&["impact", "--base", "basepoint"], &repo, &store);
    assert_eq!(
        code, 0,
        "impact --base text should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("diff sources: 1 shown of 1 total")
            && out.contains("source")
            && out.contains("hub")
            && out.contains("caller_"),
        "impact --base text should show source hub and impacted callers; got: {out}"
    );

    let (code, out, err) = run(&["impact", "--base", "basepoint", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "impact --base --json should exit 0; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid impact base json: {e}; stdout={out:?}"));
    assert_eq!(v["scope"], "diff");
    assert_eq!(v["diff_scope"], "base");
    assert_eq!(v["backend"], "git_diff_graph");
    assert_eq!(v["source_total"], 1);
    assert_eq!(v["total_exact"], 3);
    assert_eq!(v["shown"], 3);
    assert_eq!(v["merge_base"].as_str().unwrap_or("").len(), 40);
    assert_eq!(v["hits"].as_array().unwrap().len(), 3);
}

#[test]
fn brief_bundles_definition_callers_and_callees_in_one_call() {
    // do_it() is defined in helper.rs and called by caller() in lib.rs.
    // `brief do_it` must return, in one call: its definition body, its
    // CALLERS section listing `caller`, and a CALLS section.
    let (repo, store) = index_fixture("brief");
    let (code, out, err) = run(&["brief", "do_it"], &repo, &store);
    assert_eq!(code, 0, "brief should exit 0; stderr={err}");
    assert!(
        out.contains("do_it") && out.contains("pub fn do_it"),
        "brief must show the definition with source; got: {out}"
    );
    assert!(
        out.contains("(src/helper.rs:1-4)"),
        "brief header must report the actual expanded source span; got: {out}"
    );
    let header = out.find("== ").expect("brief output must contain a header");
    let source = out
        .find("    pub fn do_it")
        .expect("brief output must contain indented source");
    let between = &out[header..source];
    assert!(
        !between.lines().any(|line| line.starts_with("  - ")),
        "test-only inference bypass must preserve deterministic brief output; got: {out}"
    );
    assert!(
        out.contains("-- CALLERS") && out.contains("caller"),
        "brief must list callers incl. `caller`; got: {out}"
    );
    assert!(
        out.contains("-- CALLS"),
        "brief must have a CALLS (callees) section; got: {out}"
    );
}

#[test]
fn brief_lists_recursive_function_as_its_own_caller() {
    let root = fresh_dir("brief-recursive");
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(
        src.join("lib.rs"),
        r#"
fn recurse(n: u32) -> u32 {
    if n == 0 {
        0
    } else {
        recurse(n - 1)
    }
}
"#,
    )
    .unwrap();
    let store = root.join("store");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index must succeed; stderr={err}\nstdout={out}");

    let (code, out, err) = run(&["brief", "recurse"], &repo, &store);
    assert_eq!(code, 0, "brief recurse should exit 0; stderr={err}");
    let callers = out
        .split("-- CALLERS")
        .nth(1)
        .and_then(|tail| tail.split("-- CALLS").next())
        .unwrap_or("");
    assert!(
        callers.contains("recurse") && !callers.contains("(no callers)"),
        "brief must report a recursive self-call as a caller, got: {out}"
    );
}

#[test]
fn class_navigation_is_first_class_for_callees_brief_imports_and_search_symbols() {
    let (repo, store) = index_python_class_fixture("python-class-nav");

    let (code, out, err) = run(&["callees", "RunnerFilter"], &repo, &store);
    assert_eq!(code, 0, "callees RunnerFilter should exit 0; stderr={err}");
    assert!(
        out.contains("setup_filter") && out.contains("should_run_check"),
        "class callees must aggregate calls from owned methods/constructor; got: {out}"
    );
    assert!(
        !out.contains("(no callees)"),
        "class callees must not report an empty callable answer; got: {out}"
    );

    let (code, out, err) = run(&["brief", "RunnerFilter"], &repo, &store);
    assert_eq!(code, 0, "brief RunnerFilter should exit 0; stderr={err}");
    assert!(
        out.contains("class RunnerFilter")
            && out.contains("-- CALLERS")
            && out.contains("build_filter")
            && out.contains("-- REFERENCES")
            && out.contains("IMPORTS Module checkov/cloudformation/runner.py")
            && out.contains("-- CALLS")
            && out.contains("setup_filter"),
        "brief on a class must show definition, instantiation callers, references/imports, and member callees; got: {out}"
    );
    assert!(
        !out.contains("__file__"),
        "brief must not leak synthetic file qnames; got: {out}"
    );

    let (code, out, err) = run(&["find-usages", "RunnerFilter"], &repo, &store);
    assert_eq!(
        code, 0,
        "find-usages RunnerFilter should exit 0; stderr={err}"
    );
    assert!(
        out.contains("CALLS")
            && out.contains("build_filter")
            && out.contains("IMPORTS Module checkov/cloudformation/runner.py"),
        "find-usages on a class must include constructor calls and import dependents; got: {out}"
    );
    assert!(
        !out.contains("__file__"),
        "find-usages must not leak synthetic file qnames; got: {out}"
    );

    let (code, out, err) = run(&["impact", "RunnerFilter"], &repo, &store);
    assert_eq!(code, 0, "impact RunnerFilter should exit 0; stderr={err}");
    assert!(
        out.contains("build_filter") && out.contains("Module checkov/cloudformation/runner.py"),
        "impact on a class must include CALLS and IMPORTS dependents; got: {out}"
    );
    assert!(
        !out.contains("__file__"),
        "impact must not leak synthetic file qnames; got: {out}"
    );

    let (code, out, err) = run(&["search-symbols", "RunnerFilter"], &repo, &store);
    assert_eq!(
        code, 0,
        "search-symbols RunnerFilter should exit 0; stderr={err}"
    );
    assert!(
        out.contains("Class") && out.contains("RunnerFilter"),
        "search-symbols must still find the class definition; got: {out}"
    );
    assert!(
        !out.contains("__file__") && !out.contains(":0") && !out.contains("File "),
        "search-symbols must suppress synthetic file anchors; got: {out}"
    );
}

#[test]
fn impact_outgoing_from_a_caller_reaches_the_hub() {
    let (repo, store) = make_hot_symbol_repo("impact-out");
    let (code, _o, _e) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0);
    // caller_0_0 calls hub() → outgoing impact must reach hub.
    let (code, out, err) = run(
        &["impact", "caller_0_0", "--direction", "outgoing"],
        &repo,
        &store,
    );
    assert_eq!(code, 0, "impact outgoing should exit 0; stderr={err}");
    assert!(
        out.contains("hub"),
        "outgoing impact from caller_0_0 must reach hub; got: {out}"
    );
}
