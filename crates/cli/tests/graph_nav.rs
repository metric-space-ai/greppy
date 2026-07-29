//! Integration tests for the Track-A graph-navigation commands:
//! `who-calls` and the extended `trace` (incoming
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
    // An answer line is an address and a name, nothing else: the call site the
    // agent acts on, and who makes the call.
    let caller_line = out
        .lines()
        .find(|line| line.contains("src/lib.rs:"))
        .unwrap_or_else(|| panic!("missing caller line in stdout: {out:?}"));
    let mut fields = caller_line.split_whitespace();
    let address = fields.next().unwrap_or_default();
    assert!(
        address.starts_with("src/lib.rs:") && !address.contains('-'),
        "the address names the call site, not a span, got: {caller_line:?}"
    );
    assert!(
        fields.next().is_some_and(|name| !name.contains("::")
            && !name.contains("src/")
            && !name.contains("Function")),
        "the name is the bare symbol, with no path and no kind, got: {caller_line:?}"
    );
    // Nothing is hidden here, so nothing is counted and nothing is offered.
    assert!(
        !out.contains("Expand:") && !out.contains(" callers:"),
        "a complete answer carries no count and no expand offer, got: {out:?}"
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
fn who_calls_lists_usage_references_into_a_struct() {
    let (repo, store) = index_fixture("whocalls-struct-usage");

    let (code, out, err) = run(&["who-calls", "Widget"], &repo, &store);
    assert_eq!(
        code, 0,
        "who-calls on a referenced struct should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("src/lib.rs:") && out.contains("render"),
        "who-calls Widget must print its USAGE referrer in the normal row format; got: {out:?}"
    );
    assert!(
        !out.contains("not a function") && !out.contains("no callers"),
        "a referenced struct must produce rows, not a refusal or empty answer; got: {out:?}"
    );
}

#[test]
fn who_calls_covers_calls_and_usages_without_naming_the_edge() {
    let (repo, store) = index_fixture("references");

    // A call site and a type reference are both "places that use S". The edge
    // kind is how the graph stores them, not something the agent acts on, so it
    // is not printed -- the answer is the address and the symbol.
    let (code, out, err) = run(&["who-calls", "do_it"], &repo, &store);
    assert_eq!(code, 0, "who-calls should exit 0; stderr={err}\nstdout={out}");
    assert!(
        out.contains("caller") && !out.contains("CALLS"),
        "the caller is named, the edge kind is not; got: {out:?}"
    );

    let (code, out, err) = run(&["who-calls", "Widget"], &repo, &store);
    assert_eq!(code, 0, "who-calls should exit 0; stderr={err}\nstdout={out}");
    assert!(
        out.contains("render") && !out.contains("USAGE"),
        "a struct's referrers are listed the same way as callers; got: {out:?}"
    );
}

#[test]
fn direct_navigation_json_reports_exact_counts() {
    let (repo, store) = index_fixture("nav-json");

    // The edge kind is no longer part of any answer, so no case asserts one.
    let cases: [(&str, &str, &str, Option<&str>); 3] = [
        ("who-calls", "do_it", "caller", None),
        ("callees", "caller", "do_it", None),
        ("who-calls", "Widget", "render", None),
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

/// Small drift heals in-band (1b7135b): the triggering request reindexes and
/// serves the POST-drift truth with `fresh: true` — never the old generation.
#[test]
fn direct_navigation_json_heals_small_stale_drift_in_band() {
    let (repo, store) = index_fixture("nav-json-stale");
    std::fs::write(
        repo.join("src/lib.rs"),
        r#"
mod helper;
mod types;

fn caller() {
    helper::do_it();
}

fn caller_added_after_index() {
    helper::do_it();
}

fn render(w: types::Widget) -> u32 { w.w + 1 }
"#,
    )
    .unwrap();

    let (code, out, err) = run(&["who-calls", "do_it", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "healable small drift must serve fresh results; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid nav json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "who-calls");
    assert_eq!(
        v["fresh"], true,
        "the healed response must prove freshness: {v:?}"
    );
    let hits = v["hits"].as_array().expect("hits array");
    assert!(
        hits.iter().any(|h| {
            h["qualified_name"]
                .as_str()
                .unwrap_or_default()
                .contains("caller_added_after_index")
        }),
        "healed graph must contain the post-drift caller: {v:?}"
    );
}

/// Multi-file drift below the large-drift threshold heals in the same way:
/// one request reindexes BOTH changed files and serves post-drift truth from
/// each of them — never a mix of generations.
#[test]
fn direct_navigation_json_heals_multi_file_stale_drift_in_band() {
    let (repo, store) = index_fixture("nav-json-stale-multi");
    std::fs::write(
        repo.join("src/lib.rs"),
        r#"
mod helper;
mod types;

fn caller() {
    helper::do_it();
}

fn second_caller_added_after_index() {
    helper::do_it();
    helper::late_helper();
}

fn render(w: types::Widget) -> u32 { w.w + 1 }
"#,
    )
    .unwrap();
    std::fs::write(
        repo.join("src/helper.rs"),
        "pub fn do_it() -> u32 { 42 }\npub fn late_helper() -> u32 { 7 }\n",
    )
    .unwrap();

    let (code, out, err) = run(&["who-calls", "do_it", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "healable multi-file drift must serve fresh results; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid nav json: {e}; stdout={out:?}"));
    assert_eq!(
        v["fresh"], true,
        "the healed response must prove freshness: {v:?}"
    );
    let hits = v["hits"].as_array().expect("hits array");
    assert!(
        hits.iter().any(|h| {
            h["qualified_name"]
                .as_str()
                .unwrap_or_default()
                .contains("second_caller_added_after_index")
        }),
        "healed graph must contain the caller added in lib.rs: {v:?}"
    );

    // The same healed generation must also know the symbol added in the
    // SECOND drifted file, with its post-drift caller.
    let (code, out, err) = run(&["who-calls", "late_helper", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "the healed generation must resolve the symbol added in helper.rs; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid nav json: {e}; stdout={out:?}"));
    assert_eq!(v["fresh"], true, "second lookup must stay fresh: {v:?}");
    let hits = v["hits"].as_array().expect("hits array");
    assert!(
        hits.iter().any(|h| {
            h["qualified_name"]
                .as_str()
                .unwrap_or_default()
                .contains("second_caller_added_after_index")
        }),
        "late_helper must be called by the post-drift caller: {v:?}"
    );
}

/// Large drift starts a background refresh and fails closed. No command may
/// expose rows from the old generation while that refresh is in flight.
fn large_stale_graph_fixture(tag: &str) -> (PathBuf, PathBuf) {
    let (repo, store) = index_fixture(tag);
    std::fs::write(
        repo.join("src/helper.rs"),
        "pub fn do_it_renamed() -> u32 { 42 }\n",
    )
    .unwrap();
    for i in 0..11 {
        std::fs::write(
            repo.join(format!("src/extra_{i}.rs")),
            format!("pub fn extra_{i}() -> u32 {{ {i} }}\n"),
        )
        .unwrap();
    }
    (repo, store)
}

#[test]
fn graph_commands_refuse_rows_when_heal_budget_is_exhausted() {
    let (repo, store) = large_stale_graph_fixture("graph-stale-gate-brief");

    let (code, out, err) = run_with_env(
        &["brief", "do_it"],
        &repo,
        &store,
        &[("GREPPY_INDEX_TIME_BUDGET_MS", "0")],
    );
    assert_eq!(
        code, 75,
        "refreshing brief must return EX_TEMPFAIL; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "brief freshness refusal must stay on stdout; stderr={err:?}"
    );
    assert!(
        out.contains("graph freshness is refreshing")
            && out.contains("no stale indexed hits emitted")
            && !out.contains("== do_it"),
        "refreshing brief must explain the refusal without old evidence; got: {out:?}"
    );

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
    for (case, (args, command, collection_field)) in json_cases.into_iter().enumerate() {
        let (repo, store) = large_stale_graph_fixture(&format!("graph-stale-gate-{case}"));
        let (code, out, err) =
            run_with_env(&args, &repo, &store, &[("GREPPY_INDEX_TIME_BUDGET_MS", "0")]);
        assert_eq!(
            code, 75,
            "refreshing {command} must return EX_TEMPFAIL; stderr={err}\nstdout={out}"
        );
        assert!(
            err.is_empty(),
            "JSON freshness refusal must stay on stdout; stderr={err:?}"
        );
        let v: serde_json::Value = serde_json::from_str(&out)
            .unwrap_or_else(|e| panic!("invalid refreshing {command} json: {e}; stdout={out:?}"));
        assert_eq!(v["command"], command);
        assert_eq!(
            v["status"], "skipped_stale_index",
            "refreshing {command} must be skipped: {v:?}"
        );
        assert_eq!(
            v["fresh"], false,
            "{command} must label the result stale: {v:?}"
        );
        assert_eq!(v["freshness"]["state"], "refreshing");
        assert_eq!(
            v["freshness"]["stale_file_count"], 12,
            "{command} must report the drift extent: {v:?}"
        );
        assert!(
            v[collection_field].as_array().unwrap().is_empty(),
            "refreshing {command} must not serve rows from the old index: {v:?}"
        );
    }
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

    let (code, out, err) = run(&["search-symbol", "Widget"], &repo, &store);
    assert_eq!(
        code, 0,
        "search-symbol should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("src/types.rs:") && out.contains("Widget"),
        "search-symbol prints file:line and the name; got: {out:?}"
    );
    assert!(
        // The kind is the declaring keyword in the source, in the language's
        // own words — a Rust struct is a struct, not the index label `Class`.
        out.contains("struct") && !out.contains("Class"),
        "search-symbol prints the source kind; got: {out:?}"
    );
}

#[test]
fn search_symbols_json_reports_exact_counts_and_metadata() {
    let (repo, store) = index_fixture("symbols-json");

    let (code, out, err) = run(&["search-symbol", "Widget", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "search-symbol --json should exit 0; stderr={err}\nstdout={out}"
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

/// A one-file edit heals on the triggering symbol search, and a read issued
/// while a refresh holds the writer lock waits for that fresh snapshot
/// instead of returning an empty `refreshing` payload.
#[test]
fn symbol_queries_heal_single_file_edits_and_wait_for_edit_refresh() {
    let (repo, store) = index_fixture("symbols-json-stale");
    std::fs::write(
        repo.join("src/types.rs"),
        "pub struct WidgetRenamed { pub w: u32 }\npub struct Marker;\n",
    )
    .unwrap();

    let (code, out, err) = run(
        &["search-symbol", "WidgetRenamed", "--json"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "search-symbol must heal one-file drift in-band; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "search-symbols");
    assert_eq!(v["status"], "ok");
    assert_eq!(v["fresh"], true);
    assert!(
        v["hits"].as_array().is_some_and(|hits| hits.iter().any(|hit| {
            hit["qualified_name"]
                .as_str()
                .is_some_and(|name| name.contains("WidgetRenamed"))
        })),
        "healed symbol search must contain the edited definition: {v:?}"
    );

    let (repo, store) = index_fixture("read-waits-for-edit-refresh");
    // An edit leaves the indexed graph one file behind...
    std::fs::write(
        repo.join("src/helper.rs"),
        "pub fn do_it() -> u32 {\n    let answer = 84;\n    answer\n}\n",
    )
    .unwrap();
    // ...and the refresh that heals it pauses just before publishing, holding
    // the writer lock, so the read below meets an active writer.
    let ready = repo.parent().unwrap().join("edit-index-ready");
    let mut index = Command::new(bin());
    index
        .args(["index", "."])
        .current_dir(&repo)
        .env("GREPPY_STORE_DIR", &store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .env("GREPPY_TEST_INDEX_FAILPOINT", "after-temp-before-publish")
        .env("GREPPY_TEST_INDEX_FAILPOINT_READY", &ready)
        .env("GREPPY_TEST_INDEX_FAILPOINT_HOLD_MS", "5000")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let child = index.spawn().expect("spawn indexer with held refresh");
    for _ in 0..1500 {
        if ready.exists() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        ready.exists(),
        "indexer did not reach the publication failpoint"
    );

    let (code, out, err) = run(&["read", "do_it", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "read must wait for the edit-owned refresh; stderr={err}\nstdout={out}"
    );
    assert!(
        err.contains("graph refresh already running") && err.contains("ETA unavailable"),
        "lock wait must be explicit to the caller; stderr={err:?}"
    );
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid read json: {e}; stdout={out:?}"));
    assert_eq!(v["status"], "ok");
    assert!(
        v["source"]
            .as_str()
            .is_some_and(|source| source.contains("answer = 84")),
        "read must return the post-edit definition: {v:?}"
    );

    let index_out = child.wait_with_output().expect("wait for indexer");
    assert!(
        index_out.status.success(),
        "indexer failed\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&index_out.stdout),
        String::from_utf8_lossy(&index_out.stderr)
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
        (vec!["who-calls", "Widget", "--json"], "who-calls", "hits"),
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
        out.contains("match=enclosing"),
        "enclosing-body match must be explicit in text output; got: {out:?}"
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
        vec!["who-calls", "does_not_exist_xyz"],
        vec!["trace", "--symbol", "does_not_exist_xyz"],
    ] {
        let (code, out, _err) = run(&cmd, &repo, &store);
        assert_eq!(
            code, 1,
            "missing symbol must exit 1 for {cmd:?}; got out={out:?}"
        );
        assert!(
            out.contains("symbol not found") || out.contains("no symbol `"),
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

    // Past 25 rows the shape leads and five rows follow as an example: on a
    // long result the distribution over files is the answer, not the roll call.
    let (code, out, err) = run(&["who-calls", "hub"], &repo, &store);
    assert_eq!(code, 0, "who-calls should exit 0; stderr={err}");
    let caller_lines = out
        .lines()
        .filter(|l| l.contains("caller_") && !l.starts_with(' '))
        .count();
    assert_eq!(
        caller_lines, 5,
        "a long result shows five rows below the shape line; got {caller_lines}\n{out}"
    );
    let shape = out.lines().next().unwrap_or_default();
    assert!(
        shape.starts_with("60 callers: ") && shape.contains(".rs "),
        "the first line states how many and across which files; got: {shape:?}"
    );
    assert!(
        !out.contains("shown of") && !out.contains("--all"),
        "the count says something is missing; the output never argues for a flag: {out}"
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
    // The answer is a tree: one row per reached caller, indentation is the
    // route. No `hop N` prefixes, no truncation footer, no expand offer.
    let rows = out.lines().filter(|l| l.contains("caller_")).count();
    assert_eq!(rows, 60, "impact reaches all 60 callers; got {rows}\n{out}");
    assert!(
        !out.contains("hop ") && !out.contains("shown of") && !out.contains("Expand:"),
        "the flat hop list is gone; got: {out}"
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



#[test]
fn impact_refuses_the_retired_diff_flags() {
    // --since/--base died with the ORIENT dissolution; the prompt's impact
    // signature is `impact S [--depth N] [--direction outgoing]`. The retired
    // flags are a usage error, not a silent alias.
    let (repo, store) = make_real_git_diff_impact_repo("impact-retired-flags");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index must succeed; stderr={err}\nstdout={out}");

    for flag in [&["impact", "--since", "HEAD~1"][..], &["impact", "--base", "main"][..]] {
        let (code, out, err) = run(flag, &repo, &store);
        assert_eq!(code, 64, "retired flag must be a usage error; stdout={out} stderr={err}");
        assert!(
            out.contains("unexpected argument") || err.contains("unexpected argument"),
            "the refusal names the unexpected argument; stdout={out} stderr={err}"
        );
    }
}

#[test]
fn brief_sketches_the_body_instead_of_bundling_three_commands() {
    // caller() is defined in lib.rs and calls do_it() in helper.rs.
    // `brief caller` prints the sentence-less head verbatim and a sketch
    // naming the one call — not the old CALLERS/REFERENCES/CALLS bundle.
    let (repo, store) = index_fixture("brief");
    let (code, out, err) = run(&["brief", "caller"], &repo, &store);
    assert_eq!(code, 0, "brief should exit 0; stderr={err}\nstdout={out}");
    assert!(
        out.contains("src/lib.rs:5\nfn caller() {"),
        "brief must print the head address and the verbatim signature; got: {out}"
    );
    assert!(
        out.contains("do_it") && out.contains('}'),
        "brief's sketch must name the callee and close the block; got: {out}"
    );
    assert!(
        !out.contains("-- CALLERS")
            && !out.contains("-- REFERENCES")
            && !out.contains("-- CALLS")
            && !out.contains("== "),
        "brief must not bundle who-calls/callees behind ASCII bars; got: {out}"
    );

    // do_it calls nothing: the sketch is empty and the aggregated callers
    // line carries the one caller.
    let (code, out, err) = run(&["brief", "do_it"], &repo, &store);
    assert_eq!(code, 0, "brief should exit 0; stderr={err}\nstdout={out}");
    assert!(
        out.contains("pub fn do_it() -> u32 {"),
        "brief must show the verbatim head; got: {out}"
    );
    assert!(
        out.contains("called by caller"),
        "brief must aggregate callers into one line; got: {out}"
    );
    assert!(
        !out.contains("-- CALLERS") && !out.contains("-- CALLS"),
        "brief must not print section bars; got: {out}"
    );
}

#[test]
fn brief_sketches_match_branches_as_the_parser_sees_them() {
    let root = fresh_dir("brief-match");
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(
        src.join("lib.rs"),
        r#"fn classify(x: u32) -> u32 {
    match x {
        0 => zero(),
        _ => one(),
    }
}
fn only(spans: Vec<u32>) -> u32 {
    match spans.as_slice() {
        [] => zero(),
        [x] => one(),
        many => zero(),
    }
}
fn zero() -> u32 { 0 }
fn one() -> u32 { 1 }
"#,
    )
    .unwrap();
    let store = root.join("store");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index must succeed; stderr={err}\nstdout={out}");

    let (code, out, err) = run(&["brief", "classify"], &repo, &store);
    assert_eq!(code, 0, "brief classify should exit 0; stderr={err}");
    assert!(
        out.contains("match x"),
        "brief must sketch the match branch; got: {out}"
    );
    assert!(
        out.contains("0 — zero") && out.contains("else — one"),
        "brief must fold each arm's call into the arm's line; got: {out}"
    );

    let (code, out, err) = run(&["brief", "only"], &repo, &store);
    assert_eq!(code, 0, "brief only should exit 0; stderr={err}");
    assert!(
        out.contains("match spans.as_slice()"),
        "brief must sketch the slice match with its scrutinee; got: {out}"
    );
    assert!(
        out.contains("[] — zero") && out.contains("[x] — one") && out.contains("many — zero"),
        "slice and binding patterns keep their real shape; got: {out}"
    );
}

#[test]
fn brief_prints_a_struct_as_its_whole_definition_without_a_sketch() {
    let (repo, store) = index_fixture("brief-struct");
    let (code, out, err) = run(&["brief", "Widget"], &repo, &store);
    assert_eq!(code, 0, "brief Widget should exit 0; stderr={err}");
    assert!(
        out.contains("src/types.rs:1\npub struct Widget { pub w: u32 }"),
        "a struct's fields are its interface: whole definition, no sketch; got: {out}"
    );
}

#[test]
fn brief_refuses_an_ambiguous_name_with_addresses_not_bodies() {
    let root = fresh_dir("brief-ambiguous");
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(src.join("lib.rs"), "mod a;\nmod b;\n").unwrap();
    std::fs::write(src.join("a.rs"), "pub fn dup() -> u32 { 1 }\n").unwrap();
    std::fs::write(src.join("b.rs"), "pub fn dup() -> u32 { 2 }\n").unwrap();
    let store = root.join("store");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index must succeed; stderr={err}\nstdout={out}");

    let (code, out, err) = run(&["brief", "dup"], &repo, &store);
    assert_eq!(code, 1, "brief dup must refuse; stderr={err}\nstdout={out}");
    assert!(
        out.contains("`dup` is 2 definitions")
            && out.contains("src/a.rs:1")
            && out.contains("src/b.rs:1"),
        "brief must refuse with one address per definition; got: {out}"
    );
    assert!(
        !out.contains("pub fn dup"),
        "brief must not print the bodies of every definition; got: {out}"
    );
}

#[test]
fn brief_reports_a_missing_symbol() {
    let (repo, store) = index_fixture("brief-missing");
    let (code, out, err) = run(&["brief", "no_such_symbol"], &repo, &store);
    assert_eq!(code, 1, "brief must exit 1 for an unknown name; stderr={err}");
    assert!(
        out.contains("no symbol `no_such_symbol`"),
        "brief must say the name does not resolve; got: {out}"
    );
}

#[test]
fn brief_refuses_a_second_symbol() {
    // 0.3.0: brief sketches ONE body. The legacy `brief A B` bundle
    // (`== A ==` + full source) is gone — a second positional is a usage
    // error, like any malformed invocation.
    let (repo, store) = index_fixture("brief-two-symbols");
    let (code, out, err) = run(&["brief", "caller", "do_it"], &repo, &store);
    assert_eq!(code, 64, "a second symbol must be a usage error; stderr={err}");
    assert!(
        !out.contains("== caller ==") && !out.contains("== do_it =="),
        "the multi-symbol bundle must never print; got: {out}"
    );
}

#[test]
fn brief_refuses_a_name_that_has_nothing_to_sketch() {
    let root = fresh_dir("brief-const");
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(
        src.join("lib.rs"),
        "pub const LIMIT: u32 = 10;\npub fn within(x: u32) -> bool { x < LIMIT }\n",
    )
    .unwrap();
    let store = root.join("store");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index must succeed; stderr={err}\nstdout={out}");

    let (code, out, err) = run(&["brief", "LIMIT"], &repo, &store);
    assert_eq!(code, 1, "brief LIMIT must refuse; stderr={err}\nstdout={out}");
    assert!(
        out.contains("`LIMIT` is a ") && out.contains("not a function"),
        "brief must print the kind line for a name with no body; got: {out}"
    );
}

#[test]
fn brief_offers_the_call_tree_below_as_an_expand_pack() {
    let root = fresh_dir("brief-expand");
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(
        src.join("lib.rs"),
        "fn hub() { a(); b(); }\nfn a() {}\nfn b() {}\n",
    )
    .unwrap();
    let store = root.join("store");
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index must succeed; stderr={err}\nstdout={out}");

    let (code, out, err) = run(&["brief", "hub"], &repo, &store);
    assert_eq!(code, 0, "brief hub should exit 0; stderr={err}");
    let offer = out
        .lines()
        .find(|line| line.starts_with("expand "))
        .unwrap_or_else(|| panic!("brief must offer the call tree pack; got: {out}"));
    assert!(
        offer.contains("the call tree below hub sketched, 2 functions,"),
        "the offer must state the function and line counts; got: {offer}"
    );
    let id = offer
        .split_whitespace()
        .nth(1)
        .expect("the offer carries an expand id");
    let (code, out, err) = run(&["expand", id], &repo, &store);
    assert_eq!(code, 0, "expand should exit 0; stderr={err}");
    assert!(
        out.contains("fn a() {}") && out.contains("fn b() {}"),
        "the pack must sketch every function below hub; got: {out}"
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
    assert!(
        out.contains("called by recurse"),
        "brief must report a recursive self-call as a caller, got: {out}"
    );
    assert!(
        out.contains("  recurse\n"),
        "brief's sketch must name the recursive call, got: {out}"
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
        out.contains("class RunnerFilter:")
            && out.contains("def __init__(self):")
            && !out.contains("-- CALLERS")
            && !out.contains("-- REFERENCES")
            && !out.contains("-- CALLS"),
        "brief on a class must print the whole definition, no sketch, no bars; got: {out}"
    );
    assert!(
        !out.contains("__file__"),
        "brief must not leak synthetic file qnames; got: {out}"
    );

    let (code, out, err) = run(&["impact", "RunnerFilter"], &repo, &store);
    assert_eq!(code, 0, "impact RunnerFilter should exit 0; stderr={err}");
    // The tree lists the functions a change reaches. Module rows were file
    // anchors wearing a kind — bookkeeping, not symbols an agent can act on —
    // and they are filtered like `__file__` everywhere else.
    assert!(
        out.contains("build_filter") && out.contains("use_filter") && !out.contains("Module "),
        "impact on a class reaches its instantiating functions; got: {out}"
    );
    assert!(
        !out.contains("__file__"),
        "impact must not leak synthetic file qnames; got: {out}"
    );

    let (code, out, err) = run(&["search-symbol", "RunnerFilter"], &repo, &store);
    assert_eq!(
        code, 0,
        "search-symbol RunnerFilter should exit 0; stderr={err}"
    );
    assert!(
        out.contains("RunnerFilter") && out.contains("class"),
        "search-symbol finds the class with its source kind; got: {out}"
    );
    assert!(
        !out.contains("__file__") && !out.contains(":0") && !out.contains("File "),
        "search-symbol must suppress synthetic file anchors; got: {out}"
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
