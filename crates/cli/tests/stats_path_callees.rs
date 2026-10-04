//! Integration tests for the Track-B navigation/stats commands:
//! `stats`, `callees`, and `path`.
//!
//! These spawn the real `greppy` binary against a multi-file fixture
//! indexed end-to-end, so the cross-file CALLS edges resolved by the
//! indexer/resolver are exercised exactly as an agent would see them.
//! Each test gets an isolated `GREPPY_STORE_DIR` so parallel runs
//! never collide.
//!
//! The fixture wires a deterministic call chain across three files so a
//! path query has a real multi-hop answer:
//!
//! * `entry()`  --CALLS--> `middle()`   (lib.rs -> mid.rs)
//! * `middle()` --CALLS--> `leaf()`     (mid.rs -> leaf.rs)
//!
//! so `path --from entry --to leaf` is `entry -> middle -> leaf`, and
//! `callees entry` yields `middle`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn mark_call_provider_unavailable(store_dir: &Path) {
    fn graph_db(dir: &Path) -> Option<PathBuf> {
        for entry in std::fs::read_dir(dir).ok()?.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if let Some(db) = graph_db(&path) {
                    return Some(db);
                }
            } else if path.file_name().and_then(|name| name.to_str()) == Some("graph.db") {
                return Some(path);
            }
        }
        None
    }
    let db = graph_db(store_dir).expect("fixture graph.db");
    let store = greppy_store::Store::open(&db).expect("open provider fixture");
    let changed = store
        .conn()
        .execute(
            "UPDATE provider_state SET status = 'partial',
             supported_edge_classes = '[\"definitions\",\"usages\"]',
             unsupported_edge_classes = '[\"calls\"]', files_failed = 0
             WHERE project = 'repo' AND language = 'rust'",
            [],
        )
        .expect("mark requested call relation unavailable");
    assert_eq!(changed, 1, "fixture must contain one Rust provider row");
}

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fresh_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("greppy-cli-statspath-{tag}-{pid}-{n}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Build a git-rooted repo whose three modules form a CALLS chain
/// `entry -> middle -> leaf`. Returns (repo_root, store_dir).
fn make_chain_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).unwrap();
    // `.git` is the repo-root marker resolve_root walks up to find.
    std::fs::create_dir_all(repo.join(".git")).unwrap();

    std::fs::write(
        src.join("lib.rs"),
        r#"
mod mid;
mod leaf;

fn entry() {
    mid::middle();
}
"#,
    )
    .unwrap();

    std::fs::write(
        src.join("mid.rs"),
        r#"
use crate::leaf;

pub fn middle() {
    leaf::leaf();
}
"#,
    )
    .unwrap();

    std::fs::write(src.join("leaf.rs"), "pub fn leaf() -> u32 { 7 }\n").unwrap();

    let store = root.join("store");
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

/// Index the fixture once and assert it succeeded; shared setup.
fn index_fixture(tag: &str) -> (PathBuf, PathBuf) {
    let (repo, store) = make_chain_repo(tag);
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );
    (repo, store)
}

/// Parse a `stats` line like "  Function 3" into (key, count). Returns
/// None for non-count lines.
fn parse_count(line: &str, key: &str) -> Option<i64> {
    let line = line.trim();
    let rest = line.strip_prefix(key)?.trim();
    rest.parse::<i64>().ok()
}

// ---------------------------------------------------------------------------
// stats — file/node/edge counts and totals, deterministic + human-readable.
// ---------------------------------------------------------------------------

#[test]
fn stats_reports_files_nodes_edges_and_totals() {
    let (repo, store) = index_fixture("stats");

    let (code, out, err) = run(&["stats"], &repo, &store);
    assert_eq!(code, 0, "stats should exit 0; stderr={err}\nstdout={out}");

    // Header lines present.
    assert!(
        out.contains("project: "),
        "stats must print the project identity; got: {out:?}"
    );

    // Three source files were indexed.
    let files = out
        .lines()
        .find_map(|l| {
            l.strip_prefix("files: ")
                .and_then(|n| n.trim().parse::<i64>().ok())
        })
        .expect("stats must print a `files: N` line");
    assert_eq!(files, 3, "fixture has 3 source files; got {files}\n{out}");

    // Totals are present and positive.
    let nodes = out
        .lines()
        .find_map(|l| {
            l.strip_prefix("nodes: ")
                .and_then(|n| n.trim().parse::<i64>().ok())
        })
        .expect("stats must print a `nodes: N` line");
    let edges = out
        .lines()
        .find_map(|l| {
            l.strip_prefix("edges: ")
                .and_then(|n| n.trim().parse::<i64>().ok())
        })
        .expect("stats must print an `edges: N` line");
    assert!(
        nodes > 0,
        "indexed graph must have nodes; got {nodes}\n{out}"
    );
    assert!(
        edges > 0,
        "indexed graph must have edges; got {edges}\n{out}"
    );

    // The three Function definitions (entry/middle/leaf) are counted by
    // label, and the per-label counts sum to the node total.
    let fn_count = out
        .lines()
        .find_map(|l| parse_count(l, "Function"))
        .expect("stats must list a Function node-count line");
    assert!(
        fn_count >= 3,
        "entry/middle/leaf are 3 Functions; got {fn_count}\n{out}"
    );

    // CALLS edges are present (entry->middle, middle->leaf).
    let calls = out
        .lines()
        .find_map(|l| parse_count(l, "CALLS"))
        .expect("stats must list a CALLS edge-count line");
    assert!(calls >= 2, "chain has >=2 CALLS edges; got {calls}\n{out}");

    // Determinism: a second run produces byte-identical output.
    let (code2, out2, _err2) = run(&["stats"], &repo, &store);
    assert_eq!(code2, 0);
    assert_eq!(out, out2, "stats output must be deterministic across runs");
}

#[test]
fn stats_per_label_counts_sum_to_node_total() {
    let (repo, store) = index_fixture("stats-sum");
    let (code, out, _err) = run(&["stats"], &repo, &store);
    assert_eq!(code, 0);

    let total_nodes = out
        .lines()
        .find_map(|l| {
            l.strip_prefix("nodes: ")
                .and_then(|n| n.trim().parse::<i64>().ok())
        })
        .expect("nodes total");

    // Sum the indented per-label lines that appear after `nodes:` and
    // before `edges:`.
    let mut in_nodes = false;
    let mut sum = 0i64;
    for l in out.lines() {
        if l.starts_with("nodes: ") {
            in_nodes = true;
            continue;
        }
        if l.starts_with("edges: ") {
            break;
        }
        if in_nodes {
            // "  Label N"
            if let Some(n) = l
                .trim()
                .rsplit(' ')
                .next()
                .and_then(|n| n.parse::<i64>().ok())
            {
                sum += n;
            }
        }
    }
    assert_eq!(
        sum, total_nodes,
        "per-label node counts must sum to the node total; got sum={sum} total={total_nodes}\n{out}"
    );
}

// ---------------------------------------------------------------------------
// callees — outgoing CALLS from S, resolved to file:line.
// ---------------------------------------------------------------------------

#[test]
fn callees_lists_what_symbol_calls() {
    let (repo, store) = index_fixture("callees");

    // `entry` calls `middle` (cross-file CALLS into mid.rs).
    let (code, out, err) = run(&["callees", "entry"], &repo, &store);
    assert_eq!(code, 0, "callees should exit 0; stderr={err}\nstdout={out}");
    assert!(
        out.contains("middle"),
        "callees entry must list `middle`; got: {out:?}"
    );
    assert!(
        out.contains("src/mid.rs:"),
        "callees must print the callee's file:line (src/mid.rs); got: {out:?}"
    );
    assert!(
        !out.contains("no resolved indexed callees"),
        "entry calls middle, so callees must be non-empty; got: {out:?}"
    );
}

#[test]
fn callees_reports_no_indexed_callees_for_leaf() {
    let (repo, store) = index_fixture("callees-none");
    // Even for a true leaf, an empty graph result is not a completeness proof.
    let (code, out, _err) = run(&["callees", "leaf"], &repo, &store);
    assert_eq!(code, 0);
    assert_eq!(
        out,
        "no resolved indexed callees; external or unresolved calls may still exist\n\
inspect source with: greppy read leaf\n",
        "an empty result must state the indexed scope and offer source inspection"
    );
}

#[test]
fn callees_does_not_claim_external_calls_are_absent() {
    let (repo, store) = make_chain_repo("callees-external");
    std::fs::write(
        repo.join("src/leaf.rs"),
        "pub fn leaf() -> u32 { std::process::id() }\n",
    )
    .unwrap();
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index failed; stderr={err}\nstdout={out}");

    let (code, out, err) = run(&["callees", "leaf"], &repo, &store);
    assert_eq!(code, 0, "callees failed; stderr={err}\nstdout={out}");
    assert_eq!(
        out,
        "no resolved indexed callees; external or unresolved calls may still exist\n\
inspect source with: greppy read leaf\n",
        "the unindexed standard-library call must not become a definitive no-calls claim"
    );
}

#[test]
fn callees_reports_when_path_filter_excludes_known_call() {
    let (repo, store) = index_fixture("callees-filtered");
    // entry calls middle in mid.rs, which the leaf.rs filter excludes.
    let (code, out, err) = run(
        &["callees", "entry", "--path", "src/leaf.rs"],
        &repo,
        &store,
    );
    assert_eq!(code, 0, "callees failed; stderr={err}\nstdout={out}");
    assert_eq!(
        out,
        "no resolved indexed callees under path filter: src/leaf.rs\n\
external, unresolved or filtered calls may still exist; inspect source with: greppy read entry\n",
        "filtering out a known call must preserve the filter and uncertainty in the answer"
    );
}

#[test]
fn callees_exposes_rust_macro_token_limit_in_single_and_batch_answers() {
    let (repo, store) = make_chain_repo("callees-macro-limit");
    std::fs::write(
        repo.join("src/leaf.rs"),
        "pub(super) fn helper() -> u32 { 1 }\n\
         pub fn direct() { helper(); }\n\
         mod tests { use super::*; fn assertion() { assert_eq!(helper(), 1); } }\n",
    )
    .unwrap();
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index failed; stderr={err}\nstdout={out}");
    for args in [
        vec!["callees", "assertion", "--code"],
        vec!["callees", "assertion", "direct"],
    ] {
        let (code, out, err) = run(&args, &repo, &store);
        assert_eq!(code, 0, "callees failed; stderr={err}\nstdout={out}");
        assert!(out.contains("Rust macro argument tokens (including assert_eq!)"));
        assert!(out.contains("greppy read assertion"));
        assert!(out.contains("no resolved indexed callees"));
        if args.len() == 3 && args[2] == "direct" {
            assert!(
                out.contains("helper"),
                "the known direct edge remains visible: {out}"
            );
        }
    }
    for args in [
        vec!["callees", "assertion", "--json"],
        vec!["callees", "assertion", "direct", "--json"],
    ] {
        let (code, out, err) = run(&args, &repo, &store);
        assert_eq!(code, 0, "callees failed; stderr={err}\nstdout={out}");
        let value: serde_json::Value = serde_json::from_str(&out).unwrap();
        let limits = value["call_extraction_limits"].as_array().unwrap();
        assert_eq!(limits.len(), 1, "the direct-call target has no macro gap");
        assert!(limits.iter().any(|limit| {
            limit["reason"] == "rust_macro_argument_tokens"
                && limit["next"] == "greppy read assertion"
        }));
    }
    let (code, out, err) = run(&["callees", "direct"], &repo, &store);
    assert_eq!(code, 0, "direct callees failed; stderr={err}\nstdout={out}");
    assert!(out.contains("helper"));
    assert!(!out.contains("Rust macro argument tokens"));
    let (code, out, err) = run(&["callees", "direct", "--json"], &repo, &store);
    assert_eq!(code, 0, "direct JSON failed; stderr={err}\nstdout={out}");
    let value: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert!(value.get("call_extraction_limits").is_none());
}

#[test]
fn callees_reports_missing_symbol() {
    let (repo, store) = index_fixture("callees-missing");
    let (code, out, _err) = run(&["callees", "does_not_exist_xyz"], &repo, &store);
    assert_eq!(code, 1, "missing symbol must exit 1; got out={out:?}");
    assert_eq!(
        out, "no symbol `does_not_exist_xyz`\n",
        "missing symbol must identify the absent name; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// path — every simple path as one tree of editable edge sites.
// ---------------------------------------------------------------------------

#[test]
fn path_finds_multi_hop_chain() {
    let (repo, store) = index_fixture("path");

    // entry -> middle -> leaf over CALLS.
    let (code, out, err) = run(&["path", "--from", "entry", "--to", "leaf"], &repo, &store);
    assert_eq!(
        code, 0,
        "path entry->leaf should exist and exit 0; stderr={err}\nstdout={out}"
    );
    assert_eq!(
        out, "src/lib.rs:5  entry\n  src/lib.rs:6  middle\n    src/mid.rs:5  leaf\n",
        "the root is a definition; every indented address is the call site in its parent"
    );
}

#[test]
fn path_json_reports_shortest_path_counts_and_metadata() {
    let (repo, store) = index_fixture("path-json");

    let (code, out, err) = run(
        &[
            "path",
            "--from",
            "entry",
            "--to",
            "leaf",
            "--json",
            "--diagnostics",
        ],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "path --json entry->leaf should exist and exit 0; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid path json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "path");
    assert_eq!(v["from"], "entry");
    assert_eq!(v["to"], "leaf");
    assert_eq!(v["project"], "repo");
    assert_eq!(v["from_found"], true);
    assert_eq!(v["to_found"], true);
    assert_eq!(v["path_found"], true);
    assert!(v["reason"].is_null());
    assert_eq!(v["fresh"], true);
    assert_eq!(v["provider_complete"], true);
    assert_eq!(v["incomplete_provider_count"], 0);
    assert_eq!(v["incomplete_providers"], serde_json::json!([]));
    assert_eq!(v["scope"], "shortest_path");
    assert_eq!(v["direction"], "outgoing");
    assert_eq!(v["edge_type"], "CALLS");
    assert_eq!(v["hops"], 2);
    assert_eq!(v["total_exact"], 3);
    assert_eq!(v["shown"], 3);
    assert_eq!(v["omitted"], 0);
    assert_eq!(v["truncated"], false);
    let steps = v["steps"].as_array().expect("steps array");
    assert_eq!(steps.len(), 3);
    let names: Vec<&str> = steps
        .iter()
        .map(|s| s["name"].as_str().expect("step name"))
        .collect();
    assert_eq!(names, vec!["entry", "middle", "leaf"]);
    assert_eq!(steps[0]["file_path"], "src/lib.rs");
    assert_eq!(steps[1]["file_path"], "src/mid.rs");
    assert_eq!(steps[2]["file_path"], "src/leaf.rs");
}

#[test]
fn provider_policy_require_complete_blocks_path_json() {
    let (repo, store) = index_fixture("provider-policy-path-json");
    // A normal Rust provider is complete for CALLS. Keep this a real
    // fail-closed test by withholding the relation that path requests.
    mark_call_provider_unavailable(&store);

    let (code, out, err) = run_with_env(
        &[
            "path",
            "--from",
            "entry",
            "--to",
            "leaf",
            "--json",
            "--diagnostics",
        ],
        &repo,
        &store,
        &[("GREPPY_PROVIDER_POLICY", "require_complete")],
    );
    assert_eq!(
        code, 1,
        "strict provider policy should block path JSON; stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "strict path JSON should not require stderr parsing; stderr={err:?}"
    );
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid strict path json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "path");
    assert_eq!(v["status"], "skipped_incomplete_provider");
    assert_eq!(v["from"], "entry");
    assert_eq!(v["to"], "leaf");
    assert_eq!(v["provider_complete"], false);
    assert_eq!(v["path_found"], false);
    assert_eq!(v["total_exact"], 0);
    assert_eq!(v["shown"], 0);
    assert_eq!(v["steps"].as_array().unwrap().len(), 0);
}

/// Freshness is fail-closed even when automatic repair is disabled: old path
/// steps must never be exposed as source evidence.
#[test]
fn path_json_refuses_stale_steps_when_auto_reindex_disabled() {
    let (repo, store) = index_fixture("path-json-stale");
    std::fs::write(
        repo.join("src/leaf.rs"),
        "pub fn renamed_leaf() -> u32 { 8 }\n",
    )
    .unwrap();

    let (code, out, err) = run_with_env(
        &[
            "path",
            "--from",
            "entry",
            "--to",
            "leaf",
            "--json",
            "--diagnostics",
        ],
        &repo,
        &store,
        &[("GREPPY_AUTO_REINDEX", "0")],
    );
    assert_eq!(
        code, 75,
        "stale path must be refused; stderr={err}\nstdout={out}"
    );
    assert!(err.is_empty(), "JSON refusal must stay machine-readable");
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid stale refusal json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "path");
    assert_eq!(v["status"], "skipped_stale_index");
    assert_eq!(v["from"], "entry");
    assert_eq!(v["to"], "leaf");
    assert_eq!(v["fresh"], false, "result must be labeled stale: {v:?}");
    assert_eq!(v["freshness"]["state"], "drift");
    assert_eq!(
        v["freshness"]["stale_file_count"], 1,
        "path must report the drift extent: {v:?}"
    );
    assert_eq!(v["path_found"], false);
    assert!(
        v["steps"].as_array().unwrap().is_empty(),
        "stale path must not expose old steps: {v:?}"
    );
}

/// Automatic repair completes inside the triggering query: it observes the
/// published generation and reports that the renamed-away endpoint is gone.
#[test]
fn path_json_auto_reindexes_small_stale_drift() {
    let (repo, store) = make_chain_repo("path-json-heal");
    // `leaf` also names the file's Module node, which legitimately survives
    // renaming its function. Give the endpoint a distinct name so absence
    // proves function removal rather than making an incorrect module claim.
    std::fs::write(
        repo.join("src/leaf.rs"),
        "pub fn endpoint_before() -> u32 { 7 }\n",
    )
    .unwrap();
    std::fs::write(
        repo.join("src/mid.rs"),
        "pub fn middle() { crate::leaf::endpoint_before(); }\n",
    )
    .unwrap();
    let (index_code, index_out, index_err) = run(&["index", "."], &repo, &store);
    assert_eq!(index_code, 0, "{index_out}\n{index_err}");
    let (before_code, before_out, before_err) = run(
        &[
            "path",
            "--from",
            "entry",
            "--to",
            "endpoint_before",
            "--json",
        ],
        &repo,
        &store,
    );
    assert_eq!(before_code, 0, "{before_out}\n{before_err}");
    let before: serde_json::Value = serde_json::from_str(&before_out).unwrap();
    assert_eq!(before["path_found"], true, "{before}");
    std::fs::write(
        repo.join("src/leaf.rs"),
        "pub fn endpoint_after() -> u32 { 8 }\n",
    )
    .unwrap();

    let (code, out, err) = run(
        &[
            "path",
            "--from",
            "entry",
            "--to",
            "endpoint_before",
            "--json",
            "--diagnostics",
        ],
        &repo,
        &store,
    );
    assert_eq!(
        code, 1,
        "healed path: `endpoint_before` no longer exists, so no path; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid healed path json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "path");
    assert_eq!(
        v["fresh"], true,
        "auto-reindex must yield a fresh answer: {v:?}"
    );
    assert_eq!(v["freshness"]["state"], "fresh");
    assert_eq!(v["to_found"], false, "renamed-away endpoint must be absent");
    assert!(v["steps"].as_array().unwrap().is_empty());
    assert_eq!(v["path_found"], false);
}

#[test]
fn path_reports_no_path_when_unreachable() {
    let (repo, store) = index_fixture("path-none");
    // Reverse direction has no CALLS path: leaf does not call entry.
    let (code, out, _err) = run(&["path", "--from", "leaf", "--to", "entry"], &repo, &store);
    assert_eq!(code, 0, "no reverse path is an answer; got out={out:?}");
    assert_eq!(out, "no path from leaf to entry\n");
}

#[test]
fn path_json_reports_no_path_without_text_parsing() {
    let (repo, store) = index_fixture("path-json-none");

    let (code, out, _err) = run(
        &[
            "path",
            "--from",
            "leaf",
            "--to",
            "entry",
            "--json",
            "--diagnostics",
        ],
        &repo,
        &store,
    );
    assert_eq!(code, 1, "no reverse path -> exit 1; got out={out:?}");
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid no-path json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "path");
    assert_eq!(v["from_found"], true);
    assert_eq!(v["to_found"], true);
    assert_eq!(v["path_found"], false);
    assert_eq!(v["reason"], "no_path");
    assert!(v["hops"].is_null());
    assert_eq!(v["total_exact"], 0);
    assert_eq!(v["shown"], 0);
    assert_eq!(v["truncated"], false);
    assert!(v["steps"].as_array().expect("steps array").is_empty());
}

#[test]
fn path_requires_both_endpoints() {
    let (repo, store) = index_fixture("path-usage");
    // Missing --to is a usage error (exit 64).
    let (code, _out, err) = run(&["path", "--from", "entry"], &repo, &store);
    assert_eq!(
        code, 64,
        "missing --to must be a usage error (64); stderr={err}"
    );
}

#[test]
fn path_self_query_requires_a_real_cycle() {
    let (repo, store) = index_fixture("path-self");
    let (code, out, err) = run(&["path", "--from", "entry", "--to", "entry"], &repo, &store);
    assert_eq!(
        code, 0,
        "a self query without a call cycle is a no-path answer; stderr={err}\nstdout={out}"
    );
    assert_eq!(out, "no path from entry to entry\n");
}

#[test]
fn path_help_uses_stored_edge_names_and_has_no_code_flag() {
    let root = fresh_dir("path-help");
    let store = root.join("store");
    let (code, out, err) = run(&["path", "--help"], &root, &store);
    assert_eq!(code, 0, "path --help failed; stderr={err}");
    assert!(out.contains("CALLS, USAGE, TYPE_ASSIGN, IMPORTS"), "{out}");
    assert!(
        !out.contains("--code"),
        "path must not advertise --code: {out}"
    );
}

#[test]
fn path_rejects_code_with_actionable_recovery() {
    let root = fresh_dir("path-code");
    let store = root.join("store");
    let (code, out, err) = run(
        &["path", "--from", "entry", "--to", "leaf", "--code"],
        &root,
        &store,
    );
    assert_eq!(
        code, 64,
        "unsupported path --code must exit 64: {out}\n{err}"
    );
    assert!(
        out.contains("run it without `--code`")
            && out.contains("greppy read SYMBOL")
            && out.contains("greppy path --from SYMBOL --to SYMBOL"),
        "refusal must provide exact recovery and valid usage: {out:?}"
    );
}

#[test]
fn path_rejects_edge_names_the_store_does_not_use() {
    let root = fresh_dir("path-edge-values");
    let store = root.join("store");
    for edge in ["USES", "TYPE_REF"] {
        let (code, out, err) = run(
            &["path", "--from", "entry", "--to", "leaf", "--edge", edge],
            &root,
            &store,
        );
        assert_eq!(
            code, 64,
            "invalid --edge {edge}; stdout={out}\nstderr={err}"
        );
        assert!(
            out.contains("invalid value"),
            "usage refusal belongs on stdout; stdout={out:?}; stderr={err:?}"
        );
    }
}
