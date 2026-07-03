//! Integration tests for the Track-B navigation/stats commands:
//! `stats`, `callees`, and `path`.
//!
//! These spawn the real `grepplus` binary against a multi-file fixture
//! indexed end-to-end, so the cross-file CALLS edges resolved by the
//! indexer/resolver are exercised exactly as an agent would see them.
//! Each test gets an isolated `GREPPLUS_STORE_DIR` so parallel runs
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

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_grepplus")
}

fn fresh_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("grepplus-cli-statspath-{tag}-{pid}-{n}"));
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
        .env("GREPPLUS_STORE_DIR", store_dir);
    for (key, value) in envs {
        cmd.env(key, value);
    }
    let out = cmd.output().expect("spawn grepplus");
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
        !out.contains("(no callees)"),
        "entry calls middle, so callees must be non-empty; got: {out:?}"
    );
}

#[test]
fn callees_reports_no_callees_for_leaf() {
    let (repo, store) = index_fixture("callees-none");
    // `leaf` calls nothing.
    let (code, out, _err) = run(&["callees", "leaf"], &repo, &store);
    assert_eq!(code, 0);
    assert!(
        out.contains("(no callees)"),
        "leaf calls nothing, so callees must be empty; got: {out:?}"
    );
}

#[test]
fn callees_reports_missing_symbol() {
    let (repo, store) = index_fixture("callees-missing");
    let (code, out, _err) = run(&["callees", "does_not_exist_xyz"], &repo, &store);
    assert_eq!(code, 1, "missing symbol must exit 1; got out={out:?}");
    assert!(
        out.contains("(symbol not found)"),
        "missing symbol must report not-found; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// path — shortest path between two symbols over CALLS edges.
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
    // All three steps present, in order, each with actionable file:line.
    let entry_idx = out.find("entry").expect("path must include start `entry`");
    let middle_idx = out
        .find("middle")
        .expect("path must include the intermediate `middle`");
    let leaf_idx = out.find("leaf").expect("path must include goal `leaf`");
    assert!(
        entry_idx < middle_idx && middle_idx < leaf_idx,
        "steps must be ordered entry -> middle -> leaf; got: {out:?}"
    );
    assert!(
        out.contains("src/lib.rs:") && out.contains("src/mid.rs:") && out.contains("src/leaf.rs:"),
        "each step must carry its file:line; got: {out:?}"
    );
}

#[test]
fn path_json_reports_shortest_path_counts_and_metadata() {
    let (repo, store) = index_fixture("path-json");

    let (code, out, err) = run(
        &["path", "--from", "entry", "--to", "leaf", "--json"],
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
    assert_eq!(v["provider_complete"], false);
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

    let (code, out, err) = run_with_env(
        &["path", "--from", "entry", "--to", "leaf", "--json"],
        &repo,
        &store,
        &[("GREPPLUS_PROVIDER_POLICY", "require_complete")],
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

/// D2 fail-open: with the inline auto-reindex disabled (kill switch),
/// a stale index serves the OLD path steps, honestly labeled
/// (`fresh: false` + stderr warning), instead of refusing with exit 1.
#[test]
fn path_json_serves_labeled_stale_steps_when_auto_reindex_disabled() {
    let (repo, store) = index_fixture("path-json-stale");
    std::fs::write(
        repo.join("src/leaf.rs"),
        "pub fn renamed_leaf() -> u32 { 8 }\n",
    )
    .unwrap();

    let (code, out, err) = run_with_env(
        &["path", "--from", "entry", "--to", "leaf", "--json"],
        &repo,
        &store,
        &[("GREPPLUS_AUTO_REINDEX", "0")],
    );
    assert_eq!(
        code, 0,
        "labeled-stale path must serve the indexed path; stderr={err}\nstdout={out}"
    );
    assert!(
        err.contains("index may be stale") && err.contains("run 'grepplus index'"),
        "labeled-stale path must warn on stderr; stderr={err:?}"
    );
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid labeled-stale path json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "path");
    assert_eq!(
        v["status"],
        serde_json::Value::Null,
        "labeled-stale path must not be skipped: {v:?}"
    );
    assert_eq!(v["from"], "entry");
    assert_eq!(v["to"], "leaf");
    assert_eq!(v["fresh"], false, "result must be labeled stale: {v:?}");
    assert_eq!(v["freshness"]["state"], "stale");
    assert_eq!(
        v["freshness"]["stale_file_count"], 1,
        "path must report the drift extent: {v:?}"
    );
    assert_eq!(v["path_found"], true);
    assert!(
        !v["steps"].as_array().unwrap().is_empty(),
        "labeled-stale path must serve the old steps: {v:?}"
    );
}

/// D2: the same small drift WITH auto-reindex enabled (default) heals
/// the index inline; the renamed-away endpoint is then honestly gone.
#[test]
fn path_json_auto_reindexes_small_stale_drift() {
    let (repo, store) = index_fixture("path-json-heal");
    std::fs::write(
        repo.join("src/leaf.rs"),
        "pub fn renamed_leaf() -> u32 { 8 }\n",
    )
    .unwrap();

    let (code, out, err) = run(
        &["path", "--from", "entry", "--to", "leaf", "--json"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 1,
        "healed path: `leaf` no longer exists, so no path; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid healed path json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "path");
    assert_eq!(
        v["fresh"], true,
        "auto-reindex must yield a fresh answer: {v:?}"
    );
    assert_eq!(v["path_found"], false);
}

#[test]
fn path_reports_no_path_when_unreachable() {
    let (repo, store) = index_fixture("path-none");
    // Reverse direction has no CALLS path: leaf does not call entry.
    let (code, out, _err) = run(&["path", "--from", "leaf", "--to", "entry"], &repo, &store);
    assert_eq!(code, 1, "no reverse path -> exit 1; got out={out:?}");
    assert!(
        out.contains("(no path"),
        "unreachable goal must report no path; got: {out:?}"
    );
}

#[test]
fn path_json_reports_no_path_without_text_parsing() {
    let (repo, store) = index_fixture("path-json-none");

    let (code, out, _err) = run(
        &["path", "--from", "leaf", "--to", "entry", "--json"],
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
fn path_trivial_self_path_is_single_step() {
    let (repo, store) = index_fixture("path-self");
    let (code, out, err) = run(&["path", "--from", "entry", "--to", "entry"], &repo, &store);
    assert_eq!(
        code, 0,
        "from==to is a length-0 path and exits 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("entry") && out.contains("src/lib.rs:"),
        "self path must print the single node; got: {out:?}"
    );
}
