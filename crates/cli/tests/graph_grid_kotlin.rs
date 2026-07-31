//! Graph-certification grid for Kotlin.
//!
//! The fixture is a small, git-rooted Kotlin repository indexed through the
//! real `greppy` binary.  It deliberately keeps the three source files on
//! separate edges of the graph: `main.kt` calls and imports `helper.kt`, and
//! refers to the `Payload` type in `types.kt`.

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
    let dir = std::env::temp_dir().join(format!("greppy-cli-graph-grid-kotlin-{tag}-{pid}-{n}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Build a valid three-file Kotlin fixture with these intended graph edges:
///
/// * `caller` --CALLS--> `helperFunction` (`main.kt` -> `helper.kt`)
/// * `caller` --TYPE_REF--> `Payload` (`main.kt` -> `types.kt`)
/// * `caller` --USES--> `HELPER_VALUE` (`main.kt` -> `helper.kt`)
/// * imports in `main.kt` resolve to symbols in both other files
fn make_kotlin_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).unwrap();
    // The CLI's root discovery follows this marker upward from src/.
    std::fs::create_dir_all(repo.join(".git")).unwrap();

    std::fs::write(
        src.join("main.kt"),
        r#"package grid.main

import grid.helper.HELPER_VALUE
import grid.helper.helperFunction
import grid.types.Payload

fun caller(payload: Payload): Payload {
    val answer = helperFunction()
    val total = answer + HELPER_VALUE
    return if (total >= 0) payload else payload
}

fun render(payload: Payload): Int = payload.value
"#,
    )
    .unwrap();

    std::fs::write(
        src.join("helper.kt"),
        r#"package grid.helper

const val HELPER_VALUE: Int = 7

fun helperFunction(): Int = HELPER_VALUE

fun uncalledHelper(): Int = 99

object KotlinMarker {
    fun markerValue(): Int = HELPER_VALUE
}
"#,
    )
    .unwrap();

    std::fs::write(
        src.join("types.kt"),
        r#"package grid.types

data class Payload(val value: Int)
"#,
    )
    .unwrap();

    let store = root.join("store");
    (repo, store)
}

fn run(args: &[&str], cwd: &Path, store_dir: &Path) -> (i32, String, String) {
    let output = Command::new(bin())
        .args(args)
        .current_dir(cwd)
        .env("GREPPY_STORE_DIR", store_dir)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .output()
        .expect("spawn greppy");
    (
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

fn index_fixture(tag: &str) -> (PathBuf, PathBuf) {
    let (repo, store) = make_kotlin_repo(tag);
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "Kotlin fixture index should succeed; stderr={err}\nstdout={out}"
    );
    (repo, store)
}

fn json(stdout: &str, context: &str) -> serde_json::Value {
    serde_json::from_str(stdout)
        .unwrap_or_else(|error| panic!("{context} must return JSON: {error}; stdout={stdout:?}"))
}

fn has_hit(value: &serde_json::Value, edge_type: Option<&str>, name: &str) -> bool {
    value["hits"].as_array().into_iter().flatten().any(|hit| {
        edge_type.is_none_or(|edge| hit["edge_type"] == edge)
            && (hit["qualified_name"].as_str().unwrap_or("").contains(name)
                || hit["file_path"].as_str().unwrap_or("").contains(name))
    })
}

fn edge_keys(value: &serde_json::Value) -> Vec<(String, String, String)> {
    let mut keys = value["hits"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|hit| {
            (
                hit["edge_type"].as_str().unwrap_or("").to_string(),
                hit["qualified_name"].as_str().unwrap_or("").to_string(),
                hit["file_path"].as_str().unwrap_or("").to_string(),
            )
        })
        .collect::<Vec<_>>();
    keys.sort();
    keys
}

#[test]
fn graph_grid_kotlin_who_calls_finds_cross_file_caller() {
    let (repo, store) = index_fixture("who-calls");
    let (code, out, err) = run(&["who-calls", "helperFunction", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "who-calls should succeed; stderr={err}\nstdout={out}"
    );
    let value = json(&out, "who-calls helperFunction");
    assert_eq!(
        value["symbol_found"], true,
        "helperFunction must be indexed: {value}"
    );
    assert!(
        has_hit(&value, None, "caller") && has_hit(&value, None, "src/main.kt"),
        "CALLS incoming edge must identify caller in main.kt: {value}"
    );
}

#[test]
fn graph_grid_kotlin_who_calls_empty_for_uncalled() {
    let (repo, store) = index_fixture("who-calls-empty");
    let (code, out, err) = run(&["who-calls", "uncalledHelper", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "uncalled symbol should be a valid query; stderr={err}\nstdout={out}"
    );
    let value = json(&out, "who-calls uncalledHelper");
    assert_eq!(
        value["symbol_found"], true,
        "uncalledHelper must be indexed: {value}"
    );
    assert_eq!(
        value["total_exact"], 0,
        "uncalledHelper must have no callers: {value}"
    );
    assert!(
        value["hits"].as_array().is_some_and(Vec::is_empty),
        "uncalledHelper must return no caller rows: {value}"
    );
}

#[test]
fn graph_grid_kotlin_callees_lists_cross_file_target() {
    let (repo, store) = index_fixture("callees");
    let (code, out, err) = run(&["callees", "caller", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "callees should succeed; stderr={err}\nstdout={out}"
    );
    let value = json(&out, "callees caller");
    assert_eq!(
        value["symbol_found"], true,
        "caller must be indexed: {value}"
    );
    assert!(
        has_hit(&value, None, "helperFunction") && has_hit(&value, None, "src/helper.kt"),
        "caller must list helperFunction from helper.kt: {value}"
    );
}

#[test]
fn graph_grid_kotlin_impact_transitive_reaches_caller() {
    let (repo, store) = index_fixture("impact");
    let (code, out, err) = run(&["impact", "helperFunction", "--json"], &repo, &store);
    assert_eq!(code, 0, "impact should succeed; stderr={err}\nstdout={out}");
    let value = json(&out, "impact helperFunction");
    assert!(
        value["hits"]
            .as_array()
            .is_some_and(|hits| hits.iter().any(|hit| {
                hit["qualified_name"]
                    .as_str()
                    .unwrap_or("")
                    .contains("caller")
            })),
        "impact on helperFunction must reach caller: {value}"
    );
}

#[test]
fn graph_grid_kotlin_search_symbol_finds_all_definitions() {
    let (repo, store) = index_fixture("search-symbol");
    for symbol in [
        "caller",
        "render",
        "helperFunction",
        "uncalledHelper",
        "HELPER_VALUE",
        "KotlinMarker",
        "markerValue",
        "Payload",
    ] {
        let (code, out, err) = run(&["search-symbol", symbol, "--json"], &repo, &store);
        assert_eq!(
            code, 0,
            "search-symbol {symbol} should succeed; stderr={err}\nstdout={out}"
        );
        let value = json(&out, &format!("search-symbol {symbol}"));
        assert!(
            value["hits"]
                .as_array()
                .is_some_and(|hits| hits.iter().any(|hit| {
                    hit["qualified_name"]
                        .as_str()
                        .unwrap_or("")
                        .contains(symbol)
                })),
            "search-symbol must find Kotlin definition {symbol}: {value}"
        );
    }
}

#[test]
fn graph_grid_kotlin_brief_shows_definition_with_callers() {
    let (repo, store) = index_fixture("brief");
    let (code, out, err) = run(&["brief", "helperFunction"], &repo, &store);
    assert_eq!(code, 0, "brief should succeed; stderr={err}\nstdout={out}");
    assert!(
        out.contains("fun helperFunction") && out.contains("src/helper.kt:"),
        "brief must show helperFunction's Kotlin definition: {out}"
    );
    assert!(
        out.contains("called by caller") && !out.contains("-- CALLERS"),
        "brief must aggregate the caller of helperFunction into one line: {out}"
    );
}

#[test]
fn graph_grid_kotlin_path_connects_caller_to_helper() {
    let (repo, store) = index_fixture("path");
    let (code, out, err) = run(
        &[
            "path",
            "--from",
            "caller",
            "--to",
            "helperFunction",
            "--edge",
            "CALLS",
        ],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "path caller->helperFunction should exist; stderr={err}\nstdout={out}"
    );
    assert_eq!(
        out, "src/main.kt:7  caller\n  src/main.kt:8  helperFunction\n",
        "path must print the helperFunction call site inside caller"
    );
}

#[test]
fn graph_grid_kotlin_graph_survives_reindex() {
    let (repo, store) = index_fixture("reindex");
    let (code, first_index_out, first_index_err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "second Kotlin index should succeed; stderr={first_index_err}\nstdout={first_index_out}"
    );

    let (code, first_out, first_err) =
        run(&["who-calls", "helperFunction", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "references after reindex should succeed; stderr={first_err}"
    );
    let first = json(&first_out, "references helperFunction after reindex");
    let first_edges = edge_keys(&first);

    let (code, second_index_out, second_index_err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "third Kotlin index should succeed; stderr={second_index_err}\nstdout={second_index_out}"
    );
    let (code, second_out, second_err) =
        run(&["who-calls", "helperFunction", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "references after second reindex should succeed; stderr={second_err}"
    );
    let second = json(
        &second_out,
        "references helperFunction after second reindex",
    );
    assert_eq!(
        first_edges,
        edge_keys(&second),
        "the caller result must be stable across two reindexes"
    );
    assert!(
        has_hit(&second, None, "caller"),
        "the stable result must retain the cross-file caller relationship: {second}"
    );
}

#[test]
fn graph_grid_kotlin_stale_edit_detected() {
    let (repo, store) = index_fixture("stale-edit");
    std::fs::write(
        repo.join("src/main.kt"),
        r#"package grid.main

import grid.helper.HELPER_VALUE
import grid.helper.helperFunction
import grid.types.Payload

fun caller(payload: Payload): Payload {
    return payload
}

fun render(payload: Payload): Int = payload.value
"#,
    )
    .unwrap();

    let (code, out, err) = run(&["callees", "caller", "--json"], &repo, &store);
    assert!(
        code == 0 || code == 75,
        "stale graph query must either heal or report freshness, got code={code}; stderr={err}\nstdout={out}"
    );
    let value = json(&out, "stale callees caller");
    assert!(
        value["hits"].as_array().is_some_and(Vec::is_empty),
        "stale edit must not return the old helperFunction CALLS edge: {value}"
    );
    if code == 75 {
        assert_eq!(
            value["status"], "skipped_stale_index",
            "a refused stale query must identify the freshness gate: {value}"
        );
        assert_ne!(
            value["freshness"]["state"], "fresh",
            "a refused stale query must not claim a fresh graph: {value}"
        );
    }
}

#[test]
fn graph_grid_kotlin_declarative_or_edge_case() {
    let (repo, store) = index_fixture("declarative-object");
    // Kotlin `object` is a singleton declaration rather than a class keyword,
    // but it is still a named, importable type in the graph.  Certification
    // requires the definition to remain searchable as a Class node.
    let (code, out, err) = run(&["search-symbol", "KotlinMarker", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "Kotlin object search should succeed; stderr={err}\nstdout={out}"
    );
    let value = json(&out, "search-symbol KotlinMarker");
    assert!(
        value["hits"]
            .as_array()
            .is_some_and(|hits| hits.iter().any(|hit| {
                hit["label"] == "Class"
                    && hit["file_path"] == "src/helper.kt"
                    && hit["qualified_name"]
                        .as_str()
                        .unwrap_or("")
                        .contains("KotlinMarker")
            })),
        "Kotlin object declarations must be searchable as Class definitions: {value}"
    );
}
