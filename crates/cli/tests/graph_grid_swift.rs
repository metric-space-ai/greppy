//! End-to-end graph-certification grid for Swift.
//!
//! The fixture is a three-file Swift repository indexed through the shipped
//! binary with an isolated `GREPPY_STORE_DIR` per test. Its cross-file graph is:
//!
//! * `caller()` calls `helperFunction()` from `Helpers.swift` (CALLS),
//! * `caller()` refers to `Payload` from `Types.swift` (logical TYPE_REF),
//! * `caller()` uses `sharedConstant` from `Helpers.swift` (logical USES), and
//! * Swift module imports resolve from `Main.swift`/`Helpers.swift` to symbols
//!   defined in the other fixture files (IMPORTS).
//!
//! The navigation API persists the logical TYPE_REF/USES family under the
//! unified user-facing `USAGE` edge label, so those cells assert `USAGE`.
//! The parser registry spelling was checked explicitly: `Language::Swift`.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

use serde_json::Value;

static COUNTER: AtomicU32 = AtomicU32::new(0);

const MAIN_SWIFT: &str = r#"import helperFunction
import Payload

public func caller() -> Payload {
    let computed = helperFunction() + sharedConstant
    return Payload(value: computed)
}

public func entryPoint() -> Payload {
    caller()
}
"#;

const HELPERS_SWIFT: &str = r#"import Payload

public let sharedConstant = 7

public func helperFunction() -> Int {
    sharedConstant
}

public func uncalledFunction() -> Int {
    sharedConstant + 1
}
"#;

const TYPES_SWIFT: &str = r#"public protocol ValueProviding {
    func resolvedValue() -> Int
}

public struct Payload: ValueProviding {
    public let value: Int

    public init(value: Int) {
        self.value = value
    }

    public func resolvedValue() -> Int {
        value
    }
}
"#;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fresh_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("greppy-cli-graph-grid-swift-{tag}-{pid}-{n}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create Swift graph-grid scratch directory");
    dir
}

/// Build the same syntactically valid, three-file Swift repository for every
/// cell. The `.git` directory is the repository-root marker used by the CLI.
fn make_swift_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).expect("create Swift fixture source directory");
    std::fs::create_dir_all(repo.join(".git")).expect("create repository-root marker");
    std::fs::write(src.join("Main.swift"), MAIN_SWIFT).expect("write Main.swift");
    std::fs::write(src.join("Helpers.swift"), HELPERS_SWIFT).expect("write Helpers.swift");
    std::fs::write(src.join("Types.swift"), TYPES_SWIFT).expect("write Types.swift");
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

fn index_fixture(tag: &str) -> (PathBuf, PathBuf) {
    let (repo, store) = make_swift_repo(tag);
    assert_index_succeeds(&repo, &store);
    (repo, store)
}

fn assert_index_succeeds(repo: &Path, store: &Path) {
    let (code, out, err) = run(&["index", "."], repo, store);
    assert_eq!(
        code, 0,
        "Swift fixture index must succeed; stderr={err}\nstdout={out}"
    );
}

fn run_json(args: &[&str], repo: &Path, store: &Path) -> (i32, Value, String, String) {
    let (code, out, err) = run(args, repo, store);
    let value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid JSON for {args:?}: {e}; stderr={err:?}; stdout={out:?}"));
    (code, value, out, err)
}

fn run_json_with_env(
    args: &[&str],
    repo: &Path,
    store: &Path,
    envs: &[(&str, &str)],
) -> (i32, Value, String, String) {
    let (code, out, err) = run_with_env(args, repo, store, envs);
    let value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid JSON for {args:?}: {e}; stderr={err:?}; stdout={out:?}"));
    (code, value, out, err)
}

fn hits(value: &Value) -> &[Value] {
    value["hits"].as_array().expect("JSON hits array")
}

/// Direct-navigation JSON intentionally omits the redundant `name` field, while
/// symbol-search JSON includes it. Normalize both shapes for grid assertions.
fn hit_names(value: &Value) -> Vec<&str> {
    hits(value)
        .iter()
        .filter_map(|hit| {
            hit["name"].as_str().or_else(|| {
                hit["qualified_name"]
                    .as_str()
                    .and_then(|qname| qname.rsplit("::").next())
            })
        })
        .collect()
}

fn has_reference(value: &Value, edge_type: &str, name: &str) -> bool {
    hits(value).iter().any(|hit| {
        hit["edge_type"] == edge_type
            && (hit["name"] == name
                || hit["qualified_name"]
                    .as_str()
                    .is_some_and(|qname| qname.contains(name)))
    })
}

fn reference_signatures(value: &Value) -> Vec<String> {
    let mut signatures = hits(value)
        .iter()
        .map(|hit| {
            format!(
                "{}|{}|{}|{}",
                hit["edge_type"].as_str().unwrap_or(""),
                hit["name"].as_str().unwrap_or(""),
                hit["qualified_name"].as_str().unwrap_or(""),
                hit["file_path"].as_str().unwrap_or("")
            )
        })
        .collect::<Vec<_>>();
    signatures.sort();
    signatures
}

#[test]
fn graph_grid_swift_who_calls_finds_cross_file_caller() {
    let (repo, store) = index_fixture("who-calls");
    let (code, value, out, err) = run_json(
        &["who-calls", "helperFunction", "--json"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "who-calls helperFunction must succeed; stderr={err}\nstdout={out}"
    );
    assert!(
        hit_names(&value).contains(&"caller")
            && hits(&value)
                .iter()
                .any(|hit| hit["file_path"] == "src/Main.swift"),
        "Swift CALLS must resolve Helpers.swift::helperFunction back to Main.swift::caller; graph={value}"
    );
}

#[test]
fn graph_grid_swift_who_calls_empty_for_uncalled() {
    let (repo, store) = index_fixture("who-calls-empty");
    let (code, value, out, err) = run_json(
        &["who-calls", "uncalledFunction", "--json"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "who-calls uncalledFunction must succeed; stderr={err}\nstdout={out}"
    );
    assert_eq!(value["symbol_found"], true, "graph={value}");
    assert!(
        hits(&value).is_empty() && value["total_exact"] == 0,
        "uncalledFunction has no callers and must not acquire a false CALLS edge; graph={value}"
    );
}

#[test]
fn graph_grid_swift_callees_lists_cross_file_target() {
    let (repo, store) = index_fixture("callees");
    let (code, value, out, err) =
        run_json(&["callees", "caller", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "callees caller must succeed; stderr={err}\nstdout={out}"
    );
    assert!(
        hits(&value).iter().any(|hit| {
            hit["qualified_name"]
                .as_str()
                .is_some_and(|qname| qname.ends_with("::helperFunction"))
                && hit["file_path"] == "src/Helpers.swift"
        }),
        "Swift caller must have the cross-file helperFunction callee; graph={value}"
    );
}

#[test]
fn graph_grid_swift_find_usages_covers_call_and_import() {
    let (repo, store) = index_fixture("usages-call-import");
    let (code, helper_refs, out, err) = run_json(
        &["find-usages", "helperFunction", "--json"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "find-usages helperFunction must succeed; stderr={err}\nstdout={out}"
    );
    assert!(
        has_reference(&helper_refs, "CALLS", "caller"),
        "helperFunction usages must include caller's CALLS edge; graph={helper_refs}"
    );
    assert!(
        hits(&helper_refs).iter().any(|hit| {
            hit["edge_type"] == "IMPORTS" && hit["file_path"] == "src/Main.swift"
        }),
        "helperFunction usages must include Main.swift's Swift import; graph={helper_refs}"
    );

    let (code, constant_refs, out, err) = run_json(
        &["find-usages", "sharedConstant", "--json"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "find-usages sharedConstant must succeed; stderr={err}\nstdout={out}"
    );
    assert!(
        has_reference(&constant_refs, "USAGE", "caller"),
        "the cross-file USES relation for sharedConstant must surface as USAGE from caller; graph={constant_refs}"
    );
}

#[test]
fn graph_grid_swift_find_usages_type_reference() {
    let (repo, store) = index_fixture("usages-type-ref");
    let (code, value, out, err) =
        run_json(&["find-usages", "Payload", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "find-usages Payload must succeed; stderr={err}\nstdout={out}"
    );
    assert!(
        has_reference(&value, "USAGE", "caller"),
        "caller's cross-file Payload type reference must surface as USAGE; graph={value}"
    );
    assert!(
        hits(&value).iter().any(|hit| {
            hit["edge_type"] == "IMPORTS" && hit["file_path"] == "src/Main.swift"
        }),
        "Payload usages must retain the independent Swift import relation; graph={value}"
    );
}

#[test]
fn graph_grid_swift_impact_transitive_reaches_caller() {
    let (repo, store) = index_fixture("impact");
    let (code, value, out, err) =
        run_json(&["impact", "helperFunction", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "impact helperFunction must succeed; stderr={err}\nstdout={out}"
    );
    let names = hit_names(&value);
    assert!(
        names.contains(&"caller"),
        "impact from helperFunction must reach direct caller; graph={value}"
    );
    assert!(
        names.contains(&"entryPoint"),
        "transitive impact must continue from caller to entryPoint; graph={value}"
    );
}

#[test]
fn graph_grid_swift_search_symbols_finds_all_definitions() {
    let (repo, store) = index_fixture("symbols");
    let expected = [
        ("caller", "Function", "src/Main.swift"),
        ("entryPoint", "Function", "src/Main.swift"),
        ("helperFunction", "Function", "src/Helpers.swift"),
        ("uncalledFunction", "Function", "src/Helpers.swift"),
        ("sharedConstant", "Variable", "src/Helpers.swift"),
        ("ValueProviding", "Interface", "src/Types.swift"),
        ("Payload", "Class", "src/Types.swift"),
        ("resolvedValue", "Method", "src/Types.swift"),
        ("value", "Variable", "src/Types.swift"),
    ];

    for (name, label, file) in expected {
        let (code, value, out, err) =
            run_json(&["search-symbols", name, "--json"], &repo, &store);
        assert_eq!(
            code, 0,
            "search-symbols {name} must succeed; stderr={err}\nstdout={out}"
        );
        assert!(
            hits(&value).iter().any(|hit| {
                hit["name"] == name && hit["label"] == label && hit["file_path"] == file
            }),
            "missing Swift definition {label} {name} in {file}; graph={value}"
        );
    }
}

#[test]
fn graph_grid_swift_brief_shows_definition_with_callers() {
    let (repo, store) = index_fixture("brief");
    let (code, out, err) = run(&["brief", "helperFunction"], &repo, &store);
    assert_eq!(
        code, 0,
        "brief helperFunction must succeed; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("public func helperFunction() -> Int")
            && out.contains("-- CALLERS")
            && out.contains("caller")
            && out.contains("src/Helpers.swift:"),
        "brief must bundle the Swift helper definition and its cross-file caller; got={out:?}"
    );
}

#[test]
fn graph_grid_swift_path_connects_caller_to_helper() {
    let (repo, store) = index_fixture("path");
    let (code, value, out, err) = run_json(
        &[
            "path",
            "--from",
            "caller",
            "--to",
            "helperFunction",
            "--json",
        ],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "path caller->helperFunction must succeed; stderr={err}\nstdout={out}"
    );
    assert_eq!(value["path_found"], true, "graph={value}");
    assert_eq!(value["hops"], 1, "graph={value}");
    let steps = value["steps"].as_array().expect("path steps array");
    let names = steps
        .iter()
        .filter_map(|step| step["name"].as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        vec!["caller", "helperFunction"],
        "Swift path must cross Main.swift -> Helpers.swift; graph={value}"
    );
}

#[test]
fn graph_grid_swift_graph_survives_reindex() {
    let (repo, store) = index_fixture("reindex");

    let (code, helper_before, out, err) = run_json(
        &["find-usages", "helperFunction", "--json"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "pre-reindex helper query failed; stderr={err}\nstdout={out}"
    );
    let (code, payload_before, out, err) =
        run_json(&["find-usages", "Payload", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "pre-reindex type query failed; stderr={err}\nstdout={out}"
    );
    let before = (
        reference_signatures(&helper_before),
        reference_signatures(&payload_before),
    );
    assert!(
        before.0.iter().any(|edge| edge.starts_with("CALLS|"))
            && before.0.iter().any(|edge| edge.starts_with("IMPORTS|"))
            && before.1.iter().any(|edge| edge.starts_with("USAGE|")),
        "precondition: initial Swift graph must contain call, import, and type-reference evidence; graph={before:?}"
    );

    assert_index_succeeds(&repo, &store);

    let (code, helper_after, out, err) = run_json(
        &["find-usages", "helperFunction", "--json"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "post-reindex helper query failed; stderr={err}\nstdout={out}"
    );
    let (code, payload_after, out, err) =
        run_json(&["find-usages", "Payload", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "post-reindex type query failed; stderr={err}\nstdout={out}"
    );
    let after = (
        reference_signatures(&helper_after),
        reference_signatures(&payload_after),
    );
    assert_eq!(
        after, before,
        "a no-op second Swift index run must preserve the query-visible edge set"
    );
}

#[test]
fn graph_grid_swift_stale_edit_detected() {
    let (repo, store) = index_fixture("stale-edit");
    std::fs::write(
        repo.join("src/Main.swift"),
        r#"import helperFunction
import Payload

public func caller() -> Payload {
    let computed = sharedConstant
    return Payload(value: computed)
}

public func entryPoint() -> Payload {
    caller()
}
"#,
    )
    .expect("edit Main.swift after indexing");

    let (first_code, first, first_out, first_err) = run_json_with_env(
        &["who-calls", "helperFunction", "--json"],
        &repo,
        &store,
        &[("GREPPY_AUTO_REINDEX", "1")],
    );
    assert!(
        matches!(first_code, 0 | 75),
        "freshness may heal inline or refuse while refreshing, never serve stale; stderr={first_err}\nstdout={first_out}"
    );
    if first_code == 75 {
        assert_eq!(first["status"], "skipped_stale_index", "graph={first}");
        assert_eq!(first["fresh"], false, "graph={first}");
        assert!(
            matches!(
                first["freshness"]["state"].as_str(),
                Some("refreshing" | "drift")
            ),
            "freshness must report the edit or active repair; graph={first}"
        );
    } else {
        assert_eq!(
            first["fresh"], true,
            "an inline-healed first response must prove freshness; graph={first}"
        );
    }
    assert!(
        hits(&first).is_empty(),
        "the old caller->helperFunction edge must never escape after the edit; graph={first}"
    );

    let (second_code, second, second_out, second_err) = run_json_with_env(
        &["who-calls", "helperFunction", "--json"],
        &repo,
        &store,
        &[("GREPPY_AUTO_REINDEX", "1")],
    );
    assert_eq!(
        second_code, 0,
        "repair must publish a queryable generation; stderr={second_err}\nstdout={second_out}"
    );
    assert_eq!(second["fresh"], true, "graph={second}");
    assert!(
        hits(&second).is_empty(),
        "the healed graph must not retain the removed Swift CALLS edge; graph={second}"
    );
}

#[test]
fn graph_grid_swift_declarative_or_edge_case() {
    let (repo, store) = index_fixture("protocol-conformance");
    let (code, value, out, err) = run_json(
        &[
            "trace",
            "--symbol",
            "Payload",
            "--edge",
            "IMPLEMENTS",
            "--json",
        ],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "trace Payload --edge IMPLEMENTS must succeed; stderr={err}\nstdout={out}"
    );
    let steps = value["steps"].as_array().expect("trace steps array");
    assert!(
        steps.iter().any(|step| {
            step["name"] == "ValueProviding"
                && step["file_path"] == "src/Types.swift"
                && step["via_edge"]["edge_type"] == "IMPLEMENTS"
        }),
        "Swift protocol conformance Payload: ValueProviding must be navigable as IMPLEMENTS; graph={value}"
    );
}
