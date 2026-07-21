//! Graph certification grid for OCaml — 12 cells.
//!
//! Calls and references inside top-level `let` bindings are attributed to the
//! enclosing Function; `open`/`include`, type references, and type declarations
//! participate in graph navigation.
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);
fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}
fn fresh_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let d = std::env::temp_dir().join(format!(
        "greppy-grid-ocaml-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Valid OCaml compilation units. Main opens Helper and Types, giving IMPORTS;
/// caller calls Helper.do_it (CALLS) and reads Helper.helper_value (USES), while
/// render's annotation references Types.widget (TYPE_REF).
fn make_ocaml_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::create_dir_all(repo.join(".git")).unwrap();
    std::fs::write(
        src.join("main.ml"),
        r#"open Helper
open Types

let caller () =
  let value = Helper.do_it 0 in
  value + Helper.helper_value

let render (w : Types.widget) = w.value
let uncalled () = 42
"#,
    )
    .unwrap();
    std::fs::write(
        src.join("helper.ml"),
        r#"let helper_value = 7
let do_it x = x + helper_value
"#,
    )
    .unwrap();
    std::fs::write(
        src.join("types.ml"),
        r#"type widget = { value : int }
type status = Ready | Busy
"#,
    )
    .unwrap();
    (repo, root.join("store"))
}
fn run(args: &[&str], cwd: &Path, store: &Path) -> (i32, String, String) {
    let out = Command::new(bin())
        .args(args)
        .current_dir(cwd)
        .env("GREPPY_STORE_DIR", store)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1")
        .output()
        .unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}
fn index_fixture(tag: &str) -> (PathBuf, PathBuf) {
    let (r, s) = make_ocaml_repo(tag);
    let (c, o, e) = run(&["index", "."], &r, &s);
    assert_eq!(c, 0, "index failed: {e}\n{o}");
    (r, s)
}
fn json(args: &[&str], r: &Path, s: &Path) -> serde_json::Value {
    let (c, o, e) = run(args, r, s);
    assert_eq!(c, 0, "stderr={e}\nstdout={o}");
    serde_json::from_str(&o).unwrap_or_else(|x| panic!("invalid json {x}: {o}"))
}
fn has_qname(v: &serde_json::Value, needle: &str) -> bool {
    v["hits"].as_array().is_some_and(|a| {
        a.iter()
            .any(|h| h["qualified_name"].as_str().unwrap_or("").contains(needle))
    })
}

#[test]
fn graph_grid_ocaml_who_calls_finds_cross_file_caller() {
    let (r, s) = index_fixture("who");
    let v = json(&["who-calls", "do_it", "--json"], &r, &s);
    assert!(
        has_qname(&v, "::Function::caller"),
        "expected actual caller node, not footer/file module: {v:?}"
    );
}
#[test]
fn graph_grid_ocaml_who_calls_empty_for_uncalled() {
    let (r, s) = index_fixture("none");
    let (c, o, _) = run(&["who-calls", "uncalled"], &r, &s);
    assert_eq!(c, 0);
    assert!(o.contains("(no callers)"), "{o}");
}
#[test]
fn graph_grid_ocaml_callees_lists_cross_file_target() {
    let (r, s) = index_fixture("callees");
    let (c, o, e) = run(&["callees", "caller"], &r, &s);
    assert_eq!(c, 0, "{e}");
    assert!(o.contains("do_it") && o.contains("src/helper.ml:"), "{o}");
}
#[test]
fn graph_grid_ocaml_find_usages_covers_call_and_import() {
    let (r, s) = index_fixture("usage-call");
    let (c, o, e) = run(&["find-usages", "do_it"], &r, &s);
    assert_eq!(c, 0, "{e}");
    assert!(o.contains("CALLS") && o.contains("caller"), "{o}");
    let (c, o, e) = run(&["find-usages", "Helper"], &r, &s);
    assert_eq!(c, 0, "{e}");
    assert!(o.contains("IMPORTS") && o.contains("src/main.ml"), "{o}");
}
#[test]
fn graph_grid_ocaml_find_usages_type_reference() {
    let (r, s) = index_fixture("type-ref");
    let (c, o, e) = run(&["find-usages", "widget"], &r, &s);
    assert_eq!(c, 0, "{e}");
    // `find-usages` presents the extractor's TYPE_REF through the persisted
    // compatibility USAGE label, matching the certified Elixir grid.
    assert!(o.contains("USAGE") && o.contains("render"), "{o}");
}
#[test]
fn graph_grid_ocaml_impact_transitive_reaches_caller() {
    let (r, s) = index_fixture("impact");
    let (c, o, e) = run(&["impact", "do_it"], &r, &s);
    assert_eq!(c, 0, "{e}");
    assert!(o.contains("caller") && o.contains("hop 1"), "{o}");
}
#[test]
fn graph_grid_ocaml_search_symbols_finds_all_definitions() {
    let (r, s) = index_fixture("symbols");
    for n in [
        "caller",
        "render",
        "uncalled",
        "do_it",
        "helper_value",
        "widget",
        "status",
    ] {
        let (c, o, e) = run(&["search-symbols", n], &r, &s);
        assert_eq!(c, 0, "{e}");
        assert!(o.contains(n), "missing {n}: {o}");
    }
}
#[test]
fn graph_grid_ocaml_brief_shows_definition_with_callers() {
    let (r, s) = index_fixture("brief");
    let (c, o, e) = run(&["brief", "do_it"], &r, &s);
    assert_eq!(c, 0, "{e}");
    assert!(
        o.contains("let do_it") && o.contains("-- CALLERS") && o.contains("caller"),
        "{o}"
    );
}
#[test]
fn graph_grid_ocaml_path_connects_caller_to_helper() {
    let (r, s) = index_fixture("path");
    let (c, o, e) = run(&["path", "--from", "caller", "--to", "do_it"], &r, &s);
    assert_eq!(c, 0, "{e}\n{o}");
    assert!(
        o.contains("caller") && o.contains("do_it") && o.contains("src/helper.ml:"),
        "{o}"
    );
}
#[test]
fn graph_grid_ocaml_graph_survives_reindex() {
    let (r, s) = index_fixture("reindex");
    let (_, a, _) = run(&["who-calls", "do_it"], &r, &s);
    let (c, o, e) = run(&["index", "."], &r, &s);
    assert_eq!(c, 0, "{e}\n{o}");
    let (_, b, _) = run(&["who-calls", "do_it"], &r, &s);
    assert!(
        a.contains("src/main.ml:") && b.contains("src/main.ml:"),
        "before={a}\nafter={b}"
    );
}
#[test]
fn graph_grid_ocaml_stale_edit_detected() {
    let (r, s) = index_fixture("stale");
    std::fs::write(
        r.join("src/helper.ml"),
        "let helper_value = 7\nlet do_it_renamed x = x + helper_value\n",
    )
    .unwrap();
    let v = json(&["who-calls", "do_it_renamed", "--json"], &r, &s);
    assert_eq!(v["fresh"], true, "{v:?}");
    assert_eq!(v["symbol_found"], true, "{v:?}");
}
#[test]
fn graph_grid_ocaml_declarative_or_edge_case() {
    // OCaml-specific: a variant type declaration must remain discoverable as a
    // definition even though its constructors are declarative alternatives.
    let (r, s) = index_fixture("variant");
    let v = json(&["search-symbols", "status", "--json"], &r, &s);
    assert!(
        v["hits"]
            .as_array()
            .is_some_and(|a| a.iter().any(|h| h["name"] == "status"
                && h["file_path"].as_str().unwrap_or("").ends_with("types.ml"))),
        "{v:?}"
    );
}
