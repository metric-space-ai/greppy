//! Graph-Zertifizierungs-Grid für HASKELL — 12 Zellen.
//!
//! HASKELL ist mit diesem Grid graph-zertifiziert. Alle zwölf Zellen prüfen
//! scharf die erwarteten Navigations- und Klassifikationskanten.
//!
//! Fixture-Repo (`src/`):
//! * `Main.hs`      — `caller`, `render`, `uncalled`;
//!   `import Helper (doIt, HELPER_VALUE, marker)` und
//!   `import Types (Widget)`.
//! * `Helper.hs`    — `doIt`, `HELPER_VALUE` und `marker` (im `marker`-Body wird
//!   `HELPER_VALUE` als USAGE referenziert).
//! * `Types.hs`     — `data Widget = Widget Int`.
//!
//! Soll-Kanten über Dateigrenzen:
//! * `caller`       --CALLS-->   `doIt`            (Helper.hs)
//! * `render`       --CALLS-->   `marker`          (Helper.hs, transitiv USAGE HELPER_VALUE)
//! * `marker`-Body  --USAGE-->   `HELPER_VALUE`    (Helper.hs, via pattern auf RHS)
//! * `render`-Body  --USAGE-->   `Widget`          (Types.hs, constructor pattern)
//! * `Main.hs`      --IMPORTS--> `Helper.hs` / `Types.hs`
//!
//! Hinweise:
//! * Der HASKELL-Provider kartiert explizite Importlisten strukturell von
//!   `import` über `import_list` auf je ein `import_name`; der bespoke Pass
//!   emittiert daraus auflösbare IMPORTS-Kanten.
//! * Zelle 12 (Haskell-spezifischer Grenzfall): HASKELL `data`-Deklarationen
//!   werden im `extract_haskell` zu "Class"-Knoten (siehe
//!   `HASKELL_TYPE_KINDS = ["class", "data_type", "newtype"]`). Eine
//!   `data Widget = Widget Int`-Deklaration taucht daher als Class-Knoten
//!   im Graphen auf, nicht als "Type" oder "Struct".

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
    let dir = std::env::temp_dir().join(format!("greppy-cli-graphgrid-haskell-{tag}-{pid}-{n}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Build a git-rooted HASKELL repo with three files exercising all four
/// cross-file edge shapes:
///
/// * CALLS:   `caller` -> `doIt` (Helper.hs)
///   `render` -> `marker` (Helper.hs); `marker` body USAGE -> `HELPER_VALUE`
/// * USAGE:   `render` constructor-pattern `(Widget n)` -> `Widget` (Types.hs)
/// * IMPORTS: `import Helper (doIt, HELPER_VALUE, marker)` /
///   `import Types (Widget)` — each explicit import name persists an edge.
fn make_haskell_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).unwrap();
    // `.git` is the repo-root marker resolve_root walks up to find.
    std::fs::create_dir_all(repo.join(".git")).unwrap();

    std::fs::write(
        src.join("Main.hs"),
        r#"module Main where

import Helper (doIt, HELPER_VALUE, marker)
import Types (Widget)

caller :: Int
caller = doIt 0

render :: Widget -> Int
render (Widget n) = marker n

uncalled :: Int
uncalled = 42
"#,
    )
    .unwrap();

    std::fs::write(
        src.join("Helper.hs"),
        r#"module Helper where

doIt :: Int -> Int
doIt x = x

HELPER_VALUE :: Int
HELPER_VALUE = 7

marker :: Int -> Int
marker _ = HELPER_VALUE
"#,
    )
    .unwrap();

    std::fs::write(
        src.join("Types.hs"),
        r#"module Types where

data Widget = Widget Int
"#,
    )
    .unwrap();

    let store = root.join("store");
    (repo, store)
}

fn run(args: &[&str], cwd: &Path, store_dir: &Path) -> (i32, String, String) {
    let mut cmd = Command::new(bin());
    cmd.args(args)
        .current_dir(cwd)
        .env("GREPPY_STORE_DIR", store_dir)
        .env("GREPPY_TEST_SKIP_INFERENCE", "1");
    let out = cmd.output().expect("spawn greppy");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Index the fixture once and assert it succeeded; shared setup.
fn index_fixture(tag: &str) -> (PathBuf, PathBuf) {
    let (repo, store) = make_haskell_repo(tag);
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed; stderr={err}\nstdout={out}"
    );
    (repo, store)
}

// ---------------------------------------------------------------------------
// 1 — who-calls: incoming CALLS edge resolves to the cross-file caller.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_haskell_who_calls_finds_cross_file_caller() {
    let (repo, store) = index_fixture("who-calls-cross");
    // `doIt` is defined in Helper.hs and called by `caller` in Main.hs.
    let (code, out, err) = run(&["who-calls", "doIt"], &repo, &store);
    assert_eq!(
        code, 0,
        "who-calls should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("caller"),
        "who-calls doIt must list the caller `caller` (Main.hs); got: {out:?}"
    );
    assert!(
        out.contains("src/Main.hs:"),
        "who-calls must print the caller's file:line (src/Main.hs); got: {out:?}"
    );
    assert!(
        !out.contains("(no callers)"),
        "who-calls must find at least one caller; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 2 — who-calls: an uncalled symbol reports "(no callers)".
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_haskell_who_calls_empty_for_uncalled() {
    let (repo, store) = index_fixture("who-calls-empty");
    // `render` is defined in Main.hs but never called from anywhere.
    let (code, out, _err) = run(&["who-calls", "render"], &repo, &store);
    assert_eq!(code, 0);
    assert!(
        out.contains("(no callers)"),
        "render is uncalled, who-calls must report no callers; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 3 — callees: outgoing CALLS edges resolve to the cross-file target.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_haskell_callees_lists_cross_file_target() {
    let (repo, store) = index_fixture("callees-cross");
    // `caller` calls `doIt` which is defined in Helper.hs.
    let (code, out, err) = run(&["callees", "caller"], &repo, &store);
    assert_eq!(code, 0, "callees should exit 0; stderr={err}\nstdout={out}");
    assert!(
        out.contains("doIt"),
        "callees caller must list the cross-file callee `doIt`; got: {out:?}"
    );
    assert!(
        out.contains("src/Helper.hs:"),
        "callees must print the callee's file:line (src/Helper.hs); got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 4 — find-usages: covers CALLS AND IMPORTS reference edges.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_haskell_find_usages_covers_call_and_import() {
    let (repo, store) = index_fixture("find-usages-call-import");
    // `find-usages` aggregates REFERENCE_EDGE_TYPES = [CALLS, USAGE, USES,
    // TYPE_REF, IMPORTS] per incoming target. The CALLS leg lists `caller`;
    // the IMPORTS leg comes from Main.hs's explicit import list.
    let (code, out, err) = run(&["find-usages", "doIt"], &repo, &store);
    assert_eq!(
        code, 0,
        "find-usages should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("CALLS"),
        "find-usages doIt must show the CALLS edge kind; got: {out:?}"
    );
    assert!(
        out.contains("caller"),
        "find-usages doIt must list `caller` as a CALLS referrer; got: {out:?}"
    );
    assert!(
        out.contains("IMPORTS"),
        "find-usages doIt must show the IMPORTS edge kind (Haskell imports); got: {out:?}"
    );
    assert!(
        out.contains("src/Main.hs:"),
        "find-usages must print the IMPORTS referrer's file:line (src/Main.hs); got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 5 — find-usages: type-reference (USAGE for type / constructor references).
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_haskell_find_usages_type_reference() {
    let (repo, store) = index_fixture("find-usages-type-ref");
    // `Widget` is referenced as a constructor pattern inside `render
    // (Widget n) = ...`. Although the grammar represents it as an `apply`,
    // the LHS-pattern guard classifies it as USAGE rather than CALLS.
    let (code, out, err) = run(&["find-usages", "Widget"], &repo, &store);
    assert_eq!(
        code, 0,
        "find-usages should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("USAGE"),
        "find-usages Widget must label the edge kind USAGE; got: {out:?}"
    );
    assert!(
        out.contains("render"),
        "find-usages Widget must list `render` as the referrer; got: {out:?}"
    );
    assert!(
        out.contains("src/Main.hs:"),
        "find-usages must print the referrer's file:line (src/Main.hs); got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 6 — impact: transitive blast radius reaches the cross-file caller.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_haskell_impact_transitive_reaches_caller() {
    let (repo, store) = index_fixture("impact-transitive");
    // `impact doIt` must report `caller` as incoming CALLS dependent — the
    // single-command answer to "what breaks if I change doIt?".
    let (code, out, err) = run(&["impact", "doIt"], &repo, &store);
    assert_eq!(code, 0, "impact should exit 0; stderr={err}\nstdout={out}");
    assert!(
        out.contains("caller"),
        "impact doIt must reach `caller` at hop 1; got: {out:?}"
    );
    assert!(
        out.contains("hop 1"),
        "impact must report hop distance for direct callers; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 7 — search-symbols: finds the definition of a symbol across the repo.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_haskell_search_symbols_finds_all_definitions() {
    let (repo, store) = index_fixture("search-symbols");
    let (code, out, err) = run(&["search-symbols", "doIt"], &repo, &store);
    assert_eq!(
        code, 0,
        "search-symbols should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("doIt"),
        "search-symbols doIt must find the doIt symbol; got: {out:?}"
    );
    assert!(
        out.contains("src/Helper.hs:"),
        "search-symbols must print the symbol's file:line (src/Helper.hs); got: {out:?}"
    );
    assert!(
        out.contains("Function"),
        "search-symbols must print the node label (Function for Haskell top-level defs); got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 8 — brief: definition + callers bundled in one call.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_haskell_brief_shows_definition_with_callers() {
    let (repo, store) = index_fixture("brief");
    let (code, out, err) = run(&["brief", "doIt"], &repo, &store);
    assert_eq!(code, 0, "brief should exit 0; stderr={err}");
    assert!(
        out.contains("doIt") && out.contains("doIt x"),
        "brief must show the definition body of doIt; got: {out}"
    );
    assert!(
        out.contains("(src/Helper.hs:"),
        "brief header must report the expanded source span (src/Helper.hs); got: {out}"
    );
    assert!(
        out.contains("-- CALLERS") && out.contains("caller"),
        "brief must list callers including `caller`; got: {out}"
    );
}

// ---------------------------------------------------------------------------
// 9 — path: shortest CALLS path connects caller to helper.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_haskell_path_connects_caller_to_helper() {
    let (repo, store) = index_fixture("path");
    // `path --from caller --to doIt` over CALLS must find the single-hop
    // path caller -> doIt.
    let (code, out, err) = run(&["path", "--from", "caller", "--to", "doIt"], &repo, &store);
    assert_eq!(
        code, 0,
        "path caller->doIt should exist and exit 0; stderr={err}\nstdout={out}"
    );
    let caller_idx = out
        .find("caller")
        .expect("path must include start `caller`");
    let callee_idx = out.find("doIt").expect("path must include goal `doIt`");
    assert!(
        caller_idx < callee_idx,
        "path steps must be ordered caller -> doIt; got: {out:?}"
    );
    assert!(
        out.contains("src/Main.hs:") && out.contains("src/Helper.hs:"),
        "path steps must carry file:line for both endpoints; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 10 — graph_survives_reindex: a second index run produces the same edges.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_haskell_graph_survives_reindex() {
    let (repo, store) = index_fixture("reindex");
    // First snapshot: who-calls doIt must list `caller`.
    let (_c1, out1, _e1) = run(&["who-calls", "doIt"], &repo, &store);
    assert!(
        out1.contains("caller"),
        "first run must list caller; got: {out1:?}"
    );

    // Re-index (idempotent — store upserts on (source, target, edge_type)).
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "second index . must succeed; stderr={err}\nstdout={out}"
    );

    // Second snapshot must report the same CALLS edge, no drift, no stale refusal.
    let (code, out2, err) = run(&["who-calls", "doIt"], &repo, &store);
    assert_eq!(
        code, 0,
        "second who-calls should still exit 0 (no spurious freshness refusal); stderr={err}\nstdout={out2}"
    );
    assert!(
        out2.contains("caller"),
        "second run must still list caller (same edges after reindex); got: {out2:?}"
    );
    assert!(
        !out2.contains("refreshing"),
        "second run must not label the graph as refreshing; got: {out2:?}"
    );
}

// ---------------------------------------------------------------------------
// 11 — stale_edit_detected: editing a file after index triggers freshness.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_haskell_stale_edit_detected() {
    let (repo, store) = index_fixture("stale-edit");
    // Drift the index: rename `doIt` -> `doIt_renamed` in Helper.hs, so the
    // symbol lookup misses and the freshness gate / heal-in-band contract
    // must fire on the next navigation command.
    std::fs::write(
        repo.join("src/Helper.hs"),
        r#"module Helper where

doIt_renamed :: Int -> Int
doIt_renamed x = x

HELPER_VALUE :: Int
HELPER_VALUE = 7

marker :: Int -> Int
marker _ = HELPER_VALUE
"#,
    )
    .unwrap();

    let (code, out, err) = run(&["who-calls", "doIt_renamed", "--json"], &repo, &store);
    // Heal-in-band contract (1b7135b): a healable drift is reindexed in-band
    // and served fresh — the renamed symbol resolves against the NEW graph.
    assert_eq!(
        code, 0,
        "healable stale edit must be healed in-band and served; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(
        v["fresh"], true,
        "the healed response must prove freshness; got: {v:?}"
    );
    assert_eq!(
        v["symbol_found"], true,
        "post-drift symbol must resolve against the healed graph; got: {v:?}"
    );
}

// ---------------------------------------------------------------------------
// 12 — declarative_or_edge_case: HASKELL-spezifischer Grenzfall.
//
// Begründung: In `crates/parser/src/extract.rs` werden HASKELL-`data`- und
// `newtype`-Deklarationen (`HASKELL_TYPE_KINDS = ["class", "data_type",
// "newtype"]`) BEREITS zu "Class"-Knoten — die HASKELL-spezifische
// Entscheidung ist, dass `data Widget = Widget Int` als Class (nicht Type
// / Struct) im Graphen erscheint. search-symbols muss das Label "Class"
// und den Dateipfad `src/Types.hs` liefern, sonst fehlt die Klassifikation,
// die der Resolver für die nachfolgende USAGE-Auflösung (Zelle 5) benötigt.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_haskell_declarative_data_labelled_as_class() {
    let (repo, store) = index_fixture("data-as-class");
    // `data Widget = ...` is a Haskell algebraic data type. The bespoke
    // haskell extractor (`extract_haskell`) emits it as a "Class" node
    // (Haskell has no Enum / Interface / Struct / Type label); search-symbols
    // must surface it as such.
    let (code, out, err) = run(&["search-symbols", "Widget", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "search-symbols Widget should exit 0; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid search-symbols json: {e}; stdout={out:?}"));
    let hits = v["hits"].as_array().expect("search-symbols hits array");
    assert!(
        hits.iter().any(|hit| {
            hit["label"] == "Class"
                && hit["name"] == "Widget"
                && hit["file_path"]
                    .as_str()
                    .unwrap_or("")
                    .ends_with("Types.hs")
        }),
        "Haskell `data Widget` must surface as a Class definition in Types.hs; got: {v:?}"
    );
}
