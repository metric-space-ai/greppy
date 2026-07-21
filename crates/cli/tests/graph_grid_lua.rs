//! Graph-Zertifizierungs-Grid für LUA — 12 Zellen.
//!
//! Die Datenpfad-Extraktion (`extract_lua` in
//! `crates/parser/src/extract.rs`) emittiert `USAGE`-Kanten über
//! `lua_emit_usages` und CALLS-Kanten über `spec_calls` +
//! `lua_emit_module_scope_calls`. Dabei gelten folgende Lua-spezifische
//! Modellierungsgrenzen:
//!
//! * Es gibt **keine** separate `TYPE_REF`-Pass: Typ-Referenzen und
//!   Wert-Referenzen laufen beide über die vereinheitlichte `USAGE`-Kante
//!   (C-Reference-Parität), die per `__ref__`-Platzhalter emittiert und
//!   vom Indexer name-basiert aufgelöst wird.
//! * Die `USAGE` Targets landen nur dann im Graph, wenn der
//!   `ref_name` projektweit *eindeutig* auf eine Definition in
//!   `USAGE_LABELS` (Function / Variable / Field / etc.) auflöst —
//!   mehrdeutige oder fehlende Referenzen werden vom Resolver
//!   stillschweigend verworfen.
//! * `IMPORTS` für `require "<file>"` resolvet nur, wenn ein
//!   importable Symbol mit Namen `<file>` existiert; ein bloßer
//!   `require 'helper'` ohne ein `function helper()`-Style Top-Level
//!   Symbol findet *kein* IMPORTS-Target (Modul-Dateien sind keine
//!   IMPORTABLE_LABELS).
//! * `lua_emit_module_variables` fängt nur `variable_declaration`-
//!   Knoten (also `local X = …`) am Datei-Wurzel; ein nackter
//!   `X = …`-Assignment wird *nicht* als Variable erfasst.
//! * Lua-Deklarationen wie `function helper.do_it()` bewahren den vollen
//!   dotted Namen im `name`-Feld. Der Navigations-Lookup akzeptiert deshalb
//!   Postel-konform sowohl `helper.do_it` als auch den nackten Leaf `do_it`:
//!   qualifizierte Anfragen prüfen den vollen Namen als Kandidaten, nackte
//!   Anfragen matchen zusätzlich das letzte dotted Segment.
//!
//! Fixture-Repo (`src/`):
//! * `main.lua`   — `caller()`, `render()`, `build()`, `uncalled()`;
//!   importiert `helper.lua` und `types.lua` per `require`.
//! * `helper.lua` — `function helper.do_it()`, `function helper()`,
//!   `local LIMIT = 99` (Modul-Konstante als USAGE-Target).
//! * `types.lua`  — `local Widget = {}`, `local Marker = {}`
//!   (Modul-Variablen als USAGE-Target).
//!
//! Soll-Kanten über Dateigrenzen:
//! * `caller()`     --CALLS--> `helper.do_it()` (helper.lua)
//! * `caller()`     --USAGE--> `LIMIT`          (helper.lua)
//! * `render(w)`    --USAGE--> `Widget`         (types.lua)
//! * `build()`      --USAGE--> `Marker`         (types.lua)
//! * `main.lua`     --IMPORTS--> `function helper` (helper.lua)

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
    let dir = std::env::temp_dir().join(format!("greppy-cli-gridlua-{tag}-{pid}-{n}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create Lua scratch directory");
    dir
}

/// Build a Lua repo exercising all four expected cross-file edge shapes:
///
/// * CALLS:   `caller()`           -> `helper.do_it()`     (helper.lua)
/// * USAGE:   `caller()`            -> `LIMIT`             (helper.lua)
/// * USAGE:   `render(w)`           -> `Widget`            (types.lua)
/// * USAGE:   `build()`             -> `Marker`            (types.lua)
/// * IMPORTS: `require 'helper'`    -> `function helper`   (helper.lua)
///   `require 'types'`     -> <NOT a Function>    (no resolvable target)
fn make_lua_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).expect("create Lua fixture src directory");
    // `.git` is the repo-root marker resolve_root walks up to find.
    std::fs::create_dir_all(repo.join(".git")).expect("create repo-root marker");

    // main.lua — caller/render/build/uncalled, requires both modules.
    std::fs::write(
        src.join("main.lua"),
        r#"require 'helper'
require 'types'

function caller()
    local v = helper.do_it()
    return v + LIMIT
end

function render(w)
    local widget = Widget
    return widget + w
end

function build()
    local m = Marker
    return m
end

function uncalled()
    return 0
end
"#,
    )
    .expect("write main.lua");

    // helper.lua — `helper` (importable Function), `helper.do_it` (dotted
    // Function), `LIMIT` (module Variable via `local`).
    std::fs::write(
        src.join("helper.lua"),
        r#"function helper()
    return 0
end

function helper.do_it()
    return 42
end

local LIMIT = 99
"#,
    )
    .expect("write helper.lua");

    // types.lua — bare-module Variables as USAGE targets (no `Class` ownership
    // in Lua; label is `Variable`).
    std::fs::write(
        src.join("types.lua"),
        r#"local Widget = {}
local Marker = {}
"#,
    )
    .expect("write types.lua");

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

/// Index the fixture once; assert the indexer accepted Lua input.
fn index_fixture(tag: &str) -> (PathBuf, PathBuf) {
    let (repo, store) = make_lua_repo(tag);
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed for Lua fixture; stderr={err}\nstdout={out}"
    );
    (repo, store)
}

// ---------------------------------------------------------------------------
// 1 — who-calls: incoming CALLS edge resolves to the cross-file caller.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_lua_who_calls_finds_cross_file_caller() {
    let (repo, store) = index_fixture("who-calls");

    // `helper.do_it` is defined in helper.lua and called by `caller` in main.lua.
    let (code, out, err) = run(&["who-calls", "do_it"], &repo, &store);
    assert_eq!(
        code, 0,
        "who-calls should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("caller"),
        "who-calls do_it must list the cross-file caller `caller`; got: {out:?}"
    );
    assert!(
        out.contains("src/main.lua:"),
        "who-calls must print the caller's file:line (src/main.lua); got: {out:?}"
    );
    assert!(
        !out.contains("(no callers)"),
        "who-calls must find at least one caller; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 2 — who-calls: uncalled symbol reports "(no callers)".
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_lua_who_calls_empty_for_uncalled() {
    let (repo, store) = index_fixture("who-calls-none");

    // `Marker` is a Variable; nothing CALLS it.
    let (code, out, _err) = run(&["who-calls", "Marker"], &repo, &store);
    assert_eq!(code, 0);
    assert!(
        out.contains("(no callers)"),
        "who-calls on a never-called Variable must report no callers; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 3 — callees: outgoing CALLS edges resolve to the cross-file target.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_lua_callees_lists_cross_file_target() {
    let (repo, store) = index_fixture("callees");

    // `caller` calls `helper.do_it()` which lives in helper.lua.
    let (code, out, err) = run(&["callees", "caller"], &repo, &store);
    assert_eq!(code, 0, "callees should exit 0; stderr={err}\nstdout={out}");
    assert!(
        out.contains("do_it"),
        "callees of caller must list the cross-file `do_it`; got: {out:?}"
    );
    assert!(
        out.contains("src/helper.lua:"),
        "callees must print the callee's file:line (src/helper.lua); got: {out:?}"
    );
    assert!(
        !out.contains("(no callees)"),
        "callees must not be empty for caller; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 4 — find-usages: covers CALLS edge (dotted-target lookup is the expected gap).
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_lua_find_usages_covers_call_and_import() {
    let (repo, store) = index_fixture("usages-call-import");

    // (a) Both the provider-preserved full name and its naked leaf resolve to
    // the same dotted Function node and therefore the same CALLS edge.
    for symbol in ["do_it", "helper.do_it"] {
        let (code, calls, err) = run(&["find-usages", symbol], &repo, &store);
        assert_eq!(
            code, 0,
            "find-usages {symbol} should exit 0; stderr={err}\nstdout={calls}"
        );
        assert!(
            calls.contains("CALLS") && calls.contains("caller"),
            "find-usages {symbol} must show CALLS edge from caller; got: {calls:?}"
        );
    }

    // (b) `find-usages helper` — the IMPORTS target name. The IMPORTS edge
    // lives on `helper` (an importable Function in helper.lua).
    let (code, imports, err) = run(&["find-usages", "helper"], &repo, &store);
    assert_eq!(
        code, 0,
        "find-usages helper should exit 0; stderr={err}\nstdout={imports}"
    );
    assert!(
        imports.contains("IMPORTS") && imports.contains("src/main.lua"),
        "find-usages helper must show IMPORTS edge from src/main.lua; got: {imports:?}"
    );
}

// ---------------------------------------------------------------------------
// 5 — find-usages: cross-file USAGE on a type-like Variable.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_lua_find_usages_type_reference() {
    let (repo, store) = index_fixture("usages-type");

    // `Widget` is the receiver of `local widget = Widget` in `render`.
    // The Lua extractor emits a USAGE edge (C-reference parity for both
    // type-refs and uses) with ref_name "Widget"; the indexer's name
    // resolver must land it on `types.lua::Variable::Widget`.
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
        "find-usages Widget must list `render` as the type-like referrer; got: {out:?}"
    );
    assert!(
        out.contains("src/main.lua:"),
        "find-usages must print the referrer's file:line; got: {out:?}"
    );
    assert!(
        !out.contains("(no usages)"),
        "Widget is referenced cross-file; usages must be non-empty; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 6 — impact: transitive blast radius reaches the cross-file caller.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_lua_impact_transitive_reaches_caller() {
    let (repo, store) = index_fixture("impact");

    // Both accepted spellings must report `caller` as an incoming CALLS
    // dependent of the provider-preserved dotted declaration.
    for symbol in ["do_it", "helper.do_it"] {
        let (code, out, err) = run(&["impact", symbol], &repo, &store);
        assert_eq!(
            code, 0,
            "impact {symbol} should exit 0; stderr={err}\nstdout={out}"
        );
        assert!(
            out.contains("caller"),
            "impact {symbol} must reach the cross-file caller `caller`; got: {out:?}"
        );
        assert!(
            out.contains("src/main.lua:") || out.contains("hop"),
            "impact must print actionable file:line or hop info; got: {out:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// 7 — search-symbols: finds every Lua definition across the repo.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_lua_search_symbols_finds_all_definitions() {
    let (repo, store) = index_fixture("symbols");

    for needle in [
        "caller", "render", "build", "do_it", "Widget", "Marker", "LIMIT",
    ] {
        let (code, out, err) = run(&["search-symbols", needle], &repo, &store);
        assert_eq!(
            code, 0,
            "search-symbols {needle} should exit 0; stderr={err}\nstdout={out}"
        );
        assert!(
            out.contains(needle),
            "search-symbols {needle} must include the name; got: {out:?}"
        );
    }

    // The fully-qualified dotted def `helper.do_it` lives in helper.lua.
    let (code, out, err) = run(&["search-symbols", "helper.do_it"], &repo, &store);
    assert_eq!(code, 0, "stderr={err}");
    assert!(
        out.contains("src/helper.lua:"),
        "search-symbols helper.do_it must print the definition file:line (src/helper.lua); got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 8 — brief: bundles the definition + its callers in one call.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_lua_brief_shows_definition_with_callers() {
    let (repo, store) = index_fixture("brief");

    for symbol in ["do_it", "helper.do_it"] {
        let (code, out, err) = run(&["brief", symbol], &repo, &store);
        assert_eq!(code, 0, "brief {symbol} should exit 0; stderr={err}");
        assert!(
            out.contains("do_it") || out.contains("function helper.do_it"),
            "brief {symbol} must show the definition; got: {out}"
        );
        assert!(
            out.contains("CALLERS") && out.contains("caller"),
            "brief {symbol} must list callers incl. `caller`; got: {out}"
        );
        assert!(
            out.contains("src/helper.lua"),
            "brief must print the definition's source file (src/helper.lua); got: {out}"
        );
    }
}

// ---------------------------------------------------------------------------
// 9 — path: shortest CALLS path connects caller to helper.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_lua_path_connects_caller_to_helper() {
    let (repo, store) = index_fixture("path");

    // Both destination spellings must find the single-hop CALLS path
    // caller -> helper.do_it.
    for symbol in ["do_it", "helper.do_it"] {
        let (code, out, err) = run(&["path", "--from", "caller", "--to", symbol], &repo, &store);
        assert_eq!(
            code, 0,
            "path caller->{symbol} should exit 0; stderr={err}\nstdout={out}"
        );
        assert!(
            out.contains("caller"),
            "path caller->{symbol} must include the start symbol; got: {out:?}"
        );
        assert!(
            out.contains("do_it"),
            "path caller->{symbol} must include the destination; got: {out:?}"
        );
        assert!(
            out.contains("src/main.lua:") || out.contains("src/helper.lua:"),
            "path must print actionable file:line for one of the endpoints; got: {out:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// 10 — graph_survives_reindex: a second index run produces the same edges.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_lua_graph_survives_reindex() {
    let (repo, store) = index_fixture("reindex");

    // First snapshot: who-calls do_it must list `caller`.
    let (code1, before, _e1) = run(&["who-calls", "do_it"], &repo, &store);
    assert_eq!(code1, 0);

    // Re-index (idempotent — store upserts on (source, target, edge_type)).
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "second index . must succeed; stderr={err}\nstdout={out}"
    );

    // Second snapshot must report the same CALLS edge.
    let (code2, after, _e2) = run(&["who-calls", "do_it"], &repo, &store);
    assert_eq!(code2, 0);

    assert!(
        before.contains("caller") && after.contains("caller"),
        "who-calls do_it must keep listing caller across reindex; before={before:?}\nafter={after:?}"
    );
    assert!(
        before.contains("src/main.lua:") && after.contains("src/main.lua:"),
        "who-calls file:line anchor must persist across reindex; before={before:?}\nafter={after:?}"
    );
}

// ---------------------------------------------------------------------------
// 11 — stale_edit_detected: editing helper.lua after index triggers freshness.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_lua_stale_edit_detected() {
    let (repo, store) = index_fixture("stale");

    // Drift the index: rename the dotted def in helper.lua. The dedotted
    // name `do_it_renamed` is what we'll query.
    std::fs::write(
        repo.join("src/helper.lua"),
        r#"function helper()
    return 0
end

function helper.do_it_renamed()
    return 42
end

local LIMIT = 99
"#,
    )
    .expect("rewrite helper.lua for stale-edit scenario");

    // Heal-in-band contract (1b7135b): a healable drift is reindexed in-band
    // and served fresh against the new graph. Both accepted spellings of the
    // renamed dotted symbol must resolve (the first query performs the heal).
    for symbol in ["do_it_renamed", "helper.do_it_renamed"] {
        let (code, out, err) = run(&["who-calls", symbol, "--json"], &repo, &store);
        assert_eq!(
            code, 0,
            "healable stale edit for {symbol} must be healed in-band and served; stderr={err}\nstdout={out}"
        );
        let v: serde_json::Value = serde_json::from_str(&out)
            .unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
        assert_eq!(v["command"], "who-calls");
        assert_eq!(
            v["fresh"], true,
            "the healed response must prove freshness; got: {v:?}"
        );
        assert_eq!(
            v["symbol_found"], true,
            "post-drift symbol {symbol} must resolve against the healed graph; got: {v:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// 12 — declarative_or_edge_case: LUA-spezifischer Grenzfall.
//
// Begründung: LUA hat keine `class`-/Struct-Ownership; Typ- und
// Wert-Referenzen werden beide über die vereinheitlichte `USAGE`-Kante
// emittiert, deren Target ein `__ref__`-Platzhalter ist, den der Resolver
// projekt-eindeutig auf eine Definition in `USAGE_LABELS` auflöst. Dieser
// Test verifiziert, dass der c-reference-emittierte USAGE-Edge für `LIMIT`
// (referenziert im Body von `caller()`) auf `helper.lua::Variable::LIMIT`
// auflöst — also der name-basierte USAGE-Pfad für eine *Modul-Variable* in
// einem anderen File funktioniert. Außerdem verifiziert er, dass
// `find-usages LIMIT` die `USAGE`-Kante zurück nach `caller` zurück-
// attributed (also die `lua_enclosing_func_qname`-Walk die Quelle korrekt
// zuweist) — und dass keine `CALLS`-Kante verloren geht, da `LIMIT` keine
// aufrufbare Funktion ist.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_lua_declarative_or_edge_case() {
    let (repo, store) = index_fixture("limit-usage");

    // (a) `LIMIT` is the receiver of `v + LIMIT` in `caller()`.
    let (code, out, err) = run(&["find-usages", "LIMIT"], &repo, &store);
    assert_eq!(
        code, 0,
        "find-usages LIMIT should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("USAGE") && out.contains("caller"),
        "find-usages LIMIT must show USAGE from caller (cross-file Variable); got: {out:?}"
    );
    assert!(
        out.contains("src/main.lua:"),
        "find-usages LIMIT must print the referrer's file:line (src/main.lua); got: {out:?}"
    );
    assert!(
        !out.contains("(no usages)"),
        "LIMIT is referenced cross-file; usages must be non-empty; got: {out:?}"
    );

    // (b) `LIMIT` is a Module Variable — no one CALLS it.
    let (code, out, _err) = run(&["who-calls", "LIMIT"], &repo, &store);
    assert_eq!(code, 0);
    assert!(
        out.contains("(no callers)"),
        "who-calls LIMIT must report no callers (LIMIT is not a function); got: {out:?}"
    );
}
