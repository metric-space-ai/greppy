//! Graph-Zertifizierungs-Grid für ZIG — 12 Zellen.
//!
//! ZIG ist über alle zwölf Navigationszellen graph-zertifiziert. Das Grid hält
//! insbesondere die sprachspezifische `@import`-Auflösung, flache Methoden und
//! als Variablen modellierte `const X = struct { … }`-Deklarationen fest.
//!
//! Fixture-Repo (`src/`):
//! * `main.zig`  — `caller()`, `render()`, `uncalled()`; importiert
//!   `helper.zig` und `types.zig` per `@import("…")`.
//! * `helper.zig` — `doIt(x: u32)`, `HELPER_VALUE: u32`
//!   (`pub const HELPER_VALUE = 7` ist ZIGs einzige Modul-„Variable"-Form).
//! * `types.zig` — `pub const Widget = struct { value: u32 }`
//!   (in ZIG: ein **Variable**, nicht „Type"/„Class" — siehe Zelle 12).
//!
//! Soll-Kanten über Dateigrenzen:
//! * `caller()`    --CALLS-->  `doIt`              (helper.zig, via resolve_call_target)
//! * `render(w)`   --USAGE-->  `Widget`            (types.zig, type_identifier)
//! * `render(w)`   --USAGE-->  `HELPER_VALUE`      (helper.zig, identifier)
//! * `render(w)`   --USAGE-->  `helper`            (main.zig Variable, identifier)
//! * `render(w)`   --USAGE-->  `types`             (main.zig Variable, identifier)
//! * `main.zig`    --IMPORTS--> `helper` / `types` / `std`
//!
//! Wichtige Hinweise zu ZIG-spezifischen Eigenheiten:
//!
//! * **IMPORTS-Namensräume:** Der Zig-Pass entfernt bei `@import("helper.zig")`
//!   das `.zig`-Suffix aus `imported_name` und emittiert pro Zig-Datei einen
//!   importierbaren Class-Namensraum (`helper.zig` -> `helper`). Erweiterungslose
//!   Pakete wie `std` erhalten einen externen Namensraum-Stub. Damit resolven
//!   lokale Dateiimporte und SDK-Pakete ohne Änderungen am gemeinsamen Resolver.
//! * **USAGE als TYPE_REF-Parität:** ZIG hat keine eigene `TYPE_REF`-Pass;
//!   `extract_zig::emit_zig_usages` emittiert sowohl `identifier`- als auch
//!   `type_identifier`-Referenzen als `USAGE`-Edges (über `ref_name`).
//!   Der Resolver mappt sie auf `USAGE_LABELS` (Function/Method/Class/…
//!   + Variable/Field) — `Widget` als type_identifier in `w: types.Widget`
//!     landet damit auf `types.zig::Variable::Widget`.
//! * **Variable vs. Type:** `pub const Widget = struct { … };` ist in ZIG
//!   ein **Variable**-Knoten (`extract_zig::emit_zig_variables` fängt jede
//!   `variable_declaration` am File-Root). Die Class-Knoten des Zig-Passes sind
//!   ausschließlich Datei-/Paket-Namensräume; `struct { … }` bleibt ein
//!   anonymes Init-Literal, kein deklarativer Container. Zelle 12 dokumentiert
//!   diesen Sprung.
//! * **Flattened Methods:** Struct-Methoden werden ebenfalls als freie
//!   `Function`-Knoten emittiert, weil tree-sitter-zig's `struct_declaration`/
//!   `enum_declaration`/`union_declaration` unbenannte Container-Knoten sind
//!   und der class-def-Pfad damit nicht greift. In diesem Fixture nicht
//!   ausgeübt (helper.zig exportiert keine Struct-Methoden).
//! * **builtin_function ohne CALLS-Edge:** `@import("…")` ist eine
//!   `builtin_function`, nicht eine `call_expression`. Die `emit_zig_calls`-
//!   Pass läuft nur über `call_expression` und emittiert deshalb **keine**
//!   CALLS-Kante für `@import`. Statt dessen fängt die IMPORTS-Query
//!   `(builtin_function (builtin_identifier) @builtin (#eq? @builtin "@import"))`
//!   den Aufruf separat.

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
    let dir = std::env::temp_dir().join(format!("greppy-cli-gridzig-{tag}-{pid}-{n}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create Zig scratch directory");
    dir
}

/// Build a git-rooted Zig repo exercising all four expected cross-file edge
/// shapes:
///
/// * CALLS:    `caller()`         -> `doIt()`      (helper.zig)
/// * USAGE:    `render(w: …)`     -> `Widget`      (types.zig, type_identifier)
/// * USAGE:    `render()` body    -> `HELPER_VALUE`(helper.zig, identifier)
/// * USAGE:    `render()` body    -> `helper`      (main.zig Variable)
/// * IMPORTS:  `@import("…")`      -> Datei-/Paket-Namensraum
fn make_zig_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).expect("create Zig fixture src directory");
    // `.git` is the repo-root marker resolve_root walks up to find.
    std::fs::create_dir_all(repo.join(".git")).expect("create repo-root marker");

    // main.zig — caller / render / uncalled; bindet helper.zig und types.zig
    // per `@import("…")`; alle Identifier-Referenzen über Dateigrenzen.
    std::fs::write(
        src.join("main.zig"),
        r#"const std = @import("std");
const helper = @import("helper.zig");
const types = @import("types.zig");

pub fn caller() u32 {
    return helper.doIt(0);
}

pub fn render(w: types.Widget) u32 {
    _ = w;
    return helper.HELPER_VALUE;
}

pub fn uncalled() u32 {
    return 0;
}
"#,
    )
    .expect("write main.zig");

    // helper.zig — `doIt()` (Function) und `HELPER_VALUE` (Variable, via
    // `pub const … = …` am File-Root).
    std::fs::write(
        src.join("helper.zig"),
        r#"pub fn doIt(x: u32) u32 {
    _ = x;
    return HELPER_VALUE;
}

pub const HELPER_VALUE: u32 = 7;
"#,
    )
    .expect("write helper.zig");

    // types.zig — `pub const Widget = struct { … };` ist in ZIG ein
    // **Variable**-Knoten, nicht „Type" oder „Class".
    std::fs::write(
        src.join("types.zig"),
        r#"pub const Widget = struct {
    value: u32,
};
"#,
    )
    .expect("write types.zig");

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
    let (repo, store) = make_zig_repo(tag);
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed for Zig fixture; stderr={err}\nstdout={out}"
    );
    (repo, store)
}

// ---------------------------------------------------------------------------
// 1 — who-calls: incoming CALLS edge resolves to the cross-file caller.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_zig_who_calls_finds_cross_file_caller() {
    let (repo, store) = index_fixture("who-calls-cross");
    // `doIt` is defined in helper.zig and called by `caller` in main.zig.
    // The CALLS pass emits `target = "main.zig::Function::doIt"` (file-local),
    // but `resolve_call_target` falls back to the unique global name lookup
    // and lands on `helper.zig::Function::doIt`.
    let (code, out, err) = run(&["who-calls", "doIt"], &repo, &store);
    assert_eq!(
        code, 0,
        "who-calls should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("caller"),
        "who-calls doIt must list the cross-file caller `caller` (main.zig); got: {out:?}"
    );
    assert!(
        out.contains("src/main.zig:"),
        "who-calls must print the caller's file:line (src/main.zig); got: {out:?}"
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
fn graph_grid_zig_who_calls_empty_for_uncalled() {
    let (repo, store) = index_fixture("who-calls-empty");
    // `render` is defined in main.zig but never called from anywhere.
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
fn graph_grid_zig_callees_lists_cross_file_target() {
    let (repo, store) = index_fixture("callees-cross");
    // `caller` calls `doIt()` which is defined in helper.zig. The Zig CALLS
    // pass emits a field_expression `helper.doIt` and resolves it by the
    // trailing identifier (`doIt`); `resolve_call_target` lands on
    // `helper.zig::Function::doIt`.
    let (code, out, err) = run(&["callees", "caller"], &repo, &store);
    assert_eq!(code, 0, "callees should exit 0; stderr={err}\nstdout={out}");
    assert!(
        out.contains("doIt"),
        "callees caller must list the cross-file callee `doIt`; got: {out:?}"
    );
    assert!(
        out.contains("src/helper.zig:"),
        "callees must print the callee's file:line (src/helper.zig); got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 4 — find-usages: covers CALLS AND IMPORTS reference edges.
//
// Zig normalisiert `@import("helper.zig")` auf `imported_name = "helper"`;
// Datei-Namensräume und der externe `std`-Stub sind importierbare Ziele.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_zig_find_usages_covers_call_and_import() {
    let (repo, store) = index_fixture("find-usages-call-import");

    // (a) CALLS-Leg: `find-usages doIt` muss die CALLS-Kante aus `caller`
    // (main.zig) zurück nach `doIt` (helper.zig) zeigen.
    let (code, calls, err) = run(&["find-usages", "doIt"], &repo, &store);
    assert_eq!(
        code, 0,
        "find-usages doIt should exit 0; stderr={err}\nstdout={calls}"
    );
    assert!(
        calls.contains("CALLS") && calls.contains("caller"),
        "find-usages doIt must show CALLS edge from caller (main.zig); got: {calls:?}"
    );

    // (b) IMPORTS-Leg: `find-usages std` zeigt den IMPORTS-Edge aus main.zig
    // (`@import("std")`) zum externen Paket-Namensraum.
    let (code, imports, err) = run(&["find-usages", "std"], &repo, &store);
    assert_eq!(
        code, 0,
        "find-usages std should exit 0 (either real graph or content-fallback); stderr={err}\nstdout={imports}"
    );
    assert!(
        imports.contains("IMPORTS") && imports.contains("src/main.zig"),
        "find-usages std must show IMPORTS edge from src/main.zig (ZIG `@import(\"std\")`); got: {imports:?}"
    );

    // The `.zig` basename path follows the same contract and resolves to the
    // imported file namespace rather than being silently discarded.
    let (code, imports, err) = run(&["find-usages", "helper"], &repo, &store);
    assert_eq!(
        code, 0,
        "find-usages helper should exit 0; stderr={err}\nstdout={imports}"
    );
    assert!(
        imports.contains("IMPORTS") && imports.contains("src/main.zig"),
        "find-usages helper must show the @import(\"helper.zig\") edge; got: {imports:?}"
    );
}

// ---------------------------------------------------------------------------
// 5 — find-usages: type-reference (USAGE for `type_identifier` in Zig).
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_zig_find_usages_type_reference() {
    let (repo, store) = index_fixture("find-usages-type");
    // `Widget` is referenced as `w: types.Widget` in `render`. The Zig
    // USAGE pass emits `type_identifier`s as USAGE edges (TYPE_REF parity),
    // keyed on `ref_name = "Widget"`; the resolver lands on
    // `types.zig::Variable::Widget`.
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
        out.contains("src/main.zig:"),
        "find-usages must print the referrer's file:line (src/main.zig); got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 6 — impact: transitive blast radius reaches the cross-file caller.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_zig_impact_transitive_reaches_caller() {
    let (repo, store) = index_fixture("impact-transitive");
    // `impact doIt` must report `caller` as incoming CALLS dependent.
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
// 7 — search-symbols: finds every definition across the repo.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_zig_search_symbols_finds_all_definitions() {
    let (repo, store) = index_fixture("search-symbols");
    for needle in [
        "caller",
        "render",
        "doIt",
        "HELPER_VALUE",
        "Widget",
        "uncalled",
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
    // `doIt` lives in helper.zig — the file_path must reflect that.
    let (code, out, err) = run(&["search-symbols", "doIt"], &repo, &store);
    assert_eq!(
        code, 0,
        "search-symbols doIt should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("src/helper.zig:"),
        "search-symbols doIt must print the definition file:line (src/helper.zig); got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 8 — brief: definition + callers bundled in one call.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_zig_brief_shows_definition_with_callers() {
    let (repo, store) = index_fixture("brief");
    let (code, out, err) = run(&["brief", "doIt"], &repo, &store);
    assert_eq!(code, 0, "brief should exit 0; stderr={err}");
    assert!(
        out.contains("doIt"),
        "brief must show the definition body of doIt; got: {out}"
    );
    assert!(
        out.contains("(src/helper.zig:"),
        "brief header must report the expanded source span (src/helper.zig); got: {out}"
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
fn graph_grid_zig_path_connects_caller_to_helper() {
    let (repo, store) = index_fixture("path");
    // `path --from caller --to doIt` over CALLS must find the single-hop
    // path caller -> doIt (via resolve_call_target's unique lookup).
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
        out.contains("src/main.zig:") && out.contains("src/helper.zig:"),
        "path steps must carry file:line for both endpoints; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 10 — graph_survives_reindex: a second index run produces the same edges.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_zig_graph_survives_reindex() {
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
fn graph_grid_zig_stale_edit_detected() {
    let (repo, store) = index_fixture("stale-edit");
    // Drift the index: rename `doIt` -> `doIt_renamed` in helper.zig, so the
    // symbol lookup misses and the freshness gate / heal-in-band contract
    // must fire on the next navigation command.
    std::fs::write(
        repo.join("src/helper.zig"),
        r#"pub fn doIt_renamed(x: u32) u32 {
    _ = x;
    return HELPER_VALUE;
}

pub const HELPER_VALUE: u32 = 7;
"#,
    )
    .expect("rewrite helper.zig for stale-edit scenario");

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
// 12 — declarative_or_edge_case: ZIG-spezifischer Grenzfall.
//
// Begründung: In `crates/parser/src/extract.rs` (`emit_zig_variables`) wird
// jede `variable_declaration` am File-Root als **Variable**-Knoten erfasst —
// auch `pub const X = struct { … };`. Damit hat ZIG **keine** Type/Class/
// Struct-Knoten; Typen leben als Variablen. `search-symbols Widget` muss
// daher den Label **„Variable"** tragen (nicht „Type", „Struct" oder „Class"),
// und das `file_path` muss `src/types.zig` sein — sonst fehlt die
// Klassifikation, die der Resolver für die `USAGE`-Auflösung der
// `type_identifier`-Referenz in `w: types.Widget` (Zelle 5) benötigt.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_zig_declarative_struct_labelled_as_variable() {
    let (repo, store) = index_fixture("struct-as-variable");
    // `pub const Widget = struct { value: u32 };` is a Zig algebraic-data-
    // type literal. The bespoke `extract_zig` emits it as a "Variable" node
    // (`emit_zig_variables` walks every `variable_declaration` child of the
    // file root, no class-def/struct-def path). `search-symbols` must
    // surface it as a Variable in types.zig.
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
            hit["label"] == "Variable"
                && hit["name"] == "Widget"
                && hit["file_path"]
                    .as_str()
                    .unwrap_or("")
                    .ends_with("types.zig")
        }),
        "Zig `pub const Widget = struct {{ … }}` must surface as a Variable definition in types.zig; got: {v:?}"
    );
}
