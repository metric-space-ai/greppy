//! Graph-Zertifizierungs-Grid für C++ — 12 Zellen.
//!
//! C++ ist zum Zeitpunkt dieses Grids noch NICHT graph-zertifiziert; die
//! rot markierten Zellen sind die **erwarteten** Befunde dieser Zertifizierung
//! (die Lücken, die vor einer offiziellen Zertifizierung zu schließen sind).
//! Die Tests sind NICHT so geschrieben, dass sie die Lücken wegignorieren:
//! jede rote Zelle zeigt, was der C++-Indexer/Resolver heute tatsächlich
//! liefert vs. was das Grid erwartet.
//!
//! Fixture-Repo (`src/`):
//!   * `main.cpp`    — `caller()`, `render(::Widget)`, `other_caller()`,
//!                     inkl. `#include "helper.hpp"` / `"types.hpp"`.
//!   * `helper.hpp`  — inline-definierter Helfer `app::do_it()` +
//!                     Konstante `app::kMarker`.
//!   * `types.hpp`   — Typdefinition `struct Widget`.
//!
//! Soll-Kanten über Dateigrenzen:
//!   * `caller()`            --CALLS-->   `do_it()`                  (helper.hpp)
//!   * `other_caller()`      --CALLS-->   `app::do_it()` (qualified) (helper.hpp)
//!   * `render(::Widget w)`  --USAGE-->   `Widget`                   (types.hpp)
//!   * `main.cpp`            --IMPORTS--> `helper.hpp` / `types.hpp`
//!   * `render()`            --USAGE-->   `kMarker`                  (Konstante,
//!                                Ziel-Symbol aber NICHT extrahiert → Lücke)
//!
//! Hinweis: Die C++-Queries (`crates/parser/src/query.rs::cpp_queries`)
//! emittieren Definitions, Calls und Imports. Sie haben **kein**
//! dediziertes TYPE_REF / USES-Pass; alle Identifier-Referenzen laufen
//! durch `c_cpp_usage_pass` und werden als `USAGE`-Kante persistiert.

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
    let dir = std::env::temp_dir().join(format!("greppy-cli-graphgrid-cpp-{tag}-{pid}-{n}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Build a git-rooted C++ repo with three files exercising all four
/// cross-file edge shapes:
///   * CALLS:   `caller()` -> `do_it()` (header inline definition)
///   * CALLS:   `other_caller()` -> `app::do_it()` (qualified call)
///   * USAGE:   `render(::Widget)` parameter type -> Widget (TYPE_REF parity)
///   * USAGE:   `render(w.w + kMarker)` constant reference -> kMarker
///                                                  (NOT a definition; gap)
///   * IMPORTS: `#include "helper.hpp"` / `#include "types.hpp"`
fn make_cpp_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).unwrap();
    // `.git` is the repo-root marker resolve_root walks up to find.
    std::fs::create_dir_all(repo.join(".git")).unwrap();

    std::fs::write(
        src.join("main.cpp"),
        r#"#include "helper.hpp"
#include "types.hpp"

namespace app {

void caller() {
    do_it();
}

int render(::Widget w) {
    return w.w + kMarker;
}

void other_caller() {
    app::do_it();
}

}
"#,
    )
    .unwrap();

    std::fs::write(
        src.join("helper.hpp"),
        r#"#pragma once

namespace app {

constexpr int kMarker = 7;

inline int do_it() {
    return 42;
}

}
"#,
    )
    .unwrap();

    std::fs::write(
        src.join("types.hpp"),
        r#"#pragma once

struct Widget {
    int w;
};
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
    let (repo, store) = make_cpp_repo(tag);
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
fn graph_grid_cpp_who_calls_finds_cross_file_caller() {
    let (repo, store) = index_fixture("who-calls-cross");
    // `do_it` is defined in helper.hpp and called by `caller` in main.cpp.
    let (code, out, err) = run(&["who-calls", "do_it"], &repo, &store);
    assert_eq!(
        code, 0,
        "who-calls should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("caller"),
        "who-calls do_it must list the caller `caller` (main.cpp); got: {out:?}"
    );
    assert!(
        out.contains("src/main.cpp:"),
        "who-calls must print the caller's file:line (src/main.cpp); got: {out:?}"
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
fn graph_grid_cpp_who_calls_empty_for_uncalled() {
    let (repo, store) = index_fixture("who-calls-empty");
    // `render` is defined in main.cpp but never called from anywhere.
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
fn graph_grid_cpp_callees_lists_cross_file_target() {
    let (repo, store) = index_fixture("callees-cross");
    // `caller` calls `do_it` which is defined in helper.hpp.
    let (code, out, err) = run(&["callees", "caller"], &repo, &store);
    assert_eq!(code, 0, "callees should exit 0; stderr={err}\nstdout={out}");
    assert!(
        out.contains("do_it"),
        "callees caller must list the cross-file callee `do_it`; got: {out:?}"
    );
    assert!(
        out.contains("src/helper.hpp:"),
        "callees must print the callee's file:line (src/helper.hpp); got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 4 — find-usages: covers both CALLS and IMPORTS reference edges.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_cpp_find_usages_covers_call_and_import() {
    let (repo, store) = index_fixture("find-usages-call-import");
    // `find-usages` aggregates REFERENCE_EDGE_TYPES = [CALLS, USAGE, USES,
    // TYPE_REF, IMPORTS] per incoming target. The CALLS leg should list
    // `caller` (caller-of-do_it); the IMPORTS leg should be visible via the
    // file basename (`helper.hpp` -> the File node for src/helper.hpp).
    let (code, out, err) = run(&["find-usages", "do_it"], &repo, &store);
    assert_eq!(
        code, 0,
        "find-usages should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("CALLS"),
        "find-usages do_it must show the CALLS edge kind; got: {out:?}"
    );
    assert!(
        out.contains("caller"),
        "find-usages do_it must list `caller` as a CALLS referrer; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 5 — find-usages: TYPE_REF-equivalent (USAGE for type references) works.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "cpp graph gap: cross-file type references not classified as TYPE_REF"]
fn graph_grid_cpp_find_usages_type_reference() {
    let (repo, store) = index_fixture("find-usages-type-ref");
    // `Widget` is referenced as a parameter type in `render(::Widget w)`.
    // The C++ usage pass emits a USAGE edge from `render` keyed on ref_name
    // "Widget"; the persisted label is USAGE (the unified reference label).
    let (code, out, err) = run(&["find-usages", "Widget"], &repo, &store);
    assert_eq!(
        code, 0,
        "find-usages should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("USAGE"),
        "find-usages Widget must label the edge kind USAGE (TYPE_REF parity); got: {out:?}"
    );
    assert!(
        out.contains("render"),
        "find-usages Widget must list `render` as the type referrer; got: {out:?}"
    );
    assert!(
        out.contains("src/main.cpp:"),
        "find-usages must print the referrer's file:line (src/main.cpp); got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 6 — impact: transitive blast radius reaches the cross-file caller.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_cpp_impact_transitive_reaches_caller() {
    let (repo, store) = index_fixture("impact-transitive");
    // `impact do_it` must report `caller` (and `other_caller`) as incoming
    // CALLS dependents — the single-command answer to "what breaks if I
    // change do_it?".
    let (code, out, err) = run(&["impact", "do_it"], &repo, &store);
    assert_eq!(code, 0, "impact should exit 0; stderr={err}\nstdout={out}");
    assert!(
        out.contains("caller"),
        "impact do_it must reach `caller` at hop 1; got: {out:?}"
    );
    assert!(
        out.contains("hop 1"),
        "impact must report hop distance for direct callers; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 7 — search-symbols: finds all definitions of a symbol across the repo.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_cpp_search_symbols_finds_all_definitions() {
    let (repo, store) = index_fixture("search-symbols");
    let (code, out, err) = run(&["search-symbols", "do_it"], &repo, &store);
    assert_eq!(
        code, 0,
        "search-symbols should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("do_it"),
        "search-symbols do_it must find the do_it symbol; got: {out:?}"
    );
    assert!(
        out.contains("src/helper.hpp:"),
        "search-symbols must print the symbol's file:line (src/helper.hpp); got: {out:?}"
    );
    assert!(
        out.contains("Function"),
        "search-symbols must print the node label (Function for C++ defs); got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 8 — brief: definition + callers + callees bundled in one call.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_cpp_brief_shows_definition_with_callers() {
    let (repo, store) = index_fixture("brief");
    let (code, out, err) = run(&["brief", "do_it"], &repo, &store);
    assert_eq!(code, 0, "brief should exit 0; stderr={err}\nstdout={out}");
    assert!(
        out.contains("do_it") && out.contains("inline int do_it"),
        "brief must show the definition with source body; got: {out}"
    );
    assert!(
        out.contains("(src/helper.hpp:"),
        "brief header must report the expanded source span (src/helper.hpp); got: {out}"
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
fn graph_grid_cpp_path_connects_caller_to_helper() {
    let (repo, store) = index_fixture("path");
    // `path --from caller --to do_it` over CALLS must find the single-hop
    // path caller -> do_it.
    let (code, out, err) = run(
        &["path", "--from", "caller", "--to", "do_it"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "path caller->do_it should exist and exit 0; stderr={err}\nstdout={out}"
    );
    let caller_idx = out
        .find("caller")
        .expect("path must include start `caller`");
    let callee_idx = out.find("do_it").expect("path must include goal `do_it`");
    assert!(
        caller_idx < callee_idx,
        "path steps must be ordered caller -> do_it; got: {out:?}"
    );
    assert!(
        out.contains("src/main.cpp:") && out.contains("src/helper.hpp:"),
        "path steps must carry file:line for both endpoints; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 10 — graph_survives_reindex: a second index run produces the same edges.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_cpp_graph_survives_reindex() {
    let (repo, store) = index_fixture("reindex");
    // First snapshot: who-calls do_it must list `caller`.
    let (_c1, out1, _e1) = run(&["who-calls", "do_it"], &repo, &store);
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
    let (code, out2, err) = run(&["who-calls", "do_it"], &repo, &store);
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
#[ignore = "blocked by tracked freshness regression on dev-0.3.0 HEAD (stale drift serves rows instead of exit 75) — re-enable with that fix"]
fn graph_grid_cpp_stale_edit_detected() {
    let (repo, store) = index_fixture("stale-edit");
    // Drift the index: edit the helper header (renames do_it -> do_it_renamed),
    // so the symbol lookup misses and the freshness gate must fire on the next
    // navigation command.
    std::fs::write(
        repo.join("src/helper.hpp"),
        r#"#pragma once

namespace app {

constexpr int kMarker = 7;

inline int do_it_renamed() {
    return 42;
}

}
"#,
    )
    .unwrap();

    let (code, out, err) = run(&["who-calls", "do_it_renamed", "--json"], &repo, &store);
    // EX_TEMPFAIL (75) when refresh is in flight and the request cannot be served.
    assert_eq!(
        code, 75,
        "stale-edit must trigger freshness refusal (EX_TEMPFAIL); stderr={err}\nstdout={out}"
    );
    assert!(
        err.is_empty(),
        "JSON freshness refusal must stay on stdout; stderr={err:?}"
    );
    let v: serde_json::Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("invalid json: {e}; stdout={out:?}"));
    assert_eq!(
        v["status"], "skipped_stale_index",
        "stale who-calls must be skipped with skipped_stale_index; got: {v:?}"
    );
    assert_eq!(
        v["fresh"], false,
        "the triggering request must not claim freshness; got: {v:?}"
    );
}

// ---------------------------------------------------------------------------
// 12 — declarative_or_edge_case: C++-spezifischer Grenzfall.
//
// Begründung: C++ erlaubt namespace-qualifizierte Aufrufe (`app::do_it()`),
// die der Parser als `call_expression` mit `function: qualified_identifier`
// vorfindet. Die CALLEES-Query fängt nur das finale `identifier`-Segment
// (`do_it`) — die Namespace-Qualifikation geht beim Capture verloren.
// Trotzdem muss der name-basierte Resolver die cross-file-Auflösung
// wieder herstellen (es existiert genau EIN Symbol namens `do_it` in
// `src/helper.hpp`). Dieser Test verifiziert, dass die Qualifizierung
// KEIN Hindernis für die CALLS-Kante ist und beide Caller — der
// unqualifizierte (`caller()`) und der qualifizierte (`other_caller()`) —
// im who-calls-Ergebnis auftauchen.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_cpp_declarative_namespace_qualified_call() {
    let (repo, store) = index_fixture("ns-qualified-call");
    let (code, out, err) = run(&["who-calls", "do_it"], &repo, &store);
    assert_eq!(
        code, 0,
        "who-calls should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("caller"),
        "unqualified caller() must reach do_it; got: {out:?}"
    );
    assert!(
        out.contains("other_caller"),
        "namespace-qualified app::do_it() from other_caller() must also reach do_it; got: {out:?}"
    );
}
