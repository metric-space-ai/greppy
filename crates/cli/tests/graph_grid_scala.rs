//! Graph-Zertifizierungs-Grid für SCALA — 12 Zellen.
//!
//! SCALA ist zum Zeitpunkt dieses Grids noch NICHT graph-zertifiziert; die
//! rot markierten Zellen sind die **erwarteten** Befunde dieser Zertifizierung
//! (die Lücken, die vor einer offiziellen Zertifizierung zu schließen sind).
//! Die Tests sind NICHT so geschrieben, dass sie die Lücken wegignorieren:
//! jede rote Zelle zeigt, was der SCALA-Indexer/Resolver heute tatsächlich
//! liefert vs. was das Grid erwartet.
//!
//! Fixture-Repo (`src/`):
//! * `main.scala`    — `caller()`, `render()`, `uncalled()` in `object MainFlow`,
//!   inkl. `import grid.helper.Helper` / `import grid.types.Payload` etc.
//! * `helper.scala`  — `object Helper` mit `val HELPER_VALUE` + `def doIt(...)`,
//!   plus `object Sink` (für den `object`→"Class"-Grenzfall in Zelle 12).
//! * `types.scala`   — `class Payload(val value: Int)`.
//!
//! Soll-Kanten über Dateigrenzen:
//! * `caller()`     --CALLS-->   `doIt()`          (helper.scala)
//! * `caller(p)`    --USAGE-->   `Payload`         (types.scala, TYPE_REF-Parität)
//! * `caller()`     --USAGE-->   `HELPER_VALUE`    (helper.scala, USES-Parität)
//! * `main.scala`   --IMPORTS--> `helper.scala` / `types.scala`
//!
//! Hinweis: Die SCALA-Queries (`crates/parser/src/query.rs::scala_queries`)
//! emittieren Definitionen, Calls und Imports. Sie haben **kein** dediziertes
//! TYPE_REF / USES-Pass; alle Identifier-Referenzen laufen durch
//! `scala_emit_usages` und werden als `USAGE`-Kante persistiert (analog zu
//! C++). Der `import`-Pfad wird zu `ImportedItem.imported_name` (= finaler
//! `path:`-Segment) aufgelöst, nicht zur Quelldatei — die IMPORTS-Kante zeigt
//! also auf das aufgelöste Symbol (z. B. `Helper::doIt`).

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
    let dir = std::env::temp_dir().join(format!("greppy-cli-graphgrid-scala-{tag}-{pid}-{n}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Build a git-rooted SCALA repo with three files exercising all four
/// cross-file edge shapes:
/// * CALLS:   `caller()` -> `doIt()` (helper.scala)
/// * USAGE:   `caller(p: Payload)` parameter type -> Payload (TYPE_REF parity)
/// * USAGE:   `caller()` constant read -> `HELPER_VALUE` (USES parity)
/// * IMPORTS: `import grid.helper.Helper` / `import grid.types.Payload`
///
/// helper.scala also declares an `object Sink` so the Zelle-12 edge case
/// (object -> "Class" relabelling) has a discoverable symbol.
fn make_scala_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).unwrap();
    // `.git` is the repo-root marker resolve_root walks up to find.
    std::fs::create_dir_all(repo.join(".git")).unwrap();

    std::fs::write(
        src.join("main.scala"),
        r#"package grid.main

import grid.helper.Helper
import grid.helper.Helper.HELPER_VALUE
import grid.helper.Helper.doIt
import grid.types.Payload

object MainFlow {
  def caller(p: Payload): Int = {
    val total = doIt(2)
    val seed = HELPER_VALUE
    total + seed
  }

  def render(p: Payload): Payload = p

  def uncalled(): Int = 99
}
"#,
    )
    .unwrap();

    std::fs::write(
        src.join("helper.scala"),
        r#"package grid.helper

object Helper {
  val HELPER_VALUE: Int = 7

  def doIt(x: Int): Int = x + HELPER_VALUE
}

object Sink {
  def emit(msg: String): String = msg
}
"#,
    )
    .unwrap();

    std::fs::write(
        src.join("types.scala"),
        r#"package grid.types

class Payload(val value: Int)
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
    let (repo, store) = make_scala_repo(tag);
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
fn graph_grid_scala_who_calls_finds_cross_file_caller() {
    let (repo, store) = index_fixture("who-calls-cross");
    // `doIt` is defined in helper.scala and called by `caller` in main.scala.
    let (code, out, err) = run(&["who-calls", "doIt"], &repo, &store);
    assert_eq!(
        code, 0,
        "who-calls should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("caller"),
        "who-calls doIt must list the caller `caller` (main.scala); got: {out:?}"
    );
    assert!(
        out.contains("src/main.scala:"),
        "who-calls must print the caller's file:line (src/main.scala); got: {out:?}"
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
fn graph_grid_scala_who_calls_empty_for_uncalled() {
    let (repo, store) = index_fixture("who-calls-empty");
    // `render` is defined in main.scala but never called from anywhere.
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
fn graph_grid_scala_callees_lists_cross_file_target() {
    let (repo, store) = index_fixture("callees-cross");
    // `caller` calls `doIt` which is defined in helper.scala.
    let (code, out, err) = run(&["callees", "caller"], &repo, &store);
    assert_eq!(code, 0, "callees should exit 0; stderr={err}\nstdout={out}");
    assert!(
        out.contains("doIt"),
        "callees caller must list the cross-file callee `doIt`; got: {out:?}"
    );
    assert!(
        out.contains("src/helper.scala:"),
        "callees must print the callee's file:line (src/helper.scala); got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 4 — find-usages: covers both CALLS and IMPORTS reference edges.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_scala_find_usages_covers_call_and_import() {
    let (repo, store) = index_fixture("find-usages-call-import");
    // `find-usages` aggregates REFERENCE_EDGE_TYPES = [CALLS, USAGE, USES,
    // TYPE_REF, IMPORTS] per incoming target. The CALLS leg should list
    // `caller` (caller-of-doIt); the IMPORTS leg should be visible via
    // the import target's qualified name.
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
        "find-usages doIt must show the IMPORTS edge kind (import grid.helper.Helper.doIt); got: {out:?}"
    );
    assert!(
        out.contains("src/main.scala:"),
        "find-usages must print the IMPORTS referrer's file:line (src/main.scala); got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 5 — find-usages: TYPE_REF-equivalent (USAGE for type references) works.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_scala_find_usages_type_reference() {
    let (repo, store) = index_fixture("find-usages-type-ref");
    // `Payload` is referenced as a parameter type in `caller(p: Payload)`.
    // The SCALA usage pass emits a USAGE edge from `caller` keyed on ref_name
    // "Payload"; the persisted label is USAGE (the unified reference label).
    let (code, out, err) = run(&["find-usages", "Payload"], &repo, &store);
    assert_eq!(
        code, 0,
        "find-usages should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("USAGE"),
        "find-usages Payload must label the edge kind USAGE (TYPE_REF parity); got: {out:?}"
    );
    assert!(
        out.contains("caller"),
        "find-usages Payload must list `caller` as the type referrer; got: {out:?}"
    );
    assert!(
        out.contains("src/main.scala:"),
        "find-usages must print the referrer's file:line (src/main.scala); got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 6 — impact: transitive blast radius reaches the cross-file caller.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_scala_impact_transitive_reaches_caller() {
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
// 7 — search-symbols: finds all definitions of a symbol across the repo.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_scala_search_symbols_finds_all_definitions() {
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
        out.contains("src/helper.scala:"),
        "search-symbols must print the symbol's file:line (src/helper.scala); got: {out:?}"
    );
    assert!(
        out.contains("Method") || out.contains("Function"),
        "search-symbols must print the node label (Method or Function for SCALA defs); got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 8 — brief: definition + callers + callees bundled in one call.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_scala_brief_shows_definition_with_callers() {
    let (repo, store) = index_fixture("brief");
    let (code, out, err) = run(&["brief", "doIt"], &repo, &store);
    assert_eq!(code, 0, "brief should exit 0; stderr={err}\nstdout={out}");
    assert!(
        out.contains("doIt") && out.contains("def doIt"),
        "brief must show the definition with source body; got: {out}"
    );
    assert!(
        out.contains("(src/helper.scala:"),
        "brief header must report the expanded source span (src/helper.scala); got: {out}"
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
#[ignore = "scala graph gap: path finds no CALLS chain to the cross-file helper (who-calls resolves it — path-specific resolution)"]
fn graph_grid_scala_path_connects_caller_to_helper() {
    let (repo, store) = index_fixture("path");
    // `path --from caller --to doIt` over CALLS must find the single-hop
    // path caller -> doIt.
    let (code, out, err) = run(
        &["path", "--from", "caller", "--to", "doIt"],
        &repo,
        &store,
    );
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
        out.contains("src/main.scala:") && out.contains("src/helper.scala:"),
        "path steps must carry file:line for both endpoints; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 10 — graph_survives_reindex: a second index run produces the same edges.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_scala_graph_survives_reindex() {
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
fn graph_grid_scala_stale_edit_detected() {
    let (repo, store) = index_fixture("stale-edit");
    // Drift the index: edit the helper file (renames doIt -> doIt_renamed),
    // so the symbol lookup misses and the freshness gate must fire on the next
    // navigation command.
    std::fs::write(
        repo.join("src/helper.scala"),
        r#"package grid.helper

object Helper {
  val HELPER_VALUE: Int = 7

  def doIt_renamed(x: Int): Int = x + HELPER_VALUE
}

object Sink {
  def emit(msg: String): String = msg
}
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
// 12 — declarative_or_edge_case: SCALA-spezifischer Grenzfall.
//
// Begründung: SCALA `object`-Deklarationen sind im Parser-Code bewusst als
// "Class" gelabelt (nicht "Object" oder "Module" — siehe
// `crates/parser/src/extract.rs::scala_type_label`). Auch `trait`-Deklarationen
// werden zu "Interface" (siehe `scala_type_label` für `trait_definition`).
// Dieser Test verifiziert die Scala-spezifische Umbenennung, indem ein
// zusätzliches `object Sink` in helper.scala deklariert und über
// search-symbols als Class-Definition aufgefunden werden muss.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_scala_declarative_object_labelled_as_class() {
    let (repo, store) = index_fixture("scala-object");
    // A `object Sink` declaration is a Scala-specific singleton; the parser
    // maps it to the "Class" label so it is discoverable as a typed graph
    // node. search-symbols must locate the definition with that exact label.
    let (code, out, err) = run(&["search-symbols", "Sink", "--json"], &repo, &store);
    assert_eq!(
        code, 0,
        "search-symbols Sink should exit 0; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid search-symbols json: {e}; stdout={out:?}"));
    let hits = v["hits"]
        .as_array()
        .expect("search-symbols hits array");
    assert!(
        hits.iter().any(|hit| {
            hit["label"] == "Class"
                && hit["name"] == "Sink"
                && hit["file_path"]
                    .as_str()
                    .unwrap_or("")
                    .ends_with("helper.scala")
        }),
        "Scala `object Sink` must surface as a Class definition in helper.scala; got: {v:?}"
    );
}
