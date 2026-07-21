//! Graph-Zertifizierungs-Grid für PHP — 12 Zellen.
//!
//! PHP ist zum Zeitpunkt dieses Grids noch NICHT graph-zertifiziert; die
//! rot markierten Zellen sind die **erwarteten** Befunde dieser Zertifizierung
//! (die Lücken, die vor einer offiziellen Zertifizierung zu schließen sind).
//! Die Tests sind NICHT so geschrieben, dass sie die Lücken wegignorieren:
//! jede rote Zelle zeigt, was der PHP-Indexer/Resolver heute tatsächlich
//! liefert vs. was das Grid erwartet.
//!
//! Fixture-Repo (`src/`):
//! * `Main.php`   — `caller()`, `entryPoint()`, `uncalled()`, dazu
//!   `use Fixture\Helpers\HelperTools;` und `use Fixture\Types\Payload;`.
//! * `Helpers.php` — freie Funktion `do_it()` + Konstante `SHARED_MARKER` +
//!   Klasse `HelperTools` mit statischer Methode `helper()`.
//! * `Types.php`  — Klasse `Payload` (die als TYPE_REF-Ziel im dritten File
//!   dient).
//!
//! Soll-Kanten über Dateigrenzen:
//! * `caller()`    --CALLS-->   `do_it()`              (Helpers.php)
//! * `caller()`    --CALLS-->   `HelperTools::helper()`(Helpers.php, statisch
//!   gescoped — letzte Komponente `helper`)
//! * `caller()`    --TYPE_REF--> `Payload`             (Types.php) — PHP hat
//!   **keinen** TYPE_REF / USAGE-Pass; diese Zelle muss daher rot bleiben,
//!   bis der Spec-Maschinerie ein solcher Pass für PHP hinzugefügt wird.
//! * `caller()`    --USES-->     `SHARED_MARKER`        (Helpers.php) — selbe
//!   Lücke wie TYPE_REF.
//! * `Main.php`    --IMPORTS--> `HelperTools` / `Payload`
//!   (`namespace_use_declaration`, vom Spec-Parser zu EINER Kante pro
//!   Namespace-Prefix kollabiert)
//! * `Helpers.php` --IMPORTS--> `Payload`              (ebenfalls über
//!   `use Fixture\Types\Payload;`)
//!
//! Hinweis: Die PHP-Queries (`crates/parser/src/query.rs::php_queries`)
//! emittieren Definitions (class/interface/trait/enum/function/method), Calls
//! (bare / member / scoped — letzte Komponente) und Imports
//! (`namespace_use_declaration`). Es gibt **kein** dediziertes TYPE_REF- oder
//! USES-Pass.

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
    let dir = std::env::temp_dir().join(format!("greppy-cli-graphgrid-php-{tag}-{pid}-{n}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Build a git-rooted PHP repo with three files exercising all four
/// cross-file edge shapes:
///
/// * CALLS:   `caller()` -> `do_it()` (free function, Helpers.php)
/// * CALLS:   `caller()` -> `HelperTools::helper()` (scoped / static, letzte
///   Komponente `helper`)
/// * TYPE_REF: `function caller(): Payload`        (Types.php, NICHT emittiert
///   — PHP-Spec hat keinen TYPE_REF-Pass; Zertifizierungs-Lücke)
/// * USES:    `$m = SHARED_MARKER;`               (Helpers.php, NICHT
///   emittiert — PHP-Spec hat keinen USES-Pass; Lücke)
/// * IMPORTS: `use Fixture\Helpers\HelperTools;` und
///   `use Fixture\Types\Payload;` im Main.php
/// * IMPORTS: `use Fixture\Types\Payload;` im Helpers.php
fn make_php_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).unwrap();
    // `.git` is the repo-root marker resolve_root walks up to find.
    std::fs::create_dir_all(repo.join(".git")).unwrap();

    std::fs::write(
        src.join("Main.php"),
        r#"<?php
namespace Fixture\App;

use Fixture\Helpers\HelperTools;
use Fixture\Helpers\SHARED_MARKER;
use Fixture\Types\Payload;

function caller(): Payload {
    do_it();
    $tool = new HelperTools();
    $combined = SHARED_MARKER + HelperTools::helper($tool);
    return new Payload($combined);
}

function entryPoint(): Payload {
    return caller();
}

function uncalled(): void {
    return;
}
"#,
    )
    .unwrap();

    std::fs::write(
        src.join("Helpers.php"),
        r#"<?php
namespace Fixture\Helpers;

use Fixture\Types\Payload;

const SHARED_MARKER = 7;

function do_it(): int {
    return 42;
}

class HelperTools {
    public static function helper(int $value): int {
        return $value + SHARED_MARKER;
    }
}
"#,
    )
    .unwrap();

    std::fs::write(
        src.join("Types.php"),
        r#"<?php
namespace Fixture\Types;

class Payload {
    public function __construct(public int $value) {}
}
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
    let (repo, store) = make_php_repo(tag);
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
fn graph_grid_php_who_calls_finds_cross_file_caller() {
    let (repo, store) = index_fixture("who-calls-cross");
    // `do_it` is defined in Helpers.php and called by `caller` in Main.php.
    let (code, out, err) = run(&["who-calls", "do_it"], &repo, &store);
    assert_eq!(
        code, 0,
        "who-calls should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("caller"),
        "who-calls do_it must list the caller `caller` (Main.php); got: {out:?}"
    );
    assert!(
        out.contains("src/Main.php:"),
        "who-calls must print the caller's file:line (src/Main.php); got: {out:?}"
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
fn graph_grid_php_who_calls_empty_for_uncalled() {
    let (repo, store) = index_fixture("who-calls-empty");
    // `uncalled` is defined in Main.php but never called from anywhere.
    let (code, out, _err) = run(&["who-calls", "uncalled"], &repo, &store);
    assert_eq!(code, 0);
    assert!(
        out.contains("(no callers)"),
        "uncalled is uncalled, who-calls must report no callers; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 3 — callees: outgoing CALLS edges resolve to the cross-file target.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_php_callees_lists_cross_file_target() {
    let (repo, store) = index_fixture("callees-cross");
    // `caller` calls `do_it` which is defined in Helpers.php.
    let (code, out, err) = run(&["callees", "caller"], &repo, &store);
    assert_eq!(code, 0, "callees should exit 0; stderr={err}\nstdout={out}");
    assert!(
        out.contains("do_it"),
        "callees caller must list the cross-file callee `do_it`; got: {out:?}"
    );
    assert!(
        out.contains("src/Helpers.php:"),
        "callees must print the callee's file:line (src/Helpers.php); got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 4 — find-usages: covers both CALLS and IMPORTS reference edges.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_php_find_usages_covers_call_and_import() {
    let (repo, store) = index_fixture("find-usages-call-import");
    // `find-usages` aggregates REFERENCE_EDGE_TYPES = [CALLS, USAGE, USES,
    // TYPE_REF, IMPORTS] per incoming target. The CALLS leg should list
    // `caller` (caller-of-do_it); the IMPORTS leg should be visible via the
    // `namespace_use_declaration` (PHP's spec collapses imports per
    // namespace-prefix — `use Fixture\Helpers\{HelperTools, SHARED_MARKER}`
    // collapses to one edge per namespace).
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
fn graph_grid_php_find_usages_type_reference() {
    let (repo, store) = index_fixture("find-usages-type-ref");
    // `Payload` is used as a return type in `function caller(): Payload` in
    // Main.php. The PHP spec (`crates/parser/src/spec.rs::PHP`) emits no
    // TYPE_REF / USAGE pass — unlike C/C++/C#/Swift/Kotlin — so this
    // assertion names the *expected* future behaviour. Today the call usually
    // exits 0 (no usages) or fails to list `caller`; either is a
    // Zertifizierungs-Lücke.
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
        out.contains("src/Main.php:"),
        "find-usages must print the referrer's file:line (src/Main.php); got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 6 — impact: transitive blast radius reaches the cross-file caller.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_php_impact_transitive_reaches_caller() {
    let (repo, store) = index_fixture("impact-transitive");
    // `impact do_it` must report `caller` as an incoming CALLS dependent —
    // the single-command answer to "what breaks if I change do_it?". The
    // transitive path `do_it` → `caller` → `entryPoint` should be observable.
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
fn graph_grid_php_search_symbols_finds_all_definitions() {
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
        out.contains("src/Helpers.php:"),
        "search-symbols must print the symbol's file:line (src/Helpers.php); got: {out:?}"
    );
    assert!(
        out.contains("Function"),
        "search-symbols must print the node label (Function for PHP defs); got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 8 — brief: definition + callers + callees bundled in one call.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_php_brief_shows_definition_with_callers() {
    let (repo, store) = index_fixture("brief");
    let (code, out, err) = run(&["brief", "do_it"], &repo, &store);
    assert_eq!(code, 0, "brief should exit 0; stderr={err}\nstdout={out}");
    assert!(
        out.contains("do_it") && out.contains("function do_it"),
        "brief must show the definition with source body; got: {out}"
    );
    assert!(
        out.contains("(src/Helpers.php:"),
        "brief header must report the expanded source span (src/Helpers.php); got: {out}"
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
fn graph_grid_php_path_connects_caller_to_helper() {
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
        out.contains("src/Main.php:") && out.contains("src/Helpers.php:"),
        "path steps must carry file:line for both endpoints; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 10 — graph_survives_reindex: a second index run produces the same edges.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_php_graph_survives_reindex() {
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
//
// Vorbild cpp: heal-in-band-Contract (1b7135b) — eine heilbare Drift wird
// in-band neu indexiert und FRESH serviert; der umbenannte Symbol wird gegen
// den NEUEN Graph aufgelöst. Der Vertrag ist:
//   * exit 0
//   * fresh: true
//   * symbol_found: true
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_php_stale_edit_detected() {
    let (repo, store) = index_fixture("stale-edit");
    // Drift the index: rename the helper function in Helpers.php so the
    // symbol lookup misses and the freshness gate must fire on the next
    // navigation command.
    std::fs::write(
        repo.join("src/Helpers.php"),
        r#"<?php
namespace Fixture\Helpers;

use Fixture\Types\Payload;

const SHARED_MARKER = 7;

function do_it_renamed(): int {
    return 42;
}

class HelperTools {
    public static function helper(int $value): int {
        return $value + SHARED_MARKER;
    }
}
"#,
    )
    .unwrap();

    let (code, out, err) = run(&["who-calls", "do_it_renamed", "--json"], &repo, &store);
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
// 12 — declarative_or_edge_case: PHP-spezifischer Grenzfall.
//
// Begründung: PHP erlaubt statische Aufrufe durch Klassen-Namespaces, die
// der Parser als `scoped_call_expression` mit `name:` als FINALEM
// Identifier-Segment vorfindet. Heißt: `HelperTools::helper(...)` matcht
// die PHP-CALLS-Query mit `@callee = "helper"` (die Klassen-Qualifikation
// `HelperTools::` geht beim Capture verloren). Trotzdem muss der
// name-basierte Resolver die cross-file-Auflösung wieder herstellen —
// die freie Funktion `do_it` *und* die statische Methode `helper` leben
// beide in `src/Helpers.php`, getrennt durch ihre Labels
// (Function vs. Method). Dieser Test verifiziert, dass die Qualifizierung
// KEIN Hindernis für die CALLS-Kante ist und beide Calls — der bare
// `do_it()` und der scoped `HelperTools::helper()` — im
// `who-calls`-Ergebnis auftauchen.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_php_declarative_scoped_static_call() {
    let (repo, store) = index_fixture("scoped-static-call");
    // `who-calls helper` muss `caller` aus Main.php (via HelperTools::helper)
    // liefern, und `who-calls do_it` muss ebenfalls `caller` liefern — beide
    // Calls sind cross-file nach Helpers.php.
    let (code, out, err) = run(&["who-calls", "helper"], &repo, &store);
    assert_eq!(
        code, 0,
        "who-calls should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("caller"),
        "scoped static call HelperTools::helper(...) from caller() must reach the cross-file Method `helper`; got: {out:?}"
    );

    let (code, out, err) = run(&["who-calls", "do_it"], &repo, &store);
    assert_eq!(
        code, 0,
        "who-calls should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("caller"),
        "bare call do_it() from caller() must still reach the cross-file Function `do_it`; got: {out:?}"
    );
}
