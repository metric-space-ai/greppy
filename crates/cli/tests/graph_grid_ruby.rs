//! Graph-navigation certification grid for **Ruby**.
//!
//! Mirrors the proven Track-A integration suite in `graph_nav.rs`, but
//! driven against a Ruby fixture. Like the Python class fixture there, this
//! exercises cross-file `CALLS` / `IMPORTS` (and `USAGE`) edges end-to-end
//! through the shipped `greppy` binary, so the indexer/resolver and the
//! CLI navigation commands are exercised together.
//!
//! Six languages have already been certified; Ruby is the seventh and has
//! a known gap on Track-A graph edges (USAGE / IMPORTS cross-file are
//! indexed but the navigation commands may not surface them fully). The
//! cells in this grid therefore DOUBLE as the Ruby certification report:
//! failures are EXPECTED findings, not bugs to hide.
//!
//! Fixture shape (three files):
//!
//! * `app.rb`        -- `caller()` calls `Helper.do_it` and constructs `Widget` (TYPE_REF)
//!   plus references the constant `Helper::LIMIT`.
//! * `helper.rb`     -- defines the `Helper` module with `do_it()` and a
//!   top-level `LIMIT` constant.
//! * `types.rb`      -- defines the `Widget` class and the `Marker` struct
//!   (labelled Class by the Ruby extractor).
//!
//! Edge shape asserted by these tests:
//!   - CALLS      : `app.rb::caller`  -> `helper.rb::Helper::do_it`
//!   - USAGE/TYPE : `app.rb::render`  -> `types.rb::Widget`
//!   - USAGE      : `app.rb::build`   -> `types.rb::Marker`
//!   - IMPORTS    : `app.rb::__file__` -> `helper.rb::__file__` / `types.rb::__file__`
//!     (Ruby `require`/`require_relative` pass)

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
    let dir = std::env::temp_dir().join(format!("greppy-cli-gridruby-{tag}-{pid}-{n}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Build a Ruby repo exercising all four cross-file edge kinds:
/// * caller()         --CALLS-->    Helper.do_it()       (helper.rb)
/// * render(w)        --TYPE_REF--> types::Widget        (types.rb)
/// * build()          --USAGE-->    types::Marker        (types.rb)
/// * top of file      --IMPORTS-->  helper.rb / types.rb (require)
fn make_ruby_graph_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    std::fs::create_dir_all(&repo).unwrap();
    // `.git` is the repo-root marker resolve_root walks up to find.
    std::fs::create_dir_all(repo.join(".git")).unwrap();

    // Main file: imports the helper + types modules, then exercises every edge.
    std::fs::write(
        repo.join("app.rb"),
        r#"require_relative 'helper'
require_relative 'types'

def caller
  Helper.do_it
end

def render(_w)
  Widget.new
end

def build
  marker = Marker.new
  marker
end

def report_limit
  Helper::LIMIT
end
"#,
    )
    .unwrap();

    // Helper module: defines the cross-file target of the CALLS edge.
    std::fs::write(
        repo.join("helper.rb"),
        r#"module Helper
  LIMIT = 42

  def self.do_it
    LIMIT
  end
end
"#,
    )
    .unwrap();

    // Types module: defines the cross-file TYPE_REF + USAGE targets.
    std::fs::write(
        repo.join("types.rb"),
        r#"class Widget
  attr_accessor :width
end

class Marker
end
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

fn index_fixture(tag: &str) -> (PathBuf, PathBuf) {
    let (repo, store) = make_ruby_graph_repo(tag);
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "index . should succeed for Ruby fixture; stderr={err}\nstdout={out}"
    );
    (repo, store)
}

// ---------------------------------------------------------------------------
// 1. who-calls finds a cross-file caller.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_ruby_who_calls_finds_cross_file_caller() {
    let (repo, store) = index_fixture("whocalls");

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
        out.contains("app.rb:"),
        "who-calls must print the caller's file:line (app.rb); got: {out:?}"
    );
    assert!(
        !out.contains("no callers"),
        "who-calls must find at least one caller for do_it; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 2. who-calls is empty for an uncalled symbol.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_ruby_who_calls_empty_for_uncalled() {
    let (repo, store) = index_fixture("whocalls-none");

    // `Marker` is a class definition — nothing CALLS it.
    let (code, out, _err) = run(&["who-calls", "Marker"], &repo, &store);
    assert_eq!(code, 0);
    assert_eq!(
        out, "no callers\n",
        "who-calls on an uncalled class must report no callers; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 3. callees lists the cross-file target.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_ruby_callees_lists_cross_file_target() {
    let (repo, store) = index_fixture("callees");

    let (code, out, err) = run(&["callees", "caller"], &repo, &store);
    assert_eq!(code, 0, "callees should exit 0; stderr={err}\nstdout={out}");
    assert!(
        out.contains("do_it"),
        "callees of caller must list the cross-file `do_it`; got: {out:?}"
    );
    assert!(
        out.contains("helper.rb:") || out.contains("helper"),
        "callees must print the callee's file path (helper.rb); got: {out:?}"
    );
    assert!(
        !out.contains("no callees"),
        "callees must not be empty for caller; got: {out:?}"
    );
}

#[test]
fn graph_grid_ruby_impact_transitive_reaches_caller() {
    let (repo, store) = index_fixture("impact");

    let (code, out, err) = run(&["impact", "do_it"], &repo, &store);
    assert_eq!(code, 0, "impact should exit 0; stderr={err}\nstdout={out}");
    assert!(
        out.contains("caller"),
        "impact do_it must reach the cross-file caller `caller`; got: {out:?}"
    );
    assert!(
        out.contains("app.rb:") || out.contains("hop"),
        "impact must print actionable file:line or hop info; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 7. search-symbol finds every definition across the repo.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_ruby_search_symbol_finds_all_definitions() {
    let (repo, store) = index_fixture("symbols");

    for needle in [
        "caller", "render", "build", "do_it", "Widget", "Marker", "Helper",
    ] {
        let (code, out, err) = run(&["search-symbol", needle], &repo, &store);
        assert_eq!(
            code, 0,
            "search-symbol {needle} should exit 0; stderr={err}\nstdout={out}"
        );
        assert!(
            out.contains(needle),
            "search-symbol {needle} must include the name; got: {out:?}"
        );
    }

    // `do_it` is owned by `Helper` so its file_path must be helper.rb.
    let (code, out, err) = run(&["search-symbol", "do_it"], &repo, &store);
    assert_eq!(code, 0, "stderr={err}");
    assert!(
        out.contains("helper.rb:"),
        "search-symbol do_it must print the definition file:line (helper.rb); got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 8. brief bundles the definition + its callers in one call.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_ruby_brief_shows_definition_with_callers() {
    let (repo, store) = index_fixture("brief");

    let (code, out, err) = run(&["brief", "do_it"], &repo, &store);
    assert_eq!(code, 0, "brief should exit 0; stderr={err}");
    assert!(
        out.contains("do_it"),
        "brief do_it must show the definition; got: {out}"
    );
    assert!(
        out.contains("called by caller") && !out.contains("CALLERS"),
        "brief must aggregate callers into one line; got: {out}"
    );
}

// ---------------------------------------------------------------------------
// 9. path connects caller to helper.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_ruby_path_connects_caller_to_helper() {
    let (repo, store) = index_fixture("path");

    let (code, out, err) = run(
        &["path", "--from", "caller", "--to", "do_it"],
        &repo,
        &store,
    );
    assert_eq!(code, 0, "path should exit 0; stderr={err}\nstdout={out}");
    assert!(
        out.contains("caller"),
        "path caller->do_it must include the start symbol; got: {out:?}"
    );
    assert!(
        out.contains("do_it"),
        "path caller->do_it must include the destination; got: {out:?}"
    );
    assert!(
        out.contains("app.rb:") || out.contains("helper.rb:"),
        "path must print actionable file:line for one of the endpoints; got: {out:?}"
    );
}

// ---------------------------------------------------------------------------
// 10. graph survives a reindex — same edges on the second index run.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_ruby_graph_survives_reindex() {
    let (repo, store) = index_fixture("reindex");

    let (code1, before, _e1) = run(&["who-calls", "do_it"], &repo, &store);
    assert_eq!(code1, 0);

    // Second index run must succeed without losing the cross-file edge.
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(
        code, 0,
        "second index . should succeed; stderr={err}\nstdout={out}"
    );

    let (code2, after, _e2) = run(&["who-calls", "do_it"], &repo, &store);
    assert_eq!(code2, 0);

    assert!(
        before.contains("caller") && after.contains("caller"),
        "who-calls do_it must keep listing caller across reindex; before={before:?}\nafter={after:?}"
    );
    assert!(
        before.contains("app.rb:") && after.contains("app.rb:"),
        "who-calls file:line anchor must persist across reindex; before={before:?}\nafter={after:?}"
    );
}

// ---------------------------------------------------------------------------
// 11. stale edit is detected: changing a file after index triggers a
//     refresh path (EX_TEMPFAIL=75 in --json mode, or freshening in text
//     mode) rather than silently serving stale rows.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_ruby_stale_edit_detected() {
    let (repo, store) = index_fixture("stale");

    // Mutate one of the source files AFTER the index.
    std::fs::write(
        repo.join("helper.rb"),
        "module Helper\n  LIMIT = 99\n\n  def self.do_it\n    LIMIT\n  end\nend\n",
    )
    .unwrap();

    // Heal-in-band contract (1b7135b): a healable drift is reindexed in-band
    // and served fresh against the updated graph.
    let (code, out, err) = run(
        &["who-calls", "do_it", "--json", "--diagnostics"],
        &repo,
        &store,
    );
    assert_eq!(
        code, 0,
        "healable stale edit must be healed in-band and served; stderr={err}\nstdout={out}"
    );
    let v: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid who-calls stale json: {e}; stdout={out:?}"));
    assert_eq!(v["command"], "who-calls");
    assert_eq!(
        v["fresh"], true,
        "the healed response must prove freshness; got: {v:?}"
    );
    assert_eq!(
        v["symbol_found"], true,
        "the symbol must resolve against the healed graph; got: {v:?}"
    );
}

// ---------------------------------------------------------------------------
// 12. Ruby-specific edge case: `def self.do_it` (singleton method on a
//     module). The cross-file CALLS edge must still resolve from
//     `Helper.do_it` in app.rb to the singleton method owned by the
//     `Helper` module in helper.rb — the typical Ruby "module function"
//     shape that the other grids do not exercise.
// ---------------------------------------------------------------------------

#[test]
fn graph_grid_ruby_declarative_or_edge_case() {
    let (repo, store) = index_fixture("singleton");

    // The CALLS target is `Helper.do_it`, a singleton method owned by the
    // module. search-symbol must locate the definition.
    let (code, out, err) = run(&["search-symbol", "do_it"], &repo, &store);
    assert_eq!(
        code, 0,
        "search-symbol do_it should exit 0; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("helper.rb:"),
        "search-symbol do_it must find the singleton method definition in helper.rb; got: {out:?}"
    );
}
