//! Graph certification grid for Elixir — 12 cells.
//!
//! Fixture edges:
//! * `caller` --CALLS--> `do_it` (Helper.ex)
//! * `render` --TYPE_REF--> `Widget` (Widget.ex)
//! * `Main.ex` --IMPORTS--> `Helper` / `Widget`

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_greppy")
}

fn fresh_dir(tag: &str) -> PathBuf {
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "greppy-cli-graphgrid-elixir-{tag}-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create Elixir scratch directory");
    dir
}

fn make_elixir_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    let src = repo.join("lib");
    std::fs::create_dir_all(&src).expect("create fixture lib directory");
    std::fs::create_dir_all(repo.join(".git")).expect("create repo marker");

    std::fs::write(
        src.join("main.ex"),
        r#"defmodule Main do
  alias Widget
  import Helper
  require Helper

  def caller do
    Helper.do_it()
  end

  def render do
    %Widget{value: 1}
  end

  def uncalled do
    0
  end
end
"#,
    )
    .expect("write main.ex");

    std::fs::write(
        src.join("helper.ex"),
        r#"defmodule Helper do
  def do_it do
    private_value()
  end

  defp private_value do
    42
  end
end
"#,
    )
    .expect("write helper.ex");

    std::fs::write(
        src.join("widget.ex"),
        r#"defmodule Widget do
  defstruct [:value]
end
"#,
    )
    .expect("write widget.ex");

    (repo, root.join("store"))
}

fn run(args: &[&str], cwd: &Path, store: &Path) -> (i32, String, String) {
    let output = Command::new(bin())
        .args(args)
        .current_dir(cwd)
        .env("GREPPY_STORE_DIR", store)
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
    let (repo, store) = make_elixir_repo(tag);
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "index failed; stderr={err}\nstdout={out}");
    (repo, store)
}

#[test]
fn graph_grid_elixir_who_calls_finds_cross_file_caller() {
    let (repo, store) = index_fixture("who-calls");
    let (code, out, err) = run(&["who-calls", "do_it"], &repo, &store);
    assert_eq!(code, 0, "stderr={err}\nstdout={out}");
    assert!(
        out.contains("caller") && out.contains("lib/main.ex:"),
        "{out}"
    );
    assert!(!out.contains("(no callers)"), "{out}");
}

#[test]
fn graph_grid_elixir_who_calls_empty_for_uncalled() {
    let (repo, store) = index_fixture("who-calls-empty");
    let (code, out, err) = run(&["who-calls", "uncalled"], &repo, &store);
    assert_eq!(code, 0, "stderr={err}");
    assert!(out.contains("(no callers)"), "{out}");
}

#[test]
fn graph_grid_elixir_callees_lists_cross_file_target() {
    let (repo, store) = index_fixture("callees");
    let (code, out, err) = run(&["callees", "caller"], &repo, &store);
    assert_eq!(code, 0, "stderr={err}\nstdout={out}");
    assert!(
        out.contains("do_it") && out.contains("lib/helper.ex:"),
        "{out}"
    );
}

#[test]
fn graph_grid_elixir_impact_transitive_reaches_caller() {
    let (repo, store) = index_fixture("impact");
    let (code, out, err) = run(&["impact", "do_it"], &repo, &store);
    assert_eq!(code, 0, "stderr={err}\nstdout={out}");
    assert!(out.contains("caller") && out.contains("hop 1"), "{out}");
}

#[test]
fn graph_grid_elixir_search_symbols_finds_all_definitions() {
    let (repo, store) = index_fixture("symbols");
    for (name, file) in [
        ("caller", "lib/main.ex:"),
        ("do_it", "lib/helper.ex:"),
        ("Widget", "lib/widget.ex:"),
        ("Helper", "lib/helper.ex:"),
    ] {
        let (code, out, err) = run(&["search-symbols", name], &repo, &store);
        assert_eq!(code, 0, "{name}: stderr={err}\nstdout={out}");
        assert!(out.contains(name) && out.contains(file), "{name}: {out}");
    }
}

#[test]
fn graph_grid_elixir_brief_shows_definition_with_callers() {
    let (repo, store) = index_fixture("brief");
    let (code, out, err) = run(&["brief", "do_it"], &repo, &store);
    assert_eq!(code, 0, "stderr={err}\nstdout={out}");
    assert!(
        out.contains("def do_it") && out.contains("lib/helper.ex"),
        "{out}"
    );
    assert!(
        out.contains("called by caller") && !out.contains("CALLERS"),
        "{out}"
    );
}

#[test]
fn graph_grid_elixir_path_connects_caller_to_helper() {
    let (repo, store) = index_fixture("path");
    let (code, out, err) = run(
        &["path", "--from", "caller", "--to", "do_it"],
        &repo,
        &store,
    );
    assert_eq!(code, 0, "stderr={err}\nstdout={out}");
    let caller = out.find("caller").expect("path start");
    let helper = out.find("do_it").expect("path end");
    assert!(caller < helper, "{out}");
    assert!(
        out.contains("lib/main.ex:") && out.contains("lib/helper.ex:"),
        "{out}"
    );
}

#[test]
fn graph_grid_elixir_graph_survives_reindex() {
    let (repo, store) = index_fixture("reindex");
    let (_, before, _) = run(&["who-calls", "do_it"], &repo, &store);
    let (code, out, err) = run(&["index", "."], &repo, &store);
    assert_eq!(code, 0, "stderr={err}\nstdout={out}");
    let (code, after, err) = run(&["who-calls", "do_it"], &repo, &store);
    assert_eq!(code, 0, "stderr={err}\nstdout={after}");
    assert!(
        before.contains("caller") && after.contains("caller"),
        "before={before}\nafter={after}"
    );
}

#[test]
fn graph_grid_elixir_stale_edit_detected() {
    let (repo, store) = index_fixture("stale");
    std::fs::write(
        repo.join("lib/helper.ex"),
        r#"defmodule Helper do
  def do_it_renamed do
    private_value()
  end

  defp private_value do
    42
  end
end
"#,
    )
    .expect("rewrite helper.ex");
    let (code, out, err) = run(&["who-calls", "do_it_renamed", "--json"], &repo, &store);
    assert_eq!(code, 0, "stderr={err}\nstdout={out}");
    let value: serde_json::Value = serde_json::from_str(&out).expect("who-calls json");
    assert_eq!(value["fresh"], true, "{value}");
    assert_eq!(value["symbol_found"], true, "{value}");
}

#[test]
fn graph_grid_elixir_declarative_or_edge_case() {
    let (repo, store) = index_fixture("private-def");
    let (code, out, err) = run(&["who-calls", "private_value"], &repo, &store);
    assert_eq!(code, 0, "stderr={err}\nstdout={out}");
    assert!(
        out.contains("do_it") && out.contains("lib/helper.ex:"),
        "{out}"
    );

    let (code, symbols, err) = run(&["search-symbols", "private_value"], &repo, &store);
    assert_eq!(code, 0, "stderr={err}\nstdout={symbols}");
    assert!(
        symbols.contains("Function") && symbols.contains("private_value"),
        "{symbols}"
    );
}
