//! C# graph-certification grid.
//!
//! Every test drives the real `greppy` binary against a syntactically valid,
//! three-file C# repository and an isolated `GREPPY_STORE_DIR`. The fixture
//! deliberately asks for the full reference contract, even where the current
//! C# provider is incomplete: red cells are certification findings, not tests
//! to ignore or weaken.
//!
//! The parser registry spells the language variant `Language::CSharp`; `.cs`
//! selects that variant (`crates/parser/src/language.rs`).

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
    let dir = std::env::temp_dir().join(format!("greppy-graph-grid-csharp-{tag}-{pid}-{n}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create C# graph-grid scratch directory");
    dir
}

/// Build one valid C# repository with exactly three source files.
///
/// Cross-file relationships requested by the certification contract:
///
/// * `caller()` --CALLS--> `helper()` (`Main.cs` -> `Helpers.cs`)
/// * the `Payload` return/object types create TYPE_REF candidates into
///   `Types.cs`
/// * `caller()` reads enum constant `HelperCode.Seed`, a USES candidate into
///   `Helpers.cs`
/// * C# alias `using` directives connect `Main.cs` -> `Helpers.cs`/`Types.cs`
///   and `Helpers.cs` -> `Types.cs` through IMPORTS
/// * `entry()` --CALLS--> `caller()` gives impact a transitive second hop
fn make_csharp_repo(tag: &str) -> (PathBuf, PathBuf) {
    let root = fresh_dir(tag);
    let repo = root.join("repo");
    let src = repo.join("src");
    std::fs::create_dir_all(&src).expect("create fixture src");
    // `resolve_root` recognises this marker; no git command is needed.
    std::fs::create_dir_all(repo.join(".git")).expect("create fixture git marker");

    std::fs::write(
        src.join("Main.cs"),
        r#"using HelperTools = Fixture.Helpers.HelperTools;
using HelperCode = Fixture.Helpers.HelperCode;
using Payload = Fixture.Types.Payload;

namespace Fixture.App
{
    public static class MainFlow
    {
        public static Payload caller()
        {
            var tools = new HelperTools();
            int seed = (int)HelperCode.Seed;
            return HelperTools.helper(seed);
        }

        public static Payload entry()
        {
            return caller();
        }

        public static void uncalled()
        {
        }
    }
}
"#,
    )
    .expect("write Main.cs");

    std::fs::write(
        src.join("Helpers.cs"),
        r#"using Payload = Fixture.Types.Payload;

namespace Fixture.Helpers
{
    public enum HelperCode
    {
        Seed = 7
    }

    public sealed class HelperTools
    {
        public HelperTools()
        {
        }

        public static Payload helper(int value)
        {
            return new Payload(value);
        }
    }
}
"#,
    )
    .expect("write Helpers.cs");

    std::fs::write(
        src.join("Types.cs"),
        r#"namespace Fixture.Types
{
    public sealed record Payload(int Value);
}
"#,
    )
    .expect("write Types.cs");

    (repo, root.join("store"))
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
    let (repo, store) = make_csharp_repo(tag);
    let out = index_existing(&repo, &store);
    assert!(
        out.contains("indexed 3 files"),
        "initial index must discover all three .cs files; got: {out:?}"
    );
    (repo, store)
}

fn index_existing(repo: &Path, store: &Path) -> String {
    let (code, out, err) = run(&["index", "."], repo, store);
    assert_eq!(
        code, 0,
        "C# fixture index must succeed; stderr={err}\nstdout={out}"
    );
    assert!(
        out.contains("0 unsupported"),
        "all discovered .cs files must be accepted through Language::CSharp; got: {out:?}"
    );
    out
}

fn run_json(args: &[&str], repo: &Path, store: &Path) -> (i32, serde_json::Value, String) {
    let (code, out, err) = run(args, repo, store);
    let value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid JSON for {args:?}: {e}; stderr={err:?}; stdout={out:?}"));
    (code, value, err)
}

fn nav_hit_signature(args: &[&str], repo: &Path, store: &Path) -> Vec<String> {
    let (code, value, err) = run_json(args, repo, store);
    assert_eq!(
        code, 0,
        "navigation signature command {args:?} failed; stderr={err}; json={value}"
    );
    let mut rows = value["hits"]
        .as_array()
        .expect("navigation JSON hits array")
        .iter()
        .map(|hit| {
            format!(
                "{}|{}|{}",
                hit["edge_type"].as_str().unwrap_or(""),
                hit["qualified_name"].as_str().unwrap_or(""),
                hit["file_path"].as_str().unwrap_or("")
            )
        })
        .collect::<Vec<_>>();
    rows.sort();
    rows
}

#[test]
fn graph_grid_csharp_who_calls_finds_cross_file_caller() {
    let (repo, store) = index_fixture("who-calls");
    let (code, out, err) = run(&["who-calls", "helper"], &repo, &store);
    assert_eq!(code, 0, "who-calls failed; stderr={err}\nstdout={out}");
    assert!(
        out.contains("caller") && out.contains("src/Main.cs:"),
        "who-calls helper must return the cross-file caller from Main.cs; got: {out:?}"
    );
    assert!(
        !out.contains("(no callers)"),
        "the cross-file CALLS edge must make the result non-empty; got: {out:?}"
    );
}

#[test]
fn graph_grid_csharp_who_calls_empty_for_uncalled() {
    let (repo, store) = index_fixture("who-calls-empty");
    let (code, out, err) = run(&["who-calls", "uncalled"], &repo, &store);
    assert_eq!(code, 0, "who-calls failed; stderr={err}\nstdout={out}");
    assert!(
        out.contains("(no callers)"),
        "defined but uncalled C# method must report an empty caller set; got: {out:?}"
    );
}

#[test]
fn graph_grid_csharp_callees_lists_cross_file_target() {
    let (repo, store) = index_fixture("callees");
    let (code, out, err) = run(&["callees", "caller"], &repo, &store);
    assert_eq!(code, 0, "callees failed; stderr={err}\nstdout={out}");
    assert!(
        out.contains("helper") && out.contains("src/Helpers.cs:"),
        "callees caller must list the cross-file helper definition; got: {out:?}"
    );
    assert!(
        !out.contains("(no callees)"),
        "caller has a helper call, so callees must be non-empty; got: {out:?}"
    );
}

#[test]
fn graph_grid_csharp_find_usages_covers_call_and_import() {
    let (repo, store) = index_fixture("usages-call-import");
    // The alias import targets the HelperTools class, while `new HelperTools()`
    // targets its constructor Method. Name aggregation must expose both kinds.
    let (code, out, err) = run(&["find-usages", "HelperTools"], &repo, &store);
    assert_eq!(code, 0, "find-usages failed; stderr={err}\nstdout={out}");
    assert!(
        out.contains("CALLS") && out.contains("caller"),
        "find-usages HelperTools must include the constructor call; got: {out:?}"
    );
    assert!(
        out.contains("IMPORTS") && out.contains("src/Main.cs"),
        "find-usages HelperTools must include Main.cs's alias import; got: {out:?}"
    );
}

#[test]
fn graph_grid_csharp_find_usages_type_reference() {
    let (repo, store) = index_fixture("usages-type-ref");
    let (code, out, err) = run(&["find-usages", "Payload"], &repo, &store);
    assert_eq!(code, 0, "find-usages failed; stderr={err}\nstdout={out}");
    assert!(
        out.contains("USAGE") && out.contains("caller"),
        "Payload's cross-file return type in caller must resolve as a type-reference USAGE; got: {out:?}"
    );
}

#[test]
fn graph_grid_csharp_impact_transitive_reaches_caller() {
    let (repo, store) = index_fixture("impact");
    let (code, value, err) = run_json(
        &["impact", "helper", "--edge", "CALLS", "--json"],
        &repo,
        &store,
    );
    assert_eq!(code, 0, "impact failed; stderr={err}; json={value}");
    let hits = value["hits"].as_array().expect("impact hits array");
    assert!(
        hits.iter().any(|hit| {
            hit["qualified_name"]
                .as_str()
                .is_some_and(|q| q.contains("caller"))
                && hit["hops"] == 1
        }),
        "impact helper must reach caller at hop 1; got: {value}"
    );
    assert!(
        hits.iter().any(|hit| {
            hit["qualified_name"]
                .as_str()
                .is_some_and(|q| q.contains("entry"))
                && hit["hops"] == 2
        }),
        "impact helper must transitively reach entry at hop 2; got: {value}"
    );
}

#[test]
fn graph_grid_csharp_search_symbols_finds_all_definitions() {
    let (repo, store) = index_fixture("symbols");
    let expected = [
        ("MainFlow", "Class", "src/Main.cs"),
        ("caller", "Method", "src/Main.cs"),
        ("entry", "Method", "src/Main.cs"),
        ("uncalled", "Method", "src/Main.cs"),
        ("HelperCode", "Enum", "src/Helpers.cs"),
        ("Seed", "Variable", "src/Helpers.cs"),
        ("HelperTools", "Class", "src/Helpers.cs"),
        ("HelperTools", "Method", "src/Helpers.cs"),
        ("helper", "Method", "src/Helpers.cs"),
        ("Payload", "Class", "src/Types.cs"),
    ];

    for (name, label, file) in expected {
        let (code, value, err) = run_json(&["search-symbols", name, "--json"], &repo, &store);
        assert_eq!(
            code, 0,
            "search-symbols {name} failed; stderr={err}; json={value}"
        );
        let hits = value["hits"].as_array().expect("search-symbols hits array");
        assert!(
            hits.iter().any(|hit| hit["name"] == name
                && hit["label"] == label
                && hit["file_path"] == file),
            "missing C# definition {label} {name} in {file}; got: {value}"
        );
    }
}

#[test]
fn graph_grid_csharp_brief_shows_definition_with_callers() {
    let (repo, store) = index_fixture("brief");
    let (code, out, err) = run(&["brief", "helper"], &repo, &store);
    assert_eq!(code, 0, "brief failed; stderr={err}\nstdout={out}");
    assert!(
        out.contains("public static Payload helper(int value)")
            && out.contains("src/Helpers.cs:"),
        "brief helper must include its C# definition source; got: {out:?}"
    );
    assert!(
        out.contains("-- CALLERS") && out.contains("caller") && out.contains("src/Main.cs:"),
        "brief helper must include its cross-file caller; got: {out:?}"
    );
}

#[test]
fn graph_grid_csharp_path_connects_caller_to_helper() {
    let (repo, store) = index_fixture("path");
    let (code, value, err) = run_json(
        &[
            "path", "--from", "caller", "--to", "helper", "--json",
        ],
        &repo,
        &store,
    );
    assert_eq!(code, 0, "path failed; stderr={err}; json={value}");
    assert_eq!(value["path_found"], true, "CALLS path missing: {value}");
    assert_eq!(value["hops"], 1, "caller -> helper is one hop: {value}");
    let steps = value["steps"].as_array().expect("path steps array");
    assert_eq!(steps.len(), 2, "one-hop path has two nodes: {value}");
    assert_eq!(steps[0]["name"], "caller", "wrong path start: {value}");
    assert_eq!(steps[1]["name"], "helper", "wrong path target: {value}");
    assert_eq!(
        steps[1]["file_path"], "src/Helpers.cs",
        "helper path endpoint must be cross-file: {value}"
    );
}

#[test]
fn graph_grid_csharp_graph_survives_reindex() {
    let (repo, store) = index_fixture("reindex");
    let (stats_code, stats_before, stats_err) = run(&["stats"], &repo, &store);
    assert_eq!(
        stats_code, 0,
        "stats before reindex failed; stderr={stats_err}"
    );
    let calls_before = nav_hit_signature(&["callees", "caller", "--json"], &repo, &store);
    let refs_before = nav_hit_signature(
        &["find-usages", "HelperTools", "--json"],
        &repo,
        &store,
    );

    // Required second index run over unchanged source. Incremental reporting is
    // allowed to say zero files changed; the graph itself must remain intact.
    let _ = index_existing(&repo, &store);

    let (stats_code, stats_after, stats_err) = run(&["stats"], &repo, &store);
    assert_eq!(
        stats_code, 0,
        "stats after reindex failed; stderr={stats_err}"
    );
    let calls_after = nav_hit_signature(&["callees", "caller", "--json"], &repo, &store);
    let refs_after = nav_hit_signature(
        &["find-usages", "HelperTools", "--json"],
        &repo,
        &store,
    );

    assert_eq!(
        stats_before, stats_after,
        "second C# index must preserve all node/edge counts"
    );
    assert_eq!(
        calls_before, calls_after,
        "second C# index must preserve CALLS endpoints"
    );
    assert_eq!(
        refs_before, refs_after,
        "second C# index must preserve CALLS/IMPORTS reference endpoints"
    );
    assert!(
        !calls_after.is_empty() && refs_after.len() >= 2,
        "reindex comparison must cover real CALLS and IMPORTS edges"
    );
}

#[test]
fn graph_grid_csharp_stale_edit_detected() {
    let (repo, store) = index_fixture("stale");

    // Establish that the indexed generation really contains the edge whose
    // stale escape this cell guards against.
    let (baseline_code, baseline, baseline_err) =
        run_json(&["who-calls", "helper", "--json"], &repo, &store);
    assert_eq!(
        baseline_code, 0,
        "baseline who-calls failed; stderr={baseline_err}; json={baseline}"
    );
    assert_eq!(baseline["total_exact"], 1, "baseline edge missing: {baseline}");

    // Remove the helper call after indexing. The implementation may report
    // drift or heal before answering, but it must never return the old caller.
    std::fs::write(
        repo.join("src/Main.cs"),
        r#"using Payload = Fixture.Types.Payload;

namespace Fixture.App
{
    public static class MainFlow
    {
        public static Payload caller()
        {
            return new Payload(99);
        }

        public static Payload entry()
        {
            return caller();
        }

        public static void uncalled()
        {
        }
    }
}
"#,
    )
    .expect("edit Main.cs after index");

    let (code, out, err) = run_with_env(
        &["who-calls", "helper", "--json"],
        &repo,
        &store,
        &[("GREPPY_AUTO_REINDEX", "0")],
    );
    assert!(err.is_empty(), "JSON freshness result belongs on stdout");
    let value: serde_json::Value = serde_json::from_str(&out)
        .unwrap_or_else(|e| panic!("invalid freshness JSON: {e}; stdout={out:?}"));
    assert!(
        value["hits"].as_array().is_some_and(Vec::is_empty),
        "stale caller edges must never escape: {value}"
    );
    match code {
        75 => {
            assert_eq!(value["status"], "skipped_stale_index", "{value}");
            assert_eq!(value["fresh"], false, "{value}");
            assert!(
                matches!(
                    value["freshness"]["state"].as_str(),
                    Some("drift" | "refreshing")
                ),
                "stale refusal must identify drift/refresh: {value}"
            );
        }
        0 => {
            assert_eq!(value["fresh"], true, "healed result must be fresh: {value}");
            assert_eq!(
                value["total_exact"], 0,
                "healed result must reflect removal of helper call: {value}"
            );
        }
        other => panic!("freshness query returned unexpected exit {other}: {value}"),
    }
}

#[test]
fn graph_grid_csharp_declarative_or_edge_case() {
    let (repo, store) = index_fixture("enum-constant-usage");
    // C#-specific edge case: an enum member is a named constant declaration,
    // and a qualified `HelperCode.Seed` read in another file is a value usage.
    let (code, out, err) = run(&["find-usages", "Seed"], &repo, &store);
    assert_eq!(code, 0, "find-usages Seed failed; stderr={err}\nstdout={out}");
    assert!(
        out.contains("USAGE") && out.contains("caller") && out.contains("src/Main.cs:"),
        "qualified cross-file use of C# enum constant Seed must resolve to caller; got: {out:?}"
    );
    assert!(
        !out.contains("(no usages)"),
        "HelperCode.Seed is read cross-file and must not look unused; got: {out:?}"
    );
}
